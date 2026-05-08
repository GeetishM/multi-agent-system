"""Celery background tasks — pipeline, eval, re-eval."""
import os, json
from datetime import datetime
from worker.celery_app import celery_app


@celery_app.task(bind=True, name="run_pipeline")
def run_pipeline_task(self, job_id: str, query: str):
    import redis
    from agents.orchestrator import Orchestrator

    r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    def stream_to_redis(event: dict):
        r.publish(f"job:{job_id}", json.dumps(event, default=str))

    try:
        orch    = Orchestrator(job_id=job_id, stream_callback=stream_to_redis)
        context = orch.run(query)

        # Save to DB
        _save_job(job_id, query, context)

        r.publish(f"job:{job_id}", json.dumps({
            "event": "done",
            "data":  {"job_id": job_id, "final_answer": context.final_answer, "status": "completed"}
        }))
        return {"job_id": job_id, "status": "completed"}

    except Exception as e:
        r.publish(f"job:{job_id}", json.dumps({
            "event": "error",
            "data":  {"job_id": job_id, "error": str(e)}
        }))
        _mark_job_failed(job_id, str(e))
        raise


@celery_app.task(bind=True, name="run_eval")
def run_eval_task(self, eval_id: str):
    """Run full 15-case eval in background."""
    from eval.harness import EvalHarness
    harness = EvalHarness()
    harness.run_full_eval(eval_id=eval_id, triggered_by="api")
    return {"eval_id": eval_id, "status": "completed"}


@celery_app.task(bind=True, name="run_reeval")
def run_reeval_task(self, eval_id: str, case_ids: list, rewrite_id: str = None):
    """Re-run eval on failed cases using approved prompt rewrite."""
    from eval.harness import EvalHarness

    # Apply approved rewrite if provided
    if rewrite_id:
        _apply_rewrite(rewrite_id)

    harness = EvalHarness()
    summary = harness.run_full_eval(
        eval_id=eval_id,
        case_ids=case_ids,
        triggered_by="reeval",
    )

    # Store performance delta if rewrite was applied
    if rewrite_id:
        _store_delta(rewrite_id, eval_id, summary)

    return {"eval_id": eval_id, "status": "completed", "cases": len(case_ids)}


@celery_app.task(bind=True, name="run_meta_agent")
def run_meta_agent_task(self, eval_run_id: str):
    """Run meta agent to propose prompt rewrites after eval."""
    import sqlite3, json
    from agents.meta import MetaAgent
    from agents.decomposition import SYSTEM_PROMPT as DECOMP_PROMPT
    from agents.rag import SYSTEM_PROMPT as RAG_PROMPT
    from agents.critique import SYSTEM_PROMPT as CRITIQUE_PROMPT
    from agents.synthesis import SYSTEM_PROMPT as SYNTH_PROMPT
    from core.budget import ContextBudgetManager
    from core.logger import AgentLogger
    from ulid import ULID

    db_path = os.getenv("SQLITE_SAMPLE_DB", "/app/data/multi_agent.db").replace(
        "sqlite:////", ""
    ).replace("sqlite:///", "")

    # Load eval results from JSON file
    import glob
    files = glob.glob(f"eval_results/eval_{eval_run_id}.json")
    if not files:
        return {"error": "Eval results not found"}

    with open(files[0]) as f:
        data = json.load(f)

    results = data.get("results", [])

    agent_prompts = {
        "rag":           RAG_PROMPT,
        "critique":      CRITIQUE_PROMPT,
        "synthesis":     SYNTH_PROMPT,
        "decomposition": DECOMP_PROMPT,
    }

    job_id = str(ULID())
    budget = ContextBudgetManager(job_id)
    logger = AgentLogger(job_id)
    meta   = MetaAgent(budget, logger)

    proposal = meta.propose_rewrite(results, agent_prompts)
    if not proposal or "error" in proposal:
        return {"status": "no_rewrite_needed"}

    # Save proposal to DB
    _save_rewrite(eval_run_id, proposal)
    return {"status": "rewrite_proposed", "agent_id": proposal.get("agent_id")}


# ── DB helpers ────────────────────────────────────────────────────────────────

def _save_job(job_id: str, query: str, context):
    """Synchronously save completed job to SQLite."""
    try:
        import sqlite3, json
        db_path = _get_db_path()
        conn    = sqlite3.connect(db_path)
        conn.execute("""
            INSERT OR REPLACE INTO jobs
            (id, query, status, final_answer, context_json, completed_at)
            VALUES (?, ?, 'done', ?, ?, ?)
        """, (
            job_id, query,
            context.final_answer,
            json.dumps({"budget_used": context.budget_used}, default=str),
            datetime.utcnow().isoformat(),
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Job save failed: {e}")


def _mark_job_failed(job_id: str, error: str):
    try:
        import sqlite3
        db_path = _get_db_path()
        conn    = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE jobs SET status='failed', error=? WHERE id=?",
            (error, job_id)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _save_rewrite(eval_run_id: str, proposal: dict):
    try:
        import sqlite3, json
        from ulid import ULID
        db_path = _get_db_path()
        conn    = sqlite3.connect(db_path)
        conn.execute("""
            INSERT OR IGNORE INTO prompt_rewrites
            (id, eval_run_id, agent_id, dimension, original_prompt,
             proposed_prompt, diff_json, justification, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """, (
            str(ULID()),
            eval_run_id,
            proposal.get("agent_id",""),
            proposal.get("dimension",""),
            proposal.get("original_prompt",""),
            proposal.get("proposed_prompt",""),
            json.dumps(proposal.get("changes",[])),
            proposal.get("root_cause",""),
            datetime.utcnow().isoformat(),
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Rewrite save failed: {e}")


def _apply_rewrite(rewrite_id: str):
    """Mark rewrite as applied (prompt update is runtime-only for now)."""
    try:
        import sqlite3
        db_path = _get_db_path()
        conn    = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE prompt_rewrites SET status='applied' WHERE id=?",
            (rewrite_id,)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _store_delta(rewrite_id: str, eval_id: str, summary: dict):
    try:
        import sqlite3, json
        db_path = _get_db_path()
        conn    = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE prompt_rewrites SET delta_json=?, reeval_run_id=? WHERE id=?",
            (json.dumps(summary.get("avg_scores",{})), eval_id, rewrite_id)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _get_db_path() -> str:
    url = os.getenv("DATABASE_URL", "sqlite:////app/data/multi_agent.db")
    return url.replace("sqlite+aiosqlite:////","").replace("sqlite:////","").replace("sqlite:///","")