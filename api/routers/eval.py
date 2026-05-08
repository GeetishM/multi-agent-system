"""GET /api/v1/eval — Latest eval run summary"""
from fastapi import APIRouter, Depends
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db, EvalRun, EvalCaseResult

router = APIRouter()


@router.get("/eval")
async def get_eval_summary(db: AsyncSession = Depends(get_db)):
    """Return latest eval run summary broken down by category and dimension."""
    result = await db.execute(
        select(EvalRun).order_by(desc(EvalRun.timestamp)).limit(1)
    )
    latest = result.scalar_one_or_none()

    if not latest:
        return {
            "error_code": "NO_EVAL_RUN",
            "message":    "No eval runs found. POST /api/v1/reeval to trigger one.",
        }

    # Get per-case breakdown
    cases_result = await db.execute(
        select(EvalCaseResult)
        .where(EvalCaseResult.eval_run_id == latest.id)
        .order_by(EvalCaseResult.case_category)
    )
    cases = cases_result.scalars().all()

    by_category = {"baseline": [], "ambiguous": [], "adversarial": []}
    for c in cases:
        by_category[c.case_category].append({
            "case_id":    c.case_id,
            "query":      c.query[:80],
            "passed":     c.passed,
            "scores": {
                "correctness":        c.score_correctness,
                "citation":           c.score_citation,
                "contradiction":      c.score_contradiction,
                "tool_efficiency":    c.score_tool_efficiency,
                "budget_compliance":  c.score_budget_compliance,
                "critique_agreement": c.score_critique_agreement,
            },
        })

    return {
        "eval_run_id":  latest.id,
        "timestamp":    latest.timestamp.isoformat(),
        "triggered_by": latest.triggered_by,
        "summary": {
            "total":  latest.total_cases,
            "passed": latest.passed,
            "failed": latest.failed,
        },
        "avg_scores": {
            "correctness":        latest.avg_correctness,
            "citation":           latest.avg_citation,
            "contradiction":      latest.avg_contradiction_resolution,
            "tool_efficiency":    latest.avg_tool_efficiency,
            "budget_compliance":  latest.avg_budget_compliance,
            "critique_agreement": latest.avg_critique_agreement,
        },
        "by_category": by_category,
    }


@router.post("/eval/run")
async def trigger_eval(db: AsyncSession = Depends(get_db)):
    """Trigger a full eval run (15 test cases) as background task."""
    from worker.tasks import run_eval_task
    from ulid import ULID
    eval_id = str(ULID())
    run_eval_task.apply_async(args=[eval_id], task_id=eval_id)
    return {"eval_id": eval_id, "status": "started", "message": "Eval running in background"}