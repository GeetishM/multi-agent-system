from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from ulid import ULID

from eval.test_cases import TEST_CASES, get_case_by_id
from eval.scorer import score_case


class EvalHarness:

    def __init__(self, db_session=None):
        self.db = db_session

    def run_full_eval(
        self,
        eval_id: Optional[str] = None,
        case_ids: Optional[List[str]] = None,
        triggered_by: str = "manual",
    ) -> Dict:
        """
        Run eval on all 15 cases (or a subset if case_ids provided).
        Returns full results dict.
        """
        from agents.orchestrator import Orchestrator

        eval_id = eval_id or str(ULID())
        cases   = TEST_CASES if not case_ids else [
            get_case_by_id(cid) for cid in case_ids
        ]
        cases   = [c for c in cases if c]  # filter empty

        print(f"\n{'='*60}")
        print(f"EVAL RUN: {eval_id}")
        print(f"Cases: {len(cases)} | Triggered by: {triggered_by}")
        print(f"{'='*60}\n")

        results       = []
        all_tool_calls= []

        for i, case in enumerate(cases):
            print(f"[{i+1}/{len(cases)}] {case['category'].upper()} | {case['case_id']}")
            print(f"  Query: {case['query'][:80]}...")

            job_id = str(ULID())

            # Track tool calls for this case
            case_tool_calls = []

            def capture_tools(event: dict):
                if event.get("event") == "tool_call":
                    case_tool_calls.append(event.get("data", {}))

            try:
                orch    = Orchestrator(job_id=job_id, stream_callback=capture_tools)
                context = orch.run(case["query"])

                # Serialize context for scoring and storage
                ctx_dict = {
                    "final_answer":      context.final_answer,
                    "claims":            [
                        {
                            "text":             c.text,
                            "source_chunk_ids": c.source_chunk_ids,
                            "flagged":          c.flagged,
                            "confidence":       c.confidence,
                        }
                        for c in context.claims
                    ],
                    "messages":          [
                        {
                            "from_agent": m.from_agent.value,
                            "content":    m.content,
                            "metadata":   m.metadata,
                        }
                        for m in context.messages
                    ],
                    "budget_used":       context.budget_used,
                    "policy_violations": context.policy_violations,
                    "sub_tasks":         [
                        {"task_id": t.task_id, "description": t.description}
                        for t in context.sub_tasks
                    ],
                }

                # Score this case
                result = score_case(case, ctx_dict, case_tool_calls)
                result["job_id"]        = job_id
                result["final_answer"]  = context.final_answer
                result["context_snapshot"] = ctx_dict

                print(f"  Score: {result['avg_score']:.2f} | {'PASS' if result['passed'] else 'FAIL'}")
                print(f"  Correctness: {result['scores']['correctness']:.2f} | "
                      f"Citation: {result['scores']['citation']:.2f} | "
                      f"Budget: {result['scores']['budget_compliance']:.2f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                result = {
                    "case_id":  case["case_id"],
                    "category": case["category"],
                    "query":    case["query"],
                    "passed":   False,
                    "avg_score": 0.0,
                    "error":    str(e),
                    "scores": {
                        "correctness": 0.0, "citation": 0.0,
                        "contradiction": 0.0, "tool_efficiency": 0.0,
                        "budget_compliance": 0.0, "critique_agreement": 0.0,
                    },
                    "justifications": {k: f"Pipeline error: {e}" for k in [
                        "correctness","citation","contradiction",
                        "tool_efficiency","budget_compliance","critique_agreement"
                    ]},
                }

            results.append(result)
            all_tool_calls.extend(case_tool_calls)

        # Aggregate scores
        summary = self._aggregate(eval_id, results, triggered_by)

        # Save to DB
        if self.db:
            self._save_to_db(eval_id, summary, results, triggered_by)

        # Save to JSON file for reproducibility
        self._save_to_file(eval_id, summary, results)

        print(f"\n{'='*60}")
        print(f"EVAL COMPLETE | Avg score: {summary['avg_scores']['overall']:.2f}")
        print(f"Passed: {summary['passed']}/{summary['total']}")
        print(f"{'='*60}\n")

        return summary

    def _aggregate(self, eval_id: str, results: List[Dict], triggered_by: str) -> Dict:
        dims = ["correctness","citation","contradiction",
                "tool_efficiency","budget_compliance","critique_agreement"]

        avg_scores = {}
        for dim in dims:
            scores = [r["scores"][dim] for r in results if "scores" in r]
            avg_scores[dim] = round(sum(scores)/max(len(scores),1), 3)

        overall = round(sum(avg_scores.values())/len(dims), 3)
        avg_scores["overall"] = overall

        by_cat = {}
        for cat in ["baseline", "ambiguous", "adversarial"]:
            cat_results = [r for r in results if r.get("category") == cat]
            if cat_results:
                by_cat[cat] = {
                    "count":  len(cat_results),
                    "passed": sum(1 for r in cat_results if r.get("passed")),
                    "avg":    round(sum(r.get("avg_score",0) for r in cat_results)/len(cat_results), 3),
                }

        return {
            "eval_id":      eval_id,
            "triggered_by": triggered_by,
            "timestamp":    datetime.utcnow().isoformat(),
            "total":        len(results),
            "passed":       sum(1 for r in results if r.get("passed")),
            "failed":       sum(1 for r in results if not r.get("passed")),
            "avg_scores":   avg_scores,
            "by_category":  by_cat,
        }

    def _save_to_file(self, eval_id: str, summary: Dict, results: List[Dict]):
        """Save full eval results for reproducibility and diffing."""
        os.makedirs("eval_results", exist_ok=True)
        path = f"eval_results/eval_{eval_id}.json"
        with open(path, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2, default=str)
        print(f"Results saved to {path}")

    def _save_to_db(self, eval_id: str, summary: Dict, results: List[Dict], triggered_by: str):
        """Persist eval run to SQLite DB."""
        try:
            from core.database import EvalRun, EvalCaseResult
            from ulid import ULID

            run = EvalRun(
                id=eval_id,
                triggered_by=triggered_by,
                total_cases=summary["total"],
                passed=summary["passed"],
                failed=summary["failed"],
                avg_correctness=summary["avg_scores"]["correctness"],
                avg_citation=summary["avg_scores"]["citation"],
                avg_contradiction_resolution=summary["avg_scores"]["contradiction"],
                avg_tool_efficiency=summary["avg_scores"]["tool_efficiency"],
                avg_budget_compliance=summary["avg_scores"]["budget_compliance"],
                avg_critique_agreement=summary["avg_scores"]["critique_agreement"],
                results_json=summary,
                timestamp=datetime.utcnow(),
            )
            self.db.add(run)

            for r in results:
                case_result = EvalCaseResult(
                    id=str(ULID()),
                    eval_run_id=eval_id,
                    case_id=r["case_id"],
                    case_category=r["category"],
                    query=r["query"],
                    final_answer=r.get("final_answer",""),
                    expected_answer=get_case_by_id(r["case_id"]).get("expected_answer",""),
                    score_correctness=r["scores"]["correctness"],
                    score_citation=r["scores"]["citation"],
                    score_contradiction=r["scores"]["contradiction"],
                    score_tool_efficiency=r["scores"]["tool_efficiency"],
                    score_budget_compliance=r["scores"]["budget_compliance"],
                    score_critique_agreement=r["scores"]["critique_agreement"],
                    just_correctness=r["justifications"]["correctness"],
                    just_citation=r["justifications"]["citation"],
                    just_contradiction=r["justifications"]["contradiction"],
                    just_tool_efficiency=r["justifications"]["tool_efficiency"],
                    just_budget_compliance=r["justifications"]["budget_compliance"],
                    just_critique_agreement=r["justifications"]["critique_agreement"],
                    passed=r["passed"],
                    timestamp=datetime.utcnow(),
                )
                self.db.add(case_result)

            self.db.commit()
        except Exception as e:
            print(f"DB save failed: {e}")