from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ulid import ULID
from core.database import get_db, EvalCaseResult, PromptRewrite

router = APIRouter()


@router.post("/reeval")
async def trigger_reeval(db: AsyncSession = Depends(get_db)):
    """
    Re-run eval using latest approved prompts.
    Only runs on previously failed test cases.
    """
    # Find latest approved rewrite
    rewrite_result = await db.execute(
        select(PromptRewrite)
        .where(PromptRewrite.status == "approved")
        .order_by(PromptRewrite.reviewed_at.desc())
        .limit(1)
    )
    approved = rewrite_result.scalar_one_or_none()

    # Find previously failed cases
    failed_result = await db.execute(
        select(EvalCaseResult).where(EvalCaseResult.passed == False).limit(10)
    )
    failed_cases = failed_result.scalars().all()

    if not failed_cases:
        return {
            "message":     "No failed cases to re-evaluate.",
            "error_code":  "NO_FAILURES",
        }

    eval_id = str(ULID())
    from worker.tasks import run_reeval_task
    run_reeval_task.apply_async(
        args=[eval_id, [c.case_id for c in failed_cases], approved.id if approved else None],
        task_id=eval_id,
    )

    return {
        "eval_id":       eval_id,
        "status":        "started",
        "cases_to_rerun": len(failed_cases),
        "using_rewrite": approved.id if approved else "none (no approved rewrites)",
        "message":       "Re-eval running in background. Check /api/v1/eval for results.",
    }   