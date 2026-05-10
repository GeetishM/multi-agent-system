from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db, PromptRewrite

router = APIRouter()


class ReviewRequest(BaseModel):
    action:      str   # "approve" or "reject"
    reviewed_by: str = "human"
    notes:       str = ""


@router.post("/rewrite/{rewrite_id}")
async def review_rewrite(
    rewrite_id: str,
    request: ReviewRequest,
    db: AsyncSession = Depends(get_db),
):
    """Approve or reject a pending prompt rewrite proposed by the meta agent."""
    if request.action not in ("approve", "reject"):
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "INVALID_ACTION",
                "message":    "action must be 'approve' or 'reject'",
                "job_id":     None,
            },
        )

    result = await db.execute(
        select(PromptRewrite).where(PromptRewrite.id == rewrite_id)
    )
    rewrite = result.scalar_one_or_none()

    if not rewrite:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "REWRITE_NOT_FOUND",
                "message":    f"No pending rewrite with id {rewrite_id}",
                "job_id":     None,
            },
        )

    if rewrite.status != "pending":
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "ALREADY_REVIEWED",
                "message":    f"Rewrite already {rewrite.status}",
                "job_id":     None,
            },
        )

    rewrite.status      = request.action + "d"  # approved / rejected
    rewrite.reviewed_at = datetime.utcnow()
    rewrite.reviewed_by = request.reviewed_by
    await db.commit()

    return {
        "rewrite_id":  rewrite_id,
        "status":      rewrite.status,
        "agent_id":    rewrite.agent_id,
        "dimension":   rewrite.dimension,
        "reviewed_at": rewrite.reviewed_at.isoformat(),
        "message":     f"Rewrite {rewrite.status}. "
                       + ("Run /api/v1/reeval to test it." if rewrite.status == "approved" else ""),
    }


@router.get("/rewrite/pending")
async def list_pending_rewrites(db: AsyncSession = Depends(get_db)):
    """List all pending prompt rewrites awaiting human review."""
    result = await db.execute(
        select(PromptRewrite).where(PromptRewrite.status == "pending")
    )
    rewrites = result.scalars().all()
    return {
        "pending_count": len(rewrites),
        "rewrites": [
            {
                "rewrite_id":  r.id,
                "agent_id":    r.agent_id,
                "dimension":   r.dimension,
                "root_cause":  r.justification[:200] if r.justification else "",
                "created_at":  r.created_at.isoformat(),
            }
            for r in rewrites
        ],
    }