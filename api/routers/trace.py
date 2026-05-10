from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from core.logger import ExecutionTrace

router = APIRouter()


@router.get("/trace/{job_id}")
async def get_trace(job_id: str, db: AsyncSession = Depends(get_db)):
    """Return full execution trace for a completed job."""
    trace = await ExecutionTrace.build(job_id, db)
    if "error" in trace:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "JOB_NOT_FOUND",
                "message":    trace["error"],
                "job_id":     job_id,
            },
        )
    return trace