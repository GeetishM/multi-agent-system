"""
POST /api/v1/query  — Submit query, get SSE stream
"""
import json
import os
import asyncio
from datetime import datetime

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from ulid import ULID

from core.database import get_db, Job
from worker.tasks import run_pipeline_task

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


async def sse_generator(job_id: str, redis_url: str):
    """
    Subscribe to Redis channel for this job and forward events as SSE.
    """
    r = aioredis.from_url(redis_url)
    pubsub = r.pubsub()
    await pubsub.subscribe(f"job:{job_id}")

    yield f"data: {json.dumps({'event': 'job_created', 'data': {'job_id': job_id}})}\n\n"

    timeout = 120  # 2 min max
    start   = asyncio.get_event_loop().time()

    async for message in pubsub.listen():
        if asyncio.get_event_loop().time() - start > timeout:
            yield f"data: {json.dumps({'event': 'timeout', 'data': {'job_id': job_id}})}\n\n"
            break

        if message["type"] != "message":
            continue

        raw  = message["data"].decode("utf-8")
        data = json.loads(raw)

        yield f"data: {raw}\n\n"

        if data.get("event") in ("done", "error"):
            break

        await asyncio.sleep(0.01)

    await pubsub.unsubscribe(f"job:{job_id}")
    await r.aclose()


@router.post("/query")
async def submit_query(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit a query. Returns a streaming SSE response.
    Shows: current agent, tool calls in flight, budget remaining.
    """
    if not request.query.strip():
        return {"error": "EMPTY_QUERY", "message": "Query cannot be empty"}

    job_id = str(ULID())

    # Save job to DB
    job = Job(
        id=job_id,
        query=request.query,
        status="pending",
        created_at=datetime.utcnow(),
    )
    db.add(job)
    await db.commit()

    # Dispatch to Celery worker
    run_pipeline_task.apply_async(
        args=[job_id, request.query],
        task_id=job_id,
    )

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    return StreamingResponse(
        sse_generator(job_id, redis_url),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "X-Job-ID":          job_id,
        },
    )