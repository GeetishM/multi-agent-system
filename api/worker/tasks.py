"""Celery background tasks."""
import os
import json
from datetime import datetime
from worker.celery_app import celery_app


@celery_app.task(bind=True, name="run_pipeline")
def run_pipeline_task(self, job_id: str, query: str):
    """
    Run the full multi-agent pipeline as a background task.
    Results stored in DB. SSE streaming handled separately via Redis pub/sub.
    """
    import redis
    from agents.orchestrator import Orchestrator

    r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    def stream_to_redis(event: dict):
        """Publish SSE events to Redis channel for the API to forward."""
        r.publish(f"job:{job_id}", json.dumps(event))

    try:
        orchestrator = Orchestrator(job_id=job_id, stream_callback=stream_to_redis)
        context      = orchestrator.run(query)

        # Publish completion
        r.publish(f"job:{job_id}", json.dumps({
            "event": "done",
            "data":  {
                "job_id":       job_id,
                "final_answer": context.final_answer,
                "status":       "completed",
            }
        }))

        return {
            "job_id":       job_id,
            "final_answer": context.final_answer,
            "status":       "completed",
        }

    except Exception as e:
        r.publish(f"job:{job_id}", json.dumps({
            "event": "error",
            "data":  {"job_id": job_id, "error": str(e)},
        }))
        raise