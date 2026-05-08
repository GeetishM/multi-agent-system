"""
Structured Logger
- Consistent schema: timestamp, agent_id, event_type,
  input_hash, output_hash, latency, token_count, violations
- Writes to both console and DB
- Every event queryable by job_id
"""
from __future__ import annotations
import hashlib
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional
from ulid import ULID
import structlog

# ── Configure structlog ───────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if os.getenv("APP_ENV") == "development"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, os.getenv("LOG_LEVEL", "INFO"))
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)


def get_logger(name: str):
    return structlog.get_logger(name)


def _hash(text: Optional[str]) -> Optional[str]:
    """SHA-256 short hash for deduplication and diffing."""
    if not text:
        return None
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class AgentLogger:
    """
    Per-job structured logger. Writes events that match the required schema:
    timestamp | agent_id | event_type | input_hash | output_hash |
    latency_ms | token_count | policy_violations
    """

    def __init__(self, job_id: str, db_session=None):
        self.job_id = job_id
        self.db = db_session
        self._log = get_logger("agent")
        self._timers: Dict[str, float] = {}

    def _base_event(
        self,
        agent_id: str,
        event_type: str,
        input_text: Optional[str] = None,
        output_text: Optional[str] = None,
        latency_ms: Optional[float] = None,
        token_count: Optional[int] = None,
        policy_violation: bool = False,
        violation_detail: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        return {
            "log_id":           str(ULID()),
            "job_id":           self.job_id,
            "agent_id":         agent_id,
            "event_type":       event_type,
            "input_hash":       _hash(input_text),
            "output_hash":      _hash(output_text),
            "latency_ms":       latency_ms,
            "token_count":      token_count,
            "policy_violation": policy_violation,
            "violation_detail": violation_detail,
            "metadata":         metadata or {},
            "timestamp":        datetime.utcnow().isoformat(),
        }

    def agent_start(self, agent_id: str, input_text: str, metadata: dict = {}):
        self._timers[agent_id] = time.time()
        event = self._base_event(
            agent_id=agent_id,
            event_type="agent_start",
            input_text=input_text,
            metadata=metadata,
        )
        self._log.info("agent_start", **event)
        self._persist(event)
        return event

    def agent_end(
        self,
        agent_id: str,
        input_text: str,
        output_text: str,
        token_count: int = 0,
        metadata: dict = {},
    ):
        start = self._timers.pop(agent_id, time.time())
        latency = round((time.time() - start) * 1000, 2)
        event = self._base_event(
            agent_id=agent_id,
            event_type="agent_end",
            input_text=input_text,
            output_text=output_text,
            latency_ms=latency,
            token_count=token_count,
            metadata=metadata,
        )
        self._log.info("agent_end", **event)
        self._persist(event)
        return event

    def tool_call(
        self,
        agent_id: str,
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
        latency_ms: float,
        accepted: bool,
        retry_number: int = 0,
        error: Optional[str] = None,
    ):
        import json
        event = self._base_event(
            agent_id=agent_id,
            event_type="tool_call",
            input_text=json.dumps(tool_input, default=str),
            output_text=json.dumps(tool_output, default=str),
            latency_ms=latency_ms,
            metadata={
                "tool_name":    tool_name,
                "accepted":     accepted,
                "retry_number": retry_number,
                "error":        error,
            },
        )
        self._log.info("tool_call", **event)
        self._persist(event)
        return event

    def budget_check(
        self,
        agent_id: str,
        fits: bool,
        tokens_needed: int,
        remaining: int,
    ):
        event = self._base_event(
            agent_id=agent_id,
            event_type="budget_check",
            metadata={
                "fits":          fits,
                "tokens_needed": tokens_needed,
                "remaining":     remaining,
            },
        )
        self._log.debug("budget_check", **event)
        return event

    def policy_violation(
        self,
        agent_id: str,
        detail: str,
        token_count: Optional[int] = None,
    ):
        event = self._base_event(
            agent_id=agent_id,
            event_type="policy_violation",
            policy_violation=True,
            violation_detail=detail,
            token_count=token_count,
        )
        self._log.warning("policy_violation", **event)
        self._persist(event, is_violation=True)
        return event

    def orchestrator_decision(
        self,
        decision: str,
        justification: str,
        next_agent: Optional[str] = None,
    ):
        event = self._base_event(
            agent_id="orchestrator",
            event_type="routing_decision",
            input_text=decision,
            metadata={
                "justification": justification,
                "next_agent":    next_agent,
            },
        )
        self._log.info("routing_decision", **event)
        self._persist(event)
        return event

    def _persist(self, event: dict, is_violation: bool = False):
        """
        Write event to DB asynchronously.
        If no db session available (e.g. during testing), skip silently.
        """
        if self.db is None:
            return
        try:
            from core.database import AgentLog
            log_entry = AgentLog(
                id=event["log_id"],
                job_id=event["job_id"],
                agent_id=event["agent_id"],
                event_type=event["event_type"],
                input_hash=event.get("input_hash"),
                output_hash=event.get("output_hash"),
                latency_ms=event.get("latency_ms"),
                token_count=event.get("token_count"),
                policy_violation=is_violation,
                violation_detail=event.get("violation_detail"),
                metadata_json=event.get("metadata"),
                timestamp=datetime.utcnow(),
            )
            # Note: caller must commit the session
            self.db.add(log_entry)
        except Exception as e:
            self._log.error("logger_persist_failed", error=str(e))


# ── Execution Trace Builder ───────────────────────────────────────────────────
class ExecutionTrace:
    """
    Reconstructs the exact sequence of agent decisions, tool calls,
    and handoffs for a given job_id. Used by the /trace endpoint.
    """

    @staticmethod
    async def build(job_id: str, db: Any) -> dict:
        from sqlalchemy import select, asc
        from core.database import AgentLog, ToolCallLog, Job

        # Fetch job
        job_result = await db.execute(
            select(Job).where(Job.id == job_id)
        )
        job = job_result.scalar_one_or_none()
        if not job:
            return {"error": f"Job {job_id} not found"}

        # Fetch all agent logs in order
        logs_result = await db.execute(
            select(AgentLog)
            .where(AgentLog.job_id == job_id)
            .order_by(asc(AgentLog.timestamp))
        )
        logs = logs_result.scalars().all()

        # Fetch all tool call logs in order
        tools_result = await db.execute(
            select(ToolCallLog)
            .where(ToolCallLog.job_id == job_id)
            .order_by(asc(ToolCallLog.timestamp))
        )
        tools = tools_result.scalars().all()

        return {
            "job_id":       job_id,
            "query":        job.query,
            "status":       job.status,
            "final_answer": job.final_answer,
            "created_at":   job.created_at.isoformat() if job.created_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "sequence": [
                {
                    "log_id":           l.id,
                    "agent_id":         l.agent_id,
                    "event_type":       l.event_type,
                    "input_hash":       l.input_hash,
                    "output_hash":      l.output_hash,
                    "latency_ms":       l.latency_ms,
                    "token_count":      l.token_count,
                    "policy_violation": l.policy_violation,
                    "violation_detail": l.violation_detail,
                    "metadata":         l.metadata_json,
                    "timestamp":        l.timestamp.isoformat(),
                }
                for l in logs
            ],
            "tool_calls": [
                {
                    "call_id":      t.id,
                    "agent_id":     t.agent_id,
                    "tool_name":    t.tool_name,
                    "input":        t.input_json,
                    "output":       t.output_json,
                    "latency_ms":   t.latency_ms,
                    "accepted":     t.accepted,
                    "retry_number": t.retry_number,
                    "error":        t.error,
                    "timestamp":    t.timestamp.isoformat(),
                }
                for t in tools
            ],
            "total_events": len(logs),
            "total_tool_calls": len(tools),
            "policy_violations": sum(1 for l in logs if l.policy_violation),
        }