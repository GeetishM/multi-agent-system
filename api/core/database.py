"""
Database models and initialization.
All eval runs, jobs, logs, prompt rewrites stored here.
"""
from __future__ import annotations
import os
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, Float, Boolean,
    Text, DateTime, JSON, create_engine, event
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:////app/data/multi_agent.db")
SYNC_DATABASE_URL = DATABASE_URL.replace("+aiosqlite", "")


class Base(DeclarativeBase):
    pass


# ── Jobs ──────────────────────────────────────────────────────────────────────
class Job(Base):
    __tablename__ = "jobs"

    id              = Column(String, primary_key=True)
    query           = Column(Text, nullable=False)
    status          = Column(String, default="pending")   # pending/running/done/failed
    final_answer    = Column(Text, nullable=True)
    context_json    = Column(JSON, nullable=True)          # full SharedContext snapshot
    created_at      = Column(DateTime, default=datetime.utcnow)
    completed_at    = Column(DateTime, nullable=True)
    error           = Column(Text, nullable=True)


# ── Agent Logs ────────────────────────────────────────────────────────────────
class AgentLog(Base):
    __tablename__ = "agent_logs"

    id              = Column(String, primary_key=True)
    job_id          = Column(String, nullable=False, index=True)
    agent_id        = Column(String, nullable=False)
    event_type      = Column(String, nullable=False)  # start/end/tool_call/budget_check/violation
    input_hash      = Column(String, nullable=True)
    output_hash     = Column(String, nullable=True)
    input_text      = Column(Text, nullable=True)
    output_text     = Column(Text, nullable=True)
    latency_ms      = Column(Float, nullable=True)
    token_count     = Column(Integer, nullable=True)
    policy_violation= Column(Boolean, default=False)
    violation_detail= Column(Text, nullable=True)
    metadata_json   = Column(JSON, nullable=True)
    timestamp       = Column(DateTime, default=datetime.utcnow)


# ── Tool Call Logs ────────────────────────────────────────────────────────────
class ToolCallLog(Base):
    __tablename__ = "tool_call_logs"

    id              = Column(String, primary_key=True)
    job_id          = Column(String, nullable=False, index=True)
    agent_id        = Column(String, nullable=False)
    tool_name       = Column(String, nullable=False)
    input_json      = Column(JSON, nullable=True)
    output_json     = Column(JSON, nullable=True)
    latency_ms      = Column(Float, nullable=True)
    accepted        = Column(Boolean, nullable=True)
    retry_number    = Column(Integer, default=0)
    retry_of        = Column(String, nullable=True)   # call_id of previous attempt
    error           = Column(Text, nullable=True)
    timestamp       = Column(DateTime, default=datetime.utcnow)


# ── Eval Runs ─────────────────────────────────────────────────────────────────
class EvalRun(Base):
    __tablename__ = "eval_runs"

    id              = Column(String, primary_key=True)
    triggered_by    = Column(String, default="manual")  # manual/auto/reeval
    total_cases     = Column(Integer, default=15)
    passed          = Column(Integer, default=0)
    failed          = Column(Integer, default=0)
    avg_correctness = Column(Float, nullable=True)
    avg_citation    = Column(Float, nullable=True)
    avg_contradiction_resolution = Column(Float, nullable=True)
    avg_tool_efficiency          = Column(Float, nullable=True)
    avg_budget_compliance        = Column(Float, nullable=True)
    avg_critique_agreement       = Column(Float, nullable=True)
    results_json    = Column(JSON, nullable=True)   # full per-case breakdown
    timestamp       = Column(DateTime, default=datetime.utcnow)


# ── Eval Case Results ─────────────────────────────────────────────────────────
class EvalCaseResult(Base):
    __tablename__ = "eval_case_results"

    id                      = Column(String, primary_key=True)
    eval_run_id             = Column(String, nullable=False, index=True)
    case_id                 = Column(String, nullable=False)
    case_category           = Column(String, nullable=False)  # baseline/ambiguous/adversarial
    query                   = Column(Text, nullable=False)
    final_answer            = Column(Text, nullable=True)
    expected_answer         = Column(Text, nullable=True)

    # 6 scoring dimensions (0.0 - 1.0 each)
    score_correctness       = Column(Float, nullable=True)
    score_citation          = Column(Float, nullable=True)
    score_contradiction     = Column(Float, nullable=True)
    score_tool_efficiency   = Column(Float, nullable=True)
    score_budget_compliance = Column(Float, nullable=True)
    score_critique_agreement= Column(Float, nullable=True)

    # Justifications (required per spec)
    just_correctness        = Column(Text, nullable=True)
    just_citation           = Column(Text, nullable=True)
    just_contradiction      = Column(Text, nullable=True)
    just_tool_efficiency    = Column(Text, nullable=True)
    just_budget_compliance  = Column(Text, nullable=True)
    just_critique_agreement = Column(Text, nullable=True)

    # Full reproducibility snapshots
    prompts_sent_json       = Column(JSON, nullable=True)
    tool_calls_json         = Column(JSON, nullable=True)
    agent_outputs_json      = Column(JSON, nullable=True)

    passed                  = Column(Boolean, default=False)
    timestamp               = Column(DateTime, default=datetime.utcnow)


# ── Prompt Rewrites ───────────────────────────────────────────────────────────
class PromptRewrite(Base):
    __tablename__ = "prompt_rewrites"

    id              = Column(String, primary_key=True)
    eval_run_id     = Column(String, nullable=False)
    agent_id        = Column(String, nullable=False)
    dimension       = Column(String, nullable=False)   # which scoring dimension was worst
    original_prompt = Column(Text, nullable=False)
    proposed_prompt = Column(Text, nullable=False)
    diff_json       = Column(JSON, nullable=True)       # structured diff
    justification   = Column(Text, nullable=False)
    status          = Column(String, default="pending") # pending/approved/rejected
    reviewed_at     = Column(DateTime, nullable=True)
    reviewed_by     = Column(String, nullable=True)

    # Performance delta after re-eval (filled after approval + re-eval)
    delta_json      = Column(JSON, nullable=True)
    reeval_run_id   = Column(String, nullable=True)

    created_at      = Column(DateTime, default=datetime.utcnow)


# ── Engine + Session ──────────────────────────────────────────────────────────
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db():
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields a DB session."""
    async with AsyncSessionLocal() as session:
        yield session