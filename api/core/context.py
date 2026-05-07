"""
Shared Context Object — all inter-agent communication passes through this.
Agents NEVER call each other directly.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from python_ulid import ULID


class AgentID(str, Enum):
    ORCHESTRATOR  = "orchestrator"
    DECOMPOSITION = "decomposition"
    RAG           = "rag"
    CRITIQUE      = "critique"
    SYNTHESIS     = "synthesis"
    COMPRESSION   = "compression"
    META          = "meta"


class TaskStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    DONE       = "done"
    FAILED     = "failed"
    SKIPPED    = "skipped"


class SubTask(BaseModel):
    task_id:      str = Field(default_factory=lambda: str(ULID()))
    description:  str
    task_type:    str                        # e.g. "retrieval", "calculation", "comparison"
    dependencies: List[str] = []            # list of task_ids this depends on
    status:       TaskStatus = TaskStatus.PENDING
    result:       Optional[Any] = None
    assigned_to:  Optional[AgentID] = None


class Claim(BaseModel):
    text:             str
    confidence:       float                 # 0.0 – 1.0
    source_agent:     AgentID
    source_chunk_ids: List[str] = []
    flagged:          bool = False
    flag_reason:      Optional[str] = None


class ToolCall(BaseModel):
    call_id:    str = Field(default_factory=lambda: str(ULID()))
    tool_name:  str
    input:      Dict[str, Any]
    output:     Optional[Any] = None
    latency_ms: Optional[float] = None
    accepted:   Optional[bool] = None
    retry_of:   Optional[str] = None        # call_id of previous attempt
    error:      Optional[str] = None


class AgentMessage(BaseModel):
    message_id:  str = Field(default_factory=lambda: str(ULID()))
    from_agent:  AgentID
    content:     str
    token_count: int = 0
    timestamp:   datetime = Field(default_factory=datetime.utcnow)
    metadata:    Dict[str, Any] = {}


class SharedContext(BaseModel):
    """
    The single source of truth passed between all agents.
    Orchestrator reads and writes this. Agents receive a copy
    and return an updated copy — never modify in place externally.
    """
    job_id:          str = Field(default_factory=lambda: str(ULID()))
    original_query:  str
    sub_tasks:       List[SubTask] = []
    messages:        List[AgentMessage] = []
    tool_calls:      List[ToolCall] = []
    claims:          List[Claim] = []
    final_answer:    Optional[str] = None
    provenance_map:  Dict[str, str] = {}    # sentence → source agent + chunk
    budget_used:     Dict[str, int] = {}    # agent_id → tokens used
    policy_violations: List[str] = []
    created_at:      datetime = Field(default_factory=datetime.utcnow)
    status:          TaskStatus = TaskStatus.PENDING