
from __future__ import annotations
import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import tiktoken

# ── Default budgets from env (tokens) ─────────────────────────────────────────
BUDGETS: Dict[str, int] = {
    "orchestrator":  int(os.getenv("ORCHESTRATOR_BUDGET",  4000)),
    "decomposition": int(os.getenv("DECOMPOSITION_BUDGET", 3000)),
    "rag":           int(os.getenv("RAG_BUDGET",           5000)),
    "critique":      int(os.getenv("CRITIQUE_BUDGET",      3000)),
    "synthesis":     int(os.getenv("SYNTHESIS_BUDGET",     4000)),
    "compression":   int(os.getenv("COMPRESSION_BUDGET",   2000)),
    "meta":          int(os.getenv("META_BUDGET",          3000)),
}

# Use cl100k_base tokenizer (same as GPT-4 / most modern LLMs)
_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in a string."""
    if not text:
        return 0
    return len(_ENCODER.encode(text))


@dataclass
class BudgetViolation:
    agent_id:   str
    job_id:     str
    attempted:  int   # tokens agent tried to add
    used:       int   # tokens already used
    budget:     int   # declared max
    detail:     str


@dataclass
class AgentBudgetState:
    agent_id:   str
    budget:     int
    used:       int = 0
    violations: List[BudgetViolation] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    @property
    def is_over_budget(self) -> bool:
        return self.used > self.budget


class ContextBudgetManager:
    """
    One instance per job. Tracks token usage per agent.
    Agents must call check_budget() before building context,
    then record_usage() after.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self._states: Dict[str, AgentBudgetState] = {}
        self._violations: List[BudgetViolation] = []

    def _get_state(self, agent_id: str) -> AgentBudgetState:
        if agent_id not in self._states:
            budget = BUDGETS.get(agent_id, 3000)
            self._states[agent_id] = AgentBudgetState(
                agent_id=agent_id,
                budget=budget
            )
        return self._states[agent_id]

    def check_budget(
        self,
        agent_id: str,
        text_to_add: str,
        raise_on_violation: bool = False
    ) -> tuple[bool, int, Optional[BudgetViolation]]:
        """
        Check if adding `text_to_add` would exceed the agent's budget.

        Returns:
            (fits, tokens_needed, violation_or_None)
        """
        state = self._get_state(agent_id)
        tokens_needed = count_tokens(text_to_add)
        fits = (state.used + tokens_needed) <= state.budget

        if not fits:
            violation = BudgetViolation(
                agent_id=agent_id,
                job_id=self.job_id,
                attempted=tokens_needed,
                used=state.used,
                budget=state.budget,
                detail=(
                    f"Agent '{agent_id}' tried to add {tokens_needed} tokens "
                    f"but only {state.remaining} remain "
                    f"(used {state.used}/{state.budget})"
                )
            )
            self._violations.append(violation)
            state.violations.append(violation)

            if raise_on_violation:
                raise BudgetExceededError(violation)

            return False, tokens_needed, violation

        return True, tokens_needed, None

    def record_usage(self, agent_id: str, text: str) -> int:
        """
        Record that an agent actually used this text in its context.
        Returns tokens consumed.
        """
        state = self._get_state(agent_id)
        tokens = count_tokens(text)
        state.used += tokens
        return tokens

    def get_remaining(self, agent_id: str) -> int:
        """How many tokens does this agent have left?"""
        return self._get_state(agent_id).remaining

    def get_usage(self, agent_id: str) -> dict:
        """Full usage summary for an agent."""
        state = self._get_state(agent_id)
        return {
            "agent_id":  agent_id,
            "budget":    state.budget,
            "used":      state.used,
            "remaining": state.remaining,
            "over_budget": state.is_over_budget,
            "violations": len(state.violations),
        }

    def get_all_usage(self) -> Dict[str, dict]:
        """Full usage summary for all agents in this job."""
        return {aid: self.get_usage(aid) for aid in self._states}

    def get_all_violations(self) -> List[BudgetViolation]:
        return list(self._violations)

    def needs_compression(self, agent_id: str, threshold: float = 0.85) -> bool:
        """
        Returns True if agent has used more than `threshold` of its budget.
        Orchestrator uses this to decide whether to invoke compression agent.
        """
        state = self._get_state(agent_id)
        if state.budget == 0:
            return False
        return (state.used / state.budget) >= threshold

    def snapshot(self) -> dict:
        """Serializable snapshot for logging/storage."""
        return {
            "job_id": self.job_id,
            "agents": self.get_all_usage(),
            "total_violations": len(self._violations),
            "violations": [
                {
                    "agent_id": v.agent_id,
                    "attempted": v.attempted,
                    "used": v.used,
                    "budget": v.budget,
                    "detail": v.detail,
                }
                for v in self._violations
            ],
        }


class BudgetExceededError(Exception):
    def __init__(self, violation: BudgetViolation):
        super().__init__(violation.detail)
        self.violation = violation