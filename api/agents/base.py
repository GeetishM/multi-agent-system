from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from groq import Groq
from core.context import SharedContext, AgentID
from core.budget import ContextBudgetManager
from core.logger import AgentLogger

class BaseAgent(ABC):
    agent_id: AgentID

    def __init__(
        self,
        budget_manager: ContextBudgetManager,
        logger: AgentLogger,
        llm_client: Groq = None,
    ):
        self.budget  = budget_manager
        self.logger  = logger
        self.llm     = llm_client or Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def chat(self, messages: list, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        """Call Groq LLM and return text response."""
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def check_and_record(self, text: str) -> bool:
        """
        Check budget before adding text to context.
        Records usage if it fits. Logs violation if not.
        Returns True if fits.
        """
        fits, tokens, violation = self.budget.check_budget(
            self.agent_id, text
        )
        self.logger.budget_check(
            self.agent_id, fits, tokens,
            self.budget.get_remaining(self.agent_id)
        )
        if not fits:
            self.logger.policy_violation(
                self.agent_id,
                violation.detail,
                token_count=tokens,
            )
            return False
        self.budget.record_usage(self.agent_id, text)
        return True

    @abstractmethod
    def run(self, context: SharedContext) -> SharedContext:
        pass