"""
Compression Agent
Called when an agent exceeds 85% of its context budget.
Lossless for structured data (tool outputs, scores, citations).
Lossy only for conversational filler.
"""
from __future__ import annotations
import json
from core.context import SharedContext, AgentID, AgentMessage
from agents.base import BaseAgent


SYSTEM_PROMPT = """You are a Context Compression Agent. Compress the provided context to save tokens.

Rules (STRICTLY follow):
1. LOSSLESS (never remove): tool outputs, numeric scores, citations [chunk_id:X], JSON data, claim texts
2. LOSSY (safe to shorten): conversational filler, repeated explanations, verbose reasoning
3. Preserve all structured data exactly as-is
4. Return compressed text that is at most 40% of the original length
5. Mark compressed sections with [COMPRESSED]

Respond with the compressed text only — no JSON wrapper."""


class CompressionAgent(BaseAgent):
    agent_id = AgentID.COMPRESSION

    def run(self, context: SharedContext) -> SharedContext:
        """Not used directly — use compress_text() instead."""
        return context

    def compress_text(self, text: str, label: str = "context") -> str:
        """
        Compress a block of text.
        Called by orchestrator when an agent is near budget limit.
        """
        self.logger.agent_start(
            self.agent_id,
            input_text=f"Compressing {label} ({len(text)} chars)"
        )

        fits, _, _ = self.budget.check_budget(self.agent_id, text)
        if not fits:
            # Even compression agent is over budget — hard truncate
            truncated = text[:1000] + "\n[HARD TRUNCATED due to compression budget]"
            self.logger.policy_violation(
                self.agent_id,
                "Compression agent itself exceeded budget — hard truncated"
            )
            return truncated

        self.budget.record_usage(self.agent_id, text)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Compress this {label}:\n\n{text}"},
        ]

        try:
            compressed = self.chat(messages, max_tokens=800, temperature=0.1)
            self.logger.agent_end(
                self.agent_id,
                input_text=text[:100],
                output_text=compressed[:100],
                token_count=self.budget.get_usage(self.agent_id)["used"],
                metadata={
                    "original_length":   len(text),
                    "compressed_length": len(compressed),
                    "compression_ratio": round(len(compressed) / max(len(text), 1), 2),
                },
            )
            return compressed
        except Exception as e:
            return text[:1000] + f"\n[COMPRESSION FAILED: {e}]"