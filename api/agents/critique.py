from __future__ import annotations
import json
from core.context import SharedContext, AgentID, AgentMessage, Claim
from agents.base import BaseAgent


SYSTEM_PROMPT = """You are a Critique Agent. Your job is to review claims made by other agents and score each one individually.

For each claim:
1. Assign a confidence score 0.0-1.0 (how confident you are the claim is correct)
2. Flag the claim if confidence < 0.6
3. If flagging, specify the exact span of text you disagree with and why
4. Do NOT flag the entire output — only specific problematic spans

Respond ONLY with valid JSON:
{
  "reviewed_claims": [
    {
      "original_text": "the exact claim text",
      "confidence": 0.85,
      "flagged": false,
      "flag_reason": null,
      "flagged_span": null,
      "verdict": "supported|unsupported|uncertain"
    }
  ],
  "overall_quality": 0.82,
  "summary": "one sentence critique summary",
  "critical_issues": ["list any critical factual errors"]
}"""


class CritiqueAgent(BaseAgent):
    agent_id = AgentID.CRITIQUE

    def run(self, context: SharedContext) -> SharedContext:
        self.logger.agent_start(self.agent_id, input_text=context.original_query)

        if not context.claims:
            context.messages.append(AgentMessage(
                from_agent=self.agent_id,
                content="No claims to critique.",
            ))
            return context

        claims_text = "\n".join([
            f"[Claim {i+1}] (from {c.source_agent}): {c.text}"
            for i, c in enumerate(context.claims)
        ])

        prompt = f"""Review these claims for the query: "{context.original_query}"

Claims to review:
{claims_text}

Score each claim individually."""

        if not self.check_and_record(prompt):
            # Budget exceeded — do shallow critique
            for claim in context.claims:
                claim.confidence = max(claim.confidence - 0.1, 0.0)
            context.messages.append(AgentMessage(
                from_agent=self.agent_id,
                content="Budget exceeded — shallow critique applied.",
            ))
            return context

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]

        try:
            raw = self.chat(messages, max_tokens=1500, temperature=0.1)
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)

            # Update claims in shared context with critique scores
            reviewed = parsed.get("reviewed_claims", [])
            for i, review in enumerate(reviewed):
                if i < len(context.claims):
                    context.claims[i].confidence  = review.get("confidence", context.claims[i].confidence)
                    context.claims[i].flagged      = review.get("flagged", False)
                    context.claims[i].flag_reason  = review.get("flag_reason")

            context.messages.append(AgentMessage(
                from_agent=self.agent_id,
                content=parsed.get("summary", "Critique complete."),
                token_count=self.budget.get_usage(self.agent_id)["used"],
                metadata={
                    "overall_quality": parsed.get("overall_quality", 0.0),
                    "critical_issues": parsed.get("critical_issues", []),
                    "flagged_count":   sum(1 for r in reviewed if r.get("flagged")),
                },
            ))

            self.logger.agent_end(
                self.agent_id,
                input_text=prompt,
                output_text=parsed.get("summary", ""),
                token_count=self.budget.get_usage(self.agent_id)["used"],
            )

        except (json.JSONDecodeError, Exception) as e:
            context.messages.append(AgentMessage(
                from_agent=self.agent_id,
                content=f"Critique failed: {e}",
            ))

        return context