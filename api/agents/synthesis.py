from __future__ import annotations
import json
from core.context import SharedContext, AgentID, AgentMessage
from agents.base import BaseAgent


SYSTEM_PROMPT = """You are a Synthesis Agent. Merge all agent outputs into a single coherent final answer.

Your responsibilities:
1. Combine information from all agents
2. Resolve contradictions — pick the more confident/cited claim and explain why
3. Build a provenance map: for each sentence in your answer, note which agent and chunk it came from
4. Never surface contradictions to the user — resolve them internally

Respond ONLY with valid JSON:
{
  "final_answer": "complete, well-written answer",
  "provenance_map": {
    "sentence_1": "rag_agent + chunk doc2",
    "sentence_2": "decomposition_agent reasoning"
  },
  "resolved_contradictions": [
    {
      "contradiction": "describe what conflicted",
      "resolution": "how you resolved it",
      "chosen_claim": "the claim you kept"
    }
  ],
  "confidence": 0.88
}"""


class SynthesisAgent(BaseAgent):
    agent_id = AgentID.SYNTHESIS

    def run(self, context: SharedContext) -> SharedContext:
        self.logger.agent_start(self.agent_id, input_text=context.original_query)

        # Gather all agent messages
        rag_msgs      = [m for m in context.messages if m.from_agent == AgentID.RAG]
        critique_msgs = [m for m in context.messages if m.from_agent == AgentID.CRITIQUE]
        decomp_msgs   = [m for m in context.messages if m.from_agent == AgentID.DECOMPOSITION]

        flagged_claims = [c for c in context.claims if c.flagged]
        good_claims    = [c for c in context.claims if not c.flagged]

        prompt = f"""Query: {context.original_query}

RAG Agent findings:
{chr(10).join([m.content for m in rag_msgs]) or "No RAG output"}

Decomposition reasoning:
{chr(10).join([m.content for m in decomp_msgs]) or "No decomposition output"}

Critique summary:
{chr(10).join([m.content for m in critique_msgs]) or "No critique"}

Verified claims ({len(good_claims)}):
{chr(10).join([f"- {c.text} (confidence: {c.confidence:.2f})" for c in good_claims]) or "None"}

Flagged claims to resolve ({len(flagged_claims)}):
{chr(10).join([f"- {c.text} | Reason: {c.flag_reason}" for c in flagged_claims]) or "None"}

Synthesize a final answer resolving all contradictions."""

        if not self.check_and_record(prompt):
            # Budget exceeded — simple concatenation
            best = rag_msgs[0].content if rag_msgs else "Unable to generate answer."
            context.final_answer = best
            context.provenance_map = {"all": "rag_agent"}
            return context

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]

        try:
            raw = self.chat(messages, max_tokens=2000, temperature=0.2)
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)

            context.final_answer   = parsed.get("final_answer", "")
            context.provenance_map = parsed.get("provenance_map", {})

            context.messages.append(AgentMessage(
                from_agent=self.agent_id,
                content=context.final_answer,
                token_count=self.budget.get_usage(self.agent_id)["used"],
                metadata={
                    "resolved_contradictions": parsed.get("resolved_contradictions", []),
                    "confidence":              parsed.get("confidence", 0.0),
                    "provenance_map":          context.provenance_map,
                },
            ))

            self.logger.agent_end(
                self.agent_id,
                input_text=prompt,
                output_text=context.final_answer,
                token_count=self.budget.get_usage(self.agent_id)["used"],
            )

        except (json.JSONDecodeError, Exception) as e:
            context.final_answer = (
                rag_msgs[0].content if rag_msgs else f"Synthesis failed: {e}"
            )
            context.messages.append(AgentMessage(
                from_agent=self.agent_id,
                content=f"Synthesis error: {e}",
            ))

        return context