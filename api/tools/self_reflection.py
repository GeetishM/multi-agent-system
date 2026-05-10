from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

from tools.base import BaseTool


class SelfReflectionTool(BaseTool):
    name = "self_reflection"
    timeout_seconds = 30.0   # LLM call may take a moment

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def _validate_input(
        self,
        agent_id: str = "",
        previous_outputs: List[str] = None,
        current_claim: str = "",
        **kwargs
    ) -> Optional[str]:
        if not agent_id or not agent_id.strip():
            return "agent_id cannot be empty"
        if not previous_outputs:
            return None  # Will be caught as EMPTY_RESULTS in _execute
        if not isinstance(previous_outputs, list):
            return "previous_outputs must be a list of strings"
        return None

    def _execute(
        self,
        agent_id: str,
        previous_outputs: List[str],
        current_claim: str = "",
        **kwargs
    ) -> Dict[str, Any]:

        if not previous_outputs:
            # Failure contract: return empty → triggers EMPTY_RESULTS
            return []

        if self.llm_client and current_claim:
            return self._llm_reflect(agent_id, previous_outputs, current_claim)
        else:
            return self._rule_reflect(previous_outputs, current_claim)

    def _rule_reflect(
        self, previous_outputs: List[str], current_claim: str
    ) -> Dict[str, Any]:
        """
        Rule-based contradiction detection.
        Looks for simple numeric/boolean contradictions without LLM.
        """
        contradictions = []
        current_lower  = current_claim.lower()

        contradiction_pairs = [
            ("increase", "decrease"),
            ("higher",   "lower"),
            ("true",     "false"),
            ("yes",      "no"),
            ("always",   "never"),
            ("positive", "negative"),
            ("more",     "less"),
            ("faster",   "slower"),
        ]

        for i, prev in enumerate(previous_outputs):
            prev_lower = prev.lower()
            for word_a, word_b in contradiction_pairs:
                if word_a in current_lower and word_b in prev_lower:
                    contradictions.append({
                        "type":           "semantic_contradiction",
                        "current_claim":  current_claim[:200],
                        "conflicting_output_index": i,
                        "conflicting_text": prev[:200],
                        "pattern":        f"'{word_a}' vs '{word_b}'",
                        "confidence":     0.6,
                    })
                elif word_b in current_lower and word_a in prev_lower:
                    contradictions.append({
                        "type":           "semantic_contradiction",
                        "current_claim":  current_claim[:200],
                        "conflicting_output_index": i,
                        "conflicting_text": prev[:200],
                        "pattern":        f"'{word_b}' vs '{word_a}'",
                        "confidence":     0.6,
                    })

        return {
            "agent_id":          "self",
            "outputs_reviewed":  len(previous_outputs),
            "current_claim":     current_claim[:200],
            "contradictions":    contradictions,
            "contradiction_count": len(contradictions),
            "clean":             len(contradictions) == 0,
        }

    def _llm_reflect(
        self,
        agent_id: str,
        previous_outputs: List[str],
        current_claim: str,
    ) -> Dict[str, Any]:
        """Use LLM to detect contradictions in previous outputs."""

        combined_history = "\n\n---\n\n".join(
            [f"[Output {i+1}]: {o}" for i, o in enumerate(previous_outputs)]
        )

        prompt = f"""You are a contradiction detector. Review the agent's previous outputs and identify any contradictions with the current claim.

Previous outputs from agent '{agent_id}':
{combined_history}

Current claim to check:
{current_claim}

Respond ONLY with valid JSON in this exact format:
{{
  "contradictions": [
    {{
      "type": "factual|logical|numerical",
      "current_claim": "the part of current claim that contradicts",
      "conflicting_output_index": 0,
      "conflicting_text": "the part from previous output that contradicts",
      "explanation": "why these contradict",
      "confidence": 0.85
    }}
  ],
  "clean": true or false,
  "summary": "one sentence summary"
}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0,
            )
            import json
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)
            result["agent_id"]         = agent_id
            result["outputs_reviewed"] = len(previous_outputs)
            result["current_claim"]    = current_claim[:200]
            result["contradiction_count"] = len(result.get("contradictions", []))
            return result
        except Exception as e:
            # Fallback to rule-based
            result = self._rule_reflect(previous_outputs, current_claim)
            result["llm_error"] = str(e)
            return result