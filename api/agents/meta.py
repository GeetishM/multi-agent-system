from __future__ import annotations
import json
import difflib
from typing import Dict, List, Optional
from core.context import SharedContext, AgentID
from agents.base import BaseAgent


SYSTEM_PROMPT = """You are a Meta-Prompt Improvement Agent. Analyze evaluation failures and propose better prompts.

Given:
- The agent that performed worst
- The scoring dimension where it failed
- Example failure cases

Your job: rewrite the agent's system prompt to fix the failures.

Rules:
1. Diagnose the root cause of failure first
2. Make targeted changes — don't rewrite everything
3. Preserve the original prompt's structure
4. Explain every change with a specific justification

Respond ONLY with valid JSON:
{
  "target_agent": "agent_id",
  "target_dimension": "which scoring dimension",
  "root_cause": "why the prompt failed",
  "proposed_prompt": "the full rewritten prompt",
  "changes": [
    {
      "type": "addition|deletion|modification",
      "original": "original text or null",
      "proposed": "new text or null",
      "justification": "why this change fixes the failure"
    }
  ],
  "expected_improvement": "what metric should improve and by how much"
}"""


class MetaAgent(BaseAgent):
    agent_id = AgentID.META

    def run(self, context: SharedContext) -> SharedContext:
        return context

    def propose_rewrite(
        self,
        eval_results: List[Dict],
        agent_prompts: Dict[str, str],
    ) -> Optional[Dict]:
        """
        Analyze eval failures and propose a prompt rewrite.

        Args:
            eval_results: List of EvalCaseResult dicts from the DB
            agent_prompts: Dict mapping agent_id → current system prompt

        Returns:
            Proposal dict or None if no improvement needed
        """
        self.logger.agent_start(
            self.agent_id,
            input_text=f"Analyzing {len(eval_results)} eval results"
        )

        # Find worst-performing agent + dimension
        worst = self._find_worst(eval_results)
        if not worst:
            return None

        agent_id, dimension, failures = worst
        current_prompt = agent_prompts.get(agent_id, "")

        failure_summary = "\n".join([
            f"- Query: {f.get('query','')[:100]}\n  Score: {f.get(f'score_{dimension}', 'N/A')}\n  Justification: {f.get(f'just_{dimension}','')[:200]}"
            for f in failures[:3]  # top 3 failures
        ])

        prompt = f"""Analyze these evaluation failures for the '{agent_id}' agent on dimension '{dimension}':

Failure cases:
{failure_summary}

Current system prompt for {agent_id}:
{current_prompt[:1000]}

Propose a better prompt that fixes these failures."""

        if not self.check_and_record(prompt):
            return None

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]

        try:
            raw = self.chat(messages, max_tokens=2000, temperature=0.3)
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)

            # Build structured diff
            original_lines  = current_prompt.splitlines(keepends=True)
            proposed_lines  = parsed.get("proposed_prompt", "").splitlines(keepends=True)
            diff = list(difflib.unified_diff(
                original_lines, proposed_lines,
                fromfile=f"{agent_id}_original",
                tofile=f"{agent_id}_proposed",
                n=3,
            ))

            proposal = {
                "agent_id":           agent_id,
                "dimension":          dimension,
                "root_cause":         parsed.get("root_cause", ""),
                "original_prompt":    current_prompt,
                "proposed_prompt":    parsed.get("proposed_prompt", ""),
                "diff":               "".join(diff),
                "changes":            parsed.get("changes", []),
                "expected_improvement": parsed.get("expected_improvement", ""),
                "failure_count":      len(failures),
            }

            self.logger.agent_end(
                self.agent_id,
                input_text=prompt[:200],
                output_text=parsed.get("root_cause", ""),
                token_count=self.budget.get_usage(self.agent_id)["used"],
            )

            return proposal

        except (json.JSONDecodeError, Exception) as e:
            return {"error": str(e), "agent_id": agent_id, "dimension": dimension}

    def _find_worst(
        self, eval_results: List[Dict]
    ) -> Optional[tuple]:
        """Find the agent+dimension combination with the lowest average score."""
        dimensions = [
            "correctness", "citation", "contradiction",
            "tool_efficiency", "budget_compliance", "critique_agreement"
        ]

        # Map dimension → agent responsible
        dim_to_agent = {
            "correctness":        "rag",
            "citation":           "rag",
            "contradiction":      "synthesis",
            "tool_efficiency":    "orchestrator",
            "budget_compliance":  "orchestrator",
            "critique_agreement": "critique",
        }

        scores_by_dim: Dict[str, List[float]] = {d: [] for d in dimensions}
        failures_by_dim: Dict[str, List[Dict]] = {d: [] for d in dimensions}

        for result in eval_results:
            for dim in dimensions:
                score = result.get(f"score_{dim}")
                if score is not None:
                    scores_by_dim[dim].append(float(score))
                    if float(score) < 0.6:
                        failures_by_dim[dim].append(result)

        # Find worst dimension
        worst_dim  = None
        worst_avg  = 1.0
        for dim, scores in scores_by_dim.items():
            if scores:
                avg = sum(scores) / len(scores)
                if avg < worst_avg:
                    worst_avg  = avg
                    worst_dim  = dim

        if worst_dim is None:
            return None

        agent_id = dim_to_agent.get(worst_dim, "orchestrator")
        failures = failures_by_dim.get(worst_dim, [])

        return agent_id, worst_dim, failures