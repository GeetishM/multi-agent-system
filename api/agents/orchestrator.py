from __future__ import annotations
import json
import os
import time
from typing import Any, Callable, Dict, Generator, List, Optional
from groq import Groq
from opentelemetry import context
from core.context import SharedContext, AgentID, TaskStatus
from core.budget import ContextBudgetManager
from core.logger import AgentLogger
from agents.base import BaseAgent
from agents.decomposition import DecompositionAgent
from agents.rag import RAGAgent
from agents.critique import CritiqueAgent
from agents.synthesis import SynthesisAgent
from agents.compression import CompressionAgent


ROUTING_PROMPT = """You are a Master Orchestrator. Given a query and current pipeline state, decide the next action.

Available actions:
- "decompose"   : Break query into sub-tasks (ONLY if not in completed_actions)
- "retrieve"    : Run RAG agent to gather information (ONLY if not in completed_actions)
- "critique"    : Review current claims (ONLY if not in completed_actions)
- "synthesize"  : Merge all outputs into final answer (ONLY if not in completed_actions)
- "done"        : Pipeline is complete

Rules:
1. CRITICAL: NEVER choose an action already listed in completed_actions
2. Follow this order: decompose → retrieve → critique → synthesize → done
3. Choose the first action from remaining_actions
4. done is valid only when synthesize is in completed_actions

Respond ONLY with valid JSON:
{
  "action": "decompose|retrieve|critique|synthesize|done",
  "justification": "why you chose this action"
}"""


class Orchestrator:
    """
    Not a BaseAgent subclass — the orchestrator owns and drives all agents.
    It does not run inside the agent pipeline itself.
    """

    MAX_STEPS = 8  # prevent infinite loops

    def __init__(self, job_id: str, stream_callback: Optional[Callable] = None):
        self.job_id          = job_id
        self.stream_callback = stream_callback  # called with SSE events
        self.budget          = ContextBudgetManager(job_id)
        self.logger          = AgentLogger(job_id)
        self.llm             = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model           = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        # Instantiate all agents
        self.agents = {
            AgentID.DECOMPOSITION: DecompositionAgent(self.budget, self.logger, self.llm),
            AgentID.RAG:           RAGAgent(self.budget, self.logger, self.llm),
            AgentID.CRITIQUE:      CritiqueAgent(self.budget, self.logger, self.llm),
            AgentID.SYNTHESIS:     SynthesisAgent(self.budget, self.logger, self.llm),
            AgentID.COMPRESSION:   CompressionAgent(self.budget, self.logger, self.llm),
        }

    def _emit(self, event_type: str, data: Dict):
        """Send SSE event to client if streaming is enabled."""
        if self.stream_callback:
            self.stream_callback({
                "event": event_type,
                "data":  data,
            })

    def _decide_next_action(self, context: SharedContext, step: int) -> Dict:
        """
        Ask the LLM orchestrator what to do next.
        Logs the decision with justification.
        """
        completed_actions = []
        if context.sub_tasks:
            completed_actions.append("decompose")
        if any(m.from_agent == AgentID.RAG for m in context.messages):
            completed_actions.append("retrieve")
        if any(m.from_agent == AgentID.CRITIQUE for m in context.messages):
            completed_actions.append("critique")
        if any(m.from_agent == AgentID.SYNTHESIS for m in context.messages):
            completed_actions.append("synthesize")
        if context.final_answer:
            completed_actions.append("done")

        state_summary = {
            "step":               step,
            "completed_actions":  completed_actions,   # ← KEY FIX
            "remaining_actions":  [a for a in ["decompose","retrieve","critique","synthesize","done"] if a not in completed_actions],
            "sub_tasks":          len(context.sub_tasks),
            "claims":             len(context.claims),
            "flagged_claims":     sum(1 for c in context.claims if c.flagged),
            "has_final_answer":   bool(context.final_answer),
        }   

        prompt = f"""Query: {context.original_query}

Current pipeline state:
{json.dumps(state_summary, indent=2)}

What should the pipeline do next?"""

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ROUTING_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=256,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            decision = json.loads(raw)
        except Exception as e:
            # Fallback: deterministic routing if LLM fails
            decision = self._fallback_routing(context, step)

        self.logger.orchestrator_decision(
            decision=decision.get("action", "unknown"),
            justification=decision.get("justification", ""),
            next_agent=decision.get("action"),
        )

        self._emit("routing_decision", {
            "step":          step,
            "action":        decision.get("action"),
            "justification": decision.get("justification"),
        })

        return decision

    def _fallback_routing(self, context: SharedContext, step: int) -> Dict:
        """Deterministic fallback routing when LLM orchestrator fails."""
        if step == 0:
            return {"action": "decompose", "justification": "Always decompose first (fallback)"}
        if not any(m.from_agent == AgentID.RAG for m in context.messages):
            return {"action": "retrieve",  "justification": "Need RAG output (fallback)"}
        if not any(m.from_agent == AgentID.CRITIQUE for m in context.messages):
            return {"action": "critique",  "justification": "Need critique before synthesis (fallback)"}
        if not context.final_answer:
            return {"action": "synthesize","justification": "Need final answer (fallback)"}
        return {"action": "done",          "justification": "Pipeline complete (fallback)"}

    def _check_compression_needed(self, context: SharedContext):
        """Check if any agent needs compression before next step."""
        for agent_id_str in ["rag", "synthesis", "critique"]:
            if self.budget.needs_compression(agent_id_str, threshold=0.85):
                self._emit("compression_triggered", {
                    "agent_id": agent_id_str,
                    "usage":    self.budget.get_usage(agent_id_str),
                })
                # Compress the longest message from that agent
                agent_msgs = [
                    m for m in context.messages
                    if m.from_agent.value == agent_id_str
                ]
                if agent_msgs:
                    longest = max(agent_msgs, key=lambda m: len(m.content))
                    comp_agent = self.agents[AgentID.COMPRESSION]
                    longest.content = comp_agent.compress_text(
                        longest.content,
                        label=f"{agent_id_str}_output"
                    )

    def _run_agent(self, agent_id: AgentID, context: SharedContext) -> SharedContext:
        """Run a single agent with streaming updates."""
        self._emit("agent_start", {
            "agent_id":       agent_id.value,
            "budget_remaining": self.budget.get_remaining(agent_id.value),
        })

        agent   = self.agents[agent_id]
        context = agent.run(context)

        self._emit("agent_end", {
            "agent_id":  agent_id.value,
            "usage":     self.budget.get_usage(agent_id.value),
            "violations": len(self.budget.get_all_violations()),
        })

        return context

    def run(self, query: str) -> SharedContext:
        """
        Main pipeline execution.
        Dynamically routes through agents based on LLM decisions.
        """
        context = SharedContext(
            job_id=self.job_id,
            original_query=query,
            status=TaskStatus.RUNNING,
        )

        self._emit("pipeline_start", {
            "job_id": self.job_id,
            "query":  query,
        })

        rag_retries = 0

        for step in range(self.MAX_STEPS):
            # Check if compression needed before deciding next step
            self._check_compression_needed(context)

            # Emit current budget to client
            self._emit("budget_update", self.budget.get_all_usage())

            # Ask orchestrator what to do next
            decision = self._decide_next_action(context, step)
            action   = decision.get("action", "done")

            if action == "done":
                break

            elif action == "decompose":
                context = self._run_agent(AgentID.DECOMPOSITION, context)

            elif action == "retrieve":
                # Handle RAG retries with modified query
                refined = decision.get("refined_query")
                if refined and rag_retries < int(os.getenv("MAX_RETRIES", 2)):
                    context.original_query = refined
                    rag_retries += 1
                    self._emit("rag_retry", {
                        "retry":         rag_retries,
                        "refined_query": refined,
                    })
                context = self._run_agent(AgentID.RAG, context)

            elif action == "critique":
                context = self._run_agent(AgentID.CRITIQUE, context)

            elif action == "synthesize":
                context = self._run_agent(AgentID.SYNTHESIS, context)

            else:
                # Unknown action — log and break
                self.logger.policy_violation(
                    "orchestrator",
                    f"Unknown action '{action}' at step {step}"
                )
                break

        # Mark complete
        context.status = TaskStatus.DONE
        context.budget_used = {
            k: v["used"] for k, v in self.budget.get_all_usage().items()
        }

        # Add any policy violations to context
        for v in self.budget.get_all_violations():
            context.policy_violations.append(v.detail)

        self._emit("pipeline_end", {
            "job_id":      self.job_id,
            "final_answer": context.final_answer,
            "budget":       self.budget.get_all_usage(),
            "violations":   context.policy_violations,
        })

        return context