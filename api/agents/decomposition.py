"""
Decomposition Agent
Breaks ambiguous queries into typed sub-tasks with dependency graphs.
Dependent sub-tasks do NOT execute until dependencies resolve.
"""
from __future__ import annotations
import json
from typing import List
from core.context import SharedContext, AgentID, SubTask, TaskStatus, AgentMessage
from core.budget import ContextBudgetManager
from core.logger import AgentLogger
from agents.base import BaseAgent


SYSTEM_PROMPT = """You are a Query Decomposition Agent. Your job is to break down a user query into clear, typed sub-tasks with explicit dependency relationships.

Rules:
1. Each sub-task must have a type: one of [retrieval, calculation, comparison, summarization, analysis, code_execution, data_lookup]
2. If sub-task B needs the result of sub-task A, list A's task_id in B's dependencies
3. Sub-tasks with no dependencies can run in parallel
4. Be specific — vague sub-tasks like "research the topic" are not acceptable
5. Maximum 6 sub-tasks per query

Respond ONLY with valid JSON in this exact format:
{
  "sub_tasks": [
    {
      "task_id": "t1",
      "description": "specific description of what to do",
      "task_type": "retrieval",
      "dependencies": [],
      "assigned_to": "rag"
    },
    {
      "task_id": "t2",
      "description": "compare the results from t1 with database records",
      "task_type": "comparison",
      "dependencies": ["t1"],
      "assigned_to": "synthesis"
    }
  ],
  "reasoning": "why you decomposed it this way"
}

assigned_to must be one of: [rag, synthesis, critique, orchestrator]"""


class DecompositionAgent(BaseAgent):
    agent_id = AgentID.DECOMPOSITION

    def run(self, context: SharedContext) -> SharedContext:
        self.logger.agent_start(
            self.agent_id,
            input_text=context.original_query
        )

        prompt = f"Decompose this query into sub-tasks:\n\n{context.original_query}"

        if not self.check_and_record(prompt):
            # Budget exceeded — create a single generic task
            context.sub_tasks = [SubTask(
                task_id="t1",
                description=context.original_query,
                task_type="analysis",
                dependencies=[],
                assigned_to=AgentID.RAG,
                status=TaskStatus.PENDING,
            )]
            return context

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]

        try:
            raw = self.chat(messages, max_tokens=1024, temperature=0.2)
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)

            sub_tasks = []
            for t in parsed.get("sub_tasks", []):
                assigned_raw = t.get("assigned_to", "rag")
                try:
                    assigned = AgentID(assigned_raw)
                except ValueError:
                    assigned = AgentID.RAG

                sub_tasks.append(SubTask(
                    task_id=t.get("task_id", f"t{len(sub_tasks)+1}"),
                    description=t.get("description", ""),
                    task_type=t.get("task_type", "analysis"),
                    dependencies=t.get("dependencies", []),
                    assigned_to=assigned,
                    status=TaskStatus.PENDING,
                ))

            context.sub_tasks = sub_tasks
            reasoning = parsed.get("reasoning", "")

            msg = AgentMessage(
                from_agent=self.agent_id,
                content=f"Decomposed into {len(sub_tasks)} sub-tasks. Reasoning: {reasoning}",
                token_count=self.budget.get_usage(self.agent_id)["used"],
            )
            context.messages.append(msg)

            self.logger.agent_end(
                self.agent_id,
                input_text=prompt,
                output_text=raw,
                token_count=self.budget.get_usage(self.agent_id)["used"],
            )

        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: single task
            context.sub_tasks = [SubTask(
                task_id="t1",
                description=context.original_query,
                task_type="analysis",
                dependencies=[],
                assigned_to=AgentID.RAG,
                status=TaskStatus.PENDING,
            )]
            context.messages.append(AgentMessage(
                from_agent=self.agent_id,
                content=f"Decomposition failed ({e}), fell back to single task.",
            ))

        return context