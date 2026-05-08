import os, sys
sys.path.insert(0, '.')
os.environ.setdefault("GROQ_API_KEY", "your-key-here")  # replace with real key

from core.context import SharedContext
from core.budget import ContextBudgetManager
from core.logger import AgentLogger

job_id = "test-job-001"
budget = ContextBudgetManager(job_id)
logger = AgentLogger(job_id)

ctx = SharedContext(job_id=job_id, original_query="What is retrieval augmented generation?")

print("Testing Decomposition Agent...")
from agents.decomposition import DecompositionAgent
agent = DecompositionAgent(budget, logger)
ctx = agent.run(ctx)
print(f"  Sub-tasks created: {len(ctx.sub_tasks)}")
for t in ctx.sub_tasks:
    print(f"    - [{t.task_type}] {t.description[:60]}...")

print("\nTesting RAG Agent...")
from agents.rag import RAGAgent
agent = RAGAgent(budget, logger)
ctx = agent.run(ctx)
print(f"  Claims extracted: {len(ctx.claims)}")
print(f"  Messages: {len(ctx.messages)}")

print("\nTesting Critique Agent...")
from agents.critique import CritiqueAgent
agent = CritiqueAgent(budget, logger)
ctx = agent.run(ctx)
flagged = [c for c in ctx.claims if c.flagged]
print(f"  Flagged claims: {len(flagged)}/{len(ctx.claims)}")

print("\nTesting Synthesis Agent...")
from agents.synthesis import SynthesisAgent
agent = SynthesisAgent(budget, logger)
ctx = agent.run(ctx)
print(f"  Final answer: {ctx.final_answer[:100] if ctx.final_answer else 'NONE'}...")

print("\nBudget summary:")
for agent_id, usage in budget.get_all_usage().items():
    print(f"  {agent_id}: {usage['used']}/{usage['budget']} tokens")

print("\nAll agents OK!")