import os, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

events = []

def capture(event):
    events.append(event)
    print(f"  [{event['event']}]", event['data'].get('action') or event['data'].get('agent_id') or '')

from agents.orchestrator import Orchestrator

print("Running orchestrator pipeline...")
orch = Orchestrator(job_id="orch-test-001", stream_callback=capture)
ctx  = orch.run("What is retrieval augmented generation and how does it work?")

print(f"\nFinal answer: {ctx.final_answer[:150] if ctx.final_answer else 'NONE'}...")
print(f"Total SSE events emitted: {len(events)}")
print(f"Pipeline status: {ctx.status}")