import os, pathlib, sys
sys.path.insert(0, '.')

print("=" * 50)
print("Testing all 4 tools...")
print("=" * 50)

# ── Tool 1: Web Search ────────────────────────────
from tools.web_search import WebSearchTool
t = WebSearchTool()
r = t.run(query='large language models', max_results=2)
print(f"[1] WebSearch    | success={r.success} | results={len(r.data)}")

# ── Tool 2: Code Sandbox ──────────────────────────
from tools.code_sandbox import CodeSandboxTool
t = CodeSandboxTool()
r = t.run(code='x = 2 + 2\nprint(x)')
print(f"[2] CodeSandbox  | success={r.success} | stdout={r.data.get('stdout','').strip()}")

# ── Tool 3: SQL Lookup ────────────────────────────
from tools.sql_lookup import SQLLookupTool, seed_sample_db
pathlib.Path('../data').mkdir(exist_ok=True)
seed_sample_db('../data/sample.db')
t = SQLLookupTool(db_path='../data/sample.db')
r = t.run(question='show most expensive products')
print(f"[3] SQLLookup    | success={r.success} | rows={r.data.get('row_count')}")

# ── Tool 4: Self Reflection ───────────────────────
from tools.self_reflection import SelfReflectionTool
t = SelfReflectionTool()
r = t.run(
    agent_id='rag',
    previous_outputs=['Prices will increase next year'],
    current_claim='Prices will decrease next year'
)
print(f"[4] SelfReflect  | success={r.success} | contradictions={r.data.get('contradiction_count')}")

print("=" * 50)
print("All tools tested!")