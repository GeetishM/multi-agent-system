# Multi-Agent LLM System

> Production-grade multi-agent system with dynamic orchestration, retrieval-augmented generation, adversarial evaluation, and a self-improving prompt loop.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3%2070B-orange)](https://groq.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Quick Start

```bash
git clone <your-repo-url>
cd multi-agent-system

# Copy and configure environment
cp .env.example .env
# Open .env and set: GROQ_API_KEY=your_key_here

# Start all services
docker compose up
```

Visit **http://localhost:8000/docs** for the interactive API documentation.  
Visit **http://localhost:8081** for the Redis Commander UI.
Visit **http://localhost:8000** for the Real-Time Agent Streaming UI.

> Get your free Groq API key at [console.groq.com/keys](https://console.groq.com/keys)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client                               │
│              (SSE Stream / REST API)                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  FastAPI Server (:8000)                      │
│         5 REST endpoints + SSE streaming                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                Redis Queue (:6379)                           │
│            Celery broker + result backend                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Celery Worker                               │
│           Async pipeline execution                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Orchestrator                              │
│         LLM-driven dynamic routing engine                    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │             Shared Context Object                   │   │
│  │          (Pydantic schema, passed between agents)   │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Context Budget Manager                    │   │
│  │     (tiktoken — tracks tokens per agent per job)    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│   Agent Pipeline (dynamic order, decided at runtime):       │
│   1. Decomposition Agent  →  typed sub-tasks + dep graph    │
│   2. RAG Agent            →  multi-hop ChromaDB + web       │
│   3. Critique Agent       →  per-claim confidence scoring   │
│   4. Synthesis Agent      →  contradiction resolution       │
│   5. Compression Agent    →  on-demand context reduction    │
│   6. Meta Agent           →  post-eval prompt improvement   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Tool Registry                             │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│   │  Web Search  │  │ Code Sandbox │  │  SQL Lookup  │    │
│   │    (stub)    │  │  (Restricted │  │  (NL → SQL)  │    │
│   │  + relevance │  │   Python)    │  │   → SQLite)  │    │
│   └──────────────┘  └──────────────┘  └──────────────┘    │
│                      ┌──────────────┐                       │
│                      │Self Reflection│                      │
│                      │(contradiction │                      │
│                      │  detector)   │                       │
│                      └──────────────┘                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   Storage Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │    SQLite    │  │   ChromaDB   │  │    Redis     │    │
│  │  (jobs, logs,│  │  (vector     │  │  (pub/sub    │    │
│  │  eval, rewrite│  │   store)    │  │   + queue)   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Services

| Service | Port | Purpose |
|---|---|---|
| `mas_api` | 8000 | FastAPI server, SSE streaming, all endpoints |
| `mas_worker` | — | Celery background worker for async pipeline jobs |
| `mas_redis` | 6379 | Message broker, result backend, SSE pub/sub |
| `mas_redis_ui` | 8081 | Redis Commander — visual queue/log browser |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/query` | Submit a query; receive real-time SSE stream showing agent activity, tool calls, and budget |
| `GET` | `/api/v1/trace/{job_id}` | Full execution trace: exact agent sequence, tool calls, handoffs, timings |
| `GET` | `/api/v1/eval` | Latest eval run summary broken down by category and scoring dimension |
| `POST` | `/api/v1/eval/run` | Trigger a full 15-case eval run in the background |
| `POST` | `/api/v1/rewrite/{rewrite_id}` | Submit human approval or rejection for a pending prompt rewrite |
| `GET` | `/api/v1/rewrite/pending` | List all pending prompt rewrites awaiting human review |
| `POST` | `/api/v1/reeval` | Re-run eval on previously failed cases using latest approved prompt |
| `GET` | `/health` | Health check |

All error responses include a machine-readable `error_code`, a human-readable `message`, and `job_id` where applicable.

---

## Agents

### Orchestrator
The master controller. Uses an LLM to dynamically decide at runtime which sub-agent to invoke next, in what order, and with what context. Routing decisions are never hardcoded — they are made via structured reasoning and logged with justification. Handles RAG retries with refined queries (up to 2 retries), compression triggers at 85% budget, and policy violation logging.

**Decision boundaries:** The orchestrator calls the LLM with the current pipeline state (`completed_actions`, `remaining_actions`, `claims`, `budget_status`) and receives a JSON routing decision with justification. It falls back to deterministic routing if the LLM call fails.

### Decomposition Agent
Breaks ambiguous queries into typed sub-tasks with explicit dependency relationships. Sub-tasks have types: `retrieval`, `calculation`, `comparison`, `summarization`, `analysis`, `code_execution`, `data_lookup`. Dependent sub-tasks do not execute until their dependencies resolve. Produces a maximum of 6 sub-tasks per query.

**Decision boundaries:** If the input is short and unambiguous, produces a single sub-task. If JSON parsing fails, falls back to a single generic analysis task to avoid blocking the pipeline.

### RAG Agent
Performs multi-hop retrieval across at least two sources before forming an answer — ChromaDB (local vector store) and the web search tool. Every factual claim is tagged with `[chunk_id:X]` citations. Single-hop retrieval is explicitly not sufficient. Uses cosine similarity scoring in ChromaDB and keyword-overlap relevance scoring for web results.

**Decision boundaries:** If fewer than 2 chunks are retrieved, the agent reports insufficient context rather than hallucinating. If the LLM returns malformed JSON, the raw response is stored in context metadata for debugging.

### Critique Agent
Reviews the output of every other agent at the claim level — never at the whole-output level. Assigns a confidence score (0.0–1.0) per claim and flags the specific text span it disagrees with, along with a reason. Claims below 0.6 confidence are flagged. If the budget is exceeded, applies a shallow critique (reduces all confidence scores by 0.1) rather than skipping.

**Decision boundaries:** Only runs if there are claims in the shared context. Produces structured JSON per claim. If JSON parsing fails, preserves original confidence scores rather than zeroing them out.

### Synthesis Agent
Merges outputs from all sub-agents, resolves contradictions flagged by the critique agent, and produces a final answer with a provenance map linking each sentence to its source agent and chunk. Contradictions are resolved internally — they are never surfaced to the user. If the context budget is exceeded, falls back to returning the best RAG output directly.

**Decision boundaries:** Prioritizes higher-confidence claims when resolving contradictions. Documents every resolution decision in `resolved_contradictions` metadata.

### Compression Agent
Called by the orchestrator when any agent reaches 85% of its token budget. Applies lossy compression only to conversational filler. Structured data — tool outputs, numeric scores, citations, JSON — is always preserved losslessly. Returns compressed text marked with `[COMPRESSED]`. If the compression agent itself exceeds its own budget, hard-truncates at 1000 characters and logs a policy violation.

**Decision boundaries:** Only compresses; never summarizes structured data. Compression ratio is logged for observability.

### Meta Agent
Runs after each eval cycle. Reads the failure cases from the eval results, identifies the worst-performing agent-dimension combination, and proposes a rewritten system prompt with a structured diff and justification. The proposed rewrite is stored as `pending` in the database. It is never automatically applied — a human must approve it via the API.

**Decision boundaries:** Analyses only the top 3 failure cases to stay within context budget. If no dimension scores below 0.6 on average, returns `no_rewrite_needed`. Does not propose rewrites for the orchestrator routing logic.

---

## Tools

### Web Search
Returns structured results with source URLs, relevance scores, and chunk IDs. Currently a stub backed by a curated knowledge base with keyword-overlap scoring. In production, replace with Tavily or SerpAPI.

- **Timeout:** returns `failure_reason: TIMEOUT`
- **Empty query:** returns `failure_reason: MALFORMED`
- **No results:** returns `failure_reason: EMPTY_RESULTS`

### Code Sandbox
Executes Python snippets using RestrictedPython. Returns `stdout`, `stderr`, and `exit_code`. Blocks dangerous imports (`os`, `sys`, `subprocess`, etc.). Runtime errors are valid results (not tool failures) — they are returned with `exit_code: 1`.

- **Empty code:** returns `failure_reason: MALFORMED`
- **Blocked imports:** returns `failure_reason: MALFORMED` at validation
- **Runtime exceptions:** returned in `stderr` with `exit_code: 1`

### SQL Lookup
Converts natural language questions to SQLite SQL — via rule-based patterns locally, or via the Groq LLM when a client is provided. Queries a local SQLite database seeded with sample products, sales, and customer data. Returns column names, rows, and row count.

- **Empty question:** returns `failure_reason: MALFORMED`
- **SQL execution error:** raises `RuntimeError` caught by base tool as `EXECUTION`
- **No rows returned:** returns `failure_reason: EMPTY_RESULTS`

### Self Reflection
An agent calls this tool to re-read its own previous outputs and detect contradictions with a new claim. Uses rule-based pattern matching (e.g. "increase" vs "decrease") locally, or calls the LLM for semantic contradiction detection when a client is available.

- **No previous outputs:** returns `failure_reason: EMPTY_RESULTS`
- **Empty agent_id:** returns `failure_reason: MALFORMED`
- **No contradictions found:** returns `success: True` with empty contradictions list (this is a valid result)

---

## Evaluation Pipeline

### Test Cases (15 total)

| Category | Count | Description |
|---|---|---|
| Baseline | 5 | Straightforward queries with known correct answers |
| Ambiguous | 5 | Underspecified inputs designed to test decomposition quality |
| Adversarial | 5 | Prompt injections, factually confident wrong premises, contradiction traps |

### Scoring Dimensions (6 per case)

| Dimension | What it measures | Scoring method |
|---|---|---|
| Correctness | How accurate is the final answer vs expected | LLM judge |
| Citation | Are inline chunk citations present and accurate | Rule-based + coverage |
| Contradiction Resolution | Were flagged contradictions resolved before reaching the user | Structural check |
| Tool Efficiency | Penalises unnecessary tool calls and retries | Rule-based |
| Budget Compliance | Did agents stay within their token budgets | Policy violation count |
| Critique Agreement | Does the final answer align with critique agent's assessments | Claim exclusion rate |

Every dimension produces a numeric score (0.0–1.0) and a written justification string. Every eval run is stored in the database with full reproducibility: exact prompts sent, tool calls made, outputs received, scores, and timestamps. Re-running the eval on the same inputs produces a diff-able JSON output so regressions are immediately visible.

### Running the Eval

```bash
# Full 15-case eval via API (runs in background)
curl -X POST http://localhost:8000/api/v1/eval/run

# Quick 3-case local test
cd api
python test_eval.py

# Check results
curl http://localhost:8000/api/v1/eval
```

---

## Self-Improving Prompt Loop

```
Eval Run
   │
   ▼
Meta Agent reads failure cases
   │
   ▼
Identifies worst agent + dimension
   │
   ▼
Proposes rewrite (stored as "pending")
   │
   ▼
Human reviews: POST /api/v1/rewrite/{id}
   {"action": "approve", "reviewed_by": "your_name"}
   │
   ▼
POST /api/v1/reeval
   │
   ▼
Re-runs only previously failed cases
   │
   ▼
Performance delta stored and queryable
```

Every proposed rewrite, every approval or rejection, and every performance delta is stored with timestamps and is queryable. The loop is fully auditable end to end.

---

## Context Budget Management

Each agent declares a maximum token budget. The `ContextBudgetManager` tracks consumption per agent per job using `tiktoken` (cl100k_base encoding).

| Agent | Default Budget |
|---|---|
| Orchestrator | 4,000 tokens |
| Decomposition | 3,000 tokens |
| RAG | 5,000 tokens |
| Critique | 3,000 tokens |
| Synthesis | 4,000 tokens |
| Compression | 2,000 tokens |
| Meta | 3,000 tokens |

Agents that attempt to exceed their budget are caught and logged as policy violations — they are never silently truncated. At 85% usage, the orchestrator triggers the compression agent before proceeding. All budgets are configurable via environment variables.

---

## Structured Logging

Every event is logged with a consistent schema:

```json
{
  "log_id": "01KR3...",
  "job_id": "01KR3...",
  "agent_id": "rag",
  "event_type": "agent_end",
  "input_hash": "a1b2c3d4e5f6...",
  "output_hash": "f6e5d4c3b2a1...",
  "latency_ms": 1419.08,
  "token_count": 325,
  "policy_violation": false,
  "violation_detail": null,
  "metadata": {},
  "timestamp": "2026-05-08T10:14:22Z"
}
```

Logs are queryable via `GET /api/v1/trace/{job_id}`, which reconstructs the exact sequence of agent decisions, tool calls, and handoffs in order.

---

## Project Structure

```
multi-agent-system/
├── docker-compose.yml
├── .env.example
├── README.md
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── agents/
│   │   ├── base.py              # BaseAgent ABC
│   │   ├── orchestrator.py      # Master controller
│   │   ├── decomposition.py     # Sub-task decomposition
│   │   ├── rag.py               # Multi-hop RAG
│   │   ├── critique.py          # Per-claim critique
│   │   ├── synthesis.py         # Contradiction resolution
│   │   ├── compression.py       # Context compression
│   │   └── meta.py              # Prompt improvement
│   ├── tools/
│   │   ├── base.py              # BaseTool with failure contract
│   │   ├── web_search.py        # Structured search stub
│   │   ├── code_sandbox.py      # RestrictedPython executor
│   │   ├── sql_lookup.py        # NL→SQL→SQLite
│   │   └── self_reflection.py   # Contradiction detector
│   ├── core/
│   │   ├── context.py           # SharedContext Pydantic schema
│   │   ├── budget.py            # Token budget manager
│   │   ├── logger.py            # Structured logger + trace builder
│   │   └── database.py          # SQLAlchemy models + async engine
│   ├── eval/
│   │   ├── test_cases.py        # 15 test cases (5+5+5)
│   │   ├── scorer.py            # 6-dimensional scorer
│   │   └── harness.py           # Eval orchestration + storage
│   └── worker/
│       ├── celery_app.py        # Celery configuration
│       └── tasks.py             # Pipeline, eval, reeval tasks
└── data/
    └── knowledge/
        └── sample_docs.txt
```

---

## Environment Variables

```env
# Required
GROQ_API_KEY=gsk_...                    # Get from console.groq.com/keys
GROQ_MODEL=llama-3.3-70b-versatile      # Model to use

# Database
DATABASE_URL=sqlite+aiosqlite:////app/data/multi_agent.db

# Redis
REDIS_URL=redis://redis:6379/0

# ChromaDB
CHROMA_PERSIST_DIR=/app/data/chroma_data

# App
APP_ENV=development
LOG_LEVEL=INFO
MAX_RETRIES=2

# Context budgets (tokens per agent)
ORCHESTRATOR_BUDGET=4000
DECOMPOSITION_BUDGET=3000
RAG_BUDGET=5000
CRITIQUE_BUDGET=3000
SYNTHESIS_BUDGET=4000
COMPRESSION_BUDGET=2000
META_BUDGET=3000
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| LLM | Groq (LLaMA 3.3 70B) | Free tier, fast inference, function calling |
| API framework | FastAPI + SSE-Starlette | Async, automatic docs, native SSE support |
| Task queue | Celery + Redis | Async job processing, pub/sub for SSE |
| Vector DB | ChromaDB | Local, free, cosine similarity search |
| Embeddings | sentence-transformers | Local, no API cost |
| Database | SQLite + SQLAlchemy async | Zero-config, sufficient for single-node |
| Token counting | tiktoken | Same tokenizer as most LLMs |
| Code sandbox | RestrictedPython | Safe Python execution without Docker-in-Docker |
| Containers | Docker Compose | One-command startup |

---

## Known Limitations

**Web search is stubbed.** The web search tool uses a static 10-document knowledge base with keyword-overlap scoring. It simulates real search behaviour for the eval harness, but does not fetch live internet data. In production, replace with Tavily API or SerpAPI (both have free tiers).

**ChromaDB uses the default embedding model.** The `all-MiniLM-L6-v2` model (sentence-transformers default) is general-purpose. For a domain-specific system, a fine-tuned embedder would significantly improve retrieval quality.

**The self-improving loop applies rewrites at runtime only.** Approved prompt rewrites update the in-memory agent configuration for the re-eval run, but are not automatically written back to the source `.py` files. This is intentional — automatic code modification without review is unsafe. The proposed prompt is stored in the database and can be manually applied.

**Code sandbox is not fully isolated.** RestrictedPython prevents the most dangerous operations, but is not a true security boundary. For production, use a containerised sandbox (e.g. AWS Lambda, Firecracker) for untrusted code execution.

**Celery workers run as root in Docker.** This is acceptable for development but not for production. Use `--uid` to specify a non-root user, or switch to a rootless container runtime.

**SQLite has concurrency limits.** Write-heavy workloads with multiple concurrent Celery workers will hit SQLite's single-writer constraint. Switch to PostgreSQL for production multi-worker deployments.

**Eval scoring uses an LLM judge for correctness.** The LLM judge can be inconsistent across runs. For fully reproducible scoring, replace the correctness dimension with a deterministic metric (e.g. BERTScore, ROUGE-L) or use a dedicated eval model.

---

## What I Would Build Next

**Real web search integration.** Replace the stub with Tavily API. The interface is already defined — it's a one-file swap in `tools/web_search.py`.

**PostgreSQL backend.** Drop-in replacement for SQLite in `database.py`. Enables concurrent workers, proper indexing on `job_id` and `timestamp`, and connection pooling.

**True token-by-token SSE streaming.** Currently the system streams agent-level events. Groq supports streaming completions — wiring each agent's `chat()` method to stream individual tokens would give a much richer client experience.

**Automated rewrite CI pipeline.** A GitHub Action that runs the eval on every PR, and if a specific dimension drops below a threshold, automatically proposes a rewrite and opens a review PR.

**Agent memory with episodic recall.** Store successful reasoning chains in ChromaDB. On similar future queries, inject the most relevant past chain as a few-shot example, reducing LLM calls and improving consistency.

**Human-in-the-loop dashboard.** A lightweight React UI (or even a Streamlit app) that shows live pipeline progress, eval results, pending rewrites, and budget usage — replacing the current Swagger-only interface.

---

## Running Tests

```bash
cd api

# Test all 4 tools
python test_tools.py

# Test all agents individually
python test_agents.py

# Test the full orchestrator pipeline
python test_orchestrator.py

# Run 3-case eval (saves Groq quota)
python test_eval.py
```

---

## License

MIT — free to use, modify, and distribute.

---

*Built as a B.Tech capstone project demonstrating production-grade LLM engineering patterns.*
