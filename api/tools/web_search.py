"""
Web Search Tool — stub implementation.
Returns structured results with source URLs and relevance scores.

Failure contract:
- Timeout     → ToolResult(success=False, failure_reason=TIMEOUT)
- Empty query → ToolResult(success=False, failure_reason=MALFORMED)
- No results  → ToolResult(success=False, failure_reason=EMPTY_RESULTS)
"""
from __future__ import annotations
import hashlib
import random
from typing import Any, Dict, List, Optional
from tools.base import BaseTool, ToolResult, FailureReason


# ── Stub knowledge base ───────────────────────────────────────────────────────
# In production this would call SerpAPI / Tavily / Brave Search API.
# We simulate realistic structured results for the eval harness.

_STUB_KNOWLEDGE: List[Dict] = [
    {
        "title": "Introduction to Large Language Models",
        "url": "https://arxiv.org/abs/2307.06435",
        "snippet": "Large language models (LLMs) are neural networks trained on massive text corpora. They demonstrate emergent capabilities including reasoning, code generation, and instruction following.",
        "relevance_score": 0.95,
        "source_type": "academic",
    },
    {
        "title": "RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "url": "https://arxiv.org/abs/2005.11401",
        "snippet": "RAG combines parametric memory (LLM weights) with non-parametric memory (retrieved documents) to produce more accurate and verifiable answers.",
        "relevance_score": 0.92,
        "source_type": "academic",
    },
    {
        "title": "Python Multi-Processing Best Practices",
        "url": "https://docs.python.org/3/library/multiprocessing.html",
        "snippet": "Python's multiprocessing module supports spawning processes. Use Pool for parallel task execution and Queue for inter-process communication.",
        "relevance_score": 0.88,
        "source_type": "documentation",
    },
    {
        "title": "Climate Change: Global Temperature Trends",
        "url": "https://climate.nasa.gov/vital-signs/global-temperature/",
        "snippet": "Global average surface temperature has risen 1.1°C since the pre-industrial period. The last decade (2011-2020) was the warmest on record.",
        "relevance_score": 0.91,
        "source_type": "government",
    },
    {
        "title": "Docker Containerization Guide",
        "url": "https://docs.docker.com/get-started/",
        "snippet": "Docker packages applications into containers — standardized units of software that bundle code and dependencies together.",
        "relevance_score": 0.87,
        "source_type": "documentation",
    },
    {
        "title": "Attention Is All You Need — Transformer Architecture",
        "url": "https://arxiv.org/abs/1706.03762",
        "snippet": "The Transformer uses self-attention mechanisms to process sequences in parallel. Multi-head attention allows the model to attend to different representation subspaces.",
        "relevance_score": 0.96,
        "source_type": "academic",
    },
    {
        "title": "FastAPI Framework Documentation",
        "url": "https://fastapi.tiangolo.com/",
        "snippet": "FastAPI is a modern Python web framework built on Starlette and Pydantic. It provides automatic OpenAPI documentation and async support.",
        "relevance_score": 0.89,
        "source_type": "documentation",
    },
    {
        "title": "Vector Databases Explained",
        "url": "https://www.pinecone.io/learn/vector-database/",
        "snippet": "Vector databases store high-dimensional embeddings and support approximate nearest neighbor search. Used for semantic similarity in RAG pipelines.",
        "relevance_score": 0.90,
        "source_type": "blog",
    },
    {
        "title": "Machine Learning Model Evaluation Metrics",
        "url": "https://scikit-learn.org/stable/modules/model_evaluation.html",
        "snippet": "Evaluation metrics include accuracy, precision, recall, F1-score, ROC-AUC, and mean squared error. Choice depends on the problem type.",
        "relevance_score": 0.86,
        "source_type": "documentation",
    },
    {
        "title": "Agent-based AI Systems Survey",
        "url": "https://arxiv.org/abs/2308.11432",
        "snippet": "Multi-agent systems use specialized agents with distinct roles. Orchestration patterns include hierarchical, collaborative, and competitive frameworks.",
        "relevance_score": 0.93,
        "source_type": "academic",
    },
]


def _compute_relevance(query: str, snippet: str, title: str) -> float:
    """Simple keyword overlap relevance scoring."""
    query_words = set(query.lower().split())
    text_words  = set((snippet + " " + title).lower().split())
    overlap     = query_words & text_words
    if not query_words:
        return 0.0
    base_score = len(overlap) / len(query_words)
    # Add small random noise to simulate real search variance
    noise = random.uniform(-0.05, 0.05)
    return round(min(1.0, max(0.0, base_score * 0.4 + 0.6 + noise)), 3)


class WebSearchTool(BaseTool):
    name = "web_search"
    timeout_seconds = 8.0

    def _validate_input(self, query: str = "", max_results: int = 3, **kwargs) -> Optional[str]:
        if not query or not query.strip():
            return "Query cannot be empty"
        if len(query.strip()) < 3:
            return "Query too short (minimum 3 characters)"
        if max_results < 1 or max_results > 10:
            return "max_results must be between 1 and 10"
        return None

    def _execute(self, query: str, max_results: int = 3, **kwargs) -> List[Dict]:
        query = query.strip()

        # Score all stub results against the query
        scored = []
        for item in _STUB_KNOWLEDGE:
            score = _compute_relevance(query, item["snippet"], item["title"])
            scored.append({**item, "relevance_score": score})

        # Sort by relevance, return top N
        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        results = scored[:max_results]

        # Add chunk_id for RAG agent to cite
        for i, r in enumerate(results):
            r["chunk_id"] = hashlib.md5(r["url"].encode()).hexdigest()[:8]

        return results