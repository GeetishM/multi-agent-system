from __future__ import annotations
import json
import os
from typing import List, Dict
from core.context import SharedContext, AgentID, AgentMessage, Claim
from agents.base import BaseAgent
from tools.web_search import WebSearchTool


SYSTEM_PROMPT = """You are a Retrieval-Augmented Generation (RAG) Agent. You answer questions using ONLY the provided source chunks.

Critical rules:
1. You MUST use at least 2 different chunks in your answer
2. Every factual claim must be tagged with [chunk_id:X] showing which chunk supports it
3. Never make claims not supported by the chunks
4. If chunks are insufficient, say so explicitly
5. Perform multi-hop reasoning: use chunk A to inform how you interpret chunk B

Response format (JSON only):
{
  "answer": "Your full answer with [chunk_id:X] citations inline",
  "claims": [
    {
      "text": "exact claim text",
      "chunk_ids": ["chunk_id_1", "chunk_id_2"],
      "confidence": 0.9
    }
  ],
  "chunks_used": ["chunk_id_1", "chunk_id_2"],
  "reasoning_chain": "explain your multi-hop reasoning here",
  "insufficient": false
}"""


class RAGAgent(BaseAgent):
    agent_id = AgentID.RAG

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_tool = WebSearchTool()
        self._init_chroma()

    def _init_chroma(self):
        """Initialize ChromaDB for local vector storage."""
        try:
            # pyrefly: ignore [missing-import]
            import chromadb
            # pyrefly: ignore [missing-import]
            from chromadb.config import Settings
            persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_data")
            os.makedirs(persist_dir, exist_ok=True)
            self.chroma = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.chroma.get_or_create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"},
            )
            self._seed_knowledge_base()
        except Exception as e:
            self.chroma     = None
            self.collection = None

    def _seed_knowledge_base(self):
        """Seed ChromaDB with sample documents if empty."""
        if self.collection is None:
            return
        if self.collection.count() > 0:
            return

        docs = [
            ("doc1", "Large language models are trained on massive text corpora using transformer architectures. They learn to predict the next token, developing emergent capabilities like reasoning and code generation."),
            ("doc2", "Retrieval-Augmented Generation (RAG) combines parametric memory (model weights) with non-parametric memory (retrieved documents). This reduces hallucination and enables knowledge updates without retraining."),
            ("doc3", "Multi-agent systems use specialized agents with distinct roles. The orchestrator routes tasks dynamically, while sub-agents handle retrieval, critique, synthesis, and compression."),
            ("doc4", "Context window management is critical for production LLM systems. Token budgets prevent overflow, compression reduces context size while preserving structured data, and summarization handles conversational history."),
            ("doc5", "Prompt injection attacks attempt to override system instructions by embedding instructions in user input. Defense strategies include input sanitization, instruction hierarchy enforcement, and output validation."),
            ("doc6", "Vector databases store high-dimensional embeddings and support approximate nearest-neighbor search. ChromaDB, Qdrant, and Pinecone are popular choices for RAG pipelines."),
            ("doc7", "Evaluation of LLM systems requires multi-dimensional metrics: answer correctness, citation accuracy, hallucination rate, latency, and cost. Single-metric evaluation misses important failure modes."),
            ("doc8", "The transformer attention mechanism computes query-key-value dot products across all token positions. Self-attention enables the model to relate distant tokens, while multi-head attention captures different relationship types."),
            ("doc9", "Docker containerization packages applications with all dependencies. Docker Compose orchestrates multi-service deployments, enabling one-command startup of complex systems with databases, queues, and APIs."),
            ("doc10","FastAPI provides async request handling, automatic OpenAPI documentation, and Pydantic validation. It is built on Starlette and supports Server-Sent Events for streaming responses."),
        ]

        self.collection.add(
            ids=[d[0] for d in docs],
            documents=[d[1] for d in docs],
        )

    def _retrieve_from_chroma(self, query: str, n_results: int = 3) -> List[Dict]:
        """Retrieve relevant chunks from ChromaDB."""
        if self.collection is None:
            return []
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
            )
            chunks = []
            for i, doc in enumerate(results["documents"][0]):
                chunks.append({
                    "chunk_id":        results["ids"][0][i],
                    "text":            doc,
                    "relevance_score": 1 - results["distances"][0][i]
                    if results.get("distances") else 0.8,
                    "source":          "chroma_kb",
                })
            return chunks
        except Exception:
            return []

    def run(self, context: SharedContext) -> SharedContext:
        self.logger.agent_start(self.agent_id, input_text=context.original_query)

        # ── Step 1: Retrieve from ChromaDB ────────────────
        chroma_chunks = self._retrieve_from_chroma(context.original_query, n_results=3)

        # ── Step 2: Retrieve from Web Search (hop 2) ──────
        search_result = self.search_tool.run(
            query=context.original_query, max_results=3
        )
        web_chunks = []
        if search_result.success:
            for item in search_result.data:
                web_chunks.append({
                    "chunk_id":        item["chunk_id"],
                    "text":            item["snippet"],
                    "relevance_score": item["relevance_score"],
                    "source":          item["url"],
                })

        all_chunks = chroma_chunks + web_chunks

        # Enforce minimum 2 chunks
        if len(all_chunks) < 2:
            context.messages.append(AgentMessage(
                from_agent=self.agent_id,
                content="Insufficient chunks retrieved for multi-hop reasoning.",
            ))
            return context

        # ── Step 3: Build context with budget check ────────
        chunks_text = "\n\n".join([
            f"[Chunk {c['chunk_id']}] (relevance: {c['relevance_score']:.2f})\n{c['text']}"
            for c in all_chunks
        ])

        prompt = f"""Answer the following question using the provided chunks.
Remember: cite at least 2 chunks using [chunk_id:X] notation.

Question: {context.original_query}

Chunks:
{chunks_text}"""

        if not self.check_and_record(prompt):
            # Budget exceeded — use only top 2 chunks
            prompt = f"Answer briefly using these 2 chunks:\n{chunks_text[:500]}\nQuestion: {context.original_query}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]

        try:
            raw = self.chat(messages, max_tokens=1500, temperature=0.2)
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)

            # Add claims to shared context
            for claim_data in parsed.get("claims", []):
                context.claims.append(Claim(
                    text=claim_data.get("text", ""),
                    confidence=claim_data.get("confidence", 0.7),
                    source_agent=self.agent_id,
                    source_chunk_ids=claim_data.get("chunk_ids", []),
                ))

            # Record answer
            answer = parsed.get("answer", "")
            context.messages.append(AgentMessage(
                from_agent=self.agent_id,
                content=answer,
                token_count=self.budget.get_usage(self.agent_id)["used"],
                metadata={
                    "chunks_used":     parsed.get("chunks_used", []),
                    "reasoning_chain": parsed.get("reasoning_chain", ""),
                    "insufficient":    parsed.get("insufficient", False),
                    "all_chunks":      all_chunks,
                },
            ))

            self.logger.agent_end(
                self.agent_id,
                input_text=prompt,
                output_text=answer,
                token_count=self.budget.get_usage(self.agent_id)["used"],
            )

        except (json.JSONDecodeError, Exception) as e:
            context.messages.append(AgentMessage(
                from_agent=self.agent_id,
                content=f"RAG agent error: {e}. Raw response stored.",
                metadata={"raw_response": raw if 'raw' in dir() else ""},
            ))

        return context