"""Multi-Agent System — FastAPI Entry Point"""
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from core.database import init_db
from core.logger import get_logger
from routers import query, trace, eval, rewrite, reeval

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Multi-Agent System...")
    await init_db()

    # Seed sample DB on startup
    try:
        from tools.sql_lookup import seed_sample_db
        import pathlib
        pathlib.Path("/app/data").mkdir(parents=True, exist_ok=True)
        seed_sample_db("/app/data/sample.db")
        logger.info("Sample DB seeded.")
    except Exception as e:
        logger.warning("sample_db_seed_failed", error=str(e))

    logger.info("Multi-Agent System ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Multi-Agent LLM System",
    description="Production-grade multi-agent system with self-improving eval loop",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router,   prefix="/api/v1", tags=["Query"])
app.include_router(trace.router,   prefix="/api/v1", tags=["Trace"])
app.include_router(eval.router,    prefix="/api/v1", tags=["Eval"])
app.include_router(rewrite.router, prefix="/api/v1", tags=["Rewrite"])
app.include_router(reeval.router,  prefix="/api/v1", tags=["Re-eval"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "multi-agent-system", "version": "1.0.0"}


@app.get("/")
async def root():
    return {
        "service": "Multi-Agent LLM System",
        "docs":    "/docs",
        "health":  "/health",
        "endpoints": {
            "query":   "POST /api/v1/query",
            "trace":   "GET  /api/v1/trace/{job_id}",
            "eval":    "GET  /api/v1/eval",
            "rewrite": "POST /api/v1/rewrite/{rewrite_id}",
            "reeval":  "POST /api/v1/reeval",
        }
    }