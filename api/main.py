"""
Multi-Agent System — FastAPI Entry Point
"""
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.database import init_db
from core.logger import get_logger
from routers import query, trace, eval, rewrite, reeval

load_dotenv()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting Multi-Agent System...")
    await init_db()
    logger.info("Database initialized.")
    yield
    logger.info("Shutting down Multi-Agent System.")


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

# ── Register Routers (5 endpoints) ────────────────
app.include_router(query.router,   prefix="/api/v1", tags=["Query"])
app.include_router(trace.router,   prefix="/api/v1", tags=["Trace"])
app.include_router(eval.router,    prefix="/api/v1", tags=["Eval"])
app.include_router(rewrite.router, prefix="/api/v1", tags=["Rewrite"])
app.include_router(reeval.router,  prefix="/api/v1", tags=["Re-eval"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "multi-agent-system"}