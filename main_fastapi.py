"""
TN-LawMaster — FastAPI Backend
================================
Production REST API for the Tennessee legal analysis engine.

Endpoints:
  GET  /health              — liveness & model info
  GET  /model               — active model name
  GET  /domains             — supported TCA domains
  POST /analyze             — synchronous legal analysis
  POST /analyze/stream      — Server-Sent Events streaming analysis
  POST /ingest/text         — ingest raw TCA text into vector store
  POST /ingest/pdf          — upload and ingest a PDF
  POST /ingest/pdf/batch    — batch PDF upload
  GET  /docs                — Swagger UI (auto-generated)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Import app modules ───────────────────────────────────
from model_config import get_llm, get_active_model_name
from tn_law_agent.core import TNLawAgent

# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(
    title="TN-LawMaster API",
    description=(
        "Expert RAG + LangGraph AI agent for Tennessee Code Annotated (TCA) legal analysis. "
        "Covers TCA Titles 39 (Criminal), 36 (Family), 66 (Property), 48 (Business), and more."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────
_cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
_cors_origins = (
    ["*"] if _cors_origins_raw.strip() == "*"
    else [o.strip() for o in _cors_origins_raw.split(",")]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Global agent ─────────────────────────────────────────
_agent: TNLawAgent | None = None
_ingester = None   # Lazy-loaded only when a vector store is needed


def get_agent() -> TNLawAgent:
    global _agent
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized. Check logs.")
    return _agent


@app.on_event("startup")
async def startup():
    global _agent
    try:
        llm = get_llm()
        _agent = TNLawAgent(llm=llm)
        logger.info("✅ TN-LawMaster ready — model: %s", get_active_model_name())
    except Exception as exc:
        logger.error("❌ Agent initialization failed: %s", exc)
        # Don't crash the server; /health will report unhealthy


# ── Request / Response models ─────────────────────────────

class LegalQuery(BaseModel):
    query: str = Field(..., min_length=5, max_length=4000, example="What is the penalty for theft in Tennessee?")
    domain: str = Field("general", example="criminal")


class AnalysisResponse(BaseModel):
    query: str
    domain: str
    analysis: str
    citations: List[str]
    status: str


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=10)
    source: str = Field("manual", example="TCA Title 39")


# ── Endpoints ─────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Liveness check — reports agent and model status."""
    is_ready = _agent is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "agent_ready": is_ready,
        "model": get_active_model_name() if is_ready else "unavailable",
        "version": app.version,
    }


@app.get("/model", tags=["System"])
async def current_model():
    """Return the name of the currently active LLM."""
    return {"active_model": get_active_model_name()}


@app.get("/domains", tags=["Analysis"])
async def list_domains():
    """List all supported TCA legal domains."""
    from tn_law_agent.workflows.legal_graph import TCA_DOMAIN_COVERAGE
    return {
        "domains": [
            {"slug": slug, "description": info["desc"], "tca_titles": info["titles"]}
            for slug, info in TCA_DOMAIN_COVERAGE.items()
        ]
    }


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze(query: LegalQuery):
    """
    Perform a synchronous grounded Tennessee law analysis.

    Set `domain` to narrow the search context (e.g., `criminal`, `family`).
    """
    agent = get_agent()
    try:
        result = agent.analyze(query.query, domain=query.domain)
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Pipeline error"))
        return AnalysisResponse(
            query=result["query"],
            domain=query.domain,
            analysis=result["analysis"],
            citations=result["citations"],
            status=result["status"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error in /analyze")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze/stream", tags=["Analysis"])
async def analyze_stream(query: LegalQuery):
    """
    Streaming analysis via Server-Sent Events.

    The response is streamed in chunks so the client can display
    progress in real time without waiting for the full pipeline.
    """
    import asyncio

    agent = get_agent()

    async def _stream():
        yield "data: 🔍 Retrieving relevant Tennessee statutes...\n\n"
        await asyncio.sleep(0.05)
        yield f"data: ⚖️  Running LangGraph pipeline (domain: {query.domain})...\n\n"
        await asyncio.sleep(0.05)
        try:
            result = agent.analyze(query.query, domain=query.domain)
            analysis = result.get("analysis", "")
            citations = result.get("citations", [])
            # Stream analysis in ~200-char chunks
            chunk_size = 200
            for i in range(0, len(analysis), chunk_size):
                yield f"data: {analysis[i:i+chunk_size]}\n\n"
                await asyncio.sleep(0.01)
            if citations:
                yield f"\n\ndata: \n\n📌 **Citations**: {', '.join(citations)}\n\n"
        except Exception as exc:
            yield f"data: ❌ Error: {exc}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.post("/ingest/text", tags=["Ingestion"])
async def ingest_text(req: IngestTextRequest):
    """Ingest raw text (TCA statute text) into the vector store."""
    ingester = _get_ingester()
    n = ingester.ingest_text(req.text, source=req.source)
    return {"message": f"Ingested {n} chunks", "source": req.source, "total_docs": ingester.doc_count}


@app.post("/ingest/pdf", tags=["Ingestion"])
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload and ingest a single PDF file."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    import tempfile, shutil
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        ingester = _get_ingester()
        n = ingester.ingest_pdf(tmp_path)
        return {"filename": file.filename, "chunks_added": n, "total_docs": ingester.doc_count}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/ingest/pdf/batch", tags=["Ingestion"])
async def batch_ingest_pdf(
    files: List[UploadFile] = File(...),
    analyze_query: Optional[str] = Query(None, alias="query"),
):
    """
    Batch-upload PDFs.  Optionally also run an analysis query against
    each uploaded document immediately after ingestion.
    """
    import tempfile, shutil
    ingester = _get_ingester()
    results = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            shutil.copyfileobj(f.file, tmp)
            tmp_path = Path(tmp.name)
        try:
            n = ingester.ingest_pdf(tmp_path)
            entry: dict = {"filename": f.filename, "chunks_added": n}
            if analyze_query:
                agent = get_agent()
                r = agent.analyze(analyze_query)
                entry["analysis"] = r.get("analysis", "")[:600]
            results.append(entry)
        except Exception as exc:
            results.append({"filename": f.filename, "error": str(exc)})
        finally:
            tmp_path.unlink(missing_ok=True)
    return {"processed": len(results), "total_docs": ingester.doc_count, "results": results}


# ── Internal helpers ──────────────────────────────────────

def _get_ingester():
    """Lazy-load the vector store ingester and attach it to the agent."""
    global _ingester
    if _ingester is None:
        from tn_law_agent.knowledge.ingester import TNLawIngester
        persist_dir = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        data_dir = os.getenv("DATA_DIR", "./data")
        _ingester = TNLawIngester(data_dir=data_dir, persist_dir=persist_dir)
        if _agent is not None:
            _agent.attach_vector_store(_ingester)
    return _ingester


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_fastapi:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )