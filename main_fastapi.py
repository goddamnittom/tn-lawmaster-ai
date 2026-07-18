"""
TN-LawMaster — FastAPI Backend (v2)
=====================================
Production REST API for the Tennessee legal analysis engine.

Endpoints:
  GET  /health              — liveness & model info
  GET  /model               — active model name
  GET  /domains             — supported TCA domains
  POST /analyze             — synchronous legal analysis (cached)
  POST /analyze/stream      — token-level SSE streaming analysis
  POST /ingest/text         — ingest raw TCA text
  POST /ingest/pdf          — upload and ingest a PDF
  POST /ingest/pdf/batch    — batch PDF upload
  POST /draft               — generate legal document draft
  GET  /draft/types         — list document types
  POST /compliance/check    — check document for TN law compliance
  GET  /compliance/domains  — list compliance domains
  GET  /case-law/search     — search TN case law (CourtListener)
  GET  /tools/sol           — statute of limitations calculator
  GET  /docs                — Swagger UI
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── FastAPI & stdlib HTTP ─────────────────────────────────
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Rate limiting (optional dep: slowapi) ─────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    _limiter = Limiter(key_func=get_remote_address)
    _HAS_SLOWAPI = True
except ImportError:
    _HAS_SLOWAPI = False
    logger.info("slowapi not installed — rate limiting disabled. pip install slowapi")

# ── Response caching (optional dep: cachetools) ───────────
try:
    from cachetools import TTLCache
    _CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    _response_cache: TTLCache = TTLCache(maxsize=512, ttl=_CACHE_TTL)
    _HAS_CACHE = True
except ImportError:
    _HAS_CACHE = False
    _response_cache = {}  # type: ignore[assignment]
    logger.info("cachetools not installed — response caching disabled. pip install cachetools")

# ── API Key auth ──────────────────────────────────────────
_API_KEY = os.getenv("API_KEY", "").strip()
_AUTH_ENABLED = bool(_API_KEY)

if _AUTH_ENABLED:
    logger.info("API key authentication ENABLED")
else:
    logger.info("API key authentication DISABLED (set API_KEY env var to enable)")


def verify_api_key(request: Request) -> None:
    """FastAPI dependency: enforce API key on protected endpoints."""
    if not _AUTH_ENABLED:
        return
    key = request.headers.get("X-API-Key", "")
    if key != _API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide X-API-Key header.",
        )


# ── App init ──────────────────────────────────────────────
app = FastAPI(
    title="TN-LawMaster API",
    description=(
        "Expert RAG + LangGraph AI agent for Tennessee Code Annotated (TCA) legal analysis. "
        "Covers TCA Titles 39 (Criminal), 36 (Family), 66 (Property), 48 (Business), and more.\n\n"
        "**Authentication**: Set `X-API-Key` header if `API_KEY` env var is configured.\n"
        "**Rate limits**: 30 req/min on /analyze, 10 req/min on /ingest endpoints."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

if _HAS_SLOWAPI:
    app.state.limiter = _limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

# ── Global singletons ─────────────────────────────────────
_agent = None
_drafter = None
_checker = None
_ingester = None


def get_agent():
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized. Check startup logs.")
    return _agent


@app.on_event("startup")
async def startup():
    global _agent
    try:
        from model_config import get_llm, get_active_model_name
        from tn_law_agent.core import TNLawAgent
        llm = get_llm()
        _agent = TNLawAgent(llm=llm)
        logger.info("✅ TN-LawMaster v2 ready — model: %s", get_active_model_name())
    except Exception as exc:
        logger.error("❌ Agent initialization failed: %s", exc)


# ══════════════════════════════════════════════════════════
# Request / Response models
# ══════════════════════════════════════════════════════════

class LegalQuery(BaseModel):
    query: str = Field(..., min_length=5, max_length=4000,
                       example="What is the penalty for theft in Tennessee?")
    domain: str = Field("general", example="criminal")
    session_id: Optional[str] = Field(None, example="user-abc-123")


class AnalysisResponse(BaseModel):
    query: str
    domain: str
    analysis: str
    citations: List[str]
    status: str
    cached: bool = False


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=10)
    source: str = Field("manual", example="TCA Title 39")


class DraftRequest(BaseModel):
    doc_type: str = Field(..., example="tipa_request")
    context: Dict[str, Any] = Field(..., example={"requester_name": "Jane Smith",
                                                   "agency_name": "Nashville Metro",
                                                   "records_description": "All 2024 incident reports"})


class ComplianceRequest(BaseModel):
    document: str = Field(..., min_length=20,
                          example="This lease agreement between landlord and tenant...")
    domain: str = Field(..., example="landlord_tenant")


# ══════════════════════════════════════════════════════════
# Helper: cache key
# ══════════════════════════════════════════════════════════

def _cache_key(query: str, domain: str) -> str:
    return hashlib.sha256(f"{query}::{domain}".encode()).hexdigest()


def _get_drafter():
    global _drafter
    if _drafter is None:
        from tn_law_agent.drafts.document_drafter import TNDocumentDrafter
        from model_config import get_llm
        _drafter = TNDocumentDrafter(llm=get_llm())
    return _drafter


def _get_checker():
    global _checker
    if _checker is None:
        from tn_law_agent.compliance.checker import TNComplianceChecker
        from model_config import get_llm
        _checker = TNComplianceChecker(llm=get_llm())
    return _checker


def _get_ingester():
    global _ingester
    if _ingester is None:
        from tn_law_agent.knowledge.ingester import TNLawIngester
        persist_dir = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        data_dir = os.getenv("DATA_DIR", "./data")
        _ingester = TNLawIngester(data_dir=data_dir, persist_dir=persist_dir)
        if _agent is not None:
            _agent.attach_vector_store(_ingester)
    return _ingester


# ══════════════════════════════════════════════════════════
# System Endpoints (always public)
# ══════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
async def health():
    from model_config import get_active_model_name
    is_ready = _agent is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "agent_ready": is_ready,
        "model": get_active_model_name() if is_ready else "unavailable",
        "version": app.version,
        "auth_enabled": _AUTH_ENABLED,
        "cache_enabled": _HAS_CACHE,
        "rate_limiting_enabled": _HAS_SLOWAPI,
    }


@app.get("/model", tags=["System"])
async def current_model():
    from model_config import get_active_model_name
    return {"active_model": get_active_model_name()}


@app.get("/domains", tags=["Analysis"])
async def list_domains():
    from tn_law_agent.workflows.legal_graph import TCA_DOMAIN_COVERAGE
    return {
        "domains": [
            {"slug": slug, "description": info["desc"], "tca_titles": info["titles"]}
            for slug, info in TCA_DOMAIN_COVERAGE.items()
        ]
    }


# ══════════════════════════════════════════════════════════
# Analysis Endpoints
# ══════════════════════════════════════════════════════════

@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze(
    query: LegalQuery,
    request: Request,
    _auth: None = Depends(verify_api_key),
):
    """Synchronous grounded Tennessee law analysis with response caching."""
    # Check cache
    cache_hit = False
    ck = _cache_key(query.query, query.domain)
    if _HAS_CACHE and ck in _response_cache:
        cached_result = _response_cache[ck]
        cached_result["cached"] = True
        return AnalysisResponse(**cached_result)

    agent = get_agent()
    try:
        result = agent.analyze(query.query, domain=query.domain,
                               session_id=query.session_id)
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Pipeline error"))

        payload = {
            "query": result["query"],
            "domain": query.domain,
            "analysis": result["analysis"],
            "citations": result["citations"],
            "status": result["status"],
            "cached": False,
        }

        if _HAS_CACHE:
            _response_cache[ck] = payload

        return AnalysisResponse(**payload)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error in /analyze")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze/stream", tags=["Analysis"])
async def analyze_stream(
    query: LegalQuery,
    request: Request,
    _auth: None = Depends(verify_api_key),
):
    """
    Token-level streaming analysis via Server-Sent Events.

    Events emitted:
      data: {"event": "status", "text": "..."}    — pipeline progress
      data: {"event": "token",  "text": "..."}    — LLM token (streamed)
      data: {"event": "citations", "data": [...]} — final citations
      data: {"event": "done"}                      — stream complete
    """
    import asyncio
    import json

    agent = get_agent()

    async def _stream():
        import json

        def sse(obj: dict) -> str:
            return f"data: {json.dumps(obj)}\n\n"

        yield sse({"event": "status", "text": "🔍 Retrieving Tennessee statutes..."})

        try:
            # Run retrieve + rerank synchronously (fast steps)
            graph = agent.graph

            # Check if the graph supports async streaming
            if hasattr(graph, "astream_analysis"):
                # Use async token streaming
                full_text = ""
                async for token in graph.astream_analysis(query.query, domain=query.domain):
                    full_text += token
                    yield sse({"event": "token", "text": token})
                # Extract citations from accumulated text
                import re
                citations = re.findall(r"TCA\s*§\s*[\d]+-[\d]+-[\d]+", full_text)
                yield sse({"event": "citations", "data": list(dict.fromkeys(citations))})
            else:
                # Fallback: run pipeline sync, stream chunks of result
                yield sse({"event": "status",
                           "text": f"⚖️ Running analysis (domain: {query.domain})..."})
                result = agent.analyze(query.query, domain=query.domain,
                                       session_id=query.session_id)
                analysis = result.get("analysis", "")
                # Stream in ~150-char chunks with small delay for visual effect
                chunk_size = 150
                for i in range(0, len(analysis), chunk_size):
                    chunk = analysis[i:i + chunk_size]
                    yield sse({"event": "token", "text": chunk})
                    await asyncio.sleep(0.01)
                yield sse({"event": "citations", "data": result.get("citations", [])})

        except Exception as exc:
            yield sse({"event": "error", "text": str(exc)})

        yield sse({"event": "done"})

    return StreamingResponse(_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ══════════════════════════════════════════════════════════
# Ingestion Endpoints
# ══════════════════════════════════════════════════════════

@app.post("/ingest/text", tags=["Ingestion"])
async def ingest_text(
    req: IngestTextRequest,
    _auth: None = Depends(verify_api_key),
):
    ingester = _get_ingester()
    n = ingester.ingest_text(req.text, source=req.source)
    return {"message": f"Ingested {n} chunks", "source": req.source,
            "total_docs": ingester.doc_count}


@app.post("/ingest/pdf", tags=["Ingestion"])
async def ingest_pdf(
    file: UploadFile = File(...),
    _auth: None = Depends(verify_api_key),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    import shutil, tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        ingester = _get_ingester()
        n = ingester.ingest_pdf(tmp_path)
        return {"filename": file.filename, "chunks_added": n,
                "total_docs": ingester.doc_count}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/ingest/pdf/batch", tags=["Ingestion"])
async def batch_ingest_pdf(
    files: List[UploadFile] = File(...),
    _auth: None = Depends(verify_api_key),
):
    import shutil, tempfile
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
            results.append({"filename": f.filename, "chunks_added": n})
        except Exception as exc:
            results.append({"filename": f.filename, "error": str(exc)})
        finally:
            tmp_path.unlink(missing_ok=True)
    return {"processed": len(results), "total_docs": ingester.doc_count, "results": results}


# ══════════════════════════════════════════════════════════
# Document Drafting Endpoints
# ══════════════════════════════════════════════════════════

@app.get("/draft/types", tags=["Document Drafting"])
async def list_draft_types():
    """List all available legal document types with required fields."""
    from tn_law_agent.drafts.document_drafter import DOCUMENT_TYPES
    return {
        "document_types": [
            {
                "type": k,
                "label": v["label"],
                "tca_basis": v["tca_basis"],
                "required_fields": v["required_fields"],
                "optional_fields": v.get("optional_fields", []),
            }
            for k, v in DOCUMENT_TYPES.items()
        ]
    }


@app.post("/draft", tags=["Document Drafting"])
async def draft_document(
    req: DraftRequest,
    _auth: None = Depends(verify_api_key),
):
    """
    Generate a Tennessee legal document draft.

    Call GET /draft/types first to see available types and required context fields.
    """
    drafter = _get_drafter()
    try:
        result = drafter.draft(req.doc_type, req.context)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Document drafting error")
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════
# Compliance Checking Endpoints
# ══════════════════════════════════════════════════════════

@app.get("/compliance/domains", tags=["Compliance"])
async def list_compliance_domains():
    """List all supported Tennessee law compliance domains."""
    from tn_law_agent.compliance.checker import COMPLIANCE_DOMAINS
    return {
        "domains": [
            {
                "domain": k,
                "label": v["label"],
                "tca_refs": v["tca_refs"],
            }
            for k, v in COMPLIANCE_DOMAINS.items()
        ]
    }


@app.post("/compliance/check", tags=["Compliance"])
async def compliance_check(
    req: ComplianceRequest,
    _auth: None = Depends(verify_api_key),
):
    """
    Check a document for Tennessee law compliance issues.

    Supported domains: landlord_tenant, llc_operating, employment, consumer_protection
    """
    checker = _get_checker()
    try:
        return checker.check(req.document, req.domain)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Compliance check error")
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════
# Case Law & Tools Endpoints
# ══════════════════════════════════════════════════════════

@app.get("/case-law/search", tags=["Case Law"])
async def search_case_law(
    q: str = Query(..., min_length=3, example="DUI first offense penalty"),
    limit: int = Query(3, ge=1, le=10),
    _auth: None = Depends(verify_api_key),
):
    """Search Tennessee case law via CourtListener (free public API)."""
    from tn_law_agent.tools.legal_tools import search_tn_case_law
    try:
        result = search_tn_case_law.invoke({"query": q, "max_results": limit})
        return {"query": q, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/tools/sol", tags=["Legal Tools"])
async def statute_of_limitations(
    offense_type: str = Query(..., example="personal_injury"),
    date_of_offense: str = Query(..., example="2023-06-15"),
):
    """Calculate the Tennessee statute of limitations deadline (TCA Title 28)."""
    from tn_law_agent.tools.legal_tools import calculate_statute_of_limitations
    try:
        result = calculate_statute_of_limitations.invoke(
            {"offense_type": offense_type, "date_of_offense": date_of_offense}
        )
        return {"offense_type": offense_type, "date_of_offense": date_of_offense,
                "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/tools/forms", tags=["Legal Tools"])
async def legal_forms(
    form_type: str = Query(..., example="divorce"),
):
    """Get information about official Tennessee court forms."""
    from tn_law_agent.tools.legal_tools import get_tennessee_legal_forms
    try:
        result = get_tennessee_legal_forms.invoke({"form_type": form_type})
        return {"form_type": form_type, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════
# Export Endpoint
# ══════════════════════════════════════════════════════════

@app.post("/export/pdf", tags=["Export"])
async def export_pdf(
    query: LegalQuery,
    _auth: None = Depends(verify_api_key),
):
    """Run analysis and return the result as a downloadable PDF."""
    agent = get_agent()
    result = agent.analyze(query.query, domain=query.domain)

    try:
        from tn_law_agent.utils.pdf_exporter import export_analysis_pdf
        pdf_bytes = export_analysis_pdf(result, query=query.query, domain=query.domain)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="tn-lawmaster-analysis.pdf"'
            },
        )
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="PDF export requires reportlab. Install with: pip install reportlab",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_fastapi:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )