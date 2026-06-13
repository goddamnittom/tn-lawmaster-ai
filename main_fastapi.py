from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_config import get_llm
from tn_law_agent.core import TNLawAgent
from tn_law_agent.knowledge.ingester import TNLawIngester

app = FastAPI(title="TN-LawMaster API", version="0.6.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class LegalQuery(BaseModel):
    query: str

llm = get_llm()
agent = TNLawAgent(llm=llm)

@app.post("/analyze")
async def analyze(query: LegalQuery):
    try:
        return agent.analyze(query.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/stream")
async def analyze_stream(query: LegalQuery):
    from fastapi.responses import StreamingResponse
    import asyncio

    async def stream():
        yield "🔍 Retrieving relevant Tennessee law...\n\n"
        await asyncio.sleep(0.4)
        result = agent.analyze(query.query)
        yield result.get("analysis", "Analysis complete.")
    return StreamingResponse(stream(), media_type="text/event-stream")

@app.post("/upload-pdf/batch")
async def batch_pdf(files: List[UploadFile] = File(...), query: Optional[str] = None):
    results = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue
        # Save and process PDF (simplified)
        content = (await file.read()).decode(errors="ignore")[:12000]
        result = agent.analyze(f"Analyze this document: {query or ''}\n\n{content}")
        results.append({"filename": file.filename, "analysis": result.get("analysis", "")[:600]})
    return {"processed": len(results), "results": results}

@app.get("/model")
async def current_model():
    from model_config import ACTIVE_MODEL
    return {"active_model": ACTIVE_MODEL}

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "gemma3-1070"}