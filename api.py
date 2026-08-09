#!/usr/bin/env python3
"""
FastAPI server exposing LexisGraph as a REST API.

Run:
    uvicorn api:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive Swagger UI.
"""
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.rag_pipeline import LexisGraphPipeline

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="LexisGraph API",
    description="CPU-friendly enterprise legal RAG with hybrid retrieval",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

pipeline = LexisGraphPipeline()

# ---- API-key auth ----
# Set API_KEY in .env before deploying anywhere public. If it's left
# unset, auth is skipped entirely - fine for purely local/localhost use,
# but never leave it unset on a publicly reachable server.
API_KEY = os.environ.get("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: str = Security(api_key_header)):
    if not API_KEY:
        return  # no key configured -> auth disabled (local dev only)
    if provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_kg: bool = True


class IngestRequest(BaseModel):
    docs_dir: str = "sample_docs"


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status", dependencies=[Depends(require_api_key)])
def status():
    try:
        return pipeline.status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", dependencies=[Depends(require_api_key)])
def ingest(req: IngestRequest):
    if not Path(req.docs_dir).exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.docs_dir}")
    try:
        n_chunks = pipeline.ingest(req.docs_dir)
        return {"status": "success", "chunks_indexed": n_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", dependencies=[Depends(require_api_key)])
def query(req: QueryRequest):
    try:
        return pipeline.query(req.question, top_k=req.top_k, use_kg=req.use_kg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
