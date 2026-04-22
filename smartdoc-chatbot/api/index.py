"""
Vercel Serverless Entry Point
──────────────────────────────
This file lives in api/ so Vercel auto-detects it as a Python
serverless function. Mangum wraps the FastAPI ASGI app so Vercel
can call it via its Lambda-style event/response model.

All routes (/, /status, /ingest, /chat) are handled by FastAPI
internally — Vercel just forwards every request here.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# ── Make the project root importable ──────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi import status as http_status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mangum import Mangum
from pydantic import BaseModel, Field

from rag.embedder import collection_info, embed_and_store, get_qdrant_client
from rag.loader import load_and_chunk_pdf
from rag.retriever import generate_answer, generate_answer_stream

# ─────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "smartdoc")

# ─────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────
app = FastAPI(
    title="SmartDoc Chatbot API",
    description="RAG API powered by Gemini 2.5 Flash + Qdrant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────
class HistoryEntry(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    doc_id: Optional[str] = None
    stream: bool = True
    top_k: int = Field(6, ge=1, le=20)
    history: Optional[list[HistoryEntry]] = None


class SourceChunk(BaseModel):
    chunk_number: int
    page: Optional[int]
    content_preview: str


class ChatResponse(BaseModel):
    answer: str
    doc_id: Optional[str]
    sources: list[SourceChunk]


class IngestResponse(BaseModel):
    status: str
    doc_id: str
    filename: str
    chunks_processed: int
    collection: str


# ─────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────
@app.get("/status", tags=["Utility"])
async def health_check():
    try:
        client = get_qdrant_client()
        all_cols = [c.name for c in client.get_collections().collections]
        return {
            "status": "healthy",
            "collection": collection_info(COLLECTION_NAME),
            "all_collections": all_cols,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/ingest", response_model=IngestResponse,
          status_code=http_status.HTTP_201_CREATED, tags=["Documents"])
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    content = await file.read()
    if len(content) / (1024 * 1024) > 50:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        doc_id = Path(file.filename).stem.replace(" ", "_")[:60]
        chunks = load_and_chunk_pdf(str(tmp_path))
        embed_and_store(chunks, COLLECTION_NAME, doc_id=doc_id)

        return IngestResponse(
            status="success", doc_id=doc_id, filename=file.filename,
            chunks_processed=len(chunks), collection=COLLECTION_NAME,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.post("/chat", tags=["Chat"])
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = [h.model_dump() for h in request.history] if request.history else None

    if request.stream:
        async def token_stream():
            try:
                async for token in generate_answer_stream(
                    request.question, COLLECTION_NAME,
                    doc_id=request.doc_id, top_k=request.top_k, history=history,
                ):
                    yield token
            except Exception as exc:
                yield f"\n\n[Error: {exc}]"
        return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8")

    try:
        answer, docs = generate_answer(
            request.question, COLLECTION_NAME,
            doc_id=request.doc_id, top_k=request.top_k, history=history,
        )
        sources = [
            SourceChunk(
                chunk_number=i, page=doc.metadata.get("page"),
                content_preview=doc.page_content.strip()[:300],
            )
            for i, doc in enumerate(docs, start=1)
        ]
        return ChatResponse(answer=answer, doc_id=request.doc_id, sources=sources)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────
# Vercel handler
# ─────────────────────────────────────────────────
handler = Mangum(app, lifespan="off")
