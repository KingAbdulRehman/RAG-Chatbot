"""
SmartDoc Chatbot — FastAPI Backend
────────────────────────────────────
Exposes three endpoints:
  POST /ingest  — upload & process a PDF
  POST /chat    — answer a question (streaming or JSON)
  GET  /status  — health check + Qdrant collection stats

Run with:
    uvicorn api:app --reload --port 8000
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

from rag.embedder import collection_info, embed_and_store, get_qdrant_client
from rag.loader import load_and_chunk_pdf
from rag.retriever import generate_answer, generate_answer_stream

# ─────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "smartdoc")
MAX_UPLOAD_MB = 50

# ─────────────────────────────────────────────────
# App Initialisation
# ─────────────────────────────────────────────────
app = FastAPI(
    title="SmartDoc Chatbot API",
    description=(
        "Production-ready RAG API — upload PDFs and ask questions "
        "powered by Google Gemini 2.5 Flash and Qdrant vector search."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────

class HistoryEntry(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    doc_id: Optional[str] = Field(
        None,
        description="Filter answers to a specific ingested document.",
    )
    stream: bool = Field(True, description="Stream the response token-by-token.")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve.")
    history: Optional[list[HistoryEntry]] = Field(
        None,
        description="Previous conversation turns for memory-aware answers.",
    )


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


class StatusResponse(BaseModel):
    status: str
    collection: dict
    all_collections: list[str]


# ─────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────

@app.get(
    "/status",
    response_model=StatusResponse,
    summary="Health check",
    tags=["Utility"],
)
async def health_check() -> StatusResponse:
    """
    Returns:
    - `status`: "healthy" if Qdrant is reachable, else raises 503.
    - `collection`: stats for the configured collection.
    - `all_collections`: every collection in the Qdrant cluster.
    """
    try:
        client = get_qdrant_client()
        all_cols = [c.name for c in client.get_collections().collections]
        col_info = collection_info(COLLECTION_NAME)

        return StatusResponse(
            status="healthy",
            collection=col_info,
            all_collections=all_cols,
        )

    except EnvironmentError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Configuration error: {exc}",
        )
    except Exception as exc:
        logger.exception("Health check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Qdrant unreachable: {exc}",
        )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a PDF",
    tags=["Documents"],
)
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    """
    Upload a PDF file and process it into the Qdrant vector store.

    - Validates the file is a PDF and under 50 MB.
    - Splits the PDF into 1,000-character chunks with 200-char overlap.
    - Embeds each chunk with Gemini embedding-001.
    - Stores vectors in the configured Qdrant collection.

    Returns the `doc_id` needed for filtered `/chat` requests.
    """
    # ── Validation ───────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .pdf files are accepted.",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {size_mb:.1f} MB exceeds the {MAX_UPLOAD_MB} MB limit.",
        )

    # ── Save to temp file ────────────────────────────────────────────
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        doc_id = Path(file.filename).stem.replace(" ", "_")[:60]
        logger.info("Ingesting '%s' → doc_id='%s'", file.filename, doc_id)

        # ── Process ──────────────────────────────────────────────────
        chunks = load_and_chunk_pdf(str(tmp_path))
        embed_and_store(chunks, COLLECTION_NAME, doc_id=doc_id)

        logger.info(
            "Ingested '%s': %d chunks → collection '%s'",
            file.filename,
            len(chunks),
            COLLECTION_NAME,
        )

        return IngestResponse(
            status="success",
            doc_id=doc_id,
            filename=file.filename,
            chunks_processed=len(chunks),
            collection=COLLECTION_NAME,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except EnvironmentError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Configuration error: {exc}",
        )
    except Exception as exc:
        logger.exception("Ingestion failed for '%s'", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF: {exc}",
        )
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.post(
    "/chat",
    summary="Ask a question",
    tags=["Chat"],
    responses={
        200: {
            "description": "Streaming plain-text response OR JSON ChatResponse.",
        }
    },
)
async def chat(request: ChatRequest):
    """
    Answer a question using the RAG pipeline.

    - `stream=true` (default): returns a `text/plain` streaming response
      (compatible with `httpx` async streaming, curl, etc.).
    - `stream=false`: returns a JSON `ChatResponse` with the full answer
      and source chunk previews.

    Pass `doc_id` (returned by `/ingest`) to scope answers to one document.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    # ── Streaming mode ────────────────────────────────────────────────
    history = [h.model_dump() for h in request.history] if request.history else None

    if request.stream:
        async def token_stream():
            try:
                async for token in generate_answer_stream(
                    request.question,
                    COLLECTION_NAME,
                    doc_id=request.doc_id,
                    top_k=request.top_k,
                    history=history,
                ):
                    yield token
            except EnvironmentError as exc:
                yield f"\n\n[Configuration error: {exc}]"
            except Exception as exc:
                logger.exception("Streaming error")
                yield f"\n\n[Error: {exc}]"

        return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8")

    # ── Non-streaming mode ────────────────────────────────────────────
    try:
        answer, docs = generate_answer(
            request.question,
            COLLECTION_NAME,
            doc_id=request.doc_id,
            top_k=request.top_k,
            history=history,
        )

        sources = [
            SourceChunk(
                chunk_number=i,
                page=doc.metadata.get("page"),
                content_preview=(
                    doc.page_content.strip()[:300] + "…"
                    if len(doc.page_content) > 300
                    else doc.page_content.strip()
                ),
            )
            for i, doc in enumerate(docs, start=1)
        ]

        return ChatResponse(
            answer=answer,
            doc_id=request.doc_id,
            sources=sources,
        )

    except EnvironmentError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Configuration error: {exc}",
        )
    except Exception as exc:
        logger.exception("Chat error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ─────────────────────────────────────────────────
# Vercel serverless handler (Mangum wraps FastAPI as ASGI)
# ─────────────────────────────────────────────────
from mangum import Mangum
handler = Mangum(app, lifespan="off")


# ─────────────────────────────────────────────────
# Dev entry point
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
