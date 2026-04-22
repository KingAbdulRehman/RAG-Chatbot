"""
Gemini Embeddings + Qdrant Storage
────────────────────────────────────
Uses google-generativeai SDK directly for embeddings.
Model: models/gemini-embedding-001 (Google's current stable embedding model).
Dimension is auto-detected at runtime so collection is always created correctly.
"""

import logging
import os
from typing import List, Optional

import google.generativeai as genai
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

logger = logging.getLogger(__name__)

EMBED_MODEL = "models/gemini-embedding-001"


def _require_env(name: str) -> str:
    """Return an environment variable value or raise a clear error."""
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            "Please copy .env.example to .env and fill in your credentials."
        )
    return value


# ─────────────────────────────────────────────────
# Custom Embeddings
# ─────────────────────────────────────────────────

class GeminiEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings using google-generativeai SDK.
    Calls genai.embed_content() directly — simple and reliable.
    """

    def __init__(self, api_key: str, model: str = EMBED_MODEL) -> None:
        genai.configure(api_key=api_key)
        self._model = model
        logger.info("GeminiEmbeddings ready — model: %s", model)

    def embed_query(self, text: str) -> List[float]:
        result = genai.embed_content(
            model=self._model,
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            result = genai.embed_content(
                model=self._model,
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])
        return embeddings


def get_embeddings() -> GeminiEmbeddings:
    """Return a configured GeminiEmbeddings instance."""
    return GeminiEmbeddings(api_key=_require_env("GOOGLE_API_KEY"))


# ─────────────────────────────────────────────────
# Qdrant helpers
# ─────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    """Initialise and return a Qdrant Cloud client."""
    return QdrantClient(
        url=_require_env("QDRANT_URL"),
        api_key=_require_env("QDRANT_API_KEY"),
        timeout=30,
    )


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str,
    embedding_dim: int,
) -> None:
    """
    Create the Qdrant collection with the correct dimension.
    If the collection exists with a DIFFERENT dimension (e.g. leftover
    from a previous model), it is deleted and recreated automatically.
    """
    existing = {c.name for c in client.get_collections().collections}

    if collection_name in existing:
        info = client.get_collection(collection_name)
        try:
            actual_size = info.config.params.vectors.size
            if actual_size == embedding_dim:
                logger.info(
                    "Collection '%s' exists with correct dim=%d — reusing.",
                    collection_name, embedding_dim,
                )
                return
            else:
                logger.warning(
                    "Collection '%s' has dim=%d but model needs dim=%d. "
                    "Deleting and recreating…",
                    collection_name, actual_size, embedding_dim,
                )
                client.delete_collection(collection_name)
        except Exception:
            # Can't read dim — delete and start fresh to be safe
            logger.warning("Could not verify collection dim — recreating.")
            client.delete_collection(collection_name)

    logger.info(
        "Creating Qdrant collection '%s' with dim=%d", collection_name, embedding_dim
    )
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )

    # Create payload index on doc_id so per-document filtering works
    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.doc_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    logger.info("Collection '%s' created with doc_id index.", collection_name)


def embed_and_store(
    chunks: List[Document],
    collection_name: str,
    doc_id: Optional[str] = None,
) -> QdrantVectorStore:
    """Embed document chunks and upsert them into Qdrant."""
    if not chunks:
        raise ValueError("No chunks provided to embed_and_store.")

    embeddings = get_embeddings()
    client = get_qdrant_client()

    # Auto-detect actual vector dimension from the model
    logger.info("Detecting embedding dimension…")
    test_vector = embeddings.embed_query("dimension check")
    embedding_dim = len(test_vector)
    logger.info("Embedding dimension: %d", embedding_dim)

    ensure_collection_exists(client, collection_name, embedding_dim)

    if doc_id:
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id

    logger.info(
        "Embedding %d chunks → collection '%s' (doc_id=%s)…",
        len(chunks), collection_name, doc_id,
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    vector_store.add_documents(chunks)
    logger.info("Successfully stored %d vectors in Qdrant.", len(chunks))
    return vector_store


def get_vector_store(collection_name: str) -> QdrantVectorStore:
    """Return a QdrantVectorStore for an existing collection (query time)."""
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=collection_name,
        embedding=get_embeddings(),
    )


def collection_info(collection_name: str) -> dict:
    """Return basic stats about a Qdrant collection."""
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}

    if collection_name not in existing:
        return {"exists": False, "collection_name": collection_name}

    info = client.get_collection(collection_name)
    return {
        "exists": True,
        "collection_name": collection_name,
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
        "status": str(info.status),
    }
