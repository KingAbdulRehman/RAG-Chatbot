"""
RAG Retriever + Answer Generator
──────────────────────────────────
1. Performs semantic search against Qdrant to find the top-k most
   relevant document chunks for a user query.
2. Builds a grounded prompt — including conversation history — and
   streams the answer from Gemini 2.5 Flash.
"""

import logging
import os
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.embedder import _require_env, get_vector_store

logger = logging.getLogger(__name__)

# How many past exchanges (user + assistant pairs) to include in context
MAX_HISTORY_TURNS = 5

# ─────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────

def get_llm(streaming: bool = True) -> ChatGoogleGenerativeAI:
    """Return a Gemini 2.5 Flash instance."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=_require_env("GOOGLE_API_KEY"),
        streaming=streaming,
        temperature=0.1,
        max_retries=2,
    )


# ─────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────

def rewrite_query(question: str) -> str:
    """
    Use the LLM to rewrite a user question into a retrieval-optimised
    search query.

    This fixes cases where the user's question contains words that happen
    to appear literally in the document but in a different context.

    Example:
        User:    "what is the topic of this document?"
        Rewrite: "main subject title overview introduction key themes"

        User:    "summarise it"
        Rewrite: "document summary overview key points conclusions"
    """
    llm = get_llm(streaming=False)
    messages = [
        SystemMessage(content=(
            "You are a search query optimizer for a document retrieval system.\n"
            "Your job: convert the user's question into a SHORT, keyword-rich search query "
            "that will find the most relevant text passages in a PDF document.\n\n"
            "Rules:\n"
            "- Output ONLY the search query — no explanation, no punctuation at the end.\n"
            "- Use 5–12 descriptive keywords or phrases.\n"
            "- Focus on the MEANING of the question, not its exact words.\n"
            "- For broad/summary questions, use terms like: overview introduction summary "
            "main topic key themes conclusions.\n"
            "- For specific questions, use the core subject matter keywords."
        )),
        HumanMessage(content=f"Question: {question}\nSearch query:"),
    ]
    rewritten = (llm | StrOutputParser()).invoke(messages).strip()
    logger.info("Query rewrite: '%s' → '%s'", question[:60], rewritten[:80])
    return rewritten


def retrieve_chunks(
    query: str,
    collection_name: str,
    top_k: int = 6,
    doc_id: Optional[str] = None,
) -> List[Document]:
    """
    Retrieve the top-k most semantically similar chunks from Qdrant.
    The query is rewritten by the LLM before embedding to improve
    retrieval accuracy.

    Args:
        query:           The user's original question.
        collection_name: Qdrant collection to search.
        top_k:           Number of chunks to retrieve.
        doc_id:          If provided, filters results to a specific document.

    Returns:
        List of relevant Document objects (may be empty).
    """
    # Rewrite the query for better semantic matching
    search_query = rewrite_query(query)

    vector_store = get_vector_store(collection_name)

    search_kwargs: dict = {"k": top_k}

    if doc_id:
        search_kwargs["filter"] = Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
        )

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(search_query)

    logger.info(
        "Retrieved %d chunks (doc_id=%s)",
        len(docs), doc_id,
    )
    return docs


def format_context(docs: List[Document]) -> str:
    """Format retrieved documents into a readable context block."""
    sections = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "?")
        text = doc.page_content.strip()
        sections.append(f"[Chunk {i} | Page {page}]\n{text}")
    return "\n\n---\n\n".join(sections)


def _build_messages(
    question: str,
    context: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> list:
    """
    Build the message list for the LLM including conversation history.

    Args:
        question: Current user question.
        context:  Retrieved document context.
        history:  List of {"role": "user"|"assistant", "content": "..."} dicts.
                  Only the last MAX_HISTORY_TURNS pairs are included.

    Returns:
        List of LangChain message objects ready for the LLM.
    """
    system = SystemMessage(content=(
        "You are SmartDoc, a precise and helpful document assistant.\n"
        "Answer questions using ONLY the document context provided below.\n\n"
        "Rules:\n"
        "- Use information from the context to answer directly and concisely.\n"
        "- If the answer is not in the context, say: "
        "'I could not find that information in the uploaded document.'\n"
        "- Never fabricate facts or use knowledge outside the context.\n"
        "- Cite the page number when quoting or referencing specific content.\n"
        "- Use markdown formatting (bold, bullets) where it improves clarity.\n"
        "- You have access to the conversation history — use it to understand "
        "follow-up questions and references like 'it', 'that', 'the previous answer'."
    ))

    messages = [system]

    # Add trimmed conversation history (most recent MAX_HISTORY_TURNS exchanges)
    if history:
        trimmed = history[-(MAX_HISTORY_TURNS * 2):]  # each turn = 2 entries
        for entry in trimmed:
            if entry["role"] == "user":
                messages.append(HumanMessage(content=entry["content"]))
            else:
                messages.append(AIMessage(content=entry["content"]))

    # Add current question with the retrieved context
    messages.append(HumanMessage(content=(
        f"Document context:\n"
        f"──────────────────────────────\n"
        f"{context}\n"
        f"──────────────────────────────\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )))

    return messages


# ─────────────────────────────────────────────────
# Answer Generation
# ─────────────────────────────────────────────────

async def generate_answer_stream(
    question: str,
    collection_name: str,
    doc_id: Optional[str] = None,
    top_k: int = 5,
    history: Optional[List[Dict[str, str]]] = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator — retrieves context, builds history-aware prompt,
    and streams the Gemini answer token-by-token.

    Args:
        question:        Current user question.
        collection_name: Qdrant collection to search.
        doc_id:          Optional document filter.
        top_k:           Number of chunks to retrieve.
        history:         Previous conversation turns (list of role/content dicts).

    Yields:
        Individual string tokens from Gemini 2.5 Flash.
    """
    docs = retrieve_chunks(question, collection_name, top_k=top_k, doc_id=doc_id)

    if not docs:
        yield (
            "I could not find any relevant information in the uploaded document. "
            "Please make sure a PDF has been processed first."
        )
        return

    context = format_context(docs)
    messages = _build_messages(question, context, history)

    llm = get_llm(streaming=True)
    parser = StrOutputParser()

    async for token in (llm | parser).astream(messages):
        yield token


def generate_answer(
    question: str,
    collection_name: str,
    doc_id: Optional[str] = None,
    top_k: int = 5,
    history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, List[Document]]:
    """
    Synchronous (non-streaming) answer generation with conversation memory.
    Used by the FastAPI /chat endpoint when stream=false.

    Returns:
        Tuple of (answer_string, source_documents).
    """
    docs = retrieve_chunks(question, collection_name, top_k=top_k, doc_id=doc_id)

    if not docs:
        return (
            "I could not find any relevant information in the uploaded document.",
            [],
        )

    context = format_context(docs)
    messages = _build_messages(question, context, history)

    llm = get_llm(streaming=False)
    parser = StrOutputParser()

    answer = (llm | parser).invoke(messages)
    return answer, docs
