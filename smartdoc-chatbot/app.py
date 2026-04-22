"""
SmartDoc Chatbot — Chainlit UI
──────────────────────────────
Entry point for the chat interface.

Run with:
    chainlit run app.py --watch
"""

import asyncio
import logging
import os
from pathlib import Path

import chainlit as cl
from dotenv import load_dotenv

from rag.embedder import collection_info, embed_and_store
from rag.loader import load_and_chunk_pdf
from rag.retriever import format_context, generate_answer_stream, retrieve_chunks

load_dotenv()

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

WELCOME_MESSAGE = """\
# Welcome to SmartDoc Chatbot 📚

I can answer questions about any PDF document you upload — grounded entirely in the document's content.

---

### How to use
1. **Upload a PDF** — click the 📎 attachment icon in the message box below
2. **Ask questions** — type anything you want to know about the document

### What I can do
- Answer questions with page-level citations
- Summarise sections
- Extract key facts, dates, and figures
- Compare or explain concepts from the document

---
*Powered by **Google Gemini 2.5 Flash** + **Qdrant Cloud** vector search*
"""


# ─────────────────────────────────────────────────
# Lifecycle Hooks
# ─────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialise per-session state and send the welcome message."""
    cl.user_session.set("doc_id", None)
    cl.user_session.set("pdf_name", None)
    cl.user_session.set("pdf_uploaded", False)
    cl.user_session.set("chat_history", [])

    await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Route incoming messages to upload handler or Q&A handler."""

    # ── File upload ───────────────────────────────────────────────────
    if message.elements:
        pdf_files = [
            el for el in message.elements
            if isinstance(el, cl.File) and (
                el.name.lower().endswith(".pdf")
                or getattr(el, "mime", "") == "application/pdf"
            )
        ]

        non_pdf = [
            el for el in message.elements
            if not isinstance(el, cl.File) or not (
                el.name.lower().endswith(".pdf")
                or getattr(el, "mime", "") == "application/pdf"
            )
        ]

        if non_pdf and not pdf_files:
            await cl.Message(
                content=(
                    "⚠️ **Unsupported file type.**\n\n"
                    "Please upload a **PDF** file (.pdf). "
                    "Other formats are not supported."
                )
            ).send()
            return

        if pdf_files:
            await _handle_pdf_upload(pdf_files[0])
            return

    # ── Text question ─────────────────────────────────────────────────
    if not cl.user_session.get("pdf_uploaded"):
        await cl.Message(
            content=(
                "📎 **No document uploaded yet.**\n\n"
                "Please upload a PDF using the attachment icon (📎) first, "
                "then ask your question."
            )
        ).send()
        return

    question = message.content.strip()
    if not question:
        await cl.Message(content="Please type a question.").send()
        return

    await _handle_question(question)


# ─────────────────────────────────────────────────
# PDF Upload Handler
# ─────────────────────────────────────────────────

async def _handle_pdf_upload(file_el: cl.File) -> None:
    """Process an uploaded PDF: chunk → embed → store in Qdrant."""
    status = cl.Message(content="⏳ Reading your PDF…")
    await status.send()

    try:
        file_path: str = file_el.path
        file_name: str = file_el.name

        # ── Step 1: Load + chunk (run in thread — blocking I/O) ──────
        status.content = f"⏳ Chunking `{file_name}`…"
        await status.update()

        chunks = await asyncio.to_thread(load_and_chunk_pdf, file_path)

        # ── Step 2: Embed + store (run in thread — many API calls) ───
        doc_id = Path(file_name).stem.replace(" ", "_")[:60]
        status.content = (
            f"⏳ Embedding **{len(chunks)} chunks** into Qdrant… "
            "(this may take 20–60 seconds)"
        )
        await status.update()

        await asyncio.to_thread(embed_and_store, chunks, COLLECTION_NAME, doc_id)

        # ── Update session ───────────────────────────────────────────
        cl.user_session.set("doc_id", doc_id)
        cl.user_session.set("pdf_name", file_name)
        cl.user_session.set("pdf_uploaded", True)
        cl.user_session.set("chat_history", [])

        status.content = (
            f"✅ **`{file_name}` processed successfully!**\n\n"
            f"- 📄 Pages loaded and split into **{len(chunks)} chunks**\n"
            f"- 🗄️ Stored in Qdrant collection `{COLLECTION_NAME}`\n"
            f"- 🔖 Document ID: `{doc_id}`\n\n"
            "You can now ask questions about this document."
        )
        await status.update()

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        status.content = f"❌ **File error:** {exc}"
        await status.update()

    except ValueError as exc:
        logger.error("Invalid PDF: %s", exc)
        status.content = f"❌ **Invalid PDF:** {exc}"
        await status.update()

    except EnvironmentError as exc:
        logger.error("Missing env var: %s", exc)
        status.content = (
            f"❌ **Configuration error:** {exc}\n\n"
            "Make sure your `.env` file contains `GOOGLE_API_KEY`, "
            "`QDRANT_URL`, and `QDRANT_API_KEY`."
        )
        await status.update()

    except Exception as exc:
        error_str = str(exc)
        logger.exception("Unexpected error during PDF processing")

        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            status.content = (
                "⚠️ **Gemini API quota reached**\n\n"
                "Too many embedding requests. Wait a minute and try uploading again."
            )
        elif "401" in error_str or "API_KEY_INVALID" in error_str or "403" in error_str:
            status.content = (
                "⚠️ **Invalid API key**\n\n"
                "Your `GOOGLE_API_KEY` was rejected. Please check your `.env` file."
            )
        elif "qdrant" in error_str.lower() or "UnexpectedResponse" in error_str:
            status.content = (
                "⚠️ **Qdrant error**\n\n"
                "Could not connect to the vector database. "
                "Check your `QDRANT_URL` and `QDRANT_API_KEY`."
            )
        else:
            status.content = (
                "⚠️ **Upload failed**\n\n"
                "An unexpected error occurred while processing the PDF. "
                "Please try again or check the terminal for details."
            )
        await status.update()


# ─────────────────────────────────────────────────
# Question Handler
# ─────────────────────────────────────────────────

async def _handle_question(question: str) -> None:
    """Retrieve context, stream the LLM answer, and show source chunks."""
    doc_id: str = cl.user_session.get("doc_id")
    pdf_name: str = cl.user_session.get("pdf_name", "document")

    # Track history
    history: list = cl.user_session.get("chat_history", [])
    history.append({"role": "user", "content": question})

    # Show "Thinking…" indicator
    thinking = cl.Message(content="🤔 **Thinking…** retrieving relevant sections")
    await thinking.send()

    try:
        # ── Retrieve source chunks (blocking — run in thread) ────────
        docs = await asyncio.to_thread(
            retrieve_chunks, question, COLLECTION_NAME, 5, doc_id
        )

        if not docs:
            thinking.content = (
                "⚠️ No relevant information found in the uploaded document.\n\n"
                "Try rephrasing your question or uploading a different PDF."
            )
            await thinking.update()
            return

        # ── Stream the answer ─────────────────────────────────────────
        thinking.content = "🤔 **Generating answer…**"
        await thinking.update()

        answer_msg = cl.Message(content="")
        await answer_msg.send()

        # Remove thinking indicator once streaming starts
        await thinking.remove()

        tokens: list[str] = []
        async for token in generate_answer_stream(
            question, COLLECTION_NAME, doc_id=doc_id, history=history
        ):
            tokens.append(token)
            await answer_msg.stream_token(token)

        await answer_msg.update()

        full_answer = "".join(tokens)
        history.append({"role": "assistant", "content": full_answer})
        cl.user_session.set("chat_history", history)

        # ── Display source chunks as side elements ────────────────────
        source_elements: list[cl.Text] = []
        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page", "?")
            preview = doc.page_content.strip()
            if len(preview) > 400:
                preview = preview[:400] + "…"
            source_elements.append(
                cl.Text(
                    name=f"Source {i} — Page {page}",
                    content=(
                        f"**From `{pdf_name}` — Page {page}**\n\n"
                        f"{preview}"
                    ),
                    display="side",
                )
            )

        if source_elements:
            await cl.Message(
                content=(
                    f"📚 **{len(docs)} source chunk(s)** used from `{pdf_name}` "
                    "— click to expand:"
                ),
                elements=source_elements,
            ).send()

    except EnvironmentError as exc:
        logger.error("Missing env var: %s", exc)
        thinking.content = (
            "⚠️ **Configuration error**\n\n"
            "A required API key or setting is missing. "
            "Please check your `.env` file."
        )
        await thinking.update()

    except Exception as exc:
        error_str = str(exc)
        logger.exception("Error generating answer")

        # ── Quota / rate limit ────────────────────────────────────────
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            thinking.content = (
                "⚠️ **Gemini API quota reached**\n\n"
                "You've hit the free-tier rate limit. Please wait a moment and try again.\n\n"
                "> Free tier: 15 requests/min · 1M tokens/day"
            )

        # ── Invalid / expired API key ─────────────────────────────────
        elif "401" in error_str or "API_KEY_INVALID" in error_str or "403" in error_str:
            thinking.content = (
                "⚠️ **Invalid API key**\n\n"
                "Your Gemini API key was rejected. "
                "Please check `GOOGLE_API_KEY` in your `.env` file."
            )

        # ── Network / timeout ─────────────────────────────────────────
        elif "timeout" in error_str.lower() or "ConnectionError" in error_str:
            thinking.content = (
                "⚠️ **Connection error**\n\n"
                "Could not reach the Gemini API. "
                "Please check your internet connection and try again."
            )

        # ── Qdrant error ──────────────────────────────────────────────
        elif "qdrant" in error_str.lower() or "UnexpectedResponse" in error_str:
            thinking.content = (
                "⚠️ **Vector database error**\n\n"
                "Could not retrieve results from Qdrant. "
                "Please check your `QDRANT_URL` and `QDRANT_API_KEY`."
            )

        # ── Generic fallback (no raw error shown to user) ─────────────
        else:
            thinking.content = (
                "⚠️ **Something went wrong**\n\n"
                "An unexpected error occurred. Please try again.\n\n"
                "> If this keeps happening, restart the app or check the terminal logs."
            )

        await thinking.update()
