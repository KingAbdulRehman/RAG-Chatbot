"""
PDF Loader and Chunker
──────────────────────
Loads a PDF file and splits it into overlapping text chunks
suitable for embedding and semantic search.
"""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_and_chunk_pdf(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Load a PDF file from disk and split into overlapping text chunks.

    Args:
        file_path:     Absolute or relative path to the PDF file.
        chunk_size:    Target size (characters) for each chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        List of LangChain Document objects, each with page_content
        and metadata (source, page, start_index).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file is not a PDF or is empty.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a .pdf file, got '{path.suffix}'. "
            "Only PDF documents are supported."
        )

    logger.info("Loading PDF: %s", file_path)

    loader = PyPDFLoader(str(path))
    pages: List[Document] = loader.load()

    if not pages:
        raise ValueError(
            "The PDF could not be read or appears to be empty. "
            "Make sure it is not password-protected."
        )

    logger.info("Loaded %d pages from '%s'", len(pages), path.name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: List[Document] = splitter.split_documents(pages)

    if not chunks:
        raise ValueError("PDF was loaded but produced no text chunks after splitting.")

    logger.info(
        "Split '%s' into %d chunks (size=%d, overlap=%d)",
        path.name,
        len(chunks),
        chunk_size,
        chunk_overlap,
    )

    return chunks
