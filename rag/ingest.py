"""
rag/ingest.py — PDF ingestion, chunking, embedding, and Chroma persistence.

Role 2 deliverable covered here:
1. Load uploaded pitch deck PDF with pypdf.PdfReader (direct, no deprecated community wrapper).
2. Split text into digestible chunks.
3. Embed chunks with Gemini embeddings when available.
4. Persist chunks to Chroma so downstream graph nodes can retrieve evidence.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import List

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from rag.retriever import DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR, get_embeddings


def load_pdf_documents(pdf_path: str | Path) -> List[Document]:
    """Load a PDF into LangChain Document objects, one per non-empty page."""

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path.name}")

    reader = PdfReader(str(pdf_path))
    cleaned: List[Document] = []
    for page_num, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        # Preserve meaningful newlines because the text splitter uses them as
        # high-priority separators. Only collapse excessive blank lines/spaces.
        text = re.sub(r"\n{3,}", "\n\n", raw)
        text = re.sub(r"[ \t]+", " ", text).strip()
        if text:
            cleaned.append(
                Document(
                    page_content=text,
                    metadata={"source": str(pdf_path), "page": page_num + 1},
                )
            )
    if not cleaned:
        raise ValueError("No extractable text found. The deck may be scanned images.")
    return cleaned


def split_documents(
    documents: List[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Split documents into overlapping chunks optimized for slide/PDF text."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def ingest_pdf_to_chroma(
    pdf_path: str | Path,
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    reset_collection: bool = True,
) -> Chroma:
    """Create or refresh a Chroma index from a pitch deck PDF.

    Returns the Chroma vector store so callers can immediately call
    `.as_retriever()` without reopening it.

    Note:
        If reset_collection=False and this function is called repeatedly on
        the same PDF/collection, chunks may be duplicated in Chroma. Keep the
        default reset_collection=True for demos and tests.
    """

    documents = load_pdf_documents(pdf_path)
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = get_embeddings()

    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    if reset_collection:
        # Safe reset for repeated demos/tests on the same collection name.
        existing = Chroma(
            collection_name=collection_name,
            persist_directory=str(persist_directory),
            embedding_function=embeddings,
        )
        try:
            existing.delete_collection()
        except Exception:
            pass

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )
    return vectorstore


def ingest_and_get_retriever(
    pdf_path: str | Path,
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    k: int = 6,
):
    """Convenience helper for notebooks and smoke tests."""

    vectorstore = ingest_pdf_to_chroma(
        pdf_path,
        persist_directory,
        collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
