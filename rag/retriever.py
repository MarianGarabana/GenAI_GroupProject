"""
rag/retriever.py — Chroma retriever setup for the VC pitch evaluator.

Role 2 owns this file. It provides a single, reusable way for the rest of the
project to open the persisted Chroma vector store and retrieve relevant pitch
deck chunks.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, List, Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

DEFAULT_COLLECTION_NAME = "pitch_deck_claims"
DEFAULT_PERSIST_DIR = "chroma_db"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"


class HashEmbeddings(Embeddings):
    """Small deterministic local embedding fallback for tests and demos.

    Gemini embeddings are used when GOOGLE_API_KEY is present. This fallback
    keeps the pipeline runnable in class/CI without paid API access. It is not
    meant for production search quality, but it preserves the same LangChain
    interface as Gemini embeddings.
    """

    def __init__(self, size: int = 768) -> None:
        self.size = size

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.size
        tokens = [tok.strip(".,:;!?()[]{}'\"$%+-").lower() for tok in text.split()]
        for token in tokens:
            if not token:
                continue
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.size
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def get_embeddings(use_local_fallback: bool = True) -> Embeddings:
    """Return Gemini embeddings, or a deterministic local fallback.

    Set USE_LOCAL_EMBEDDINGS=true in .env to force offline mode.
    """

    force_local = os.getenv("USE_LOCAL_EMBEDDINGS", "").lower() in {"1", "true", "yes"}
    has_key = bool(os.getenv("GOOGLE_API_KEY"))
    if force_local or (use_local_fallback and not has_key):
        return HashEmbeddings()
    return GoogleGenerativeAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)


def get_vectorstore(
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embeddings: Optional[Embeddings] = None,
    require_existing: bool = False,
) -> Chroma:
    """Open an existing Chroma vector store for pitch deck chunks.

    Args:
        require_existing: When True, raise a clear error if the Chroma
            persistence folder is missing. This catches common demo mistakes
            such as running retrieval before ingestion or from the wrong
            working directory.
    """

    persist_directory = Path(persist_directory)
    if require_existing and not persist_directory.exists():
        raise RuntimeError(
            f"Chroma directory not found: {persist_directory}. "
            "Run ingest_pdf_to_chroma(...) before opening the retriever."
        )

    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=embeddings or get_embeddings(),
    )


def get_retriever(
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embeddings: Optional[Embeddings] = None,
    k: int = 6,
    require_existing: bool = True,
):
    """Return a LangChain retriever over the persisted Chroma store."""

    vectorstore = get_vectorstore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embeddings=embeddings,
        require_existing=require_existing,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_context(query: str, retriever, max_chars: int = 6000) -> str:
    """Retrieve pitch deck chunks and format them as prompt context."""

    docs: Iterable[Document] = retriever.invoke(query)
    parts = []
    total = 0
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        text = doc.page_content.strip()
        if not text:
            continue
        block = f"[Chunk {i} | source={source} | page={page}]\n{text}"
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(block) > remaining:
            if parts:
                break
            block = block[:remaining]
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)
