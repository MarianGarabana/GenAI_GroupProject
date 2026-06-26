"""Role 2 RAG package: PDF ingestion, Chroma retrieval, and LCEL claim extraction."""

from rag.ingest import ingest_pdf_to_chroma, load_pdf_documents, split_documents
from rag.retriever import get_retriever, get_vectorstore
from rag.extractor_chain import StartupClaims, extract_claims_from_pdf, extract_claims_dict, build_extraction_chain

__all__ = [
    "ingest_pdf_to_chroma",
    "load_pdf_documents",
    "split_documents",
    "get_retriever",
    "get_vectorstore",
    "StartupClaims",
    "extract_claims_from_pdf",
    "extract_claims_dict",
    "build_extraction_chain",
]
