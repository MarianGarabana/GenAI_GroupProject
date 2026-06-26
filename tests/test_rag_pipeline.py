"""Unit tests for Role 2: PDF ingestion, Chroma retrieval, and claim extraction."""

from pathlib import Path

from langchain_core.documents import Document

from rag.extractor_chain import StartupClaims, extract_claims_from_pdf
from rag.retriever import retrieve_context
from rag.ingest import ingest_pdf_to_chroma, load_pdf_documents, split_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PDF = PROJECT_ROOT / "data" / "ecocart_pitch.pdf"


def test_load_and_split_sample_pitch_pdf():
    docs = load_pdf_documents(SAMPLE_PDF)
    chunks = split_documents(docs, chunk_size=450, chunk_overlap=50)

    assert docs
    assert chunks
    assert any("EcoCart AI" in chunk.page_content for chunk in chunks)
    assert all("chunk_id" in chunk.metadata for chunk in chunks)


def test_chroma_retriever_returns_relevant_chunks(tmp_path):
    vectorstore = ingest_pdf_to_chroma(
        SAMPLE_PDF,
        persist_directory=tmp_path / "chroma",
        collection_name="test_pitch_deck_claims",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    docs = retriever.invoke("What traction, customers, or revenue does the startup have?")

    assert docs
    # HashEmbeddings is an offline fallback based on token overlap, not true semantic similarity.
    joined = " ".join(doc.page_content for doc in docs)
    assert "MRR" in joined or "customers" in joined or "traction" in joined.lower()


def test_extract_claims_from_pdf_offline_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("USE_LOCAL_EMBEDDINGS", "true")

    claims = extract_claims_from_pdf(
        SAMPLE_PDF,
        persist_directory=tmp_path / "chroma",
        collection_name="test_pitch_deck_extract",
    )

    assert isinstance(claims, StartupClaims)
    assert "EcoCart AI" in claims.product_description or "EcoCart AI" in claims.team_background
    assert claims.market_size != "Not mentioned in pitch deck"
    assert claims.traction != "Not mentioned in pitch deck"


def test_retrieve_context_truncates_large_first_chunk_instead_of_empty():
    class MockRetriever:
        def invoke(self, query):
            return [Document(page_content="x" * 8000, metadata={"source": "mock.pdf", "page": 1})]

    result = retrieve_context("any query", MockRetriever(), max_chars=6000)

    assert result
    assert len(result) <= 6000
    assert "Chunk 1" in result
