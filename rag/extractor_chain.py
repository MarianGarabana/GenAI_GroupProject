"""
rag/extractor_chain.py — LCEL RAG chain for structured pitch claim extraction.

Role 2 deliverable covered here:
Chroma retriever -> prompt -> Gemini -> structured JSON/Pydantic output parser.
The chain extracts market size, team background, traction, and product description.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from rag.ingest import ingest_pdf_to_chroma
from rag.retriever import DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR, retrieve_context


class StartupClaims(BaseModel):
    """Structured output expected from the RAG extractor."""

    market_size: str = Field(description="Claimed TAM/SAM/SOM or market opportunity, including numbers if present.")
    team_background: str = Field(description="Founder/team credentials, prior roles, domain expertise, and advisors.")
    traction: str = Field(description="Revenue, users, growth, pilots, LOIs, partnerships, retention, or other proof.")
    product_description: str = Field(description="What the product does, target customer, and pain point solved.")


def _get_llm(model: str = "gemini-2.5-flash", temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    # Keep this lazy. Do not create a module-level LLM/chain object, otherwise
    # importing rag.extractor_chain would fail on machines without GOOGLE_API_KEY.
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set. Use fallback=True for offline extraction.")
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def build_extraction_chain(retriever, model: str = "gemini-2.5-flash"):
    """Build the LCEL RAG chain used by Role 2.

    Input: a user question string.
    Output: StartupClaims Pydantic object.
    """

    parser = PydanticOutputParser(pydantic_object=StartupClaims)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful venture capital analyst. Extract only facts supported by the supplied pitch deck context. "
                "If a field is absent, write 'Not mentioned in pitch deck'. Return structured JSON only.\n\n{format_instructions}",
            ),
            (
                "human",
                "Question: {question}\n\nRetrieved pitch deck context:\n{context}",
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    context_step = RunnableLambda(lambda q: retrieve_context(q, retriever))
    chain = (
        {"context": context_step, "question": RunnablePassthrough()}
        | prompt
        | _get_llm(model=model)
        | parser
    )
    return chain


def _keyword_fallback_from_text(text: str) -> StartupClaims:
    """Low-tech fallback when no LLM key is available.

    It selects sentences around investor keywords. This is intentionally simple,
    transparent, and useful for demos/tests where Gemini cannot be called.
    """

    normalized = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", normalized)

    def pick(patterns: list[str], default: str = "Not mentioned in pitch deck") -> str:
        hits = []
        for sentence in sentences:
            lower = sentence.lower()
            if any(pattern in lower for pattern in patterns):
                hits.append(sentence.strip())
            if len(" ".join(hits)) > 550:
                break
        return " ".join(hits)[:700] if hits else default

    return StartupClaims(
        market_size=pick(["market", "tam", "sam", "som", "billion", "$", "opportunity"]),
        team_background=pick(["founder", "team", "ex-", "experience", "advisor", "ceo", "cto"]),
        traction=pick(["revenue", "mrr", "arr", "users", "customers", "growth", "pilot", "partnership", "retention", "traction"]),
        product_description=pick(["product", "platform", "app", "software", "solution", "automates", "helps", "problem"]),
    )


def extract_claims_from_pdf(
    pdf_path: str | Path,
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    query: str = "Extract the startup's market size, team background, traction, and product description from this pitch deck.",
    fallback: bool = True,
    k: int = 6,
) -> StartupClaims:
    """End-to-end Role 2 helper: PDF -> Chroma -> Retriever -> LCEL extractor.

    With GOOGLE_API_KEY set, this uses Gemini. Without it, fallback=True returns
    a deterministic keyword-based extraction so the demo and tests still run.
    """

    vectorstore = ingest_pdf_to_chroma(
        pdf_path=pdf_path,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    if os.getenv("GOOGLE_API_KEY"):
        chain = build_extraction_chain(retriever)
        try:
            return chain.invoke(query)
        except OutputParserException:
            if not fallback:
                raise
            docs = retriever.invoke(query)
            return _keyword_fallback_from_text(" ".join(doc.page_content for doc in docs))
        except Exception as exc:
            raise RuntimeError(f"RAG extraction failed: {exc}") from exc

    if not fallback:
        raise RuntimeError("GOOGLE_API_KEY is not set and fallback=False.")

    docs = retriever.invoke(query)
    return _keyword_fallback_from_text(" ".join(doc.page_content for doc in docs))


def extract_claims_dict(*args, **kwargs) -> dict:
    """Same as extract_claims_from_pdf, but returns a plain dict for graph state."""

    return extract_claims_from_pdf(*args, **kwargs).model_dump()
