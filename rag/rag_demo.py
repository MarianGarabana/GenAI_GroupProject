"""
rag/rag_demo.py — Standalone RAG Pipeline Demo

WHAT IS THIS?
-------------
This file demonstrates the complete RAG (Retrieval-Augmented Generation)
pipeline as taught in Sessions 5-6.

RAG solves a key problem: LLMs only know what they were trained on.
If you want the LLM to answer questions about a SPECIFIC document
(like a pitch deck), you need to "augment" it with that external information.

THE 3 STEPS OF RAG (as taught in Session 6):
  1. INDEXING   — Load the document and split it into chunks.
                  Convert chunks into vectors (embeddings) and store in Chroma.
  2. RETRIEVAL  — When a question comes in, find the most relevant chunks.
  3. AUGMENTATION — Put the relevant chunks into the LLM prompt as context.

SIMPLE ANALOGY:
  Imagine you're taking an open-book exam. Before the exam:
    1. INDEXING: You read the textbook and write index cards (embeddings).
       You organize the cards by topic in a box (Chroma vector store).
  During the exam:
    2. RETRIEVAL: You read the question and flip to the right index cards.
    3. AUGMENTATION: You use those cards + your own knowledge to write the answer.

CLASS CONCEPTS:
  - "The Power of Embeddings" .............. Session 5
  - "Retrieval-Augmented Generation (RAG)" . Session 6
  - Chroma vector database ................. Sessions 5-6
  - LCEL RAG chain ......................... Sessions 8-9

NOTEBOOK REFERENCE:
  Conchita_LangChain_RAG_june26.ipynb (Block 3)
"""

import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()


# ============================================================
# EMBEDDINGS MODEL
#
# WHAT: Embeddings convert text into a list of numbers (a "vector").
# WHY: Computers can't directly compare text, but they CAN compare numbers.
# Two texts with similar meaning will have similar vectors — they'll be
# "close" to each other in mathematical space.
#
# SIMPLE ANALOGY: Imagine a map of the world. Cities that are geographically
# close are also close on the map. Embeddings create a "meaning map" where
# texts with similar meaning are close together.
#
# CLASS CONCEPT: "The Power of Embeddings" — Session 5
# ============================================================
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# ============================================================
# STEP 1: INDEXING — Load document → Split → Embed → Store in Chroma
# ============================================================

def build_vectorstore(text: str, persist_dir: str = "/tmp/pitch_chroma") -> Chroma:
    """
    Takes raw text (from the pitch deck), splits it into chunks,
    converts each chunk into an embedding vector, and stores everything
    in a Chroma vector database.

    STEP 1 of RAG: INDEXING
    CLASS CONCEPT: Sessions 5-6

    Args:
        text: The full text extracted from the pitch deck PDF
        persist_dir: Where to save the Chroma database on disk

    Returns:
        A Chroma vectorstore ready for retrieval
    """

    # Split the text into overlapping chunks
    # WHY: LLMs have token limits. We can't put 50 pages into one prompt.
    # Instead we split into small pieces and only retrieve the relevant ones.
    # The overlap (200 chars) prevents losing context at chunk boundaries.
    #
    # CLASS CONCEPT: Text splitting — Session 6 (RAG Indexing)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # Each chunk is ~1000 characters
        chunk_overlap=200,  # 200 characters overlap between chunks
    )
    chunks = text_splitter.create_documents([text])

    # Convert text chunks into embedding vectors and store in Chroma
    # This is the "vector database" mentioned in Session 6
    #
    # CLASS CONCEPT: "Vector databases" — Session 6
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_dir,
    )

    return vectorstore


# ============================================================
# STEP 2 & 3: RETRIEVAL + AUGMENTATION — The RAG Chain
# ============================================================

def build_rag_chain(vectorstore: Chroma):
    """
    Builds a RAG chain that:
    1. Takes a question as input
    2. Retrieves the most relevant chunks from the vectorstore (RETRIEVAL)
    3. Puts those chunks into the Gemini prompt as context (AUGMENTATION)
    4. Returns Gemini's answer

    STEPS 2 & 3 of RAG: RETRIEVAL + AUGMENTATION
    CLASS CONCEPT: Sessions 5-6, 8-9

    This is the COMPLETE RAG pipeline in one chain.
    """

    # The retriever searches the vectorstore for relevant chunks
    # SESSION 6: "Retrieval" phase of RAG
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Find most similar chunks by vector distance
        search_kwargs={"k": 4},    # Return top 4 most relevant chunks
    )

    # Prompt template for the RAG chain
    # {context} gets filled with retrieved chunks
    # {question} gets filled with the user's question
    prompt = ChatPromptTemplate.from_template(
        """You are an expert VC analyst. Use ONLY the following pitch deck excerpts
to answer the question. If the answer is not in the excerpts, say "Not found in pitch deck."

PITCH DECK EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    def format_docs(docs):
        """Concatenate retrieved document chunks into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    # The complete RAG chain using LCEL (Sessions 8-9):
    #   retriever → format → into prompt → Gemini → parse to string
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ============================================================
# CONVENIENCE FUNCTION — Used by presentation demo
# ============================================================

def answer_pitch_question(pitch_text: str, question: str) -> str:
    """
    Full RAG pipeline: takes pitch text + a question, returns an answer
    grounded in the actual pitch deck content.

    Example usage:
        answer = answer_pitch_question(
            pitch_text=raw_pdf_text,
            question="What is the startup's total addressable market?"
        )
    """
    vectorstore = build_vectorstore(pitch_text)
    rag_chain = build_rag_chain(vectorstore)
    return rag_chain.invoke(question)
