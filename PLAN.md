# VC Pitch Evaluator — Full Backend Plan (Role 1: Graph Architect)

**Date:** 2026-06-24  
**Course:** Generative AI — IE University MBDS  
**Professor:** Conchita Diaz Cantarero (Head of AI Education, Google Cloud EMEA)  
**Role:** Role 1 — Graph Architect (LangGraph StateGraph, edges, human interrupt)

---

## What Are We Building?

We are building a **smart assistant for venture capital analysts**. The analyst uploads a startup's pitch deck (a PDF), and our AI system automatically:

1. **Reads** the PDF and extracts all the text
2. **Identifies** the 4 key claims investors care about (market size, team, traction, product)
3. **Searches the web** to check if those claims are realistic
4. **Scores** each dimension from 0 to 10
5. **Asks a human** to review if any score is below 6
6. **Writes** a professional investment memo with a final recommendation

> **Simple analogy:** Think of it like an assembly line at a factory. The PDF goes in one end, passes through 6 different "stations" (agents), and comes out the other end as a complete investment report. Each station does one specific job and passes its work to the next.

---

## Architecture Overview

```
PDF File
   │
   ▼
[INGEST] ──► [EXTRACT] ──► [VALIDATE] ──► [SCORE]
                                              │
                              ┌───────────────┤
                              │               │
                     any score < 6?      all scores ≥ 6?
                              │               │
                              ▼               ▼
                       [HUMAN REVIEW]   [WRITE MEMO]
                              │               │
                              └───────────────┘
                                      │
                                      ▼
                                [Investment Memo]
```

---

## Class Session Map — Every Concept Used

| Concept / Class Topic | Session Taught | Where Used in Project |
|---|---|---|
| **LangGraph — Defining The State (TypedDict)** | Session 9-10 | `graph/state.py` — `PitchState` |
| **LangGraph — StateGraph, add_node, set_entry_point** | Session 9-10 | `graph/graph.py` — `build_graph()` |
| **LangGraph — add_edge (linear connections)** | Session 9-10 | `graph/graph.py` — ingest→extract→validate→score |
| **LangGraph — add_conditional_edges (routing)** | Session 9-10 | `graph/graph.py` — score→human or score→memo |
| **LangGraph — compile, graph.stream()** | Session 9-10 | `graph/graph.py` — `build_graph()` |
| **LangGraph — MemorySaver (checkpointing)** | Session 9-10 (advanced) | `graph/graph.py` — `MemorySaver()` |
| **LangGraph — interrupt() human-in-the-loop** | Session 9-10 (advanced) | `graph/nodes.py` — `human_review_node()` |
| **Gemini Model (ChatGoogleGenerativeAI)** | Sessions 3, 10-11 | `graph/nodes.py` — all LLM calls |
| **LCEL Chains (prompt \| llm \| parser)** | Sessions 8-9 | `graph/nodes.py` — extract, score, write_memo |
| **ChatPromptTemplate** | Sessions 8-9 | `graph/nodes.py` — every node that calls Gemini |
| **Agents — Tool Use** | Session 7 | `graph/nodes.py` — `validate_node()` |
| **DuckDuckGo Search Tool** | Sessions 8-9 | `graph/nodes.py` — `validate_node()` |
| **Function Calling / Tool Binding** | Sessions 10-11 | `graph/nodes.py` — search tool invocation |
| **RAG — Indexing (text extraction from PDF)** | Session 6 | `graph/nodes.py` — `ingest_node()` |
| **Embeddings (vector representation)** | Session 5 | `rag/rag_demo.py` — Chroma demo |
| **RAG — Full Pipeline (Chroma + retrieval)** | Sessions 5-6 | `rag/rag_demo.py` — standalone demo |
| **Graph Visualization (draw_mermaid_png)** | Session 9-10 | `graph/graph.py` — `get_graph_image()` |

---

## File Structure

```
GenAI_GroupProject/
│
├── graph/                        ← ROLE 1 OWNS THIS ENTIRE FOLDER
│   ├── __init__.py               ← Makes 'graph' a Python package
│   ├── state.py                  ← PitchState TypedDict (shared state)
│   ├── nodes.py                  ← All 6 node implementations
│   └── graph.py                  ← StateGraph build + compile
│
├── rag/                          ← Supplementary RAG demo
│   └── rag_demo.py               ← Standalone Chroma + embeddings demo
│
├── app.py                        ← Streamlit UI (Role 5)
├── requirements.txt              ← Updated dependencies
├── .env.example                  ← API key template
├── PLAN.md                       ← This file
└── PROJECT_HISTORY.md            ← What was built + pending tasks
```

---

## Detailed File Plans

---

### File 1: `graph/state.py` — The Shared State

**What it does:** Defines the "folder" that gets passed between every agent. Every node can read from it and write to it.

**Simple explanation:** Think of the state like a relay race baton. Each runner (agent) grabs it, does their part, and hands it to the next runner. The baton remembers everything that happened so far.

**Class concept:** `TypedDict` for state — *taught in Session 9-10*, from `Conchita_LangGraph_Core_Concepts_june26.ipynb`, section "Defining The State"

```python
class PitchState(TypedDict):
    pdf_path: str                 # Input from user
    raw_text: str                 # After ingest
    extracted_claims: dict        # After extract
    validation_results: dict      # After validate
    scores: dict                  # After score
    human_review_required: bool   # Graph control
    human_feedback: Optional[str] # From human analyst
    investment_memo: Optional[str] # Final output
    error: Optional[str]          # Error tracking
```

---

### File 2: `graph/nodes.py` — The Agent Functions

Each "node" is just a Python function that:
- Takes the current state as input
- Does its job
- Returns a dict with the fields it updated

**Node 1 — `ingest_node`**
- **What:** Reads the PDF and extracts all text
- **Tools:** PyPDF (`PdfReader`)
- **Class concept:** RAG Pipeline Step 1 — *Session 6* ("Indexing" phase of RAG)

**Node 2 — `extract_node`**
- **What:** Uses Gemini to find the 4 key claims in the text
- **Tools:** `ChatPromptTemplate | ChatGoogleGenerativeAI | JsonOutputParser`
- **Class concept:** LCEL Chains — *Sessions 8-9*. Gemini — *Session 3*

**Node 3 — `validate_node`**
- **What:** Searches the web for each claim using DuckDuckGo, then uses Gemini to assess whether the claim is realistic
- **Tools:** `DuckDuckGoSearchRun` (web tool), LCEL chain for assessment
- **Class concept:** AI Agents + Tool Use — *Session 7*. Function calling — *Sessions 10-11*

**Node 4 — `score_node`**
- **What:** Gemini scores each dimension 0-10 based on claims + validation evidence
- **Tools:** LCEL chain with `JsonOutputParser`
- **Class concept:** LCEL Chains — *Sessions 8-9*

**Node 5 — `human_review_node`**
- **What:** PAUSES the graph and waits for a human analyst's feedback
- **Tools:** LangGraph `interrupt()`
- **Class concept:** Human-in-the-loop interrupt — *Session 9-10* (advanced)

**Node 6 — `write_memo_node`**
- **What:** Gemini writes the final investment memo combining all evidence + scores + human feedback
- **Tools:** LCEL chain
- **Class concept:** LCEL Chains — *Sessions 8-9*

---

### File 3: `graph/graph.py` — The StateGraph

**What it does:** Connects all 6 nodes into a graph with edges and routing logic.

**Class concept:** All of Session 9-10 — `StateGraph`, `add_node`, `add_edge`, `add_conditional_edges`, `set_entry_point`, `compile`

**Code pattern (identical to class notebook):**
```python
workflow = StateGraph(PitchState)
workflow.add_node("ingest", ingest_node)
workflow.set_entry_point("ingest")
workflow.add_edge("ingest", "extract")
workflow.add_conditional_edges("score", needs_human_review)
graph = workflow.compile(checkpointer=MemorySaver())
```

---

### File 4: `rag/rag_demo.py` — RAG Pipeline Demo

**What it does:** A standalone demo of the full RAG pipeline — exactly as taught in Sessions 5-6. Uses Chroma to store pitch deck text as embeddings, then retrieves relevant chunks for answering questions.

**Class concept:** Embeddings — *Session 5*. RAG (Indexing → Retrieval → Augmentation) — *Session 6*

---

## How the Human Interrupt Works

This is the most advanced feature — it goes beyond the class exercises.

```
1. Graph runs: ingest → extract → validate → score
2. Score node finds team score = 4 (below 6)
3. Conditional edge routes to "human_review"
4. human_review_node calls interrupt({scores: ..., claims: ...})
5. *** GRAPH FREEZES HERE ***
6. Streamlit UI shows the scores and a form for the analyst
7. Analyst types feedback and clicks "Submit Review"
8. Streamlit calls: graph.invoke(Command(resume=feedback), config)
9. *** GRAPH UNFREEZES ***
10. Continues to write_memo with the human feedback included
```

**Why this impresses:** This is production-grade GenAI — not just a chatbot, but a real workflow with human oversight built in.

---

## How to Run (after integration with Role 5)

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## Grading Alignment

| Grading Criterion | How Project Addresses It |
|---|---|
| Business Use Case (20%) | VC pitch evaluation = real business problem with clear ROI |
| Technical Depth (25%) | LangGraph + RAG + Agents + Human interrupt + Gemini |
| MVP Integration (25%) | Streamlit → graph.invoke → live results (Role 5) |
| Presentation (20%) | Architecture diagram + every concept mapped to class session |
| Live Demo (10%) | End-to-end: upload PDF → get investment memo in real time |
