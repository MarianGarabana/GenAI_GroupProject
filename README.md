# VC Startup Pitch Evaluator

> **IE University — Generative AI | Master in Business Analytics & Data Science**
> Final Group Project — Built with LangGraph, Google Gemini, RAG, and Multi-Agent orchestration

---
## Streamlit App Link

https://vc-analyst.streamlit.app/ 

## What It Does

A VC analyst uploads a startup pitch deck (PDF). The system automatically:

1. **Reads** the PDF and extracts all text
2. **Extracts** the 4 key investor metrics — market size, team, traction, product
3. **Validates** each claim by searching the web for real-world evidence
4. **Scores** each dimension 0–10 using Gemini AI
5. **Pauses** for a human analyst review if any score is below 6
6. **Writes** a professional investment recommendation memo

The result: a full due diligence report in seconds instead of hours.

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────┐    ┌─────────┐    ┌──────────┐    ┌───────┐
│  INGEST │───▶│ EXTRACT │───▶│ VALIDATE │───▶│ SCORE │
│ (PyPDF) │    │(Gemini) │    │(DuckDuck)│    │(Gemini│
└─────────┘    └─────────┘    └──────────┘    └───┬───┘
                                                   │
                                    ┌──────────────┤
                                    │              │
                             score < 6?      all ≥ 6?
                                    │              │
                                    ▼              │
                             ┌────────────┐        │
                             │   HUMAN    │        │
                             │   REVIEW   │        │
                             │(interrupt) │        │
                             └─────┬──────┘        │
                                   │               │
                                   ▼               ▼
                              ┌──────────────────────┐
                              │      WRITE MEMO      │
                              │      (Gemini)        │
                              └──────────────────────┘
                                         │
                                         ▼
                                  Investment Memo
```

Built with **LangGraph StateGraph** — a directed graph where each box is an autonomous AI agent passing state to the next.

---

## Generative AI Concepts Applied

Every component in this project maps directly to a concept taught in class:

| Class Concept | Session | Where Used in Project |
|---|---|---|
| **LangGraph — TypedDict State** | 9–10 | `graph/state.py` — `PitchState` shared across all agents |
| **LangGraph — StateGraph, add\_node, add\_edge** | 9–10 | `graph/graph.py` — full pipeline graph |
| **LangGraph — Conditional Edges** | 9–10 | `graph/graph.py` — routes to human review if score < 6 |
| **LangGraph — MemorySaver + interrupt()** | 9–10 | `graph/graph.py` + `nodes.py` — human-in-the-loop pause/resume |
| **Google Gemini (ChatGoogleGenerativeAI)** | 3, 10–11 | Every node that calls the LLM — uses `gemini-2.5-flash` |
| **LCEL Chains (prompt \| llm \| parser)** | 8–9 | `extract_node`, `score_node`, `validate_node`, `write_memo_node` |
| **Pydantic structured output (PydanticOutputParser)** | 8–9 | `chains/scorer.py` + `chains/memo_writer.py` — validates Gemini output against typed models |
| **ChatPromptTemplate** | 8–9 | Every Gemini call uses structured prompt templates |
| **AI Agents + Tool Use** | 7 | `agents/validator_agent.py` — ReAct agent with DuckDuckGo + Wikipedia |
| **ReAct framework (Reason + Act)** | 7 | `agents/validator_agent.py` — agent reasons which tool to call, acts, observes result, repeats until done |
| **llm.bind_tools + ToolNode** | 9–10 | `agents/validator_agent.py` — same pattern as class news writer notebook |
| **Function Calling** | 10–11 | `@tool` decorated functions in `agents/tools.py` |
| **RAG Pipeline (Indexing → Retrieval → Augmentation)** | 6 | `rag/rag_demo.py` — Chroma vector store + retrieval chain |
| **Embeddings + Vector Search** | 5 | `rag/rag_demo.py` — `GoogleGenerativeAIEmbeddings` + cosine similarity |

---

## Tech Stack

| Technology | Role |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Multi-agent orchestration, human-in-the-loop |
| [LangChain](https://github.com/langchain-ai/langchain) | LCEL chains, prompt templates, tools |
| [Google Gemini 2.5 Flash](https://ai.google.dev/) | LLM for extraction, scoring, memo writing |
| [Chroma](https://www.trychroma.com/) | Vector database for RAG pipeline |
| [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) | Web search tool for claim validation |
| [Wikipedia](https://pypi.org/project/wikipedia/) | Background research tool for claim validation |
| [PyPDF](https://pypdf.readthedocs.io/) | PDF text extraction |
| [Streamlit](https://streamlit.io/) | Web UI |

---

## Project Structure

```
GenAI_GroupProject/
│
├── graph/                      # Role 1 — Graph Architect
│   ├── state.py                # PitchState TypedDict (shared state contract)
│   ├── nodes.py                # All 6 agent node implementations
│   └── graph.py                # StateGraph: nodes, edges, compile, visualization
│
├── rag/                        # Role 2 — RAG Engineer
│   ├── ingest.py               # PDF loader + text splitter → Chroma index
│   ├── retriever.py            # Similarity search wrapper over Chroma
│   ├── extractor_chain.py      # LCEL retrieval chain: query → docs → Gemini answer
│   ├── rag_demo.py             # End-to-end RAG demo script
│   └── __init__.py
│
├── agents/                     # Role 3 — Agent Engineer
│   ├── tools.py                # @tool: search_web (DuckDuckGo) + search_wikipedia
│   ├── validator_agent.py      # ReAct sub-graph: validator node + ToolNode loop
│   └── __init__.py             # Exports validate_claims(state) for Role 1 to wire in
│
├── chains/                     # Role 4 — Output Engineer
│   ├── output_models.py        # Pydantic models: ScoreResult + InvestmentMemo
│   ├── scorer.py               # LCEL chain: claims+evidence → Gemini → ScoreResult
│   └── memo_writer.py          # LCEL chain: full state → Gemini → formatted memo
│
├── data/
│   ├── sample_pitch.pdf        # Demo deck — JobAnyDay (used in pipeline smoke test)
│   └── ecocart_pitch.pdf       # EcoCart AI deck (used in RAG unit tests)
│
├── tests/                      # Role 5 — Integration Lead
│   ├── test_chains.py          # Unit tests: scorer + memo writer chains
│   ├── test_rag_pipeline.py    # Unit tests: Chroma ingestion + retrieval
│   └── test_validator_agent.py # Unit tests: ReAct agent + tool calls
│
├── conftest.py                 # Global pytest config — IE University SSL proxy fix
├── test_r4.py                  # Smoke test: full pipeline end-to-end (run directly)
├── app.py                      # Streamlit UI
├── requirements.txt            # pip dependencies
├── pyproject.toml              # uv/build dependencies (mirrors requirements.txt)
└── .env.example                # API key template
```

---

## Team Roles

| Role | Responsibility | Key Technology |
|---|---|---|
| **Marian Garabana** — Graph Architect | LangGraph StateGraph, edges, human interrupt, state schema | LangGraph, MemorySaver |
| **Siddharth Murali** — RAG Engineer | PDF ingestion, text splitting, Chroma vector store, claim extraction | Chroma, Embeddings |
| **Lea Hochar** — Agent Engineer | ReAct validation agent, DuckDuckGo + Wikipedia tools, ToolNode sub-graph | `llm.bind_tools`, `ToolNode` |
| **Stephan Pentchev** — Output Engineer | Scoring chain, investment memo writer | LCEL, Gemini |
| **Dominique Robson** — Integration Lead | Streamlit UI, end-to-end wiring, live demo | Streamlit |
| **Smaragda Apostolou & Andrea Sabatés** — Presentation Lead | Slides, business case, architecture story, live demo script | — |

---

## How the Human-in-the-Loop Works

One of the most advanced features — **LangGraph's `interrupt()` mechanism**:

```
1. Graph runs ingest → extract → validate → score
2. Score node finds team score = 4  (below threshold of 6)
3. Conditional edge routes to human_review node
4. human_review_node calls interrupt()  ◄── GRAPH FREEZES HERE
5. Streamlit UI shows scores + review form to analyst
6. Analyst types feedback and clicks Submit
7. Streamlit calls graph.invoke(Command(resume=feedback), config)
8. GRAPH UNFREEZES — continues with human feedback included
9. write_memo_node generates final memo incorporating analyst notes
```

This pattern mirrors how real VC firms operate — AI does the heavy lifting, humans make the final call.

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/MarianGarabana/GenAI_GroupProject.git
cd GenAI_GroupProject

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
# Get a free key at: https://aistudio.google.com

# 5. Run
streamlit run app.py
```

---

## Development Timeline

| Days | Milestone |
|---|---|
| Tue–Wed | Role 1 graph backend + Role 2 RAG scaffold + repo setup |
| Thu–Fri | Roles 3 & 4 implement their nodes, Role 5 wires Streamlit |
| Sat–Sun | Full integration, demo dry-runs with real pitch decks |
| Mon | Final polish, live demo, presentation |
