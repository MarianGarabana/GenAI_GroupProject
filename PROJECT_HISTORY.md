# PROJECT HISTORY — VC Startup Pitch Evaluator

**Project:** VC Startup Pitch Evaluator  
**Course:** Generative AI — IE University MBDS  
**Professor:** Conchita Diaz Cantarero (Head of AI Education, Google Cloud EMEA)  
**Group Presentation weight:** 30% of final grade  

---

## Session 1 — 2026-06-24

**Author:** Marian Garabana (Role 1 — Graph Architect) + Claude (GenAI Expert)

### What Was Built

---

#### `PLAN.md` (new)

Full architectural plan document covering:
- What the project builds and why
- Architecture flow diagram (ingest → extract → validate → score → conditional → memo)
- Table mapping every class concept (Sessions 1-11) to the exact file and function where it's used
- Detailed plan for each file with simple analogies
- Grading alignment table

---

#### `graph/__init__.py` (new)

Makes the `graph/` folder a proper Python package. Exports `graph`, `build_graph`, `get_graph_image`, and `PitchState` so the Streamlit app only needs to do `from graph import graph`.

---

#### `graph/state.py` (new)

**Class concept used:** "Defining The State" using `TypedDict` — **Session 9-10**  
**Notebook:** `Conchita_LangGraph_Core_Concepts_june26.ipynb`, section "Defining The State"

Defines `PitchState` — the shared "folder" that flows through every node in the graph.

```python
class PitchState(TypedDict):
    pdf_path: str               # Input: path to uploaded PDF
    raw_text: str               # After ingest_node
    extracted_claims: dict      # After extract_node
    validation_results: dict    # After validate_node
    scores: dict                # After score_node
    human_review_required: bool # Routing flag
    human_feedback: Optional[str]  # From human analyst
    investment_memo: Optional[str] # Final output
    error: Optional[str]        # Error tracking
```

Every node receives this state and returns only the fields it changed. LangGraph merges the changes automatically.

---

#### `graph/nodes.py` (new)

Contains all 6 node (agent) functions. Each node = one specialist in the pipeline.

| Node | What It Does | Class Concept | Session |
|------|-------------|--------------|---------|
| `ingest_node` | PyPDF extracts all text from the PDF | RAG Pipeline — Indexing | 6 |
| `extract_node` | LCEL chain → Gemini → JSON claims dict | LCEL Chains, Gemini | 3, 8-9 |
| `validate_node` | DuckDuckGo search + Gemini assessment per claim | Tool Use, Function Calling | 7, 8-9, 10-11 |
| `score_node` | LCEL chain → Gemini → JSON scores 0-10 | LCEL Chains | 8-9 |
| `human_review_node` | LangGraph `interrupt()` — pauses graph for analyst | Human-in-the-loop | 9-10 |
| `write_memo_node` | LCEL chain → Gemini → professional investment memo | LCEL Chains, Gemini | 3, 8-9 |

**Key design decision:** LLM and search tool are lazy-loaded via `get_llm()` / `get_search_tool()` rather than instantiated at module level. This means the module imports cleanly even without a `GOOGLE_API_KEY` set (important for testing and CI).

**LLM used:** `gemini-2.5-flash` — same model used by Conchita in class (`Conchita_News_Writer_Agent_in_LangGraph_june26.ipynb`).

---

#### `graph/graph.py` (new)

**Class concept used:** Everything from Session 9-10 LangGraph.

Builds the complete `StateGraph` following the exact same code pattern from Conchita's notebook:

```python
workflow = StateGraph(PitchState)       # Create graph
workflow.add_node("ingest", ingest_node) # Add nodes
workflow.set_entry_point("ingest")       # Set start
workflow.add_edge("ingest", "extract")   # Add edges
workflow.add_conditional_edges("score", needs_human_review)  # Routing
graph = workflow.compile(checkpointer=MemorySaver())  # Compile
```

**`needs_human_review(state)` function:**  
Conditional edge router — same pattern as `bad_manager_node_assigner` in Conchita's notebook.  
Returns `"human_review"` if any score < 6, otherwise `"write_memo"`.

**`get_graph_image(compiled_graph)` function:**  
Returns Mermaid PNG bytes of the graph — same visualization used in class notebooks.  
In Streamlit: `st.image(get_graph_image(graph))`

**`MemorySaver` checkpointer:**  
Required for `interrupt()` to work. Saves graph state after every node.  
This is what allows the graph to pause at `human_review_node` and resume when the analyst submits feedback.

---

#### `rag/rag_demo.py` (new)

**Class concepts:** Embeddings (Session 5) + RAG Pipeline (Session 6)

Standalone demo of the complete RAG pipeline:
1. `build_vectorstore(text)` — splits text → embeds with `GoogleGenerativeAIEmbeddings` → stores in Chroma
2. `build_rag_chain(vectorstore)` — retriever + LCEL RAG chain (retriever | prompt | llm | parser)
3. `answer_pitch_question(text, question)` — convenience function for demo

Uses `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` — standard RAG chunking strategy.

---

#### `requirements.txt` (updated)

Pinned to working versions. Added `langchain-chroma` for the RAG demo.

---

### Verification

All files pass syntax check. Import test output:
```
✓ PitchState
✓ All 6 nodes
✓ graph.py
✓ graph package

Graph nodes: ['__start__', 'ingest', 'extract', 'validate', 'score', 'human_review', 'write_memo']
✓ needs_human_review routes to human_review when traction=3
✓ needs_human_review routes to write_memo when all scores ≥ 6

All checks passed ✓
```

---

### How to Use the Graph (for Role 5 — Integration)

**Normal run (all scores ≥ 6):**
```python
from graph import graph

config = {"configurable": {"thread_id": "session-001"}}  # unique per user session
initial_state = {"pdf_path": "/tmp/uploaded_pitch.pdf"}

for event in graph.stream(initial_state, config, stream_mode="values"):
    print(event)  # Each step's state update — use for live Streamlit updates
```

**Run with human interrupt:**
```python
from langgraph.types import Command

# Step 1: Run until interrupt
result = graph.invoke(initial_state, config)

# Step 2: Check if graph paused for human review
if "__interrupt__" in result:
    interrupt_data = result["__interrupt__"][0].value
    # interrupt_data contains: scores, extracted_claims, validation_results
    
    # Step 3: Human fills out the Streamlit review form...
    human_feedback = "Team is strong. Low traction is expected at seed stage. Proceed."
    
    # Step 4: Resume the graph with the human's feedback
    final_result = graph.invoke(Command(resume=human_feedback), config)
    memo = final_result["investment_memo"]
```

**Graph visualization in Streamlit:**
```python
from graph import graph, get_graph_image
img_bytes = get_graph_image(graph)
st.image(img_bytes, caption="Live Agent Pipeline")
```

---

## Pending Tasks

### Role 1 (Marian) — Remaining
- [ ] Wire graph into `app.py` with Role 5 (Streamlit integration)
- [ ] Implement human interrupt form in Streamlit (show scores + textarea for feedback)
- [ ] Handle `__interrupt__` detection in Streamlit session state
- [ ] Add graph visualization to sidebar or "How it works" section

### Role 2 — RAG Engineer (not yet started)
- [ ] Extend `rag/rag_demo.py` into a proper module used by `extract_node`
- [ ] Optionally replace direct Gemini extraction in `extract_node` with RAG-based extraction (Chroma retrieval → Gemini prompt)
- [ ] Test with multiple pitch deck PDFs of varying length

### Role 3 — Agent Engineer ✅ Complete
- [x] Built `agents/tools.py` — `@tool` decorated `search_web` (DuckDuckGo) and `search_wikipedia` (Wikipedia)
- [x] Built `agents/validator_agent.py` — ReAct sub-graph following professor's news writer pattern exactly: `create_agent()` → `prompt | llm.bind_tools(tools)`, `agent_node()` with `functools.partial`, `ToolNode` from `langgraph.prebuilt`, `should_continue()` conditional edge
- [x] Exposes `validate_claims(state) -> dict` — reads `state["extracted_claims"]`, writes `state["validation_results"]`
- [x] `agents/__init__.py` exports `validate_claims` for Role 1 to wire as a node
- [x] Added `wikipedia` and `ddgs` to `requirements.txt` and `pyproject.toml`
- [x] Created `pyproject.toml` for `uv lock` / `uv sync` environment setup

**Why an agent and not a regular pipeline:**
A fixed pipeline would hardcode the search query (e.g. `search("market size " + claim)`). The agent reads each claim, writes its own query, picks the right tool, and can search again if the first result is thin. This matters because the right query depends on what the claim actually says — something you can't know in advance.

**Value of the agent:**
- Goes beyond the pitch deck — independently checks whether the startup's claims hold up against real-world data
- Replaces 1-2 hours of manual analyst work per deck (Googling claims, cross-referencing sources, summarising)
- Makes scoring credible — without it, the scorer would grade the startup on its own claims, which is circular
- One line: the agent is the fact-checker — it makes sure the AI isn't just taking the startup's word for it

**Architecture — ReAct framework:**
The validator agent implements the ReAct (Reason + Act) loop:
1. **Reason** — Gemini reads the claims and decides which tool to call
2. **Act** — `ToolNode` executes DuckDuckGo or Wikipedia search
3. **Observe** — result is appended to messages and passed back to Gemini
4. **Repeat** — loop continues via `should_continue` conditional edge until Gemini stops making tool calls

This is the same pattern as `create_react_agent` from `langgraph.prebuilt`, built explicitly step-by-step following the professor's news writer notebook.

**Design decisions:**
- `RunnableWithMessageHistory` was dropped — it conflicts with compiled LangGraph sub-graphs and Gemini's message format. Memory is not needed since the validator runs once per pipeline execution.
- All 4 claims are sent to the agent in a single prompt so it can search for all of them in one loop, rather than running 4 separate sub-graph invocations.
- Follows professor's `Conchita_News_Writer_Agent_in_LangGraph_june26.ipynb` pattern exactly.

### Role 4 — Output Engineer ✅ COMPLETE

- [x] Refine `score_node` prompt — float scores 0.0–10.0 with explicit rubric bands
- [x] Add confidence field (`high`/`medium`/`low`) based on evidence quality
- [x] Replace `JsonOutputParser` with `PydanticOutputParser(ScoreResult)` in scorer
- [x] Replace raw string memo with `PydanticOutputParser(InvestmentMemo)` + formatted markdown
- [x] `composite_score` computed automatically (30% market, 30% team, 25% product, 15% traction)
- [x] `risks` field is `list[str]` with Pydantic-enforced 3–5 items (not a text blob)
- [x] `recommendation` is a Literal — Pydantic rejects any string not in the allowed set
- [x] Both chains are lazy-loaded singletons (same pattern as `get_llm()` in nodes.py)
- [x] `score_node` and `write_memo_node` in `nodes.py` delegate to chain functions

### Role 5 — Integration Lead (not yet started)
- [ ] Rewrite `app.py` to wire graph into the Streamlit UI
- [ ] Implement file upload → save to temp path → pass to graph
- [ ] Stream node-by-node progress with `st.status()` or `st.spinner()`
- [ ] Implement the human review form (shows interrupt payload, submits feedback)
- [ ] Add graph visualization panel
- [ ] End-to-end test with a real pitch deck PDF

### Role 6 — Presentation Lead (not yet started)
- [ ] Create presentation slides (15 min limit)
- [ ] Map architecture diagram to class sessions (PLAN.md has the content)
- [ ] Prepare live demo script with a compelling example pitch deck
- [ ] Build the business case narrative (time saved, cost reduction for VC firms)

### General
- [ ] Add `.env` file with `GOOGLE_API_KEY` (everyone needs this)
- [ ] Test with a real pitch deck PDF end-to-end
- [ ] Add error handling UI in Streamlit (show user-friendly error messages)
- [ ] Write end-to-end tests in `tests/` folder




## Session 2 — 2026-06-25 (continued)

**Author:** Dom (Role 5 — Integration Lead)

### End-to-End Integration Complete

✅ **Graph execution verified** — PDF upload → ingest → extract → validate → score → human interrupt
✅ **Human-in-the-loop working** — Graph pauses correctly when any score < 6
✅ **Analyst review form functional** — Captures feedback textarea
✅ **Error handling in place** — Shows user-friendly errors

### Status

**Backend:** Complete (Marian's 6-node graph)  
**Frontend:** Complete (Streamlit wired to graph)  
**Integration:** Complete (E2E tested)

**Known issues to fix Thursday:**
- Scores currently returning 0 (likely placeholder data in scorer node)
- Need R3 (Agent Engineer) to enhance validate_node with proper DuckDuckGo integration
- Need R4 (Output Engineer) to refine scorer and memo writer chains

### Next steps

- Thursday: R3 & R4 arrive, enhance agent + output chains
- Friday: Refine prompts, test with real pitch data
- Saturday–Sunday: Demo rehearsal + final polish
- Monday: Presentation

---

## Session 3 — 2026-06-26

**Authors:** Lea (Role 3 — Agent Engineer) + Steve (Role 4 — Scoring & Output Engineer)

### Role 3 Complete — Validation Agent

Built the full `agents/` module following the professor's news writer notebook pattern.

**Files created:**
- `agents/tools.py` — `@tool` decorated `search_web` (DuckDuckGo) and `search_wikipedia`
- `agents/validator_agent.py` — ReAct sub-graph: `create_agent()`, `agent_node()`, `ToolNode`, `should_continue()` conditional edge, `validate_claims(state)` public interface
- `agents/__init__.py` — exports `validate_claims`
- `pyproject.toml` — added for `uv` environment management

**Dependencies added:** `wikipedia`, `ddgs` (required by newer `langchain-community` DuckDuckGo wrapper)

**Handoff to Role 1 (Marian):**
Replace `validate_node` in `graph/nodes.py` with:
```python
from agents import validate_claims
workflow.add_node("validate", validate_claims)
```

---

### Role 4 Complete — Scoring & Output Chains

#### `chains/output_models.py` (new)

Pydantic data contracts for the two output chains. These are the types that
`PydanticOutputParser` sends as a JSON schema to Gemini and validates the response against.

**`ScoreResult`**

| Field | Type | Constraint |
| --- | --- | --- |
| `market` | `float` | `ge=0.0, le=10.0` — Pydantic rejects anything outside range |
| `team` | `float` | `ge=0.0, le=10.0` |
| `traction` | `float` | `ge=0.0, le=10.0` |
| `product` | `float` | `ge=0.0, le=10.0` |
| `reasoning` | `ScoreReasoning` | Nested model — enforces exactly 4 keys, no freeform dict |
| `confidence` | `Literal["high","medium","low"]` | Rejected if any other string |
| `composite_score` | `@computed_field float` | Auto-computed, never sent to Gemini |

`composite_score` formula: `0.30×market + 0.30×team + 0.25×product + 0.15×traction`
Reflects real VC seed-stage weighting (team + market dominate; traction is a smaller signal early).

**`InvestmentMemo`**

| Field | Type | Constraint |
| --- | --- | --- |
| `executive_summary` | `str` | — |
| `business_overview` | `str` | — |
| `team_assessment` | `str` | — |
| `traction_evidence` | `str` | — |
| `risks` | `list[str]` | `min_length=3, max_length=5` — never a paragraph blob |
| `recommendation` | `Literal["INVEST","PASS","CONDITIONAL INVEST"]` | Hard-typed |
| `recommendation_rationale` | `str` | — |

---

#### `chains/scorer.py` (new)

LCEL scoring chain replacing the inline `JsonOutputParser` chain in `score_node`.

**Key upgrades over the previous implementation:**

- `PydanticOutputParser(ScoreResult)` instead of `JsonOutputParser` — invalid output raises instead of silently passing
- Prompt asks for **float with one decimal place** (e.g. 7.5, not 7) — LLMs cluster on integers
- Evidence-adjustment rule in prompt: `VERIFIED` claims score at least 0.5 higher; `UNVERIFIED` caps at 6.5
- `temperature=0` for scoring — maximum consistency across runs of the same deck
- Lazy singleton via `get_scorer_chain()` — same pattern as `get_llm()` in `nodes.py`
- `score_claims(state: dict) → dict` is the node function wired into the graph

**HITL trigger:** any individual score `< 6.0` (not composite) — a startup with composite 6.5 but traction 2.0 still triggers human review.

---

#### `chains/memo_writer.py` (new)

LCEL memo writer chain replacing the inline `StrOutputParser` chain in `write_memo_node`.

**Key upgrades:**

- `PydanticOutputParser(InvestmentMemo)` validates the full structured memo before it becomes a string
- `_format_memo_to_markdown(memo)` converts validated `InvestmentMemo` → clean markdown — Gemini never touches formatting
- `_format_human_section(feedback)` returns empty string when no HITL occurred — no empty headings in the prompt
- `composite_score` from `state["scores"]` is passed explicitly to the prompt so the memo rationale references the actual number
- `temperature=0.3` (slightly higher than scorer) — scoring needs consistency, writing benefits from mild variation

---

#### `graph/nodes.py` (updated)

`score_node` and `write_memo_node` bodies replaced with single-line delegation:

```python
def score_node(state):    return score_claims(state)
def write_memo_node(state): return write_memo(state)
```

All prompt logic, parsing, and error handling now lives in `chains/scorer.py` and `chains/memo_writer.py`.
Docstrings retained for class concept mapping used in the presentation.

---

#### `app.py` (updated)

Added SSL proxy fix at the very top of the file. IE University's network runs an SSL
inspection proxy that replaces server certificates. The `google-genai` SDK uses `httpx`
with its own SSL context that doesn't inherit Python's default cert store — this
caused `[SSL: CERTIFICATE_VERIFY_FAILED]` on every Gemini API call.

Fix: monkey-patch `httpx.Client.__init__` to set `verify=False` before any imports.

---

#### `data/sample_pitch.pdf` (new)

JobAnyDay pitch deck added as the "known good" demo PDF. Text-based (not scanned),
6,382 characters extracted cleanly across all pages.

---

### Issues Resolved

| Issue | Root Cause | Fix |
| --- | --- | --- |
| Scores returning 0 | `JsonOutputParser` with no Pydantic validation; integers requested | `PydanticOutputParser(ScoreResult)` with float prompt |
| Circular import | `chains/scorer.py` imported `graph.state` → triggered `graph/__init__.py` → `graph.nodes` → back to `chains/scorer.py` | Removed `PitchState` import from chain files; dependency direction is `graph → chains`, never `chains → graph` |
| SSL certificate error | IE University SSL inspection proxy; `google-genai` httpx client ignores `SSL_CERT_FILE` and `certifi` | `httpx.Client.__init__` monkey-patch with `verify=False` at top of `app.py` |
| API 429 error | Prepaid Gemini credits depleted | Top up at [aistudio.google.com](https://aistudio.google.com) → Billing |

---

### Status

**R3 agent:** Complete  
**R4 chains:** Complete and verified (29/29 unit tests passing)  
**SSL fix:** Applied to `app.py`  
**Demo PDF:** `data/sample_pitch.pdf` (JobAnyDay)

### Next Steps — Friday 27 Jun

- Top up GOOGLE_API_KEY credits and run full end-to-end with real Gemini output
- R1 wires `validate_claims` from `agents/` into `graph/nodes.py`
- R1 confirms `graph/nodes.py` wiring is compatible with R4's output shape
- R5 verifies Streamlit renders `investment_memo` markdown correctly (uses `st.markdown`, not `st.text`)
- All: prompt-tune using JobAnyDay deck as the reference input

---

## Session 3 (continued) — 2026-06-26

**Author:** Steve (Role 4 — Scoring & Output Engineer)

### Merge: main → R4 branch

Merged Lea's R3 work from `main` into the `R4` branch. Three-way conflict resolved across:

| File | Conflict | Resolution |
| --- | --- | --- |
| `requirements.txt` | Duplicate `pytest` placement + comment wording | Kept R3's updated comment + `ddgs`/`wikipedia`; `pytest` moved to end |
| `README.md` | `agents/` vs `chains/` in project structure tree | Merged both — agents with R3's files, chains with R4's files |
| `PROJECT_HISTORY.md` | Two Session 3 entries (Lea vs Steve) | Kept both under one heading, R3 first then R4 |

New files brought in from R3: `agents/tools.py`, `agents/validator_agent.py`, `agents/__init__.py`, `pyproject.toml`, `tests/test_validator_agent.py`.

---

### scorer.py prompt update — R3 compatibility

R3's agent outputs **rich text paragraphs** of research evidence (not simple `VERIFIED`/`UNVERIFIED` labels). Updated the EVIDENCE ADJUSTMENT RULE in `chains/scorer.py` to judge evidence by text quality rather than labels:

- **Before:** "A VERIFIED claim should score at least 0.5 higher than UNVERIFIED"
- **After:** "If the evidence contains specific numbers, statistics, or named sources → treat as strong; vague or generic text → treat as weak; no relevant findings → cap at 5.0"

This makes scoring more accurate — Gemini now reads real evidence rather than single-word labels.

---

### pytest configuration fixed

`test_r4.py` (smoke test) was being collected by pytest as a test module because the filename starts with `test_`. Module-level code including `raise SystemExit(1)` caused an INTERNALERROR during collection.

Fix: wrapped all executable code in `if __name__ == "__main__":` so pytest can safely import the file without running it.

---

### Test results

```
pytest tests/test_chains.py      → 29/29 passed (R4 unit tests, no API calls)
pytest                           → 29 passed, 6 skipped, 1 failed
```

The 1 remaining failure (`test_not_mentioned_claims` in `test_validator_agent.py`) is a bug in R3's error handler — when the API returns 403, the exception catches ALL claims (including the ones pre-filtered as "Not mentioned") and overwrites them with `UNVERIFIED`. R3 skip guard only covers 429 (rate limit), not 403 (blocked key). Not an R4 issue.

---

### Issues Resolved This Session

| Issue | Root Cause | Fix |
| --- | --- | --- |
| pytest INTERNALERROR on `test_r4.py` | File collected as test module; `raise SystemExit(1)` ran during collection | Wrapped all code in `if __name__ == "__main__":` |
| `ddgs` ModuleNotFoundError | Merge brought in R3's new dependency but venv not synced | `pip install -r requirements.txt` |
| 403 PERMISSION_DENIED | API key belonged to GCP project where Gemini API was blocked | Create new API key from AI Studio on a new project (free tier) |
| Duplicate `ddgs` in requirements.txt | User manually added bare `ddgs` after `ddgs>=0.1.0` already existed | Removed duplicate |

---

### Status after Session 3

**R3 agent:** Complete  
**R4 chains:** Complete and verified  
**Merge:** R3 + R4 integrated on R4 branch  
**Unit tests:** 29/29 R4 tests passing  
**Blocked on:** Valid Gemini API key with free-tier access (create at aistudio.google.com → "Create API key in new project")

### Next Steps — Friday 27 Jun

- Get working API key (free tier, new project in AI Studio)
- Run `python test_r4.py` to get real scores + memo output
- R1 wires `validate_claims` from `agents/__init__.py` into `graph/nodes.py`
- R5 confirms `st.markdown(memo)` is used (not `st.text`) for investment memo rendering
- All: end-to-end demo rehearsal with JobAnyDay pitch deck
