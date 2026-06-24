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

### Role 3 — Agent Engineer (not yet started)
- [ ] Enhance `validate_node` with a proper LangChain `AgentExecutor` or `create_react_agent`
- [ ] Add Wikipedia tool alongside DuckDuckGo
- [ ] Add memory to the validation agent (track previously validated claims)
- [ ] Consider using `ToolNode` from LangGraph prebuilt (as used in Conchita's news writer notebook)

### Role 4 — Output Engineer (not yet started)
- [ ] Refine `score_node` prompt for more consistent scoring
- [ ] Add confidence intervals to scores
- [ ] Improve `write_memo_node` with a structured Pydantic output model
- [ ] Consider a separate "summary card" output (key metrics at a glance)

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
