"""
graph/graph.py — The LangGraph StateGraph

WHAT IS THIS?
-------------
This is the "brain map" that connects all 6 agents together into a
single pipeline. Think of it like designing the floor plan of an office:
  - Which rooms exist (nodes)
  - Which hallways connect them (edges)
  - Which hallways have a fork in the road (conditional edges)
  - How to save all work if there's a power cut (MemorySaver)

CLASS CONCEPTS USED:
  - StateGraph, add_node, set_entry_point ......... Session 9-10
  - add_edge (regular connections) ................ Session 9-10
  - add_conditional_edges (decision routing) ...... Session 9-10
  - compile(checkpointer=MemorySaver()) ........... Session 9-10
  - graph.stream() ................................ Session 9-10
  - Graph visualization (draw_mermaid_png) ........ Session 9-10

KEY NOTEBOOK REFERENCE:
  Conchita_LangGraph_Core_Concepts_june26.ipynb  — "Build Graph" section
  Conchita_News_Writer_Agent_in_LangGraph_june26.ipynb — multi-agent example

THE GRAPH STRUCTURE:
  ingest → extract → validate → score → [DECISION POINT]
                                              ↓ any score < 6
                                         human_review
                                              ↓
                                         write_memo → END
                                              ↑
                                    (also if all scores ≥ 6)
"""

from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import PitchState
from graph.nodes import (
    ingest_node,
    extract_node,
    validate_node,
    score_node,
    human_review_node,
    write_memo_node,
)


# ============================================================
# CONDITIONAL EDGE FUNCTION
# ============================================================

def needs_human_review(state: PitchState) -> Literal["human_review", "write_memo"]:
    """
    This function is called by the conditional edge after score_node.
    It looks at the scores and decides which node to run next.

    Returns "human_review" if any score is below 6.
    Returns "write_memo" if all scores are 6 or above.

    SIMPLE ANALOGY:
    Like a traffic light at a junction. Green (all good) → go to write_memo.
    Red (something is wrong) → stop at human_review first.

    CLASS CONCEPT: "Conditional Edges" — Session 9-10
    This is the SAME PATTERN as Conchita's notebook:

        def bad_manager_node_assigner(state) -> Literal["node 1", "node 2"]:
            assigned_node = state['assigned_node']
            if assigned_node == 1:
                return "node 1"
            elif assigned_node == 2:
                return "node 2"

    Our version just looks at scores instead of node numbers.
    """
    scores = state.get("scores", {})

    # Extract the 4 numeric scores
    numeric_scores = [
        scores.get("market", 10),
        scores.get("team", 10),
        scores.get("traction", 10),
        scores.get("product", 10),
    ]

    # If ANY score is below 6, route to human review
    if any(isinstance(s, (int, float)) and s < 6 for s in numeric_scores):
        return "human_review"

    # All scores are fine — go straight to writing the memo
    return "write_memo"


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_graph():
    """
    Builds and compiles the complete LangGraph StateGraph.

    Returns a compiled graph ready to be invoked with graph.invoke() or
    streamed with graph.stream().

    CLASS CONCEPT: Everything from Session 9-10.

    This function follows the EXACT SAME STEPS as Conchita's notebook:
    1. Create StateGraph(State)
    2. workflow.add_node(...)
    3. workflow.set_entry_point(...)
    4. workflow.add_edge(...)
    5. workflow.add_conditional_edges(...)
    6. graph = workflow.compile()

    The one addition beyond class: checkpointer=MemorySaver()
    This is needed for the human interrupt to work (saves graph state).
    """

    # ── Step 1: Create the graph ──────────────────────────────────────
    # StateGraph(PitchState) tells LangGraph what our state looks like.
    # SESSION 9-10: from langgraph.graph import StateGraph
    workflow = StateGraph(PitchState)

    # ── Step 2: Add all the nodes (agents) ────────────────────────────
    # Each node is a Python function defined in nodes.py
    # SESSION 9-10: workflow.add_node("name", function)
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("score", score_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("write_memo", write_memo_node)

    # ── Step 3: Set the entry point ───────────────────────────────────
    # This is where the graph starts every time it's invoked.
    # SESSION 9-10: workflow.set_entry_point(...)
    workflow.set_entry_point("ingest")

    # ── Step 4: Add regular edges (always go A → B) ───────────────────
    # These are the "always go this way" connections.
    # SESSION 9-10: workflow.add_edge("from", "to")
    workflow.add_edge("ingest", "extract")
    workflow.add_edge("extract", "validate")
    workflow.add_edge("validate", "score")

    # ── Step 5: Add the CONDITIONAL edge after scoring ────────────────
    # This is a "fork in the road" — the graph decides which way to go
    # by calling the needs_human_review() function.
    # SESSION 9-10: workflow.add_conditional_edges("from", routing_function)
    #
    # Same pattern as class notebook:
    #   workflow.add_conditional_edges("manager", bad_manager_node_assigner)
    workflow.add_conditional_edges("score", needs_human_review)

    # ── Step 6: After human review, always go to write_memo ──────────
    workflow.add_edge("human_review", "write_memo")

    # ── Step 7: After writing the memo, we're done ───────────────────
    # END is a special LangGraph constant that terminates the graph.
    # SESSION 9-10: from langgraph.graph import END
    workflow.add_edge("write_memo", END)

    # ── Step 8: Compile with MemorySaver ──────────────────────────────
    # MemorySaver is a checkpointer — it saves the graph state to memory
    # after EVERY node runs. This has two purposes:
    #
    # 1. HUMAN INTERRUPT: The graph can pause (at human_review_node) and
    #    resume later. The saved state lets it pick up exactly where it left off.
    #    Without MemorySaver, interrupt() would not work.
    #
    # 2. PERSISTENCE: If something fails, you can inspect the intermediate
    #    state to debug what went wrong.
    #
    # SIMPLE ANALOGY: Like a video game save point. Before the boss fight
    # (human review), the game saves. If you die (fail), you can restart
    # from the save point, not from the very beginning.
    #
    # SESSION 9-10: workflow.compile() — MemorySaver is the advanced addition.
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    return graph


# ============================================================
# GRAPH VISUALIZATION HELPER
# ============================================================

def get_graph_image(compiled_graph) -> bytes:
    """
    Returns a PNG image of the graph structure as bytes.
    Used in Streamlit to display the architecture visually.

    SIMPLE ANALOGY: Like printing a map of the office floor plan so
    everyone can see how the rooms connect before starting work.

    CLASS CONCEPT: Graph visualization — Session 9-10
    NOTEBOOK: Conchita_LangGraph_Core_Concepts_june26.ipynb:
        display(Image(graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API
        )))

    In Streamlit we use: st.image(get_graph_image(graph))
    """
    from langchain_core.runnables.graph import MermaidDrawMethod

    return compiled_graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API
    )


# ============================================================
# MODULE-LEVEL GRAPH INSTANCE
# ============================================================
# Build once when the module is imported — reuse across all requests.
# Each invocation gets its own thread_id via config, so they don't interfere.
graph = build_graph()
