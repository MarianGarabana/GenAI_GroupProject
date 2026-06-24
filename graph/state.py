"""
graph/state.py — The Shared State Definition

WHAT IS THIS?
-------------
This file defines the "folder" (called PitchState) that every agent in our
graph can read from and write to. Think of it like a relay race baton — each
runner (agent) grabs it, does their part, and passes it to the next one.
The baton keeps a record of EVERYTHING that has happened so far.

CLASS CONCEPT: "Defining The State" using TypedDict
TAUGHT IN: Session 9-10
NOTEBOOK: Conchita_LangGraph_Core_Concepts_june26.ipynb, section "Defining The State"

Conchita's class example:
    class GraphState(TypedDict):
        input: str
        assigned_node: int
        prev_node_to_write: str

Our project version is more complex because we have more agents and more data
flowing through the system — but the exact same pattern.
"""

from typing import TypedDict, Optional


class PitchState(TypedDict):
    """
    The shared state object that flows through the entire LangGraph pipeline.

    Every node (agent) in the graph receives this state as input and returns
    a dict with only the fields it changed. LangGraph automatically merges
    those changes back into the state before passing it to the next node.

    Fields are grouped by which node fills them in.
    """

    # ----------------------------------------------------------------
    # INPUT — provided by the user via the Streamlit app
    # ----------------------------------------------------------------
    pdf_path: str
    # Example: "/tmp/uploads/startup_pitch.pdf"
    # This is the ONLY field the user must provide. Everything else
    # gets filled in automatically as the graph runs.

    # ----------------------------------------------------------------
    # AFTER ingest_node
    # Concept: RAG Pipeline Step 1 "Indexing" — Session 6
    # ----------------------------------------------------------------
    raw_text: str
    # All the text extracted from every page of the PDF.
    # Example: "We are building the Airbnb of dog grooming..."

    # ----------------------------------------------------------------
    # AFTER extract_node
    # Concept: LCEL Chains (prompt | llm | parser) — Sessions 8-9
    # ----------------------------------------------------------------
    extracted_claims: dict
    # The 4 key investor metrics pulled from the pitch deck.
    # Example:
    # {
    #   "market_size": "Global pet grooming market is $5.2B",
    #   "team_background": "Ex-Google engineer + 10yr vet industry founder",
    #   "traction": "2,000 bookings in 3 months, $40k MRR",
    #   "product_description": "Mobile app connecting pet owners with groomers"
    # }

    # ----------------------------------------------------------------
    # AFTER validate_node
    # Concept: AI Agents + Tool Use — Session 7
    # Concept: Function Calling — Sessions 10-11
    # ----------------------------------------------------------------
    validation_results: dict
    # Web search evidence for each claim.
    # Example:
    # {
    #   "market_size": "VERIFIED — IBISWorld confirms $4.8B US market in 2024",
    #   "team_background": "PLAUSIBLE — LinkedIn confirms Google background",
    #   ...
    # }

    # ----------------------------------------------------------------
    # AFTER score_node
    # Concept: LCEL Chains + Gemini structured output — Sessions 8-9
    # ----------------------------------------------------------------
    scores: dict
    # Scores 0-10 for each dimension + reasoning.
    # Example:
    # {
    #   "market": 8,
    #   "team": 9,
    #   "traction": 4,
    #   "product": 7,
    #   "reasoning": {
    #     "market": "Large addressable market with strong growth trends",
    #     "traction": "Revenue too low for the stage, needs more validation"
    #   }
    # }

    # ----------------------------------------------------------------
    # GRAPH CONTROL — used by the conditional edge logic
    # Concept: Conditional Edges — Session 9-10
    # ----------------------------------------------------------------
    human_review_required: bool
    # Set to True by score_node if ANY score is below 6.
    # The conditional edge "needs_human_review" reads this to decide
    # whether to route to human_review or write_memo.

    # ----------------------------------------------------------------
    # AFTER human_review_node (only runs if human_review_required=True)
    # Concept: LangGraph interrupt() — Session 9-10 (advanced)
    # ----------------------------------------------------------------
    human_feedback: Optional[str]
    # What the human analyst typed in the review form.
    # Example: "The traction is low but the team is exceptional. Proceed."

    # ----------------------------------------------------------------
    # AFTER write_memo_node
    # Concept: LCEL Chains + Gemini — Sessions 8-9
    # ----------------------------------------------------------------
    investment_memo: Optional[str]
    # The final investment recommendation memo in plain text.
    # Formatted like a real VC investment committee memo.

    # ----------------------------------------------------------------
    # ERROR HANDLING
    # ----------------------------------------------------------------
    error: Optional[str]
    # If any node fails, the error message is stored here.
    # The UI can display it to the user.
