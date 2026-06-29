"""
graph/nodes.py — All 6 Node (Agent) Implementations

WHAT IS THIS?
-------------
Each "node" is a Python function that acts as one agent in the pipeline.
It receives the current PitchState, does its specific job, and returns
a dict with ONLY the fields it changed. LangGraph merges those changes
into the state automatically.

Think of each node like a specialist at a hospital:
  - The receptionist (ingest) gets your information
  - The nurse (extract) takes your vitals
  - The lab (validate) runs tests
  - The doctor (score) reads the results
  - The senior doctor (human_review) checks for anything risky
  - The discharge nurse (write_memo) prepares the final report

CLASS CONCEPTS USED IN THIS FILE:
  - Gemini (ChatGoogleGenerativeAI) .............. Session 3, 10-11
  - LCEL Chains (prompt | llm | parser) ......... Sessions 8-9
  - ChatPromptTemplate ........................... Sessions 8-9
  - Tool Use / DuckDuckGo Search ................ Session 7, 8-9
  - Function Calling ............................ Sessions 10-11
  - RAG — text extraction from PDF .............. Session 6
  - LangGraph interrupt() ....................... Session 9-10
"""

import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.types import interrupt
from pypdf import PdfReader

from graph.state import PitchState
from chains.scorer import score_claims
from chains.memo_writer import write_memo
from agents.validator_agent import validate_claims

# Load environment variables from .env file (GOOGLE_API_KEY)
load_dotenv()


# ============================================================
# LAZY-LOADED SINGLETONS
#
# We initialise the LLM and search tool ONLY on first use, not at import
# time. This means the module loads fine even without a GOOGLE_API_KEY set
# (e.g. during syntax checks, CI, or before .env is configured).
#
# CLASS CONCEPT: "Introduction to Gemini" — Session 3
# CLASS CONCEPT: "Google AI SDK" — Sessions 10-11
# NOTEBOOK: Conchita_News_Writer_Agent_in_LangGraph_june26.ipynb
#   llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
# ============================================================

_llm = None
_search_tool = None


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Returns (and caches) the Gemini LLM.

    WHY gemini-2.5-flash: Same model Conchita used in class. Fast, cheap,
    very capable. temperature=0.1 means 'be consistent and factual'.
    """
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
        )
    return _llm


def get_search_tool() -> DuckDuckGoSearchRun:
    """
    Returns (and caches) the DuckDuckGo search tool.

    CLASS CONCEPT: "AI Agents + Tool Use" — Session 7
    CLASS CONCEPT: "LangChain Tools" — Sessions 8-9
    CLASS CONCEPT: "Function Calling" — Sessions 10-11
    """
    global _search_tool
    if _search_tool is None:
        _search_tool = DuckDuckGoSearchRun()
    return _search_tool


# ============================================================
# NODE 1: INGEST
# ============================================================

def ingest_node(state: PitchState) -> dict:
    """
    WHAT THIS NODE DOES:
    Reads the uploaded PDF file and extracts all the text from every page.

    SIMPLE ANALOGY:
    Like a person opening a 20-page business report and reading it out loud,
    word by word, so the rest of the team can hear what's inside.

    INPUT FROM STATE: pdf_path (the file location of the uploaded PDF)
    OUTPUT TO STATE:  raw_text (all the text from the PDF as one big string)

    CLASS CONCEPT: "RAG Pipeline — Step 1: Indexing/Document Loading" — Session 6
    The first step of any RAG system is loading and extracting the document.
    """
    try:
        # PyPDF's PdfReader opens the PDF file
        reader = PdfReader(state["pdf_path"])

        # Loop through every page and extract its text
        # This is the "document loading" step from the RAG pipeline (Session 6)
        full_text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                full_text += f"\n--- Page {i + 1} ---\n{page_text}"

        if not full_text.strip():
            return {
                "raw_text": "",
                "error": "Could not extract text from PDF. It may be a scanned image. Try a text-based PDF.",
            }

        return {"raw_text": full_text, "error": None}

    except FileNotFoundError:
        return {"raw_text": "", "error": f"PDF file not found at: {state['pdf_path']}"}
    except Exception as e:
        return {"raw_text": "", "error": f"PDF ingestion failed: {str(e)}"}


# ============================================================
# NODE 2: EXTRACT
# ============================================================

def extract_node(state: PitchState) -> dict:
    """
    WHAT THIS NODE DOES:
    Uses Gemini AI to read through the pitch deck text and pull out the
    4 most important claims that investors evaluate.

    SIMPLE ANALOGY:
    Like having a very smart assistant who reads a 20-page business plan
    and summarizes ONLY the key facts onto a single Post-It note:
    "How big is the market? Who are the founders? What growth do they have?
    What does the product actually do?"

    INPUT FROM STATE: raw_text (the full PDF text from ingest_node)
    OUTPUT TO STATE:  extracted_claims (dict with 4 key facts)

    CLASS CONCEPT: "LCEL Chains (prompt | llm | parser)" — Sessions 8-9
    This is the EXACT chain pattern Conchita taught:
        chain = prompt | llm | parser
    Each | means "pipe the output into the next step"

    CLASS CONCEPT: "Gemini multimodal / text understanding" — Session 3
    """

    # If ingestion failed, skip this node
    if not state.get("raw_text"):
        return {
            "extracted_claims": {},
            "error": state.get("error", "No text to extract from."),
        }

    # STEP 1: Define the prompt template
    # ChatPromptTemplate is how we tell Gemini what to do — taught Sessions 8-9
    # {pitch_text} is a placeholder that gets filled with the actual text
    prompt = ChatPromptTemplate.from_template(
        """You are an expert venture capital analyst. Your job is to read a startup
pitch deck and extract EXACTLY 4 key facts that investors care about.

Here is the FULL text of the pitch deck:
===
{pitch_text}
===

Extract these 4 things. If the pitch deck doesn't mention something, write exactly
"Not mentioned in pitch deck".

Return ONLY a valid JSON object. No extra text, no markdown, just JSON:
{{
    "market_size": "What total addressable market does the startup claim? Include any numbers.",
    "team_background": "Who are the founders? What relevant experience do they have?",
    "traction": "What growth metrics, revenue, users, or partnerships does the startup have?",
    "product_description": "What does the product do? What specific problem does it solve?"
}}"""
    )

    # STEP 2: Define the output parser
    # JsonOutputParser converts Gemini's text response into a Python dict
    parser = JsonOutputParser()

    # STEP 3: Build the LCEL chain
    # This is THE core LangChain pattern — taught in Sessions 8-9
    # prompt | llm | parser means:
    #   1. Fill the prompt template with our variables
    #   2. Send the filled prompt to Gemini
    #   3. Parse Gemini's response from text into a Python dict
    chain = prompt | get_llm() | parser

    try:
        result = chain.invoke({"pitch_text": state["raw_text"]})
        return {"extracted_claims": result, "error": None}

    except Exception as e:
        return {
            "extracted_claims": {
                "market_size": "Extraction failed",
                "team_background": "Extraction failed",
                "traction": "Extraction failed",
                "product_description": "Extraction failed",
            },
            "error": f"Claim extraction failed: {str(e)}",
        }


# ============================================================
# NODE 3: VALIDATE
# ============================================================

def validate_node(state: PitchState) -> dict:
    """
    WHAT THIS NODE DOES:
    For each claim the startup made, searches the web to find evidence
    about whether the claim is realistic. Then uses Gemini to summarize
    whether the claim is VERIFIED, PLAUSIBLE, or UNVERIFIED.

    SIMPLE ANALOGY:
    Like a fact-checker at a newspaper. Before publishing a story, they
    Google every claim the journalist made. "The startup says the market
    is $5B? Let me search 'global pet grooming market size' and see what
    industry reports say..."

    INPUT FROM STATE: extracted_claims (dict of 4 facts from extract_node)
    OUTPUT TO STATE:  validation_results (dict with web evidence for each fact)

    CLASS CONCEPT: "AI Agents + Tool Use" — Session 7
    Agents have TOOLS they can use to take actions. Here, the tool is web search.

    CLASS CONCEPT: "LangChain Tools (DuckDuckGoSearchRun + WikipediaQueryRun)" — Sessions 8-9
    CLASS CONCEPT: "Function Calling" — Sessions 10-11
    The agent sub-graph uses llm.bind_tools() so Gemini decides which tool to call
    and how many times — this is the full ReAct (Reason + Act) loop pattern.

    CLASS CONCEPT: "Multi-agent sub-graphs" — Sessions 9-10
    validate_claims() runs its own internal StateGraph with a validator node and
    a ToolNode, looping until Gemini stops calling tools.
    """
    return validate_claims(state)


# ============================================================
# NODE 4: SCORE
# ============================================================

def score_node(state: PitchState) -> dict:
    """
    WHAT THIS NODE DOES:
    Gemini reads ALL the information gathered so far (claims + web evidence)
    and gives each of the 4 dimensions a score from 0 to 10.
    It also provides one sentence of reasoning for each score.

    SIMPLE ANALOGY:
    Like judges at a cooking competition scoring dishes on 4 criteria:
    taste, presentation, originality, and technique. Each judge sees the
    dish AND reads the chef's description AND the food critic's review
    before giving a score.

    INPUT FROM STATE: extracted_claims + validation_results
    OUTPUT TO STATE:  scores (dict with numeric scores + reasoning)
                      human_review_required (True if any score < 6)

    CLASS CONCEPT: "LCEL Chains (prompt | llm | parser)" — Sessions 8-9
    CLASS CONCEPT: "Gemini structured output" — Session 3, 10-11
    """
    return score_claims(state)


# ============================================================
# NODE 5: HUMAN REVIEW
# ============================================================

def human_review_node(state: PitchState) -> dict:
    """
    WHAT THIS NODE DOES:
    Completely PAUSES the graph and waits for a human analyst to review
    the scores and provide feedback before the graph continues.

    The graph will not move forward until a human submits their review
    through the Streamlit UI.

    SIMPLE ANALOGY:
    Like an alarm on a car production line. If a robot detects a potential
    defect, it rings an alarm and the whole assembly line STOPS. A human
    supervisor walks over, inspects the car, and either says "it's fine,
    continue" or "fix this before moving on." Only after the human's decision
    does the line start moving again.

    INPUT FROM STATE: scores, extracted_claims, validation_results
    OUTPUT TO STATE:  human_feedback (what the analyst typed)
                      human_review_required = False (done with review)

    CLASS CONCEPT: "LangGraph interrupt() — Human-in-the-loop" — Session 9-10
    This is an ADVANCED LangGraph feature that goes beyond the class exercises.

    HOW interrupt() WORKS:
    1. When this node runs, it calls interrupt(payload)
    2. LangGraph FREEZES the entire graph right here
    3. The payload is sent back to the caller (Streamlit)
    4. The graph state is SAVED by MemorySaver (nothing is lost)
    5. When the human submits feedback, Streamlit calls:
           graph.invoke(Command(resume=human_feedback), config=config)
    6. The graph UNFREEZES and continues from exactly this point
    7. interrupt() returns the human's feedback as its value
    8. We store that feedback in state and continue to write_memo

    REQUIRES: Graph must be compiled with checkpointer=MemorySaver()
              (see graph.py — this is set up there)
    """

    # interrupt() pauses the graph and sends this data to the UI
    # The UI will display scores and ask the analyst for feedback
    human_feedback = interrupt(
        {
            "message": (
                "⚠️ One or more scores are below 6. "
                "Please review the analysis and provide your feedback before the memo is written."
            ),
            "scores": state.get("scores", {}),
            "extracted_claims": state.get("extracted_claims", {}),
            "validation_results": state.get("validation_results", {}),
        }
    )

    # When we reach this line, the human has submitted their feedback
    # and the graph has been resumed with Command(resume=feedback)
    return {
        "human_feedback": human_feedback,
        "human_review_required": False,
    }


# ============================================================
# NODE 6: WRITE MEMO
# ============================================================

def write_memo_node(state: PitchState) -> dict:
    """
    WHAT THIS NODE DOES:
    Uses Gemini to write a complete, professional investment memo that
    synthesizes everything gathered: claims, web evidence, scores, and
    (if applicable) the human analyst's feedback.

    SIMPLE ANALOGY:
    Like the final report card a teacher writes after seeing all the test
    scores, homework, and class participation. It summarizes everything
    and gives a clear final recommendation: INVEST, PASS, or CONDITIONAL.

    INPUT FROM STATE: extracted_claims + validation_results + scores + human_feedback
    OUTPUT TO STATE:  investment_memo (the complete written memo)

    CLASS CONCEPT: "LCEL Chains (prompt | llm | parser)" — Sessions 8-9
    CLASS CONCEPT: "Gemini text generation" — Session 3
    """

    return write_memo(state)
