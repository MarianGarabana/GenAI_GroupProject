"""
chains/scorer.py — LCEL Scoring Chain

Reads extracted_claims + validation_results from graph state,
calls Gemini, and returns a validated ScoreResult written back to state["scores"].

Replaces the inline score_node chain in graph/nodes.py with:
  - PydanticOutputParser instead of JsonOutputParser (validation, not just parsing)
  - Float scores with one decimal place instead of integers
  - Evidence-aware scoring rubric (VERIFIED claims score higher)
  - composite_score computed by the model, not by Gemini

CLASS CONCEPTS:
  - LCEL chain (prompt | llm | parser) ........... Sessions 8-9
  - PydanticOutputParser ......................... Sessions 8-9
  - ChatPromptTemplate.partial() ................. Sessions 8-9
  - Gemini (ChatGoogleGenerativeAI) .............. Sessions 3, 10-11
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from chains.output_models import ScoreResult


# ============================================================
# PROMPT
# ============================================================

_SCORING_PROMPT = """\
You are a senior partner at a tier-1 venture capital firm scoring a startup pitch deck.

You have been given:
1. The startup's own claims extracted from their pitch deck
2. Independent web-search validation evidence for each claim

STARTUP CLAIMS:
{claims}

VALIDATION EVIDENCE:
{validation}

Score each of the 4 investment dimensions from 0.0 to 10.0.
Use ONE DECIMAL PLACE — write 7.5, not 7. Write 4.0, not 4.

SCORING RUBRIC:
  0.0 – 2.9  : Critical flaw or no data — strong do-not-invest signal
  3.0 – 4.9  : Significant weakness — major concerns that need addressing
  5.0 – 6.9  : Average — acceptable but with notable gaps or uncertainty
  7.0 – 8.4  : Strong — compelling, well-supported claim
  8.5 – 10.0 : Exceptional — world-class, rare, directly verifiable

DIMENSION GUIDE:
  market   — Total addressable market size and how well the startup captures it
  team     — Founder credentials, domain expertise, and team completeness
  traction — Revenue, user growth, and partnerships weighted by stage (seed = less traction expected)
  product  — Differentiation, defensibility, technical quality, problem-solution fit

EVIDENCE ADJUSTMENT RULE:
  - A VERIFIED claim should score at least 0.5 higher than the same claim marked UNVERIFIED
  - An UNVERIFIED claim caps the dimension score at 6.5 unless the claim is inherently private
    (e.g. a proprietary algorithm detail that cannot appear in public search results)
  - A PLAUSIBLE claim sits between VERIFIED and UNVERIFIED — use your judgment

CONFIDENCE GUIDE:
  high   — 3 or more claims are VERIFIED by web evidence
  medium — 1 or 2 claims are VERIFIED, rest are PLAUSIBLE
  low    — all claims are UNVERIFIED or evidence is missing

Do NOT include a composite_score field — that is computed automatically.

{format_instructions}"""


# ============================================================
# CHAIN FACTORY (lazy singleton)
# ============================================================

_scorer_chain = None


def get_scorer_chain():
    """
    Builds and caches the scoring chain on first call.

    Why lazy? The chain instantiates the LLM (which reads the API key from
    the environment). Deferring this to first use means the module imports
    cleanly even before .env is loaded — same pattern as get_llm() in nodes.py.

    Why cache? Building a chain is cheap, but there's no reason to rebuild
    it on every request. One chain instance handles unlimited invocations.

    Chain anatomy (LCEL pipe — Sessions 8-9):
      prompt  → fills {claims}, {validation}, {format_instructions}
      llm     → Gemini 2.5 Flash generates the JSON response
      parser  → PydanticOutputParser validates response → ScoreResult instance
    """
    global _scorer_chain

    if _scorer_chain is None:
        parser = PydanticOutputParser(pydantic_object=ScoreResult)

        # partial() bakes format_instructions into the prompt permanently.
        # This means callers only pass {claims} and {validation} — not three variables.
        # format_instructions is the JSON schema PydanticOutputParser generates
        # from ScoreResult so Gemini knows the exact shape to return.
        prompt = ChatPromptTemplate.from_template(_SCORING_PROMPT).partial(
            format_instructions=parser.get_format_instructions()
        )

        # temperature=0 for maximum consistency in scoring.
        # We want the same pitch deck to produce the same scores every run.
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

        _scorer_chain = prompt | llm | parser

    return _scorer_chain


# ============================================================
# NODE FUNCTION — this is what graph/nodes.py calls
# ============================================================

def score_claims(state: dict) -> dict:
    """
    LangGraph node function: reads claims + validation from state,
    runs the scoring chain, writes results back to state.

    OUTPUT TO STATE:
      scores               — ScoreResult as a dict (includes composite_score)
      human_review_required — True if any individual dimension score < 6.0
      error                — None on success, error message on failure

    Why check individual scores for human_review_required rather than composite?
    A startup could have composite_score = 6.2 but traction = 2.0 — a single
    critical weakness that an analyst must see, even if the average looks okay.
    """
    claims = state.get("extracted_claims", {})
    validation = state.get("validation_results", {})

    if not claims:
        return {
            "scores": {},
            "human_review_required": True,
            "error": "Cannot score — no claims were extracted.",
        }

    try:
        result: ScoreResult = get_scorer_chain().invoke(
            {
                "claims": str(claims),
                "validation": str(validation),
            }
        )

        # Flag for human review if any single dimension falls below 6.0.
        # Uses the four raw scores, not composite_score — see docstring above.
        needs_review = any(
            s < 6.0
            for s in [result.market, result.team, result.traction, result.product]
        )

        # model_dump() serialises ScoreResult → plain dict.
        # Includes all fields: market, team, traction, product,
        # reasoning, confidence, AND composite_score (computed field).
        return {
            "scores": result.model_dump(),
            "human_review_required": needs_review,
            "error": None,
        }

    except Exception as e:
        return {
            "scores": {},
            "human_review_required": True,
            "error": f"Scoring failed: {str(e)}",
        }
