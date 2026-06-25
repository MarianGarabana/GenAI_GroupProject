"""
chains/memo_writer.py — LCEL Memo Writer Chain

Takes the full graph state (claims, validation, scores, optional human feedback)
and produces a structured InvestmentMemo, then formats it as a markdown string
written to state["investment_memo"].

Why format to string at the end?
  state["investment_memo"] is typed Optional[str] and app.py renders it as text.
  We still get all the Pydantic validation benefits (risks as a list, recommendation
  as a Literal) during generation — the formatting step happens after validation,
  so bad LLM output is caught before it ever becomes a string.

CLASS CONCEPTS:
  - LCEL chain (prompt | llm | parser) ........... Sessions 8-9
  - PydanticOutputParser ......................... Sessions 8-9
  - ChatPromptTemplate.partial() ................. Sessions 8-9
  - Gemini (ChatGoogleGenerativeAI) .............. Sessions 3, 10-11
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from chains.output_models import InvestmentMemo


# ============================================================
# PROMPT
# ============================================================

_MEMO_PROMPT = """\
You are a partner at a tier-1 venture capital firm writing an internal investment \
committee memo after reviewing a startup pitch deck.

ANALYSIS DATA:

Startup claims extracted from pitch deck:
{claims}

Independent web validation evidence:
{validation}

Dimension scores (0.0–10.0 scale):
{scores}

Composite weighted score: {composite_score} / 10.0
(Weighted: 30% market, 30% team, 25% product, 15% traction)

{human_feedback_section}

WRITING GUIDELINES:
  - Be direct and honest — do not sugarcoat weak scores
  - Reference specific numbers from the claims and validation evidence
  - risks must be 3–5 distinct, standalone sentences — not a single paragraph
  - recommendation must be consistent with the composite score:
      composite >= 7.0  → lean toward INVEST
      composite 5.0–6.9 → CONDITIONAL INVEST
      composite < 5.0   → lean toward PASS
  - If human analyst feedback is present, factor it into the recommendation
  - recommendation_rationale must name the composite score and cite one key
    strength and one key concern

{format_instructions}"""


# ============================================================
# HUMAN FEEDBACK FORMATTER
# ============================================================

def _format_human_section(feedback: str | None) -> str:
    """
    Returns a formatted prompt block when a human analyst reviewed the pitch,
    or an empty string when no human review occurred.

    Keeping this logic here (rather than in the prompt template) means the
    prompt stays clean — no Jinja-style conditionals, no empty headings.
    """
    if not feedback:
        return ""
    return f"HUMAN ANALYST REVIEW (overrides scores where noted):\n{feedback}"


# ============================================================
# MEMO FORMATTER
# ============================================================

def _format_memo_to_markdown(memo: InvestmentMemo) -> str:
    """
    Converts a validated InvestmentMemo into a readable markdown string.

    risks is a list[str] at this point — guaranteed 3–5 items by Pydantic.
    We convert it here to a markdown bullet list rather than asking Gemini
    to format it, which avoids inconsistent spacing and numbering.
    """
    risks_md = "\n".join(f"- {risk}" for risk in memo.risks)

    return f"""\
## 1. EXECUTIVE SUMMARY
{memo.executive_summary}

## 2. BUSINESS OVERVIEW
{memo.business_overview}

## 3. TEAM ASSESSMENT
{memo.team_assessment}

## 4. TRACTION & EVIDENCE
{memo.traction_evidence}

## 5. RISKS & CONCERNS
{risks_md}

## 6. FINAL RECOMMENDATION
**{memo.recommendation}**

{memo.recommendation_rationale}"""


# ============================================================
# CHAIN FACTORY (lazy singleton)
# ============================================================

_memo_chain = None


def get_memo_chain():
    """
    Builds and caches the memo writer chain on first call.

    Same lazy-singleton pattern as get_scorer_chain() in scorer.py.

    Chain anatomy:
      prompt  → fills {claims}, {validation}, {scores}, {composite_score},
                {human_feedback_section}, {format_instructions}
      llm     → Gemini 2.5 Flash generates each memo section as JSON
      parser  → PydanticOutputParser validates → InvestmentMemo instance

    temperature=0.3 (slightly higher than scorer):
      The memo is a written document — some variation in phrasing is fine
      and makes the output feel less robotic. Scoring needs consistency;
      writing benefits from mild creativity.
    """
    global _memo_chain

    if _memo_chain is None:
        parser = PydanticOutputParser(pydantic_object=InvestmentMemo)

        prompt = ChatPromptTemplate.from_template(_MEMO_PROMPT).partial(
            format_instructions=parser.get_format_instructions()
        )

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

        _memo_chain = prompt | llm | parser

    return _memo_chain


# ============================================================
# NODE FUNCTION — this is what graph/nodes.py calls
# ============================================================

def write_memo(state: dict) -> dict:
    """
    LangGraph node function: reads full state, generates and validates the
    investment memo, formats it to markdown, writes to state.

    OUTPUT TO STATE:
      investment_memo — formatted markdown string (replaces the raw Gemini text
                        previously produced by write_memo_node in nodes.py)
      error           — None on success, error message on failure

    The scores dict from state already contains composite_score (written there
    by score_claims() via ScoreResult.model_dump()) — we extract it directly
    rather than recomputing.
    """
    claims = state.get("extracted_claims", {})
    validation = state.get("validation_results", {})
    scores = state.get("scores", {})
    human_feedback = state.get("human_feedback")

    composite_score = scores.get("composite_score", "N/A")

    try:
        result: InvestmentMemo = get_memo_chain().invoke(
            {
                "claims": str(claims),
                "validation": str(validation),
                "scores": str(scores),
                "composite_score": composite_score,
                "human_feedback_section": _format_human_section(human_feedback),
            }
        )

        return {
            "investment_memo": _format_memo_to_markdown(result),
            "error": None,
        }

    except Exception as e:
        return {
            "investment_memo": f"Memo generation failed. Error: {str(e)}",
            "error": str(e),
        }
