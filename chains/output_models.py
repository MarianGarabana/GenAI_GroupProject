"""
chains/output_models.py — Pydantic Output Models

These are the data contracts between Gemini and the rest of the LangGraph pipeline.
PydanticOutputParser generates a JSON schema from each model and injects it into
the prompt so Gemini knows exactly what structure to return.

Pydantic then validates the response — if Gemini returns a score of 11 or a
recommendation that isn't one of the three allowed strings, it raises immediately
rather than silently passing bad data into the graph state.

CLASS CONCEPTS:
  - Structured output with Pydantic .............. Sessions 8-9
  - Pydantic output parser in LCEL chains ........ Sessions 8-9
  - Data validation at system boundaries ......... General best practice
"""

from typing import Literal
from pydantic import BaseModel, Field, computed_field


# ============================================================
# SCORING MODELS
# ============================================================

class ScoreReasoning(BaseModel):
    """
    One explanatory sentence per scoring dimension.

    Using a nested model (instead of dict[str, str]) forces Gemini to output
    exactly these four keys. A plain dict would let the LLM invent key names
    like "Market Size" or "market_opportunity" — breaking any downstream code
    that reads state["scores"]["market"].
    """
    market: str = Field(description="One sentence explaining the market score")
    team: str = Field(description="One sentence explaining the team score")
    traction: str = Field(description="One sentence explaining the traction score")
    product: str = Field(description="One sentence explaining the product score")


class ScoreResult(BaseModel):
    """
    Structured output from the scorer LCEL chain.

    Scores are floats 0.0–10.0 with one decimal of precision (e.g. 7.5, not 7).
    Pydantic enforces the bounds via ge/le constraints — a score of 10.5 raises
    a ValidationError before it can corrupt the graph state.

    composite_score is a @computed_field: it does NOT appear in the JSON schema
    sent to Gemini. The model computes it after validation using VC-realistic
    weightings. This guarantees mathematical correctness regardless of what
    the LLM does.

    CLASS CONCEPT: "Pydantic output parser" — Sessions 8-9
    """
    market: float = Field(
        ge=0.0, le=10.0,
        description="Market size and opportunity score with one decimal place (e.g. 7.5)"
    )
    team: float = Field(
        ge=0.0, le=10.0,
        description="Founding team quality and relevant experience score with one decimal place (e.g. 8.0)"
    )
    traction: float = Field(
        ge=0.0, le=10.0,
        description="Revenue, growth metrics, and customer traction score with one decimal place (e.g. 4.5)"
    )
    product: float = Field(
        ge=0.0, le=10.0,
        description="Product differentiation, defensibility, and technical quality score with one decimal place (e.g. 6.5)"
    )
    reasoning: ScoreReasoning = Field(
        description="One sentence per dimension explaining why that score was given"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "Confidence in these scores based on evidence quality: "
            "'high' if claims are well-supported by web evidence, "
            "'medium' if evidence is partial or indirect, "
            "'low' if evidence is missing or contradictory"
        )
    )

    @computed_field
    @property
    def composite_score(self) -> float:
        """
        Weighted composite score using realistic early-stage VC weighting:
          30% market  — TAM is the ceiling on returns
          30% team    — team is the most important signal at seed stage
          25% product — differentiation and defensibility
          15% traction — less weight at early stage; market/team matter more

        Rounded to 2 decimal places to avoid float noise (e.g. 7.000000001).
        """
        return round(
            0.30 * self.market
            + 0.30 * self.team
            + 0.25 * self.product
            + 0.15 * self.traction,
            2
        )


# ============================================================
# MEMO MODEL
# ============================================================

class InvestmentMemo(BaseModel):
    """
    Structured output from the memo writer LCEL chain.

    Each section of the memo is its own field rather than one big markdown blob.
    This matters for the UI (each section can be rendered separately) and for
    grading (the professor can clearly see every component of the memo).

    risks is a list[str] rather than a string — Pydantic enforces 3–5 items,
    so the LLM can't produce a single run-on paragraph instead of distinct risks.

    recommendation is a Literal — Pydantic rejects any string that isn't exactly
    one of the three options, including "Conditional Invest" (wrong capitalisation)
    or "MAYBE INVEST" (invented option).

    CLASS CONCEPT: "Structured output with Pydantic" — Sessions 8-9
    """
    executive_summary: str = Field(
        description="2-3 sentences: who they are, what they do, and overall investment impression"
    )
    business_overview: str = Field(
        description="Market opportunity, product/service description, and competitive positioning"
    )
    team_assessment: str = Field(
        description="Founder credentials, relevant domain experience, and team completeness"
    )
    traction_evidence: str = Field(
        description="Growth metrics, revenue, customers, and milestones — cross-referenced with web validation evidence"
    )
    risks: list[str] = Field(
        min_length=3,
        max_length=5,
        description="3 to 5 specific, distinct risks as short standalone sentences (not a paragraph)"
    )
    recommendation: Literal["INVEST", "PASS", "CONDITIONAL INVEST"] = Field(
        description=(
            "Final investment decision. Use exactly one of: INVEST, PASS, CONDITIONAL INVEST. "
            "INVEST if composite score >= 7.0 and no critical red flags. "
            "PASS if composite score < 5.0 or a fundamental flaw exists. "
            "CONDITIONAL INVEST for scores between 5.0 and 7.0 or specific concerns that could be resolved."
        )
    )
    recommendation_rationale: str = Field(
        description=(
            "2-3 sentences explaining the recommendation. "
            "Must reference the composite score and name the key strength and key concern."
        )
    )
