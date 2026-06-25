"""
tests/test_chains.py — Unit tests for R4 chain modules

These tests make NO real API calls to Gemini.
They cover three things:
  1. Pydantic model constraints in output_models.py
  2. Pure helper functions in memo_writer.py
  3. Early-exit / error-handling behaviour of the node functions

Run with:  pytest tests/test_chains.py -v

CLASS CONCEPT: Testing structured outputs — Sessions 8-9
"""

import pytest
from pydantic import ValidationError

from chains.output_models import InvestmentMemo, ScoreReasoning, ScoreResult
from chains.memo_writer import _format_human_section, _format_memo_to_markdown
from chains.scorer import score_claims
from chains.memo_writer import write_memo


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _reasoning(**kwargs):
    defaults = dict(market="m", team="t", traction="tr", product="p")
    return ScoreReasoning(**{**defaults, **kwargs})


def _score_result(**kwargs):
    defaults = dict(
        market=7.0, team=8.0, traction=6.0, product=7.0,
        reasoning=_reasoning(),
        confidence="medium",
    )
    return ScoreResult(**{**defaults, **kwargs})


def _memo(**kwargs):
    defaults = dict(
        executive_summary="A fintech startup.",
        business_overview="Large underserved market.",
        team_assessment="Experienced founders.",
        traction_evidence="200 customers, 15% MoM growth.",
        risks=["Regulatory risk.", "High CAC.", "Single revenue stream."],
        recommendation="CONDITIONAL INVEST",
        recommendation_rationale="Composite score of 6.8 with strong team.",
    )
    return InvestmentMemo(**{**defaults, **kwargs})


# ─────────────────────────────────────────────
# ScoreResult — Pydantic constraints
# ─────────────────────────────────────────────

class TestScoreResult:

    def test_composite_score_computed_correctly(self):
        s = _score_result(market=8.0, team=9.0, traction=4.0, product=7.0)
        expected = round(0.30*8.0 + 0.30*9.0 + 0.25*7.0 + 0.15*4.0, 2)
        assert s.composite_score == expected

    def test_composite_not_a_simple_average(self):
        # Market + team dominate (30% each). A great team should pull composite up
        # more than equal weight would.
        high_team = _score_result(market=6.0, team=10.0, traction=6.0, product=6.0)
        simple_avg = (6.0 + 10.0 + 6.0 + 6.0) / 4  # = 7.0
        assert high_team.composite_score != simple_avg

    def test_score_above_max_rejected(self):
        with pytest.raises(ValidationError):
            _score_result(market=10.1)

    def test_score_below_min_rejected(self):
        with pytest.raises(ValidationError):
            _score_result(team=-0.1)

    def test_score_at_boundary_accepted(self):
        s = _score_result(market=0.0, team=10.0)
        assert s.market == 0.0
        assert s.team == 10.0

    def test_invalid_confidence_rejected(self):
        with pytest.raises(ValidationError):
            _score_result(confidence="very high")

    def test_all_valid_confidence_values_accepted(self):
        for level in ("high", "medium", "low"):
            s = _score_result(confidence=level)
            assert s.confidence == level

    def test_reasoning_enforces_four_keys(self):
        # ScoreReasoning is a typed model — missing a key raises
        with pytest.raises((ValidationError, TypeError)):
            ScoreReasoning(market="ok", team="ok")  # missing traction + product

    def test_model_dump_includes_composite_score(self):
        s = _score_result()
        d = s.model_dump()
        assert "composite_score" in d
        assert "market" in d
        assert "confidence" in d


# ─────────────────────────────────────────────
# InvestmentMemo — Pydantic constraints
# ─────────────────────────────────────────────

class TestInvestmentMemo:

    def test_valid_memo_accepted(self):
        m = _memo()
        assert m.recommendation == "CONDITIONAL INVEST"

    def test_invalid_recommendation_rejected(self):
        with pytest.raises(ValidationError):
            _memo(recommendation="MAYBE")

    def test_all_valid_recommendations_accepted(self):
        for rec in ("INVEST", "PASS", "CONDITIONAL INVEST"):
            m = _memo(recommendation=rec)
            assert m.recommendation == rec

    def test_risks_too_short_rejected(self):
        with pytest.raises(ValidationError):
            _memo(risks=["Only one risk."])

    def test_risks_too_long_rejected(self):
        with pytest.raises(ValidationError):
            _memo(risks=["r1", "r2", "r3", "r4", "r5", "r6"])

    def test_risks_at_minimum_accepted(self):
        m = _memo(risks=["r1", "r2", "r3"])
        assert len(m.risks) == 3

    def test_risks_at_maximum_accepted(self):
        m = _memo(risks=["r1", "r2", "r3", "r4", "r5"])
        assert len(m.risks) == 5


# ─────────────────────────────────────────────
# _format_human_section
# ─────────────────────────────────────────────

class TestFormatHumanSection:

    def test_none_returns_empty_string(self):
        assert _format_human_section(None) == ""

    def test_empty_string_returns_empty_string(self):
        assert _format_human_section("") == ""

    def test_feedback_appears_in_output(self):
        result = _format_human_section("Team is exceptional. Proceed.")
        assert "Team is exceptional. Proceed." in result

    def test_output_includes_header(self):
        result = _format_human_section("Some feedback")
        assert "HUMAN ANALYST REVIEW" in result


# ─────────────────────────────────────────────
# _format_memo_to_markdown
# ─────────────────────────────────────────────

class TestFormatMemoToMarkdown:

    def test_all_six_sections_present(self):
        md = _format_memo_to_markdown(_memo())
        for heading in [
            "## 1. EXECUTIVE SUMMARY",
            "## 2. BUSINESS OVERVIEW",
            "## 3. TEAM ASSESSMENT",
            "## 4. TRACTION & EVIDENCE",
            "## 5. RISKS & CONCERNS",
            "## 6. FINAL RECOMMENDATION",
        ]:
            assert heading in md, f"Missing: {heading}"

    def test_recommendation_is_bold(self):
        md = _format_memo_to_markdown(_memo(recommendation="INVEST"))
        assert "**INVEST**" in md

    def test_risks_formatted_as_bullets(self):
        md = _format_memo_to_markdown(_memo(
            risks=["Risk A.", "Risk B.", "Risk C."]
        ))
        assert "- Risk A." in md
        assert "- Risk B." in md
        assert "- Risk C." in md

    def test_risks_not_a_single_line(self):
        # Three risks should produce three separate bullet lines
        md = _format_memo_to_markdown(_memo(
            risks=["Risk A.", "Risk B.", "Risk C."]
        ))
        lines_with_bullets = [l for l in md.splitlines() if l.startswith("- ")]
        assert len(lines_with_bullets) == 3


# ─────────────────────────────────────────────
# score_claims — early-exit / error handling
# (no API call — triggers the empty-claims guard)
# ─────────────────────────────────────────────

class TestScoreClaimsGuards:

    def test_empty_claims_returns_error_state(self):
        result = score_claims({"extracted_claims": {}, "validation_results": {}})
        assert result["human_review_required"] is True
        assert "Cannot score" in result["error"]
        assert result["scores"] == {}

    def test_missing_claims_key_handled(self):
        result = score_claims({})
        assert result["human_review_required"] is True

    def test_returns_expected_keys(self):
        result = score_claims({"extracted_claims": {}, "validation_results": {}})
        assert "scores" in result
        assert "human_review_required" in result
        assert "error" in result


# ─────────────────────────────────────────────
# write_memo — error handling without API
# ─────────────────────────────────────────────

class TestWriteMemoGuards:

    def test_returns_expected_keys(self):
        # Will fail with SSL/API error but must still return correct keys
        result = write_memo({
            "extracted_claims": {},
            "validation_results": {},
            "scores": {},
            "human_feedback": None,
        })
        assert "investment_memo" in result
        assert "error" in result

    def test_no_crash_on_missing_state_keys(self):
        result = write_memo({})
        assert "investment_memo" in result
