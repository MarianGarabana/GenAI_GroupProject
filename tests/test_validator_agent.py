"""
tests/test_validator_agent.py — Unit tests for the Role 3 validator agent

Run with: python -m pytest tests/test_validator_agent.py -v
"""

import pytest
from agents.validator_agent import validate_claims

MOCK_CLAIMS = {
    "market_size": "The electric vehicle market is projected to reach $800B by 2027",
    "team_background": "CEO has 15 years in renewable energy, CTO ex-Tesla engineer",
    "traction": "5,000 paying customers and $200K ARR after 12 months",
    "product_description": "A smart home energy management platform using AI",
}


@pytest.fixture(scope="module")
def agent_result():
    """Run validate_claims once and reuse the result across all tests."""
    state = {"extracted_claims": MOCK_CLAIMS}
    result = validate_claims(state)

    # If running in a region where the API is blocked or credits are exhausted,
    # skip these tests as they require a live API call.
    error = result.get("error", "")
    if "PERMISSION_DENIED" in str(error) or "RESOURCE_EXHAUSTED" in str(error) or "429" in str(error):
        pytest.skip(f"Skipped due to Gemini API issue: {str(error)[:100]}")

    return result


def test_returns_four_claims(agent_result):
    """Agent should return exactly one result per input claim."""
    assert len(agent_result["validation_results"]) == 4


def test_all_claim_keys_present(agent_result):
    """All 4 expected keys should be in the output."""
    assert set(agent_result["validation_results"].keys()) == set(
        MOCK_CLAIMS.keys()
    )


def test_no_error(agent_result):
    """Agent should not return an error on valid claims."""
    assert agent_result.get("error") is None


def test_each_result_is_string(agent_result):
    """Each validation result should be a non-empty string."""
    for key, value in agent_result["validation_results"].items():
        assert isinstance(value, str), f"{key} is not a string"
        assert len(value) > 0, f"{key} is empty"


def test_minimum_output_length(agent_result):
    """Each result should be long enough for the scorer to work with."""
    for key, value in agent_result["validation_results"].items():
        assert len(value) >= 100, (
            f"{key} is too short ({len(value)} chars) for scoring"
        )


def test_no_duplicate_sections(agent_result):
    """Each result should not contain content from other claim sections."""
    keys = list(agent_result["validation_results"].keys())
    for key in keys:
        other_keys = [k for k in keys if k != key]
        content = agent_result["validation_results"][key].lower()
        for other in other_keys:
            other_label = other.replace("_", " ")
            assert other_label not in content or content.index(
                other_label
            ) == 0, f"{key} result contains content from {other}"


def test_not_mentioned_claims():
    """Not-mentioned claims should pass through without hitting the agent."""
    state = {
        "extracted_claims": {
            "market_size": "The EV market is worth $500 billion",
            "team_background": "Not mentioned in pitch deck",
            "traction": "Not mentioned in pitch deck",
            "product_description": "An AI-powered energy management system",
        }
    }
    outcome = validate_claims(state)

    error = outcome.get("error", "")
    if "RESOURCE_EXHAUSTED" in str(error) or "429" in str(error):
        pytest.skip("Skipped due to Gemini API rate limit")

    assert "Not mentioned" in outcome["validation_results"]["team_background"]
    assert "Not mentioned" in outcome["validation_results"]["traction"]
    assert len(outcome["validation_results"]["market_size"]) > 100
