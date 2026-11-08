"""Backend contracts and COMET null-reporting, without loading any weights."""
import pytest

from taf.comet import CometScorer
from taf.config import StreamingConfig
from taf.hf_backends import HFLLM, HFMT
from taf.policy import build_policy


def test_hf_backend_modules_expose_expected_contract():
    assert hasattr(HFLLM, "sample")
    assert hasattr(HFMT, "translate")
    assert hasattr(HFMT, "token_strings")
    assert hasattr(HFMT, "decode")


def test_comet_unavailable_is_null_not_fabricated():
    scorer = CometScorer.unavailable("no model")
    result = scorer.score(["Hallo"], ["Hello"], ["Hello"])
    assert result["score"] is None
    assert result["status"] == "unavailable"
    assert "no model" in result["reason"]
    assert result["kind"] == "none"


def test_taf_requires_llm():
    cfg = StreamingConfig(policy="taf")
    with pytest.raises(ValueError):
        build_policy(None, object(), cfg)


def test_candidate_budget_for_divisible_and_non_divisible_n():
    assert StreamingConfig(n=10, beam_width=40).candidate_budget == 40
    assert StreamingConfig(n=8, beam_width=40).candidate_budget == 40
    assert StreamingConfig(n=4, beam_width=40).candidate_budget == 40
    # 40/6 is not integral; floor is auditable rather than hidden.
    cfg = StreamingConfig(n=6, beam_width=40)
    assert cfg.effective_beam_width == 6
    assert cfg.candidate_budget == 36
