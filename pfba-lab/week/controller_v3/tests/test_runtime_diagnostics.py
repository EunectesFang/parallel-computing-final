"""Tests for risk-head calibration, candidate parity, and runtime diagnostics.

All fakes are CPU-only and deterministic. No model weights are downloaded.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from controller_v3.action import Action
from controller_v3.config import ControllerConfig
from controller_v3.evaluator import risk_calibration
from controller_v3.features import NUMERIC_FEATURES, StateInput
from controller_v3.labeler import CounterfactualLabeler
from controller_v3.runtime import ControlledRuntime
from controller_v3.trainer import ActionTarget

WRITE = Action(2, 4)


def _state() -> StateInput:
    return StateInput(
        numeric=(0.0,) * len(NUMERIC_FEATURES),
        predictor_id=0,
        translator_id=0,
    )


def _target(error_target: float, no_emit_target: float, action=WRITE):
    return ActionTarget(
        state=_state(),
        action=action,
        value_target=0.0,
        error_target=float(error_target),
        no_emit_target=float(no_emit_target),
        emit_target=1.0,
        state_key="k",
    )


# ----------------------------------------------------------------------
# Risk calibration
# ----------------------------------------------------------------------
class RiskNetwork:
    """Constant risk logits so Brier/ECE have hand-computable values."""

    def __init__(self, error_logit: float = 0.0, no_emit_logit: float = 2.0):
        self.error_logit = float(error_logit)
        self.no_emit_logit = float(no_emit_logit)
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self

    def encode(self, numeric, predictor_id, translator_id, src_ids, tgt_ids):
        return torch.zeros((numeric.size(0), 2), dtype=torch.float32)

    def write_values(self, z, actions):
        batch = z.size(0)
        count = len(actions)
        return SimpleNamespace(
            value=torch.zeros((batch, count)),
            error_risk=torch.full((batch, count), self.error_logit),
            no_emit=torch.full((batch, count), self.no_emit_logit),
            expected_emit=torch.ones((batch, count)),
        )


def test_risk_calibration_brier_and_ece_are_computed_from_sigmoid():
    network = RiskNetwork(error_logit=0.0, no_emit_logit=2.0)
    targets = [
        _target(1.0, 0.0),
        _target(0.0, 1.0),
    ]

    result = risk_calibration(network, targets, buckets=10)

    assert result["status"] == "ok"
    assert result["n"] == 2
    # sigmoid(0) = 0.5; Brier over labels 1,0 is 0.25 and the reliability
    # bucket gap is zero because the mean label is also 0.5.
    assert result["error_risk"]["brier"] == pytest.approx(0.25)
    assert result["error_risk"]["ece"] == pytest.approx(0.0)

    probability = 1.0 / (1.0 + math.exp(-2.0))
    assert result["no_emit"]["mean_predicted"] == pytest.approx(probability)
    assert result["no_emit"]["base_rate"] == pytest.approx(0.5)
    expected_brier = ((probability - 0.0) ** 2 + (probability - 1.0) ** 2) / 2
    assert result["no_emit"]["brier"] == pytest.approx(expected_brier)
    assert result["no_emit"]["ece"] == pytest.approx(abs(probability - 0.5))
    assert "uncalibrated" in result["threshold_note"]

    # eval() is restored to the previous training flag.
    assert network.training is True


def test_risk_calibration_skips_read_and_reports_empty_as_unavailable():
    network = RiskNetwork()
    assert risk_calibration(network, [])["status"] == "unavailable"
    read_only = [_target(0.0, 0.0, action=Action(0, 0))]
    assert risk_calibration(network, read_only)["status"] == "unavailable"


# ----------------------------------------------------------------------
# Candidate-generation parity
# ----------------------------------------------------------------------
class ParityPredictor:
    """One candidate per call, seeded exactly like the runtime."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int, int]] = []

    def sample(self, observed, branches, horizon, seed):
        self.calls.append((observed, int(branches), int(horizon), int(seed)))
        assert int(branches) == 1
        return [
            " ".join(f"w{int(seed) + depth}" for depth in range(int(horizon)))
        ]


class IdentityTranslator:
    def translate_batch(self, sources, prefix, final=False, **kwargs):
        return [str(source) for source in sources]


def test_per_candidate_labeling_matches_runtime_candidate_rows():
    cfg = ControllerConfig(actions=(WRITE,))
    predictor = ParityPredictor()
    translator = IdentityTranslator()
    labeler = CounterfactualLabeler(
        predictor,
        translator,
        cfg,
        candidate_mode="per_candidate",
    )
    assert labeler.candidate_mode == "per_candidate"

    outcomes = labeler.label_state(
        "a b", (), "a b c", [WRITE], seed=5
    )
    runtime = ControlledRuntime(predictor, translator, cfg)
    result = runtime.execute("a b", (), WRITE, seed=5)

    # Same action, prefix, and seed: the translated candidate pool is
    # identical because both paths sample candidate i with seed + i.
    assert outcomes[0].candidates == result.candidates
    assert [outcome.new_words for outcome in outcomes] == [
        tuple(result.selected.split())
    ]
    # Both drew exactly two single-branch calls with seeds 5 and 6.
    assert predictor.calls[:2] == predictor.calls[2:4]
    assert [call[3] for call in predictor.calls[:2]] == [5, 6]
    assert all(call[1] == 1 and call[2] == 4 for call in predictor.calls)


class WordTokenizer:
    eos_token_id = 99

    def decode(self, token_ids, skip_special_tokens=True,
               clean_up_tokenization_spaces=False):
        return " ".join(f"t{int(token)}" for token in token_ids)

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [index for index, _ in enumerate(str(text).split())]}


class BatchedPredictor:
    """Only the legacy batched sampling path; records its calls."""

    def __init__(self) -> None:
        self.tok = WordTokenizer()
        self.calls: list[tuple[str, int, int, int]] = []

    def sample(self, observed, branches, horizon, seed):
        self.calls.append((observed, int(branches), int(horizon), int(seed)))
        return [f"candidate-{index}" for index in range(int(branches))]


def test_batched_candidate_mode_is_an_explicit_ablation():
    predictor = BatchedPredictor()
    labeler = CounterfactualLabeler(
        predictor,
        IdentityTranslator(),
        ControllerConfig(actions=(WRITE,)),
        candidate_mode="batched",
    )
    assert labeler.candidate_mode == "batched"
    rows = labeler.sample_candidates("obs", 2, 4, seed=5)
    assert len(rows) == 2
    # The ablation draws the whole pool with one call, unlike the runtime.
    assert predictor.calls == [("obs", 2, 4, 5)]


class RowsOnlyPredictor:
    def sample_token_rows(self, observed, max_b, max_h, seed):
        return [[1, 2] for _ in range(int(max_b))]


def test_per_candidate_request_falls_back_to_batched_without_sample():
    labeler = CounterfactualLabeler(
        RowsOnlyPredictor(),
        IdentityTranslator(),
        ControllerConfig(actions=(WRITE,)),
        candidate_mode="per_candidate",
    )
    # The predictor cannot draw one candidate at a time, so the labeler records
    # the batched mode it actually used instead of claiming parity.
    assert labeler.candidate_mode == "batched"
