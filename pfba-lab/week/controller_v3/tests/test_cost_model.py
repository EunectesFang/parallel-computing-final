"""Tests for the measured/power-law/pipeline cost models and the runtime."""

import math

import pytest

from controller_v3.action import READ, Action
from controller_v3.config import ControllerConfig
from controller_v3.cost_model import (
    MIN_COST_S,
    CostEstimate,
    MeasuredCostTable,
    PipelineCostModel,
    PowerLawCostModel,
)
from controller_v3.runtime import ControlledRuntime, StepResult, vote_statistics

WRITE = Action(2, 4)


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FakeCostModel:
    """Constant write cost used by the pipeline test."""

    def __init__(self, pred_s: float = 1.0, mt_s: float = 1.0) -> None:
        self.pred_s = pred_s
        self.mt_s = mt_s
        self.calls: list[tuple] = []

    def predict(self, predictor_id, translator_id, action, src_words, tgt_words):
        self.calls.append((predictor_id, translator_id, action, src_words, tgt_words))
        if action.is_read:
            return CostEstimate(0.0, 0.0)
        return CostEstimate(self.pred_s, self.mt_s)


class FakePredictor:
    """One continuation per call; records the per-candidate seeds."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int, int]] = []

    def sample(self, prefix: str, branches: int, horizon: int, seed: int):
        self.calls.append((prefix, branches, horizon, seed))
        return [f"cont-{seed}"]


class ScriptedTranslator:
    """Returns queued translations, one per requested source."""

    def __init__(self, outputs) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[list[str], str, dict]] = []

    def translate_batch(self, sources, prefix, **kwargs):
        self.calls.append((list(sources), prefix, dict(kwargs)))
        count = len(sources)
        batch = self.outputs[:count]
        self.outputs = self.outputs[count:]
        return batch


# ----------------------------------------------------------------------
# Measured table
# ----------------------------------------------------------------------
def test_measured_table_exact_nearest_global_and_floor():
    table = MeasuredCostTable(
        [
            (0, 0, Action(2, 4), 0.2, 0.3),
            (0, 0, Action(2, 4), 0.4, 0.5),
            (0, 0, Action(4, 6), 0.8, 0.9),
            (1, 1, Action(2, 4), 1.0, 1.2),
        ]
    )

    exact = table.predict(0, 0, Action(2, 4), src_words=5, tgt_words=4)
    assert exact.pred_s == pytest.approx(0.3)  # median of 0.2, 0.4
    assert exact.mt_s == pytest.approx(0.4)  # median of 0.3, 0.5
    assert exact.wall_s == pytest.approx(0.7)

    # (6, 6) is closer to the measured (4, 6) than to (2, 4).
    nearest = table.predict(0, 0, Action(6, 6), src_words=5, tgt_words=4)
    assert nearest.pred_s == pytest.approx(0.8)
    assert nearest.mt_s == pytest.approx(0.9)

    # Same model pair, exact action kept even when the record set has others.
    assert table.predict(1, 1, Action(2, 4), 0, 0).pred_s == pytest.approx(1.0)

    # Unseen model pair: global median over all four records.
    global_estimate = table.predict(9, 9, Action(3, 5), src_words=5, tgt_words=4)
    assert global_estimate.pred_s == pytest.approx(0.6)
    assert global_estimate.mt_s == pytest.approx(0.7)

    empty = MeasuredCostTable().predict(0, 0, WRITE, 1, 1)
    assert empty.pred_s == pytest.approx(MIN_COST_S)
    assert empty.mt_s == pytest.approx(MIN_COST_S)
    assert empty.wall_s == pytest.approx(2 * MIN_COST_S)


def test_measured_table_read_is_free():
    table = MeasuredCostTable([(0, 0, Action(2, 4), 0.5, 0.5)])
    estimate = table.predict(0, 0, READ, src_words=3, tgt_words=2)
    assert estimate == CostEstimate(0.0, 0.0, 0.0)
    assert estimate.wall_s == 0.0


def test_measured_table_never_uses_read_for_write_fallback():
    # A stray READ row must not turn a write action into a zero-cost action.
    only_read = MeasuredCostTable([(0, 0, READ, 0.0, 0.0)])
    estimate = only_read.predict(0, 0, WRITE, src_words=3, tgt_words=2)
    assert estimate.pred_s == pytest.approx(MIN_COST_S)
    assert estimate.mt_s == pytest.approx(MIN_COST_S)

    mixed = MeasuredCostTable(
        [
            (0, 0, READ, 0.0, 0.0),
            (0, 0, Action(8, 10), 0.9, 0.8),
        ]
    )
    nearest = mixed.predict(0, 0, WRITE, src_words=3, tgt_words=2)
    assert nearest.pred_s == pytest.approx(0.9)  # the write row, not READ
    assert nearest.mt_s == pytest.approx(0.8)


# ----------------------------------------------------------------------
# Power law
# ----------------------------------------------------------------------
def _power_law_records(predictor_id=0, translator_id=0):
    actions = (
        Action(2, 4),
        Action(4, 2),
        Action(6, 10),
        Action(8, 6),
        Action(10, 8),
    )
    records = []
    for action in actions:
        pred = 0.02 * action.branches**0.5 * action.horizon**0.25
        mt = 0.03 * action.branches**0.4 * action.horizon**0.2
        records.append((predictor_id, translator_id, action, pred, mt))
    return records


def test_power_law_recovers_monotone_exponents():
    model = PowerLawCostModel(_power_law_records())
    exponents = model.exponents(0, 0)
    assert exponents is not None
    r_b, r_h, s_b, s_h = exponents
    assert r_b == pytest.approx(0.5, rel=1e-6)
    assert r_h == pytest.approx(0.25, rel=1e-6)
    assert s_b == pytest.approx(0.4, rel=1e-6)
    assert s_h == pytest.approx(0.2, rel=1e-6)
    assert all(0.0 <= value <= 1.0 for value in exponents)

    low = model.predict(0, 0, Action(2, 4), 5, 4)
    more_branches = model.predict(0, 0, Action(4, 4), 5, 4)
    more_horizon = model.predict(0, 0, Action(2, 8), 5, 4)
    assert more_branches.pred_s > low.pred_s
    assert more_branches.mt_s > low.mt_s
    assert more_horizon.pred_s > low.pred_s
    assert more_horizon.mt_s > low.mt_s
    assert low.wall_s == pytest.approx(low.pred_s + low.mt_s)


def test_power_law_clamps_exponents_that_would_exceed_one():
    # value = branches^3 exactly. The unclamped least-squares solution is
    # (r_b=3, r_h=0) for both stages; the model must clamp r_b to 1.
    records = [
        (0, 0, Action(2, 3), 2.0**3, 2.0**3),
        (0, 0, Action(3, 2), 3.0**3, 3.0**3),
        (0, 0, Action(4, 5), 4.0**3, 4.0**3),
        (0, 0, Action(5, 4), 5.0**3, 5.0**3),
        (0, 0, Action(6, 7), 6.0**3, 6.0**3),
    ]
    model = PowerLawCostModel(records)
    exponents = model.exponents(0, 0)
    assert exponents is not None
    r_b, r_h, s_b, s_h = exponents
    assert r_b == pytest.approx(1.0)  # clamped down from 3
    assert s_b == pytest.approx(1.0)  # clamped down from 3
    assert r_h == pytest.approx(0.0)
    assert s_h == pytest.approx(0.0)


def test_power_law_falls_back_with_fewer_than_four_records():
    records = [
        (0, 0, Action(2, 4), 0.2, 0.3),
        (0, 0, Action(4, 6), 0.4, 0.5),
        (0, 0, Action(6, 10), 0.6, 0.7),
    ]
    model = PowerLawCostModel(records)
    assert model.exponents(0, 0) is None
    # The measured table still answers: exact for an observed action,
    # nearest for an unobserved one on the same pair.
    assert model.predict(0, 0, Action(4, 6), 0, 0).pred_s == pytest.approx(0.4)
    assert model.predict(0, 0, Action(5, 6), 0, 0).pred_s == pytest.approx(0.4)


def test_power_law_read_is_free():
    model = PowerLawCostModel(_power_law_records())
    assert model.predict(0, 0, READ, 5, 4) == CostEstimate(0.0, 0.0, 0.0)


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def test_pipeline_overlap_beats_serial_sum():
    pipeline = PipelineCostModel(FakeCostModel(pred_s=1.0, mt_s=1.0))
    action = Action(4, 6)

    # The translation device is busy for 0.5 s; prediction runs during that
    # window, so the action adds 1.5 s instead of pred_s + mt_s = 2.0 s.
    overlapped = pipeline.marginal_wall(
        t_pred_free=0.0,
        t_mt_free=0.5,
        release_s=0.0,
        dependency_s=0.0,
        predictor_id=0,
        translator_id=0,
        action=action,
        src_words=5,
        tgt_words=4,
    )
    assert overlapped == pytest.approx(1.5)
    assert overlapped < 1.0 + 1.0

    # Both devices free: the strict sum, no artificial discount.
    serial = pipeline.marginal_wall(
        0.0, 0.0, 0.0, 0.0, 0, 0, action, 5, 4
    )
    assert serial == pytest.approx(2.0)

    # Released late: prediction cannot start before release_s.
    released = pipeline.marginal_wall(0.0, 0.0, 5.0, 0.0, 0, 0, action, 5, 4)
    assert released == pytest.approx(7.0)

    # Translation waits for its dependency even when its device is idle.
    dependent = pipeline.marginal_wall(0.0, 0.0, 0.0, 10.0, 0, 0, action, 5, 4)
    assert dependent == pytest.approx(11.0)

    # Never negative, even if the recorded free times look inconsistent.
    assert pipeline.marginal_wall(5.0, 5.0, 0.0, 0.0, 0, 0, action, 5, 4) >= 0.0


def test_pipeline_predict_read_is_free():
    pipeline = PipelineCostModel(MeasuredCostTable([(0, 0, Action(2, 4), 0.5, 0.5)]))
    estimate = pipeline.predict(0, 0, READ, src_words=3, tgt_words=2)
    assert estimate == CostEstimate(0.0, 0.0, 0.0)
    assert estimate.wall_s == 0.0


# ----------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------
def _runtime(outputs):
    predictor = FakePredictor()
    translator = ScriptedTranslator(outputs)
    return ControlledRuntime(predictor, translator, ControllerConfig()), predictor, translator


def test_execute_returns_candidates_and_vote_stats():
    runtime, predictor, translator = _runtime(["hello world", "hello there"])
    committed = ["x"]

    result = runtime.execute("a b c", committed, Action(2, 4), seed=10)

    assert isinstance(result, StepResult)
    assert result.candidates == ("hello world", "hello there")
    assert result.selected == "hello"
    assert result.committed_after == ("x", "hello")
    assert result.no_emit is False
    assert result.spent_s == pytest.approx(result.pred_s + result.mt_s)

    # One sample call per candidate, each with branches=1 and seed + index.
    assert predictor.calls == [
        ("a b c", 1, 4, 10),
        ("a b c", 1, 4, 11),
    ]
    # Only the two new requests are translated, with the committed prefix.
    assert len(translator.calls) == 1
    sources, prefix, _kwargs = translator.calls[0]
    assert sources == ["a b c cont-10", "a b c cont-11"]
    assert prefix == "x"

    # Decisive depth is 2: "hello" wins at depth 1, depth 2 splits 1-1.
    assert result.vote_top1 == pytest.approx(0.5)
    assert result.vote_margin == pytest.approx(0.0)
    assert result.vote_entropy == pytest.approx(math.log(2.0))

    # The caller's committed list is untouched.
    assert committed == ["x"]


def test_execute_no_emit_on_empty_translations():
    runtime, _predictor, _translator = _runtime(["", ""])
    committed = ["x"]

    result = runtime.execute("a b", committed, Action(2, 4), seed=1)

    assert result.selected == ""
    assert result.no_emit is True
    assert result.committed_after == ("x",)
    assert result.spent_s == pytest.approx(result.pred_s + result.mt_s)
    # All-empty candidates are unanimous under split(" "): one "" prefix.
    assert result.vote_top1 == pytest.approx(1.0)
    assert result.vote_margin == pytest.approx(1.0)
    assert result.vote_entropy == pytest.approx(0.0)
    assert committed == ["x"]


def test_execute_rejects_read():
    runtime, _predictor, _translator = _runtime([])
    with pytest.raises(ValueError, match="READ"):
        runtime.execute("a b", ["x"], READ, seed=0)


def test_predict_and_ralcp_do_not_mutate_inputs():
    runtime, _predictor, _translator = _runtime([])
    already_sampled = ["hello world", "hello there"]
    assert runtime.ralcp(already_sampled) == "hello"
    assert already_sampled == ["hello world", "hello there"]

    # Bias-corrected RALCP over a list also leaves the list intact.
    candidates = ["a b", "a c"]
    assert runtime.ralcp(candidates) == "a"
    assert candidates == ["a b", "a c"]


def test_candidate_seeds_are_stable_when_budget_grows():
    predictor_small = FakePredictor()
    translator_small = ScriptedTranslator(["a", "a"])
    small = ControlledRuntime(predictor_small, translator_small, ControllerConfig())
    small.execute("s", [], Action(2, 4), seed=7)

    predictor_large = FakePredictor()
    translator_large = ScriptedTranslator(["a", "a", "a", "a"])
    large = ControlledRuntime(predictor_large, translator_large, ControllerConfig())
    large.execute("s", [], Action(4, 4), seed=7)

    assert predictor_small.calls == predictor_large.calls[:2]
    assert [call[3] for call in predictor_large.calls] == [7, 8, 9, 10]


def test_extend_only_samples_and_translates_new_candidates():
    runtime, predictor, translator = _runtime(
        ["hello world", "hello there", "hello again", "hello friend"]
    )
    committed = ["x"]
    current = ["hello world", "hello there"]

    first = runtime.execute("a b", committed, Action(2, 4), seed=10)
    assert first.candidates == tuple(current)

    result = runtime.extend(
        observed="a b",
        committed=committed,
        current_candidates=current,
        extra_branches=2,
        horizon=4,
        seed=10,
        spent_s=0.25,
    )

    # The first two samples line up with a single B=4 execute; the extension
    # continues with seeds 12 and 13.
    assert [call[3] for call in predictor.calls] == [10, 11, 12, 13]
    assert len(translator.calls) == 2
    assert translator.calls[1][0] == ["a b cont-12", "a b cont-13"]

    assert result.candidates == tuple(
        ["hello world", "hello there", "hello again", "hello friend"]
    )
    assert result.selected == "hello"
    assert result.committed_after == ("x", "hello")
    assert result.spent_s == pytest.approx(0.25 + result.pred_s + result.mt_s)
    assert result.no_emit is False

    # Both inputs are untouched and the existing candidates were not retranslated.
    assert current == ["hello world", "hello there"]
    assert committed == ["x"]


def test_extend_rejects_non_positive_arguments():
    runtime, _predictor, _translator = _runtime([])
    for extra in (0, -1):
        with pytest.raises(ValueError, match="extra_branches"):
            runtime.extend("a", [], [], extra, 4, seed=0)
    for horizon in (0, -2):
        with pytest.raises(ValueError, match="horizon"):
            runtime.extend("a", [], [], 2, horizon, seed=0)


@pytest.mark.parametrize(
    "candidates",
    [
        ["a  b", "a b"],  # repeated space: one candidate has an extra "" token
        [" a b", "a b"],  # leading space: an extra leading "" token
        ["a b ", "a b"],  # trailing space: an extra trailing "" token
    ],
)
def test_vote_statistics_uses_ralcp_space_tokenization(candidates):
    # split(" ") sees the whitespace difference as a real token, so the pool
    # splits 1-1 at the decisive depth. A naive split() would call this
    # unanimous (top1 = 1.0) and report the wrong decisive depth.
    top1, margin, entropy = vote_statistics(candidates, 0.6)
    assert top1 == pytest.approx(0.5)
    assert margin == pytest.approx(0.0)
    assert entropy == pytest.approx(math.log(2.0))


def test_vote_statistics_all_empty_candidates_is_defined():
    top1, margin, entropy = vote_statistics(["", ""])
    assert top1 == pytest.approx(1.0)
    assert margin == pytest.approx(1.0)
    assert entropy == pytest.approx(0.0)

    assert vote_statistics([]) == (0.0, 0.0, 0.0)


def test_vote_statistics_consensus_and_split():
    top1, margin, entropy = vote_statistics(["a b", "a b", "a b"])
    assert top1 == pytest.approx(1.0)
    assert margin == pytest.approx(1.0)
    assert entropy == pytest.approx(0.0)

    top1, margin, entropy = vote_statistics(["a b", "a b", "a c"])
    # Depth 1: "a" 3/3; depth 2: "a b" 2/3 which is above 0.6*3, so the
    # threshold never fails and the pool's deepest depth is decisive.
    assert top1 == pytest.approx(2 / 3)
    assert margin == pytest.approx(1 / 3)
    assert entropy == pytest.approx(-(2 / 3) * math.log(2 / 3) - (1 / 3) * math.log(1 / 3))

    top1, margin, entropy = vote_statistics(["a b", "a c", "a d"])
    # Depth 2 fails (1/3 < 0.6), so depth 2 is decisive.
    assert top1 == pytest.approx(1 / 3)
    assert margin == pytest.approx(0.0)
    assert entropy == pytest.approx(math.log(3.0))
