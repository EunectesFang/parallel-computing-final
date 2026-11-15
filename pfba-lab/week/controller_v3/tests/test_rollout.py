"""Tests for the rollout, post-candidate integration, and evaluator.

Every predictor, translator, runtime, network, cost model, and timer here is a
fake. No model weights, GPU, torch tensor construction, or network access is
needed: the fake network returns plain Python sequences, and `run_rollout`
reads every score through `float(...)`.
"""

from types import SimpleNamespace

import pytest

from controller_v3 import evaluator
from controller_v3.action import Action
from controller_v3.config import ControllerConfig
from controller_v3.cost_model import CostEstimate
from controller_v3.evaluator import MetricUnavailableError, paired_bootstrap, summarize_traces
from controller_v3.features import FeatureBuilder
from controller_v3.post_controller import PostAction, PostDecision
from controller_v3.rollout import RolloutTrace, run_rollout
from controller_v3.runtime import StepResult


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FakeNetwork:
    """Returns scripted READ values and constant write Q values.

    `read_values` is consumed one entry per decision; the last entry repeats.
    The fake ignores its tensor inputs and returns plain lists, which the
    rollout converts with `float(...)`.
    """

    def __init__(self, read_values, write_values, no_emit_logit=-10.0):
        self.read_values = list(read_values)
        self.write_q = dict(write_values)
        self.no_emit_logit = float(no_emit_logit)
        self.decisions = 0
        self.write_queries = []

    def encode(self, numeric, predictor_id, translator_id, src_ids, tgt_ids):
        return None

    def read_value(self, z):
        index = min(self.decisions, len(self.read_values) - 1)
        self.decisions += 1
        return [float(self.read_values[index])]

    def write_values(self, z, actions):
        self.write_queries.append(tuple(actions))
        values = [[float(self.write_q.get(action, 0.0)) for action in actions]]
        logits = [[self.no_emit_logit for _ in actions]]
        return SimpleNamespace(
            value=values,
            no_emit=logits,
            error_risk=logits,
            expected_emit=values,
        )


class RecordingCost:
    """Fixed wall cost, recording which actions were priced."""

    def __init__(self, wall_s=0.3):
        self.wall_s = float(wall_s)
        self.calls = []

    def predict(self, predictor_id, translator_id, action, src_words, tgt_words):
        self.calls.append(action)
        return CostEstimate(pred_s=self.wall_s / 2.0, mt_s=self.wall_s / 2.0)


class RecordingPredictor:
    def __init__(self, continuation="part"):
        self.continuation = continuation
        self.calls = []

    def sample(self, observed, branches, horizon, seed):
        self.calls.append((observed, branches, horizon, seed))
        return [self.continuation] * int(branches)


class PoolTranslator:
    """Returns the same candidate text for every request."""

    def __init__(self, output="x y"):
        self.output = output
        self.calls = []

    def translate_batch(self, sources, prefix, final=False, **kwargs):
        self.calls.append((list(sources), prefix, bool(final)))
        return [self.output] * len(sources)


class FixedTimer:
    """Every timed stage costs the same fixed number of seconds."""

    def __init__(self, seconds=1.0):
        self.seconds = float(seconds)
        self.calls = []

    def run(self, name, function):
        value = function()
        self.calls.append(name)
        return value, self.seconds


class ReadOnlyRuntime:
    """Fails loudly if any write path is touched; serves only the EOS flush."""

    def __init__(self, final="final"):
        self.final = final
        self.execute_calls = 0
        self.extend_calls = 0
        self.translate_calls = []

    def execute(self, *args, **kwargs):
        self.execute_calls += 1
        raise AssertionError("READ must not call runtime.execute")

    def extend(self, *args, **kwargs):
        self.extend_calls += 1
        raise AssertionError("READ must not call runtime.extend")

    def translate(self, requests, prefix, final=False):
        self.translate_calls.append((list(requests), prefix, bool(final)))
        return [self.final], 0.5


class ScriptedRuntime:
    """Returns a fixed provisional / extended StepResult and records calls."""

    def __init__(self, provisional, extended=None, final="eosword"):
        self.provisional = provisional
        self.extended = extended
        self.final = final
        self.execute_calls = 0
        self.extend_calls = []
        self.last_execute = None

    def execute(self, observed, committed, action, seed):
        self.execute_calls += 1
        self.last_execute = (observed, tuple(committed), action, seed)
        return self.provisional

    def extend(
        self,
        observed,
        committed,
        current_candidates,
        extra_branches,
        horizon,
        seed,
        spent_s=0.0,
    ):
        self.extend_calls.append(
            {
                "observed": observed,
                "committed": tuple(committed),
                "current_candidates": tuple(current_candidates),
                "extra_branches": int(extra_branches),
                "horizon": int(horizon),
                "seed": seed,
                "spent_s": float(spent_s),
            }
        )
        return self.extended

    def translate(self, requests, prefix, final=False):
        return [self.final], 0.5


class FixedPostController:
    """Always proposes the same post action and records the arguments."""

    def __init__(self, action: PostAction):
        self.action = action
        self.calls = []

    def decide(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return PostDecision(self.action, "test")


def sentence(source="a b c d e", reference="x y z", sentence_id="s1"):
    return SimpleNamespace(
        sentence_id=sentence_id, source=source, reference=reference, split="test"
    )


def provisional_result(selected="prov", committed_after=("prov",), spent_s=0.3):
    return StepResult(
        candidates=("prov-a", "prov-b"),
        selected=selected,
        committed_after=committed_after,
        pred_s=0.1,
        mt_s=0.2,
        spent_s=spent_s,
        vote_top1=0.5,
        vote_margin=0.0,
        vote_entropy=0.9,
        no_emit=False,
    )


def extended_result(selected="chosen", committed_after=("chosen",)):
    return StepResult(
        candidates=("prov-a", "prov-b", "extra-a", "extra-b"),
        selected=selected,
        committed_after=committed_after,
        pred_s=0.3,
        mt_s=0.4,
        spent_s=0.8,
        vote_top1=0.75,
        vote_margin=0.5,
        vote_entropy=0.2,
        no_emit=False,
    )


# ----------------------------------------------------------------------
# Rollout: write path, commit, history, EOS flush, LAAL
# ----------------------------------------------------------------------
def test_write_then_read_commits_and_flushes_at_eos():
    cfg = ControllerConfig(
        actions=(Action(2, 4),),
        advantage_baseline=(4, 6),  # not selectable; must still be priced
        min_start=3,
        arrival_gap=0.3,
    )
    network = FakeNetwork(
        read_values=[-5.0, -5.0, 5.0],
        write_values={Action(2, 4): 0.0},
    )
    predictor = RecordingPredictor("part")
    translator = PoolTranslator("x y")
    cost = RecordingCost(0.3)
    timer = FixedTimer(1.0)

    trace = run_rollout(
        sentence(source="a b c d e f"),
        predictor,
        translator,
        network,
        FeatureBuilder(),
        cost,
        cfg,
        timer=timer,
    )

    assert isinstance(trace, RolloutTrace)
    assert trace.source == "a b c d e f"
    assert trace.reference == "x y z"
    # Two writes of "x y" at read positions 3 and 4, then the EOS flush appends
    # one more "x y" at position 6.
    assert trace.committed == ("x", "y") * 3
    assert trace.hypothesis == "x y x y x y"
    assert trace.action_histogram == {"b2h4": 2, "read": 1}
    assert trace.read_fraction == pytest.approx(1.0 / 3.0)
    assert trace.laal is not None
    assert trace.completion_s == pytest.approx(7.6)
    assert trace.delays == (3, 3, 4, 4, 6, 6)

    actions = [step["action"] for step in trace.steps]
    assert actions == ["b2h4", "b2h4", "read", "eos"]
    assert [step["read_pos"] for step in trace.steps] == [3, 4, 5, 6]

    first = trace.steps[0]
    assert first["committed_before"] == 0
    assert first["committed_after"] == 2
    assert first["new_words"] == ("x", "y")
    assert first["q_read"] == pytest.approx(-5.0)
    assert first["q_write"] == pytest.approx(0.0)
    assert first["cost"] == pytest.approx(0.3)
    assert first["no_emit"] is False
    assert first["post_action"] is None

    last = trace.steps[-1]
    assert last["action"] == "eos"
    assert last["committed_after"] == 6

    # The advantage baseline is scored and priced even though it is not in
    # cfg.actions and never appears in the histogram.
    assert Action(4, 6) in network.write_queries[0]
    assert Action(4, 6) in cost.calls
    assert "b4h6" not in trace.action_histogram
    # Two writes x two branches: READ contributed no prediction calls.
    assert len(predictor.calls) == 4
    # One translation per write plus exactly one EOS flush with the committed
    # prefix and `final=True`.
    assert len(translator.calls) == 3
    assert translator.calls[-1] == (["a b c d e f"], "x y x y", True)


# ----------------------------------------------------------------------
# Rollout: READ path
# ----------------------------------------------------------------------
def test_read_path_advances_read_pos_without_touching_the_runtime():
    cfg = ControllerConfig(
        actions=(Action(2, 4),),
        advantage_baseline=(4, 6),
        min_start=3,
        arrival_gap=0.3,
    )
    network = FakeNetwork(read_values=[5.0], write_values={Action(2, 4): 0.0})
    runtime = ReadOnlyRuntime(final="final")

    trace = run_rollout(
        sentence(source="a b c d e f", reference="final", sentence_id="read"),
        None,
        None,
        network,
        FeatureBuilder(),
        None,
        cfg,
        runtime=runtime,
    )

    assert runtime.execute_calls == 0
    assert runtime.extend_calls == 0
    assert runtime.translate_calls == [(["a b c d e f"], "", True)]
    assert trace.action_histogram == {"read": 3}
    assert trace.read_fraction == 1.0
    assert [step["action"] for step in trace.steps] == ["read", "read", "read", "eos"]
    assert [step["read_pos"] for step in trace.steps] == [3, 4, 5, 6]
    assert trace.committed == ("final",)
    assert trace.completion_s == pytest.approx(2.0)
    assert trace.laal is not None


# ----------------------------------------------------------------------
# Rollout: post-candidate controller integration
# ----------------------------------------------------------------------
def _run_with_post(action: PostAction, runtime: ScriptedRuntime):
    cfg = ControllerConfig(
        actions=(Action(2, 4),),
        advantage_baseline=(4, 6),
        max_branches=10,
        min_start=3,
        arrival_gap=0.3,
    )
    network = FakeNetwork(read_values=[-5.0, 5.0], write_values={Action(2, 4): 0.0})
    post = FixedPostController(action)
    trace = run_rollout(
        sentence(),
        None,
        None,
        network,
        FeatureBuilder(),
        None,
        cfg,
        runtime=runtime,
        post_controller=post,
    )
    assert len(post.calls) >= 1
    return trace, post


def test_post_controller_add_2_extends_pool_and_commits_final_result():
    runtime = ScriptedRuntime(provisional_result(), extended_result())
    trace, _post = _run_with_post(PostAction.ADD_2, runtime)

    assert runtime.execute_calls == 1
    assert len(runtime.extend_calls) == 1
    call = runtime.extend_calls[0]
    assert call["extra_branches"] == 2
    assert call["horizon"] == 4
    assert call["spent_s"] == pytest.approx(0.3)
    assert call["current_candidates"] == ("prov-a", "prov-b")
    assert call["committed"] == ()
    assert call["seed"] == runtime.last_execute[3]

    write_step = trace.steps[0]
    assert write_step["post_action"] == "add_2"
    assert write_step["new_words"] == ("chosen",)
    assert write_step["committed_after"] == 1
    assert write_step["spent_s"] == pytest.approx(0.8)
    assert write_step["vote_entropy"] == pytest.approx(0.2)
    assert write_step["vote_margin"] == pytest.approx(0.5)
    assert trace.committed == ("chosen", "eosword")


def test_post_controller_stop_commits_the_initial_result():
    runtime = ScriptedRuntime(provisional_result())
    trace, _post = _run_with_post(PostAction.STOP, runtime)

    assert runtime.extend_calls == []
    write_step = trace.steps[0]
    assert write_step["post_action"] == "stop"
    assert write_step["new_words"] == ("prov",)
    assert trace.committed == ("prov", "eosword")


def test_post_controller_wait_discards_the_provisional_result():
    runtime = ScriptedRuntime(provisional_result())
    trace, _post = _run_with_post(PostAction.WAIT, runtime)

    assert runtime.extend_calls == []
    write_step = trace.steps[0]
    assert write_step["post_action"] == "wait"
    assert write_step["new_words"] == ()
    assert write_step["committed_before"] == 0
    assert write_step["committed_after"] == 0
    assert write_step["no_emit"] is True
    assert write_step["spent_s"] == pytest.approx(0.3)
    # Only the EOS flush contributes words when the write waited.
    assert trace.committed == ("eosword",)


# ----------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------
def make_trace(
    sentence_id,
    laal,
    completion_s=1.0,
    histogram=None,
    steps=(),
    hypothesis="x",
    reference="x",
    source="a",
):
    return SimpleNamespace(
        sentence_id=sentence_id,
        source=source,
        reference=reference,
        hypothesis=hypothesis,
        laal=laal,
        completion_s=completion_s,
        read_fraction=0.0,
        action_histogram=dict(histogram or {}),
        steps=tuple(steps),
    )


class UnavailableRunPfba:
    @staticmethod
    def corpus_bleu(hypotheses, references):
        return {"status": "unavailable", "reason": "no sacrebleu"}

    @staticmethod
    def comet_score(hypotheses, sources, references, model_name, device):
        return {"status": "error", "reason": "no comet"}


class OkRunPfba(UnavailableRunPfba):
    @staticmethod
    def corpus_bleu(hypotheses, references):
        return {"status": "ok", "score": 12.5}


def test_corpus_bleu_returns_the_delegated_unavailable_status(monkeypatch):
    monkeypatch.setattr(evaluator, "run_pfba", UnavailableRunPfba)

    result = evaluator.corpus_bleu(["a"], ["a"])

    assert result["status"] == "unavailable"
    assert "sacrebleu" in result["reason"]


def test_corpus_bleu_and_comet_return_unavailable_without_run_pfba(monkeypatch):
    monkeypatch.setattr(evaluator, "run_pfba", None)

    assert evaluator.corpus_bleu(["a"], ["a"])["status"] == "unavailable"
    assert evaluator.comet_score(["a"], ["a"], ["a"])["status"] == "unavailable"


def test_summarize_traces_aggregates_status_aware_metrics(monkeypatch):
    monkeypatch.setattr(evaluator, "run_pfba", OkRunPfba)
    traces = [
        make_trace("a", 1.0, 1.0, {"read": 1, "b2h4": 1}, ({"action": "read"}, {"action": "b2h4"})),
        make_trace("b", None, 2.0, {"read": 1}, ({"action": "read"},)),
        make_trace("c", 3.0, 3.0, {"b2h4": 1}),
    ]

    summary = summarize_traces(traces)

    assert summary["n"] == 3
    assert summary["bleu"] == {"status": "ok", "score": 12.5}
    assert summary["comet"]["status"] == "skipped"
    assert summary["laal"] == pytest.approx(2.0)  # None is skipped
    assert summary["completion_s"]["mean"] == pytest.approx(2.0)
    assert summary["completion_s"]["p95"] == pytest.approx(2.9)
    assert summary["action_histogram"] == {"read": 2, "b2h4": 2}
    assert summary["read_fraction"] == pytest.approx(0.5)
    assert len(summary["steps"]) == 3
    assert summary["steps"][0]["sentence_id"] == "a"


def test_summarize_traces_runs_comet_only_when_requested(monkeypatch):
    monkeypatch.setattr(evaluator, "run_pfba", UnavailableRunPfba)
    summary = summarize_traces(
        [make_trace("a", 1.0)], include_comet=True
    )

    assert summary["comet"]["status"] == "error"


def test_paired_bootstrap_identical_traces_contains_zero():
    traces_a = [make_trace("s1", 1.0), make_trace("s2", 3.0)]
    traces_b = [make_trace("s1", 1.0), make_trace("s2", 3.0)]

    result = paired_bootstrap(traces_a, traces_b, "laal", samples=500, seed=7)

    assert result["n"] == 2
    assert result["point"] == pytest.approx(0.0)
    assert result["low"] <= 0.0 <= result["high"]


def test_paired_bootstrap_skips_none_laal_values():
    traces_a = [make_trace("s1", 1.0), make_trace("s2", None)]
    traces_b = [make_trace("s1", 0.5), make_trace("s2", 2.0)]

    result = paired_bootstrap(traces_a, traces_b, "laal", samples=100)

    assert result["n"] == 1
    assert result["point"] == pytest.approx(0.5)
    assert result["low"] == pytest.approx(0.5)
    assert result["high"] == pytest.approx(0.5)


def test_paired_bootstrap_bleu_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(
        evaluator,
        "corpus_bleu",
        lambda hypotheses, references: {"status": "unavailable", "reason": "no sacrebleu"},
    )
    traces_a = [make_trace("s1", 1.0)]
    traces_b = [make_trace("s1", 1.0)]

    with pytest.raises(MetricUnavailableError):
        paired_bootstrap(traces_a, traces_b, "bleu")


def test_paired_bootstrap_rejects_unknown_metric():
    with pytest.raises(ValueError):
        paired_bootstrap([make_trace("s1", 1.0)], [make_trace("s1", 1.0)], "comet")
