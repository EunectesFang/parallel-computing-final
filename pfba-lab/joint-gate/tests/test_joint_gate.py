"""CPU-only tests for joint_gate.  No models, GPU, network, or training run needed."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import joint_gate as jg


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------

class StaticTimer:
    """Deterministic timer: every stage costs exactly ``seconds``."""

    def __init__(self, seconds=0.25):
        self.seconds = seconds
        self.calls = []

    def run(self, name, fn):
        self.calls.append(name)
        return fn(), self.seconds


class ScriptedPredictor:
    def __init__(self, continuations=None):
        self.calls = []
        self.continuations = continuations

    def sample(self, observed, branches, horizon, seed):
        self.calls.append((observed, branches, horizon, seed))
        if self.continuations is not None:
            return list(self.continuations)
        return [f"future{i}" for i in range(branches)]


class ScriptedTranslator:
    def __init__(self, output_fn=None):
        self.calls = []
        self.output_fn = output_fn or (lambda source, prefix: "alpha beta")

    def translate_batch(self, sources, prefix):
        self.calls.append((list(sources), prefix))
        return [self.output_fn(source, prefix) for source in sources]


class BoomPredictor:
    def sample(self, *args, **kwargs):
        raise AssertionError("READ must not call the predictor")


class BoomTranslator:
    def translate_batch(self, *args, **kwargs):
        raise AssertionError("READ must not call the translator")


class ScriptedPolicy:
    def __init__(self, actions, name="scripted"):
        self.actions = list(actions)
        self.name = name
        self.seen_features = []

    def decide(self, features):
        self.seen_features.append(dict(features))
        if not self.actions:
            return jg.READ
        return self.actions.pop(0)


def make_features(**overrides):
    base = {
        "observed_len": 2.0,
        "observed_chars": 10.0,
        "observed_punct": 0.0,
        "observed_punct_ratio": 0.0,
        "observed_ends_punct": 0.0,
        "committed_words": 1.0,
        "committed_chars": 4.0,
    }
    base.update(overrides)
    return base


def make_row(sentence_id="s1", split="train", read_pos=2, action="b2h2",
             matched=2, incorrect=0, success=None, utility=None,
             predict_s=0.1, translate_s=0.2, features=None, observed="a b",
             committed=("x",), new_words=("x", "y"), feature_s=0.01):
    if success is None:
        success = int(matched > 0)
    if utility is None:
        utility = float(matched - incorrect)
    return jg.TrainingRow(
        sentence_id=sentence_id,
        split=split,
        read_pos=read_pos,
        observed=observed,
        committed=tuple(committed),
        features=dict(make_features() if features is None else features),
        feature_s=feature_s,
        action=action,
        new_words=tuple(new_words),
        matched=int(matched),
        incorrect=int(incorrect),
        success=int(success),
        utility=float(utility),
        predict_s=float(predict_s),
        translate_s=float(translate_s),
        total_s=float(predict_s + translate_s),
    )


def synthetic_rows(prefix, n_states=12, actions=None, split="train", base_sentence=0):
    actions = list(jg.ACTION_GRID if actions is None else actions)
    rows = []
    for i in range(n_states):
        sid = f"{prefix}-s{base_sentence + i:03d}"
        for j, action in enumerate(actions):
            # Balanced-ish classes and varied features/costs.
            ok = ((i + j) % 3) != 0
            matched = 2 if ok else 0
            incorrect = 0 if ok else (1 + (i % 2))
            features = make_features(
                observed_len=float(1 + (i % 7)),
                observed_chars=float(3 + (i % 7)),
                observed_punct=float(i % 2),
                observed_punct_ratio=0.1 * (i % 3),
                observed_ends_punct=float(i % 2),
                committed_words=float(i % 4),
                committed_chars=float(2 * (i % 4)),
            )
            rows.append(make_row(
                sentence_id=sid,
                split=split,
                read_pos=1 + (i % 7),
                action=action.label,
                matched=matched,
                incorrect=incorrect,
                features=features,
                predict_s=0.05 + 0.01 * j,
                translate_s=0.10 + 0.02 * i,
                observed="a b c",
                committed=("x",),
                new_words=("x", "y") if ok else ("x", "z"),
            ))
    return rows


# ---------------------------------------------------------------------------
# Action grid and helpers
# ---------------------------------------------------------------------------

class ActionGridTests(unittest.TestCase):
    def test_action_grid_and_baselines(self):
        self.assertEqual(
            jg.ACTION_GRID,
            (jg.Action(2, 2), jg.Action(2, 6), jg.Action(4, 2), jg.Action(4, 6)),
        )
        self.assertEqual(jg.B_ONLY_ACTIONS, (jg.Action(2, 6), jg.Action(4, 6)))
        self.assertEqual(jg.H_ONLY_ACTIONS, (jg.Action(4, 2), jg.Action(4, 6)))
        self.assertEqual(jg.FIXED_BASELINE, jg.Action(4, 6))
        self.assertTrue(jg.READ.is_read)
        self.assertFalse(jg.Action(2, 2).is_read)
        self.assertEqual(jg.Action(2, 2).label, "b2h2")
        self.assertEqual(jg.Action.from_label("b2h2"), jg.Action(2, 2))
        self.assertEqual(jg.READ.label, "read")

    def test_upstream_ralcp_prefix_changes_between_depths(self):
        # Depth 1 support 2/3 >= 0.6, depth 2 support 1/3 < 0.6.
        self.assertEqual(jg.upstream_ralcp(["a b", "a c", "a d"], 0.6), "a")
        # Duplicates count and ties keep the first-appearing prefix.
        self.assertEqual(jg.upstream_ralcp(["a b", "a b", "a c"], 0.6), "a b")
        self.assertEqual(jg.upstream_ralcp([], 0.6), "")

    def test_prefix_matching_counts_correct_and_incorrect(self):
        matched, incorrect, success, utility = jg.score_commit(
            ("a", "b"), ("a", "b", "c"), incorrect_penalty=1.0)
        self.assertEqual((matched, incorrect, success, utility), (2, 0, 1, 2.0))

        matched, incorrect, success, utility = jg.score_commit(
            ("a", "x", "y"), ("a", "b"), incorrect_penalty=1.0)
        self.assertEqual((matched, incorrect, success, utility), (1, 2, 1, -1.0))

        self.assertEqual(jg.score_commit((), ("a",), 1.0), (0, 0, 0, 0.0))

    def test_state_seed_is_deterministic(self):
        a = jg.derive_state_seed(13, "dev", "dev:0001", 4)
        b = jg.derive_state_seed(13, "dev", "dev:0001", 4)
        c = jg.derive_state_seed(13, "dev", "dev:0001", 5)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


# ---------------------------------------------------------------------------
# Offline features: no source length / future, and timing
# ---------------------------------------------------------------------------

class FeatureTests(unittest.TestCase):
    def test_only_observed_and_committed_affect_features(self):
        extractor = jg.FeatureExtractor()
        f1, _ = extractor.extract("der Hund", ("the",))
        f2, _ = extractor.extract("der Hund", ("the",))
        self.assertEqual(f1, f2)
        self.assertEqual(f1["observed_len"], 2.0)
        self.assertEqual(f1["committed_words"], 1.0)
        self.assertNotIn("source_len", f1)
        self.assertNotIn("future", f1)

    def test_extract_signature_has_no_source_or_future(self):
        params = list(inspect.signature(jg.FeatureExtractor.extract).parameters)
        self.assertEqual(params, ["self", "observed", "committed", "timer"])

    def test_surface_punctuation(self):
        extractor = jg.FeatureExtractor()
        features, _ = extractor.extract("Hallo , Welt .", ())
        self.assertEqual(features["observed_punct"], 2.0)
        self.assertEqual(features["observed_ends_punct"], 1.0)

        features, _ = extractor.extract("Hallo Welt", ())
        self.assertEqual(features["observed_punct"], 0.0)
        self.assertEqual(features["observed_ends_punct"], 0.0)

    def test_optional_entropy_feature_is_timed(self):
        calls = []

        def entropy_fn(observed):
            calls.append(observed)
            return 1.5, 0.25

        extractor = jg.FeatureExtractor(entropy_fn=entropy_fn)
        timer = StaticTimer(0.125)
        features, seconds = extractor.extract("a b", (), timer=timer)
        self.assertEqual(calls, ["a b"])
        self.assertEqual(features["entropy"], 1.5)
        self.assertEqual(features["max_prob"], 0.25)
        self.assertEqual(seconds, 0.125)
        self.assertIn("features", timer.calls)


# ---------------------------------------------------------------------------
# Action execution and offline labelling
# ---------------------------------------------------------------------------

class ActionExecutionTests(unittest.TestCase):
    def test_compute_action_uses_observed_and_forced_prefix(self):
        predictor = ScriptedPredictor()
        translator = ScriptedTranslator()
        timer = StaticTimer(0.25)
        outcome = jg.execute_action(
            predictor, translator, observed="der Hund", committed=("the",),
            action=jg.Action(2, 4), seed=7, ralcp_threshold=0.6, timer=timer)
        self.assertEqual(outcome.new_words, ("alpha", "beta"))
        self.assertEqual(predictor.calls[0][0], "der Hund")
        self.assertEqual(predictor.calls[0][1:3], (2, 4))
        self.assertEqual(translator.calls[0][1], "the")
        self.assertEqual(outcome.predict_s, 0.25)
        self.assertEqual(outcome.translate_s, 0.25)
        self.assertEqual(outcome.total_s, 0.75)  # predict + translate + aggregate

    def test_read_action_does_no_model_work(self):
        timer = StaticTimer(0.25)
        outcome = jg.execute_action(
            BoomPredictor(), BoomTranslator(), observed="a", committed=(),
            action=jg.READ, seed=1, ralcp_threshold=0.6, timer=timer)
        self.assertEqual(outcome.new_words, ())
        self.assertEqual(outcome.total_s, 0.0)
        self.assertEqual(timer.calls, [])

    def test_label_state_uses_one_oracle_and_all_actions(self):
        source = "a b c"
        state = jg.State(
            sentence_id="dev:0001", split="train", read_pos=2,
            observed="a b", committed=("x",), features=make_features(),
            feature_s=0.01)
        oracle_calls = []

        def output_fn(src, prefix):
            if src == source:
                oracle_calls.append((src, prefix))
                return "x y z"
            return "x y"

        translator = ScriptedTranslator(output_fn=output_fn)
        predictor = ScriptedPredictor()
        timer = StaticTimer(0.25)
        rows = jg.label_state(
            state, source, predictor, translator,
            actions=(jg.Action(2, 2), jg.Action(4, 6)),
            seed=13, ralcp_threshold=0.6, timer=timer, incorrect_penalty=1.0)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(oracle_calls), 1)
        for row in rows:
            self.assertEqual(row.observed, "a b")
            self.assertEqual(row.committed, ("x",))
            self.assertEqual(row.features, state.features)
            self.assertEqual((row.matched, row.incorrect, row.success), (2, 0, 1))
            self.assertEqual(row.utility, 2.0)
            self.assertEqual(row.feature_s, 0.01)
        self.assertEqual([r.action for r in rows], ["b2h2", "b4h6"])

    def test_label_state_includes_incorrect_words_separately(self):
        source = "a b c"
        state = jg.State(
            sentence_id="dev:0002", split="train", read_pos=2,
            observed="a b", committed=(), features=make_features(),
            feature_s=0.0)

        def output_fn(src, prefix):
            return "x y z" if src == source else "x q"

        rows = jg.label_state(
            state, source, ScriptedPredictor(),
            ScriptedTranslator(output_fn=output_fn),
            actions=(jg.Action(2, 2),), seed=1, ralcp_threshold=0.6,
            timer=StaticTimer(0.0))
        self.assertEqual(rows[0].matched, 1)
        self.assertEqual(rows[0].incorrect, 1)
        self.assertEqual(rows[0].success, 1)
        self.assertEqual(rows[0].utility, 0.0)


# ---------------------------------------------------------------------------
# State collection and splits
# ---------------------------------------------------------------------------

class SplitAndCollectionTests(unittest.TestCase):
    def test_dev_split_is_sentence_disjoint(self):
        sentences = [jg.Sentence(f"dev:{i:04d}", "a b", "x y") for i in range(10)]
        mapping = jg.split_dev_sentences(sentences, tune_fraction=0.3, seed=13)
        self.assertEqual(set(mapping), {s.sentence_id for s in sentences})
        self.assertTrue(set(mapping.values()) <= {"train", "tune"})
        self.assertEqual(sum(v == "tune" for v in mapping.values()), 3)
        self.assertIn("train", mapping.values())

    def test_collect_states_dedupes_and_limits(self):
        sentences = [jg.Sentence("dev:0001", "a b c d e", "w x y z")]
        behaviors = [
            jg.FixedBehavior(jg.Action(2, 2)),
            jg.PeriodicBehavior(jg.Action(4, 6), every=2),
        ]
        translator = ScriptedTranslator()
        states = jg.collect_states(
            sentences, "train", ScriptedPredictor(), translator,
            behaviors, jg.FeatureExtractor(),
            jg.GateConfig(min_start=1), timer=StaticTimer(0.0),
            max_states_per_sentence=4)
        self.assertLessEqual(len(states), 4)
        keys = [(s.sentence_id, s.read_pos, s.committed) for s in states]
        self.assertEqual(len(keys), len(set(keys)))
        for state in states:
            self.assertEqual(state.split, "train")
            self.assertEqual(set(state.features), set(jg.FeatureExtractor().feature_names()))
            self.assertNotIn("source_len", state.features)

    def test_validate_sentence_disjoint_rejects_overlap(self):
        train = [make_row(sentence_id="a", split="train")]
        tune = [make_row(sentence_id="a", split="tune")]
        test = [make_row(sentence_id="b", split="test")]
        with self.assertRaises(ValueError):
            jg.validate_sentence_disjoint(train, tune, test)
        jg.validate_sentence_disjoint(
            [make_row(sentence_id="a", split="train")],
            [make_row(sentence_id="b", split="tune")],
            [make_row(sentence_id="c", split="test")])


# ---------------------------------------------------------------------------
# Models, threshold tuning, checkpoints
# ---------------------------------------------------------------------------

class ModelTests(unittest.TestCase):
    def _fit(self):
        train = synthetic_rows("tr", n_states=15, split="train")
        tune = synthetic_rows("tu", n_states=15, split="tune", base_sentence=100)
        model = jg.fit_gate_model(
            train, tune, jg.ACTION_GRID, seed=13, cost_weight=0.1,
            incorrect_penalty=1.0)
        return model, train, tune

    def test_fit_produces_calibrated_logistic_and_cost_utility_estimators(self):
        from sklearn.calibration import CalibratedClassifierCV

        model, _, _ = self._fit()
        self.assertEqual(set(model.estimators), set(jg.ACTION_GRID))
        for action in jg.ACTION_GRID:
            estimator = model.estimators[action]
            self.assertIsInstance(estimator.success, CalibratedClassifierCV)
            self.assertTrue(hasattr(estimator.utility, "predict"))
            self.assertTrue(hasattr(estimator.cost, "predict"))
            p = estimator.success.predict_proba(np.zeros((1, len(model.feature_names))))[0, 1]
            self.assertTrue(0.0 <= p <= 1.0)

    def test_threshold_chosen_on_tune_only(self):
        model, _, _ = self._fit()
        self.assertIsInstance(model.threshold, float)
        test_rows = synthetic_rows("te", n_states=10, split="test", base_sentence=200)
        with self.assertRaises(ValueError):
            jg.choose_threshold(model, test_rows)
        with self.assertRaises(ValueError):
            jg.evaluate_offline(model, synthetic_rows(
                "tu2", n_states=3, split="tune", base_sentence=300))

    def test_decision_is_read_or_a_trained_action(self):
        model, _, _ = self._fit()
        decision = model.decide(make_features())
        self.assertTrue(decision is jg.READ or decision in jg.ACTION_GRID)
        scores = model.score_actions(make_features())
        self.assertEqual(set(scores), set(jg.ACTION_GRID))

    def test_baseline_action_subsets(self):
        train = synthetic_rows("tr", n_states=15, split="train")
        tune = synthetic_rows("tu", n_states=15, split="tune", base_sentence=100)
        b_model = jg.fit_gate_model(train, tune, jg.B_ONLY_ACTIONS, seed=2)
        h_model = jg.fit_gate_model(train, tune, jg.H_ONLY_ACTIONS, seed=2)
        self.assertEqual(set(b_model.estimators), set(jg.B_ONLY_ACTIONS))
        self.assertEqual(set(h_model.estimators), set(jg.H_ONLY_ACTIONS))

    def test_checkpoint_roundtrip(self):
        model, _, _ = self._fit()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "joint.joblib"
            jg.save_checkpoint(model, path)
            self.assertTrue(path.is_file())
            loaded = jg.load_checkpoint(path)
        features = make_features()
        self.assertEqual(model.decide(features), loaded.decide(features))
        self.assertEqual(model.threshold, loaded.threshold)

    def test_fixed_and_learned_policies(self):
        fixed = jg.FixedActionPolicy(jg.FIXED_BASELINE)
        self.assertEqual(fixed.decide(make_features()), jg.FIXED_BASELINE)
        model, _, _ = self._fit()
        learned = jg.LearnedGatePolicy(model)
        self.assertIn(learned.decide(make_features()), (jg.READ, *jg.ACTION_GRID))

    def test_neural_confidence_is_fitted_and_calibrated(self):
        train = synthetic_rows("tr", n_states=18, split="train")
        tune = synthetic_rows("tu", n_states=15, split="tune")
        model = jg.fit_gate_model(train, tune, neural=True)
        confidence = model.estimators[jg.FIXED_BASELINE].success
        self.assertEqual(type(confidence.shared.estimator[-1]).__name__, "MLPClassifier")
        self.assertTrue(all(estimator.success.shared is confidence.shared
                            for estimator in model.estimators.values()))
        self.assertIn(model.decide(make_features()), (jg.READ, *jg.ACTION_GRID))


# ---------------------------------------------------------------------------
# Online rollouts and metrics
# ---------------------------------------------------------------------------

class RolloutTests(unittest.TestCase):
    def _translator(self):
        return ScriptedTranslator(
            output_fn=lambda source, prefix: "z" if source == "a b c d" else "x y")

    def test_online_rollout_reads_and_computes_with_forced_prefix(self):
        sentence = jg.Sentence("test:0001", "a b c d", "x y z")
        policy = ScriptedPolicy([jg.Action(2, 2), jg.READ, jg.READ])
        translator = self._translator()
        timer = StaticTimer(0.25)
        trace = jg.run_online_rollout(
            sentence, ScriptedPredictor(), translator, policy,
            jg.FeatureExtractor(), jg.GateConfig(min_start=1), split="test",
            timer=timer)
        self.assertEqual(trace.committed, ("x", "y", "z"))
        self.assertEqual(trace.hypothesis, "x y z")
        self.assertEqual(trace.action_histogram.get("read"), 2)
        self.assertEqual(trace.action_histogram.get("b2h2"), 1)
        self.assertEqual(trace.action_histogram.get("eos"), 1)
        self.assertIsNotNone(trace.laal)
        self.assertGreaterEqual(trace.completion_s, 0.0)
        for sources, prefix in translator.calls:
            self.assertIn(prefix, ("", "x y"))
        for features in policy.seen_features:
            self.assertNotIn("source_len", features)
            self.assertNotIn("future", features)

    def test_online_rollout_is_deterministic(self):
        sentence = jg.Sentence("test:0002", "a b c d", "x y z")
        cfg = jg.GateConfig(min_start=1)
        a = jg.run_online_rollout(
            sentence, ScriptedPredictor(), self._translator(),
            ScriptedPolicy([jg.Action(2, 2), jg.READ, jg.READ]),
            jg.FeatureExtractor(), cfg, split="test", timer=StaticTimer(0.25))
        b = jg.run_online_rollout(
            sentence, ScriptedPredictor(), self._translator(),
            ScriptedPolicy([jg.Action(2, 2), jg.READ, jg.READ]),
            jg.FeatureExtractor(), cfg, split="test", timer=StaticTimer(0.25))
        self.assertEqual(a.committed, b.committed)
        self.assertEqual(a.delays, b.delays)
        self.assertEqual(a.action_histogram, b.action_histogram)
        self.assertEqual(a.completion_s, b.completion_s)

    def test_rollout_metrics_include_bleu_comet_laal_completion_and_histogram(self):
        sentence = jg.Sentence("test:0001", "a b c d", "x y z")
        trace = jg.run_online_rollout(
            sentence, ScriptedPredictor(), self._translator(),
            ScriptedPolicy([jg.Action(2, 2), jg.READ, jg.READ]),
            jg.FeatureExtractor(), jg.GateConfig(min_start=1), split="test",
            timer=StaticTimer(0.25))
        result = jg.rollout_metrics(
            [trace], metrics=("bleu", "comet"),
            bleu_fn=lambda hypotheses, references: {"status": "ok", "score": 12.5},
            comet_fn=lambda hypotheses, sources, references, device="cpu": {
                "status": "ok", "score": 0.75})
        self.assertEqual(result["bleu"]["score"], 12.5)
        self.assertEqual(result["comet"]["score"], 0.75)
        self.assertIsNotNone(result["laal"])
        self.assertIn("completion_s", result)
        self.assertIn("action_histogram", result)
        self.assertEqual(result["n"], 1)

    def test_requested_but_unavailable_metric_fails(self):
        with self.assertRaises(jg.MetricUnavailableError):
            jg.ensure_metrics_available(
                ("bleu", "comet"), availability={"bleu": True, "comet": False})
        jg.ensure_metrics_available(
            ("bleu",), availability={"bleu": True, "comet": False})

        sentence = jg.Sentence("test:0001", "a b c d", "x y z")
        trace = jg.run_online_rollout(
            sentence, ScriptedPredictor(), self._translator(),
            ScriptedPolicy([jg.Action(2, 2), jg.READ, jg.READ]),
            jg.FeatureExtractor(), jg.GateConfig(min_start=1), split="test",
            timer=StaticTimer(0.25))
        with self.assertRaises(jg.MetricUnavailableError):
            jg.rollout_metrics(
                [trace], metrics=("comet",),
                comet_fn=lambda *a, **k: {"status": "error", "reason": "missing"})


# ---------------------------------------------------------------------------
# End-to-end feasibility with fake adapters (no training on real data)
# ---------------------------------------------------------------------------

class EndToEndFeasibilityTests(unittest.TestCase):
    def test_small_pipeline_runs_and_saves_artifacts(self):
        dev = [
            jg.Sentence(f"dev:{i:04d}",
                        "a b c d e" if i % 2 == 0 else "a b c",
                        "x y z")
            for i in range(6)
        ]
        test = [
            jg.Sentence(f"test:{i:04d}", "a b c d e", "x y z")
            for i in range(3)
        ]
        translator = ScriptedTranslator(
            output_fn=lambda source, prefix: "z" if source == "a b c d e" else "x y")
        cfg = jg.GateConfig(min_start=1, seed=13)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            results = jg.run_feasibility(
                dev, test, ScriptedPredictor(), translator, out, cfg,
                behaviors=(
                    jg.FixedBehavior(jg.Action(2, 2)),
                    jg.PeriodicBehavior(jg.Action(4, 6), every=2),
                ),
                metrics=("bleu",),
                bleu_fn=lambda hypotheses, references: {
                    "status": "ok", "score": 10.0},
                tune_fraction=0.34,
                max_states_per_sentence=3)
            self.assertTrue((out / "results.json").is_file())
            self.assertTrue((out / "checkpoints" / "joint.joblib").is_file())
            self.assertTrue((out / "checkpoints" / "b_only.joblib").is_file())
            self.assertTrue((out / "checkpoints" / "h_only.joblib").is_file())
            saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["config"]["seed"], 13)
            policies = saved["rollouts"]
            self.assertEqual(set(policies), {"fixed_4_6", "fixed_2_6", "b_only", "h_only", "joint"})
            for cell in policies.values():
                self.assertIn("bleu", cell)
                self.assertIn("laal", cell)
                self.assertIn("completion_s", cell)
                self.assertIn("action_histogram", cell)
            self.assertIn("offline", saved)
            self.assertIn("thresholds", saved)


if __name__ == "__main__":
    unittest.main()
