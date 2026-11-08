"""CPU-only tests for source-streaming TAF/PFBA.

Run:
    D:/MINICONDA/envs/py312/python.exe -m unittest discover -s tests -v

No torch/transformers/sacrebleu import is required at module import time.
"""

from __future__ import annotations

import unittest

import run_pfba as pfba


class RALCPMathTests(unittest.TestCase):
    """Mirror upstream agents/utils.py::ralcp exactly."""

    def test_common_prefix_over_original_denominator(self):
        self.assertEqual(
            pfba.upstream_ralcp(["a b c", "a b d", "a b c"], 0.6), "a b c"
        )

    def test_winner_can_extend_between_depths_first_tie_wins(self):
        # Depth 1 winner is "a"; at depth 2 the support is tied and the
        # first-appearing prefix wins while the threshold still passes.
        self.assertEqual(pfba.upstream_ralcp(["a b", "a c"], 0.5), "a b")

    def test_candidates_that_ended_stay_in_denominator(self):
        # "a b c" is supported by 1/3 candidates, below 0.6, so the prefix
        # stops at depth 1 even though one candidate is longer.
        self.assertEqual(pfba.upstream_ralcp(["a b c", "a", "a"], 0.6), "a")

    def test_duplicate_votes_are_not_collapsed(self):
        # If duplicates were deduplicated, "a b" would have 1/2 support and
        # fail; upstream keeps both votes, giving 2/3.
        self.assertEqual(
            pfba.upstream_ralcp(["a b", "a b", "a c"], 2 / 3), "a b"
        )

    def test_exact_threshold_passes(self):
        self.assertEqual(pfba.upstream_ralcp(["a", "b"], 0.5), "a")

    def test_empty_candidates(self):
        self.assertEqual(pfba.upstream_ralcp([], 0.5), "")


class RecordingPredictor:
    def __init__(self, continuation: str = "", repeat: int = 1):
        self.continuation = continuation
        self.repeat = repeat
        self.prompts: list[tuple[str, int]] = []

    def sample(self, prefix, branches, horizon, seed):
        self.prompts.append((prefix, seed))
        if not self.continuation:
            return []
        return [self.continuation] * min(branches, self.repeat)


class RecordingTranslator:
    """Returns only the NEW target text after the forced prefix."""

    def __init__(self, table=None, default: str = ""):
        self.table = table or {}
        self.default = default
        self.calls: list[tuple[list[str], str]] = []
        self.rows = 0

    def translate_batch(self, sources, target_prefix):
        sources = list(sources)
        self.calls.append((sources, target_prefix))
        self.rows += len(sources)
        return [self.table.get((s, target_prefix), self.default) for s in sources]


def config(**overrides):
    values = dict(branches=3, horizon=2, arrival_gap=0.0, split="test")
    values.update(overrides)
    return pfba.StreamConfig(**values)


class NoFutureLeakageTests(unittest.TestCase):
    def test_latest_prefix_coalesces_only_already_arrived_words(self):
        class ReplayTimer(pfba.NullTimer):
            def run(self, name, fn):
                return fn(), {"predict": 0.11, "translate": 0.10}.get(name, 0.0)

        source = "eins zwei drei vier fünf sechs sieben"
        predictor = RecordingPredictor()
        trace = pfba.run_streaming(
            "s1", "test", source, "", predictor, RecordingTranslator(),
            "pfba-latest", config(arrival_gap=0.1), ReplayTimer(),
        )
        self.assertEqual([step.read_pos for step in trace.steps], [3, 5, 7])
        self.assertEqual([text for text, _ in predictor.prompts],
                         ["eins zwei drei", "eins zwei drei vier fünf"])

    def test_policy_never_sees_unobserved_source(self):
        source = "eins zwei drei vier fünf"
        predictor = RecordingPredictor()  # no continuations at all
        translator = RecordingTranslator()
        trace = pfba.run_streaming(
            "s1", "test", source, "one two three four five",
            predictor, translator, "taf", config(), pfba.NullTimer(),
        )
        self.assertEqual(
            [call[0][0] for call in translator.calls[:-1]],
            ["eins zwei drei", "eins zwei drei vier"],
        )
        for sources, _ in translator.calls[:-1]:
            for text in sources:
                self.assertTrue(source.startswith(text))
                self.assertNotIn("fünf", text)
        self.assertEqual(translator.calls[-1][0], [source])
        self.assertFalse(any("fünf" in prompt for prompt, _ in predictor.prompts))
        self.assertEqual(trace.hypothesis, "")

    def test_committed_prefix_is_forced_into_every_translation(self):
        table = {
            ("a b c", ""): "x y",
            ("a b c d", "x y"): "z",
            ("a b c d e", "x y z"): "",
        }
        translator = RecordingTranslator(table=table)
        trace = pfba.run_streaming(
            "s2", "test", "a b c d e", "x y z", RecordingPredictor(),
            translator, "taf", config(), pfba.NullTimer(),
        )
        prefixes = [prefix for _, prefix in translator.calls]
        self.assertEqual(prefixes, ["", "x y", "x y z"])
        self.assertEqual(trace.committed, ("x", "y", "z"))
        self.assertEqual(trace.hypothesis, "x y z")
        # Every forced prefix is a prefix of the final committed hypothesis.
        for prefix in prefixes:
            self.assertTrue(trace.hypothesis.startswith(prefix))


class PFBAReuseTests(unittest.TestCase):
    def test_duplicate_requests_are_reused_but_votes_preserved(self):
        source = "a b c d"
        pred = RecordingPredictor(continuation="d", repeat=3)
        taf_tr = RecordingTranslator()
        taf = pfba.run_streaming(
            "s3", "test", source, "e", pred, taf_tr, "taf",
            config(), pfba.NullTimer(),
        )
        pfba_tr = RecordingTranslator()
        fast = pfba.run_streaming(
            "s3", "test", source, "e", pred, pfba_tr, "pfba",
            config(), pfba.NullTimer(),
        )
        # The released TAF policy votes over the three sampled futures only.
        self.assertEqual(taf_tr.calls[0][0], ["a b c d"] * 3)
        # PFBA sends each exact request once.
        self.assertEqual(pfba_tr.calls[0][0], ["a b c d"])
        self.assertGreater(pfba_tr.calls[0][0].count("a b c d"), 0)
        self.assertLess(pfba_tr.rows, taf_tr.rows)
        # The final full-source request was already translated earlier.
        self.assertEqual(fast.full_reuse_hits, 1)
        self.assertGreaterEqual(fast.reuse_hits, 3)
        self.assertEqual(taf.reuse_hits, 0)
        # Duplicate votes are still present for RALCP.
        self.assertEqual(taf.steps[0].n_candidates, 3)
        self.assertEqual(fast.steps[0].n_candidates, 3)
        self.assertEqual(fast.hypothesis, taf.hypothesis)


class LAALTests(unittest.TestCase):
    def test_hand_computed_length_adaptive_no_clamp(self):
        # S=5, ref=pred=4, delays 1..4 -> slope 5/4.
        value = pfba.laal_score([1, 2, 3, 4], 5, 4, 4)
        self.assertAlmostEqual(value, 0.625)

    def test_cutoff_at_first_full_source_emission(self):
        # Only delays up to and including the first d >= S count.
        value = pfba.laal_score([1, 2, 3, 3], 3, 4, 4)
        self.assertAlmostEqual(value, 1.25)

    def test_terms_are_not_clamped(self):
        value = pfba.laal_score([0, 0], 5, 2, 2)
        self.assertAlmostEqual(value, -1.25)

    def test_empty_prediction_is_undefined(self):
        self.assertIsNone(pfba.laal_score([], 3, 3, 0))


class SeedTests(unittest.TestCase):
    def test_prediction_microbatches_keep_all_branches_and_stable_seeds(self):
        class Predictor(pfba.QwenPredictor):
            def __init__(self):
                self.prediction_batch_size = 2
                self.calls = []

            def sample(self, prefix, branches, horizon, seed):
                if branches <= self.prediction_batch_size:
                    self.calls.append((prefix, branches, horizon, seed))
                    return [str(seed)] * branches
                return super().sample(prefix, branches, horizon, seed)

        predictor = Predictor()
        self.assertEqual(predictor.sample("true prefix", 5, 6, 13), ["13", "13", "15", "15", "17"])
        self.assertEqual(predictor.calls, [("true prefix", 2, 6, 13),
                                          ("true prefix", 2, 6, 15),
                                          ("true prefix", 1, 6, 17)])
    def test_seed_depends_on_sentence_and_read_pos_not_mode(self):
        a = pfba.derive_step_seed(13, "test", "s1", 3)
        b = pfba.derive_step_seed(13, "test", "s1", 3)
        c = pfba.derive_step_seed(13, "test", "s1", 4)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class DataAndCLITests(unittest.TestCase):
    def test_upstream_local_split_loads(self):
        samples = pfba.load_split("dev", limit=2)
        self.assertEqual(len(samples), 2)
        self.assertTrue(samples[0].source)
        self.assertTrue(samples[0].reference)
        self.assertTrue(samples[0].sentence_id.startswith("dev:"))

    def test_defaults(self):
        args = pfba.build_arg_parser().parse_args([])
        cfg = pfba.resolve_config(args)
        self.assertEqual(cfg.min_start, 3)
        self.assertEqual(cfg.branches, 10)
        self.assertEqual(cfg.horizon, 10)
        self.assertEqual(cfg.ralcp_threshold, 0.6)
        self.assertEqual(cfg.split, "test")

    def test_ralcp_threshold_validated(self):
        args = pfba.build_arg_parser().parse_args(["--ralcp-threshold", "1.5"])
        with self.assertRaises(ValueError):
            pfba.resolve_config(args)


if __name__ == "__main__":
    unittest.main()
