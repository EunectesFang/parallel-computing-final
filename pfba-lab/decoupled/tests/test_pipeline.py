from threading import Event
from time import sleep
import pytest

from pipeline import Config, run_sentence

SOURCE = "a b c d e f g h"
CONFIG = Config(arrival_gap=0.01, min_start=2, ready_limit=1)


class Predictor:
    def sample(self, observed, branches, horizon, seed):
        return [" imagined"] * branches


class Translator:
    def __init__(self):
        self.calls = []

    def translate_batch(self, requests, prefix):
        self.calls.append((requests, prefix))
        return ["x"] * len(requests)


def test_fifo_matches_serial_and_preserves_committed_prefix():
    translator = Translator()
    sequential = run_sentence(SOURCE, "x x x", Predictor(), translator, CONFIG, "taf")
    translator = Translator()
    pipelined = run_sentence(SOURCE, "x x x", Predictor(), translator, CONFIG, "overlap-fifo")
    assert pipelined.hypothesis == sequential.hypothesis
    assert pipelined.delays == sequential.delays
    assert [step.read_pos for step in pipelined.steps] == list(range(2, 9))
    assert [prefix for _, prefix in translator.calls] == [" ".join(["x"] * i) for i in range(7)]
    assert translator.calls[-1][0] == [SOURCE]
    for span in pipelined.predictions:
        assert span.start_s >= (span.read_pos - 1) * CONFIG.arrival_gap


def test_prediction_can_finish_while_translation_is_waiting():
    translated_started, prediction_finished = Event(), Event()

    class ProbePredictor(Predictor):
        def sample(self, observed, *args):
            if observed == "a b c":
                assert translated_started.wait(2)
                prediction_finished.set()
            return super().sample(observed, *args)

    class ProbeTranslator(Translator):
        def translate_batch(self, requests, prefix):
            if not self.calls:
                translated_started.set()
                assert prediction_finished.wait(2)
            return super().translate_batch(requests, prefix)

    trace = run_sentence(SOURCE, "", ProbePredictor(), ProbeTranslator(), CONFIG, "overlap-fifo")
    assert trace.overlap_s > 0.0
    assert trace.peak_ready <= CONFIG.ready_limit


def test_latest_flushes_actual_source_after_backlog():
    class SlowTranslator(Translator):
        def translate_batch(self, requests, prefix):
            if not self.calls:
                sleep(0.09)
            return super().translate_batch(requests, prefix)

    translator = SlowTranslator()
    trace = run_sentence(SOURCE, "", Predictor(), translator, CONFIG, "overlap-latest")
    assert trace.steps[-1].read_pos == 8
    assert translator.calls[-1][0] == [SOURCE]
    assert len(trace.steps) < 7
    assert all(span.start_s >= (span.read_pos - 1) * CONFIG.arrival_gap
               for span in trace.predictions)


def test_prediction_failure_reaches_caller():
    class BrokenPredictor:
        def sample(self, *_):
            raise RuntimeError("broken predictor")

    import pytest
    with pytest.raises(RuntimeError, match="broken predictor"):
        run_sentence(SOURCE, "", BrokenPredictor(), Translator(), CONFIG, "overlap-fifo")

def test_anchor_veto_only_keeps_target_words_agreeing_with_true_prefix():
    class AnchorTranslator(Translator):
        def translate_batch(self, requests, prefix):
            self.calls.append((requests, prefix))
            return ["shared speculative" if item.endswith("imagined") else "shared observed"
                    for item in requests]

    translator = AnchorTranslator()
    config = Config(arrival_gap=0.001, min_start=2, anchor_policy="veto")
    trace = run_sentence("a b c", "", Predictor(), translator, config, "taf-anchor")
    assert translator.calls[0][0][-1] == "a b"
    assert trace.steps[0].new_words == ("shared",)
    assert translator.calls[-1] == (["a b c"], "shared")


def test_anchor_vote_can_corroborate_a_source_prefix_candidate():
    class DiversePredictor:
        def sample(self, *args):
            return ["next", "other", "third"]

    class DiverseTranslator(Translator):
        def translate_batch(self, requests, prefix):
            self.calls.append((requests, prefix))
            outputs = {"a b next": "stable", "a b other": "uncertain", "a b third": "alternative",
                       "a b": "stable", "a b c": "last"}
            return [outputs[item] for item in requests]

    config = Config(branches=3, arrival_gap=0.001, min_start=2, threshold=0.5, anchor_policy="vote")
    trace = run_sentence("a b c", "", DiversePredictor(), DiverseTranslator(), config,
                         "overlap-fifo-anchor")
    assert trace.steps[0].new_words == ("stable",)

def test_bounded_catch_up_respects_initial_commit_and_skip_limit():
    class BusyTranslator(Translator):
        def translate_batch(self, requests, prefix):
            self.calls.append((requests, prefix))
            sleep(0.03)
            return [""] * len(requests) if len(self.calls) <= 2 else ["x"] * len(requests)

    source = "a b c d e f g h i j k l m n"
    translator = BusyTranslator()
    config = Config(arrival_gap=0.005, min_start=2, ready_limit=1,
                    bounded_initial_steps=2, bounded_initial_words=1)
    trace = run_sentence(source, "", Predictor(), translator, config, "bounded-2")
    positions = [step.read_pos for step in trace.steps]
    assert positions[:3] == [2, 3, 4]
    assert positions[-2:] == [13, 14]
    gaps = [right - left for left, right in zip(positions, positions[1:])]
    assert max(gaps) in (2, 3)
    assert trace.peak_ready <= 1
    assert trace.discarded_predictions == 0
    assert translator.calls[-1][0] == [source]
    assert all(span.start_s >= (span.read_pos - 1) * config.arrival_gap
               for span in trace.predictions)


def test_bounded_never_skips_if_no_initial_word_was_committed():
    class SilentTranslator(Translator):
        def translate_batch(self, requests, prefix):
            self.calls.append((requests, prefix))
            sleep(0.01)
            return [""] * len(requests)

    config = Config(arrival_gap=0.001, min_start=2, ready_limit=2,
                    bounded_initial_steps=1, bounded_initial_words=1)
    trace = run_sentence(SOURCE, "", Predictor(), SilentTranslator(), config, "bounded-3")
    assert [step.read_pos for step in trace.steps] == list(range(2, 9))
    assert trace.peak_ready <= 2


def test_bounded_threshold_does_not_change_taf_threshold():
    class SplitTranslator(Translator):
        def translate_batch(self, requests, prefix):
            return ["agree", "agree", "other", "other"] if len(requests) == 4 else ["last"]

    config = Config(arrival_gap=0.001, min_start=2, threshold=0.6, bounded_threshold=0.5)
    sequential = run_sentence("a b c", "", Predictor(), SplitTranslator(), config, "taf")
    bounded = run_sentence("a b c", "", Predictor(), SplitTranslator(), config, "bounded-1")
    assert sequential.steps[0].new_words == ()
    assert bounded.steps[0].new_words == ("agree",)

@pytest.mark.parametrize("mode,cadence,cap", [
    ("bounded-2", 3, 2), ("bounded-1-every2", 2, 1), ("bounded-1-every3", 3, 1),
])
def test_bounded_skip_cadence_counts_completed_mt_steps_only(mode, cadence, cap):
    class BusyTranslator(Translator):
        def translate_batch(self, requests, prefix):
            self.calls.append((requests, prefix))
            sleep(0.015)
            return ["x"] * len(requests)

    source = " ".join(f"word{i}" for i in range(24))
    config = Config(arrival_gap=0.001, min_start=2, ready_limit=1,
                    bounded_initial_steps=2, bounded_initial_words=1,
                    bounded_skip_every=3 if mode == "bounded-2" else 1)
    trace = run_sentence(source, "", Predictor(), BusyTranslator(), config, mode)
    skips = [span for span in trace.predictions if span.skipped_before > 0]
    assert len(skips) >= 2
    assert all(span.completed_mt_steps_at_submit >= 2 for span in skips)
    assert all(right.completed_mt_steps_at_submit - left.completed_mt_steps_at_submit >= cadence
               for left, right in zip(skips, skips[1:]))
    positions = [step.read_pos for step in trace.steps]
    assert max(right - left - 1 for left, right in zip(positions, positions[1:])) <= cap
    assert trace.peak_ready <= config.ready_limit
    assert trace.discarded_predictions == 0
    assert all(span.start_s >= (span.read_pos - 1) * config.arrival_gap
               for span in trace.predictions)


def test_early_fifo_uses_the_same_schedule_and_marks_only_nonfinal_steps():
    class AwareTranslator(Translator):
        def translate_batch(self, requests, prefix, final=False, early_stop_k=0, threshold=0.6):
            self.calls.append((final, early_stop_k, list(requests), prefix))
            return ["x"] * len(requests)

    source = SOURCE
    plain = AwareTranslator()
    early = AwareTranslator()
    fifo = run_sentence(source, "x x x", Predictor(), plain, CONFIG, "overlap-fifo")
    stopped = run_sentence(source, "x x x", Predictor(), early, CONFIG, "overlap-fifo-early")
    assert stopped.hypothesis == fifo.hypothesis
    assert stopped.delays == fifo.delays
    assert [step.read_pos for step in stopped.steps] == [step.read_pos for step in fifo.steps]
    assert all(call[0] is False and call[1] == 4 for call in early.calls[:-1])
    assert early.calls[-1][0] is True and early.calls[-1][1] == 0
    assert all(call[1] == 0 for call in plain.calls)
    assert early.calls[-1][2] == [source]


def test_bounded_skip_cadence_must_be_positive():
    import pytest
    with pytest.raises(ValueError, match="skip_every"):
        Config(bounded_skip_every=0)
