"""Tests for reward functions and the counterfactual labeler.

Every predictor, translator, and timer here is a fake; no model or torch is
loaded. The labeler's own `model.generate` path is exercised only by real
adapters, so these tests pin the deterministic parts: nesting, the shared
translation pool, matched/incorrect against a fixed oracle, READ, and the
leakage-adjacent separation of oracle and candidate requests.
"""

import pytest

from controller_v3.action import READ, Action
from controller_v3.config import ControllerConfig
from controller_v3.labeler import CounterfactualLabeler
from controller_v3.reward import (
    RewardConfig,
    delay_penalty,
    immediate_reward,
    quality_reward,
    terminal_reward,
)

# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
WORDS = {1: "alpha", 2: "beta", 3: "gamma", 4: "delta", 5: "sits", 6: "sat"}
IDS = {word: token for token, word in WORDS.items()}

OBSERVED = "Hallo Welt."
SOURCE = "Hallo Welt."
COMMITTED = ["the", "cat"]
ORACLE = "the cat sits"
CANDIDATE = "the cat sat"
# `_request_text` produces this for a decoded continuation "alpha beta".
REQUEST = "Hallo Welt. alpha beta"


class FakeTokenizer:
    eos_token_id = 99

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        return " ".join(
            WORDS.get(int(token), f"t{int(token)}")
            for token in token_ids
            if int(token) != self.eos_token_id
        )

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [IDS[word] for word in text.split()]}


class RowsPredictor:
    """Exposes `sample_token_rows`; returns the first `max_b` scripted rows."""

    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.calls = []
        self.tok = FakeTokenizer()

    def sample_token_rows(self, observed, max_b, max_h, seed):
        self.calls.append((observed, max_b, max_h, seed))
        return [list(row) for row in self.rows[:max_b]]


class SamplePredictor:
    """Exposes only `sample` (plus an optional tokenizer)."""

    def __init__(self, samples, with_tokenizer=True):
        self.samples = list(samples)
        self.calls = []
        if with_tokenizer:
            self.tok = FakeTokenizer()

    def sample(self, observed, max_b, max_h, seed):
        self.calls.append((observed, max_b, max_h, seed))
        return list(self.samples[:max_b])


class EmptyPredictor:
    """No sampling interface at all."""


class MappingTranslator:
    def __init__(self, mapping=None):
        self.mapping = dict(mapping or {})
        self.calls = []

    def translate_batch(self, sources, prefix, **kwargs):
        self.calls.append((list(sources), prefix, dict(kwargs)))
        return [self.mapping.get(source, "") for source in sources]


class RecordingTimer:
    """Returns an increasing fake duration per `run` call."""

    def __init__(self):
        self.calls = []
        self.clock = 0.0

    def run(self, name, function):
        self.clock += 1.0
        value = function()
        self.calls.append((name, self.clock))
        return value, self.clock


def make_labeler(predictor, translator, timer=None):
    return CounterfactualLabeler(
        predictor, translator, ControllerConfig(), timer=timer
    )


def basic_case(actions, timer=None):
    rows = [[1, 2], [1, 2], [1, 2], [1, 2]]
    predictor = RowsPredictor(rows)
    translator = MappingTranslator({SOURCE: ORACLE, REQUEST: CANDIDATE})
    labeler = make_labeler(predictor, translator, timer=timer)
    outcomes = labeler.label_state(OBSERVED, COMMITTED, SOURCE, actions, seed=3)
    return predictor, translator, outcomes


# ----------------------------------------------------------------------
# Reward
# ----------------------------------------------------------------------
def test_quality_reward_uses_matched_and_incorrect_weights():
    cfg = RewardConfig(weight_m=1.0, weight_e=2.0)
    assert quality_reward(3, 1, cfg) == 1.0
    assert quality_reward(0, 0, cfg) == 0.0
    assert quality_reward(2, 4, cfg) == -6.0


def test_delay_penalty_clamps_non_positive_factors():
    cfg = RewardConfig(lambda_l=0.5)
    assert delay_penalty(2.0, 3.0, cfg) == pytest.approx(3.0)
    assert delay_penalty(-1.0, 3.0, cfg) == 0.0
    assert delay_penalty(2.0, -3.0, cfg) == 0.0


def test_immediate_reward_subtracts_delay_penalty():
    cfg = RewardConfig(weight_m=1.0, weight_e=2.0, lambda_l=0.1)
    expected = (4 - 2 * 1) - 0.1 * 2.0 * 5.0
    assert immediate_reward(4, 1, 2.0, 5.0, cfg) == pytest.approx(expected)


def test_terminal_reward_is_negative_absolute_length_gap():
    cfg = RewardConfig()
    assert terminal_reward("a b c", "a b c d", cfg) == -1.0
    assert terminal_reward("a", "x y z", cfg) == -2.0
    assert terminal_reward("a b", "x y", cfg) == 0.0


# ----------------------------------------------------------------------
# Candidate sampling
# ----------------------------------------------------------------------
def test_sample_candidates_is_nested_b2_is_first_two_of_b4():
    predictor = RowsPredictor([[1, 2, 3], [4, 5, 6], [1, 4], [2, 5]])
    labeler = make_labeler(predictor, MappingTranslator())

    small = labeler.sample_candidates("obs", 2, 3, 7)
    large = labeler.sample_candidates("obs", 4, 3, 7)

    assert small == large[:2]
    assert small == [[1, 2, 3], [4, 5, 6]]
    assert predictor.calls == [("obs", 2, 3, 7), ("obs", 4, 3, 7)]


def test_sample_candidates_falls_back_to_sample_plus_tokenizer():
    predictor = SamplePredictor(["alpha beta", "gamma"])
    labeler = make_labeler(predictor, MappingTranslator())

    rows = labeler.sample_candidates("obs", 2, 4, 5)

    assert rows == [[1, 2], [3]]
    assert predictor.calls == [("obs", 2, 4, 5)]


def test_sample_candidates_without_tokenizer_raises():
    predictor = SamplePredictor(["alpha"], with_tokenizer=False)
    labeler = make_labeler(predictor, MappingTranslator())

    with pytest.raises(RuntimeError):
        labeler.sample_candidates("obs", 2, 4, 5)


def test_sample_candidates_with_no_sampling_interface_raises():
    labeler = make_labeler(EmptyPredictor(), MappingTranslator())

    with pytest.raises(RuntimeError):
        labeler.sample_candidates("obs", 2, 4, 5)


# ----------------------------------------------------------------------
# Oracle and candidate paths
# ----------------------------------------------------------------------
def test_oracle_words_translates_source_only_with_committed_prefix():
    predictor = RowsPredictor([])
    translator = MappingTranslator({SOURCE: "  the cat sits  "})
    labeler = make_labeler(predictor, translator)

    words = labeler.oracle_words(SOURCE, ["the", "cat"])

    assert words == ("the", "cat", "sits")
    assert translator.calls == [([SOURCE], "the cat", {"final": False})]
    assert predictor.calls == []  # the oracle path samples nothing


def test_label_state_separates_oracle_and_candidate_translation():
    _predictor, translator, outcomes = basic_case([Action(2, 4)])

    assert len(outcomes) == 1
    # First translation is the oracle source alone; the second is the
    # observed-prefix request pool. They never share a batch.
    assert translator.calls[0][0] == [SOURCE]
    assert translator.calls[1][0] == [REQUEST] * 2
    assert translator.calls[0][1] == translator.calls[1][1] == "the cat"


# ----------------------------------------------------------------------
# Matching and the shared pool
# ----------------------------------------------------------------------
def test_matched_incorrect_and_emitted_against_fixed_oracle():
    _predictor, _translator, outcomes = basic_case([Action(2, 4), Action(4, 4)])

    assert [outcome.action for outcome in outcomes] == [Action(2, 4), Action(4, 4)]
    expected_pools = {
        Action(2, 4): (CANDIDATE, CANDIDATE),
        Action(4, 4): (CANDIDATE,) * 4,
    }
    for outcome in outcomes:
        assert outcome.new_words == ("the", "cat", "sat")
        assert outcome.matched == 2          # "the cat" agree, "sat" != "sits"
        assert outcome.incorrect == 1
        assert outcome.emitted == 3
        assert outcome.no_emit is False
        assert outcome.next_committed == ("the", "cat", "the", "cat", "sat")
        assert outcome.candidates == expected_pools[outcome.action]
        assert outcome.vote_top1 == 1.0
        assert outcome.vote_margin == 1.0
        assert outcome.vote_entropy == 0.0


def test_translate_once_per_unique_horizon_not_per_branch_pair():
    rows = [[1, 2, 3, 4, 5, 6] for _ in range(4)]
    predictor = RowsPredictor(rows)
    request_4 = "Hallo Welt. alpha beta gamma delta"
    request_6 = "Hallo Welt. alpha beta gamma delta sits sat"
    translator = MappingTranslator(
        {SOURCE: ORACLE, request_4: "one two", request_6: "three four"}
    )
    timer = RecordingTimer()
    labeler = make_labeler(predictor, translator, timer=timer)
    actions = [Action(2, 4), Action(4, 4), Action(2, 6), Action(4, 6)]

    outcomes = labeler.label_state(OBSERVED, COMMITTED, SOURCE, actions, seed=9)
    by_action = {outcome.action: outcome for outcome in outcomes}

    candidate_calls = [call for call in translator.calls if call[0] != [SOURCE]]
    assert len(candidate_calls) == 2  # one per horizon, not one per (b, h)
    assert candidate_calls[0][0] == [request_4] * 4
    assert candidate_calls[1][0] == [request_6] * 4

    # b variants share the max-b translation time for their horizon.
    assert by_action[Action(2, 4)].mt_s == by_action[Action(4, 4)].mt_s == 3.0
    assert by_action[Action(2, 6)].mt_s == by_action[Action(4, 6)].mt_s == 4.0
    # pred_s is the single shared generation measurement.
    assert {outcome.pred_s for outcome in outcomes} == {1.0}


# ----------------------------------------------------------------------
# READ
# ----------------------------------------------------------------------
def test_read_outcome_has_zero_compute_and_no_words():
    labeler = make_labeler(RowsPredictor([]), MappingTranslator())

    outcome = labeler.read_outcome(COMMITTED)

    assert outcome.action is READ
    assert outcome.new_words == ()
    assert outcome.matched == 0
    assert outcome.incorrect == 0
    assert outcome.emitted == 0
    assert outcome.no_emit is True
    assert outcome.next_committed == ("the", "cat")
    assert outcome.pred_s == 0.0
    assert outcome.mt_s == 0.0
    assert outcome.candidates == ()


def test_label_state_fills_read_without_extra_translation():
    _predictor, translator, outcomes = basic_case([READ, Action(2, 4)])

    assert outcomes[0] == make_labeler(
        RowsPredictor([]), MappingTranslator()
    ).read_outcome(COMMITTED)
    assert len(translator.calls) == 2  # oracle + one horizon, READ adds none


# ----------------------------------------------------------------------
# Immutability
# ----------------------------------------------------------------------
def test_label_state_does_not_mutate_input_committed():
    committed = ["the", "cat"]
    predictor = RowsPredictor([[1, 2]] * 4)
    translator = MappingTranslator({SOURCE: ORACLE, REQUEST: CANDIDATE})
    labeler = make_labeler(predictor, translator)

    outcomes = labeler.label_state(
        OBSERVED, committed, SOURCE, [Action(2, 4)], seed=1
    )

    assert committed == ["the", "cat"]
    assert isinstance(outcomes[0].next_committed, tuple)
    assert outcomes[0].next_committed == ("the", "cat", "the", "cat", "sat")
