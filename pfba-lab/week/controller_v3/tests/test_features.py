"""Tests for the 34-value online state feature contract."""

import inspect
import math

import pytest

from controller_v3.features import (
    CONFIDENCE_FEATURES,
    HISTORY_FEATURES,
    MASK_FEATURES,
    NUMERIC_FEATURES,
    SURFACE_FEATURES,
    FeatureBuilder,
    FeatureVector,
    History,
    OptionalConfidence,
    RuntimeInfo,
    estimate_length_ratio,
    feature_diagnostics,
)


def as_dict(vector: FeatureVector) -> dict:
    """Map a feature vector back to its named values for readable assertions."""
    return dict(zip(NUMERIC_FEATURES, vector.numeric))


def test_feature_groups_form_exactly_thirty_four_unique_names():
    assert len(SURFACE_FEATURES) == 26
    assert len(CONFIDENCE_FEATURES) == 4
    assert len(MASK_FEATURES) == 4
    assert NUMERIC_FEATURES == SURFACE_FEATURES + CONFIDENCE_FEATURES + MASK_FEATURES
    assert len(NUMERIC_FEATURES) == 34
    assert len(set(NUMERIC_FEATURES)) == 34
    assert MASK_FEATURES == tuple(f"{name}_available" for name in CONFIDENCE_FEATURES)


def test_build_returns_thirty_four_floats_in_order():
    builder = FeatureBuilder()
    vector = builder.build("Hallo Welt.", ["Hello", "there", "friend"])
    assert isinstance(vector, FeatureVector)
    assert isinstance(vector.numeric, tuple)
    assert len(vector.numeric) == 34
    assert all(isinstance(value, float) for value in vector.numeric)
    # The tuple follows NUMERIC_FEATURES order.
    values = as_dict(vector)
    assert values["src_words"] == 2.0
    assert values["tgt_words"] == 3.0
    assert values["length_ratio"] == pytest.approx(1.5)


def test_surface_feature_values_follow_the_existing_conventions():
    builder = FeatureBuilder()
    vector = builder.build("Hallo Welt.", ["Hello", "there", "friend"])
    values = as_dict(vector)
    assert values["src_words"] == 2.0
    assert values["src_chars"] == float(len("Hallo Welt."))
    assert values["tgt_words"] == 3.0
    assert values["tgt_chars"] == float(len("Hello there friend"))
    assert values["length_ratio"] == pytest.approx(3 / 2)
    assert values["normalized_lag"] == 0.0  # 2 - 3 / 1.0 is negative
    assert values["src_end_punct"] == 1.0
    assert values["tgt_end_punct"] == 0.0
    assert values["last_capital"] == 1.0  # "Welt." starts with a capital
    assert values["current_clause_len"] == 0.0  # the last word closes a clause


def test_length_ratio_is_current_ratio_not_the_training_estimate():
    builder = FeatureBuilder(length_ratio=2.0)
    vector = builder.build("a b c d e f", ["x", "y", "z", "w"])
    values = as_dict(vector)
    assert values["length_ratio"] == pytest.approx(4 / 6)
    # rho is used only for the lag feature.
    assert values["normalized_lag"] == pytest.approx(6 - 4 / 2.0)
    assert builder.rho == 2.0


def test_normalized_lag_never_goes_negative():
    builder = FeatureBuilder(length_ratio=0.5)
    vector = builder.build("a b", ["x", "y", "z", "w"])
    assert as_dict(vector)["normalized_lag"] == 0.0


def test_history_features_are_copied():
    builder = FeatureBuilder()
    history = History(
        last_branches=4,
        last_horizon=6,
        last_vote_top1=0.75,
        last_vote_margin=0.5,
        last_vote_entropy=0.25,
        last_commit_count=2,
        last_empty=True,
        words_since_commit=7,
        no_emit_streak=3,
    )
    values = as_dict(builder.build("a b", ["x"], history=history))
    expected = {
        "last_B": history.last_branches,
        "last_H": history.last_horizon,
        "last_vote_top1": history.last_vote_top1,
        "last_vote_margin": history.last_vote_margin,
        "last_vote_entropy": history.last_vote_entropy,
        "last_commit_count": history.last_commit_count,
        "last_empty": history.last_empty,
    }
    assert tuple(expected) == HISTORY_FEATURES
    for name, value in expected.items():
        assert values[name] == float(value), name
    assert values["words_since_commit"] == 7.0
    assert values["no_emit_streak"] == 3.0


def test_runtime_features_are_copied():
    builder = FeatureBuilder()
    runtime = RuntimeInfo(
        pred_cost_ref=1.25,
        mt_cost_ref=2.5,
        pred_queue_work=0.5,
        mt_queue_work=1.5,
        time_since_commit=0.75,
        arrival_slack=0.125,
    )
    values = as_dict(builder.build("a b", ["x"], runtime=runtime))
    assert values["pred_cost_ref"] == 1.25
    assert values["mt_cost_ref"] == 2.5
    assert values["pred_queue_work"] == 0.5
    assert values["mt_queue_work"] == 1.5
    assert values["time_since_commit"] == 0.75
    assert values["arrival_slack"] == 0.125


def test_confidence_values_and_masks():
    builder = FeatureBuilder()
    confidence = OptionalConfidence(
        values=(0.5, 0.25, 0.75, 0.125),
        available=(True, False, True, False),
    )
    values = as_dict(builder.build("a b", ["x"], confidence=confidence))
    assert [values[name] for name in CONFIDENCE_FEATURES] == [0.5, 0.0, 0.75, 0.0]
    assert [values[name] for name in MASK_FEATURES] == [1.0, 0.0, 1.0, 0.0]


def test_confidence_defaults_are_zero_and_masked_out():
    values = as_dict(FeatureBuilder().build("a b", ["x"]))
    assert [values[name] for name in CONFIDENCE_FEATURES] == [0.0, 0.0, 0.0, 0.0]
    assert [values[name] for name in MASK_FEATURES] == [0.0, 0.0, 0.0, 0.0]


def test_punctuation_features():
    builder = FeatureBuilder()
    values = as_dict(builder.build("Er kam. Dann ging es weiter", ["Er", "kam."]))
    assert values["src_end_punct"] == 0.0
    assert values["tgt_end_punct"] == 1.0
    assert values["current_clause_len"] == 4.0  # Dann ging es weiter
    assert values["subordinate_marker"] == 0.0

    values = as_dict(builder.build("Er kam", ["Er", "kam"]))
    assert values["src_end_punct"] == 0.0
    assert values["tgt_end_punct"] == 0.0
    assert values["current_clause_len"] == 2.0


def test_subordinate_marker_detects_german_conjunctions():
    builder = FeatureBuilder()
    values = as_dict(builder.build("Ich glaube dass er", ["I", "think"]))
    assert values["subordinate_marker"] == 1.0
    assert values["current_clause_len"] == 4.0

    values = as_dict(builder.build("Katzen schlafen weil es regnet", ["Cats", "sleep"]))
    assert values["subordinate_marker"] == 1.0

    values = as_dict(builder.build("Es passiert oft, sodass wir", ["It", "happens"]))
    assert values["subordinate_marker"] == 1.0

    values = as_dict(builder.build("Katzen schlafen", ["Cats", "sleep"]))
    assert values["subordinate_marker"] == 0.0


def test_articles_are_not_subordinate_markers():
    builder = FeatureBuilder()
    for source in ("Die Katze schlaeft", "Der Hund bellt", "Das Kind singt"):
        values = as_dict(builder.build(source, ["it"]))
        assert values["subordinate_marker"] == 0.0, source


def test_build_cannot_see_source_reference_or_future_words():
    names = set(inspect.signature(FeatureBuilder.build).parameters)
    forbidden = {"source", "reference", "future", "full_source"}
    assert forbidden.isdisjoint(names)
    assert {"observed", "committed"} <= names
    action_names = set(inspect.signature(FeatureBuilder.__init__).parameters)
    assert forbidden.isdisjoint(action_names)


def test_text_ids_stay_empty_without_vocabularies():
    vector = FeatureBuilder().build("eins zwei", ["one"])
    assert vector.src_ids == ()
    assert vector.tgt_ids == ()


def test_text_ids_use_the_configured_window():
    class FakeVocab:
        def __init__(self):
            self.calls = []

        def encode(self, words, window):
            words = tuple(words)
            self.calls.append((words, window))
            return tuple(range(1, min(window, len(words)) + 1))

    src_vocab = FakeVocab()
    tgt_vocab = FakeVocab()
    builder = FeatureBuilder(src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    vector = builder.build("eins zwei drei", ["one", "two"])
    assert vector.src_ids == (1, 2, 3)
    assert vector.tgt_ids == (1, 2)
    assert src_vocab.calls == [(("eins", "zwei", "drei"), 16)]
    assert tgt_vocab.calls == [(("one", "two"), 16)]


def test_empty_prefixes_do_not_crash():
    vector = FeatureBuilder().build("", [])
    values = as_dict(vector)
    assert values["src_words"] == 0.0
    assert values["tgt_words"] == 0.0
    assert values["length_ratio"] == 0.0
    assert values["src_end_punct"] == 0.0
    assert values["tgt_end_punct"] == 0.0
    assert len(vector.numeric) == 34


def test_estimate_length_ratio_uses_non_empty_pairs_and_clamps():
    assert estimate_length_ratio([("a b", "c d e"), ("x", "y")]) == pytest.approx(1.25)
    assert estimate_length_ratio([("", "x y"), ("a b", "")]) == 1.0
    assert estimate_length_ratio([]) == 1.0
    assert estimate_length_ratio([("a b c d e", "x")]) == 0.5
    assert estimate_length_ratio([("a", "b c d e f g")]) == 2.0


def test_numeric_values_are_finite():
    vector = FeatureBuilder().build("Hallo Welt.", ["Hello", "world"])
    assert all(math.isfinite(value) for value in vector.numeric)


# ----------------------------------------------------------------------
# Feature diagnostics
# ----------------------------------------------------------------------
def _vector(**overrides) -> FeatureVector:
    values = [0.0] * len(NUMERIC_FEATURES)
    for name, value in overrides.items():
        values[NUMERIC_FEATURES.index(name)] = float(value)
    return FeatureVector(
        numeric=tuple(values),
        predictor_id=0,
        translator_id=0,
        src_ids=(),
        tgt_ids=(),
    )


def test_feature_diagnostics_reports_mean_std_nonzero_and_missing():
    vectors = [
        _vector(src_words=1, tgt_words=0),
        _vector(src_words=3, tgt_words=2),
    ]
    diagnostics = feature_diagnostics(vectors)

    assert diagnostics["n"] == 2
    src = diagnostics["features"]["src_words"]
    assert src["mean"] == pytest.approx(2.0)
    assert src["std"] == pytest.approx(1.0)
    assert src["nonzero"] == 2
    assert src["missing"] == 0
    tgt = diagnostics["features"]["tgt_words"]
    assert tgt["mean"] == pytest.approx(1.0)
    assert tgt["nonzero"] == 1


def test_feature_diagnostics_counts_non_finite_values_as_missing():
    values = [0.0] * len(NUMERIC_FEATURES)
    values[NUMERIC_FEATURES.index("src_words")] = float("nan")
    vector = FeatureVector(
        numeric=tuple(values),
        predictor_id=0,
        translator_id=0,
        src_ids=(),
        tgt_ids=(),
    )
    diagnostics = feature_diagnostics([vector])
    src = diagnostics["features"]["src_words"]
    assert src["missing"] == 1
    assert src["mean"] is None
    assert src["std"] is None


def test_feature_diagnostics_flags_constant_training_feature_that_moves():
    training = [_vector(src_words=2) for _ in range(3)]
    serving = [_vector(src_words=2), _vector(src_words=5)]
    diagnostics = feature_diagnostics(serving, reference=training)

    src = diagnostics["features"]["src_words"]
    assert src["train_mean"] == pytest.approx(2.0)
    assert src["serve_mean"] == pytest.approx(3.5)
    assert src["mean_diff"] == pytest.approx(1.5)
    assert src["constant_train_varies_serving"] is True
    assert "src_words" in diagnostics["constant_train_varies_serving"]
    assert any("src_words" in warning for warning in diagnostics["warnings"])

    # A feature constant in both sets is not flagged.
    assert (
        diagnostics["features"]["tgt_words"]["constant_train_varies_serving"]
        is False
    )


def test_feature_diagnostics_rejects_a_wrong_width_vector():
    vector = FeatureVector(
        numeric=(0.0, 1.0),
        predictor_id=0,
        translator_id=0,
        src_ids=(),
        tgt_ids=(),
    )
    with pytest.raises(ValueError, match="NUMERIC_FEATURES"):
        feature_diagnostics([vector])
