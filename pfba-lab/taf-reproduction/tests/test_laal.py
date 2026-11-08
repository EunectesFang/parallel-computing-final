"""SimulEval LAAL invariants: reference length, cutoff, no per-token clamp."""
import pytest

from taf.laal import (
    average_lagging,
    laal,
    source_completion_cutoff,
)


def test_monotone_equal_length_matches_average_lagging():
    delays = [1, 2, 3, 4]
    assert laal(4, 4, delays) == pytest.approx(1.0)
    assert average_lagging(4, 4, delays) == pytest.approx(1.0)


def test_everything_written_at_source_end():
    assert laal(4, 4, [4, 4, 4, 4]) == pytest.approx(2.5)


def test_length_adaptive_slope():
    # S=10, R=4, P=4 -> T=max(4,4)=4; slope (S-1)/(T-1) = 3.
    # Terms 10, 7, 4, 1 -> mean 5.5 (SimulEval does not clamp, but all >= 0).
    assert laal(10, 4, [10, 10, 10, 10]) == pytest.approx(5.5)


def test_simuleval_does_not_clamp_negative_terms():
    # is: 1,0,-1,...,-8; sum -35; /T=10 -> -3.5.
    assert laal(10, 10, [1] * 10) == pytest.approx(-3.5)


def test_target_length_is_max_reference_and_prediction():
    # Prediction is 4 words but the reference is 10 words: T=10, slope=1.
    # Terms 1, 0, -1, -2 -> sum -2; /10 -> -0.2.
    assert laal(10, 10, [1, 1, 1, 1]) == pytest.approx(-0.2)
    # Prediction longer than reference: T=12.
    assert laal(4, 4, [4] * 12) == pytest.approx(2.5)


def test_reference_longer_than_single_emission():
    # P=1, R=3 -> T=3; slope=(5-1)/(3-1)=2; term 2; /3.
    assert laal(5, 3, [2]) == pytest.approx(2 / 3)


def test_cutoff_drops_post_source_emissions():
    delays = [1, 2, 10, 10, 12]
    # First full-source emission begins at index 2 (delay 10).  A whole
    # emission is included: both tokens with delay 10.  The later delay-12
    # token is dropped.
    assert source_completion_cutoff(delays, 10) == 4
    cutoff_value = laal(10, 4, delays, cutoff=4)
    # P_used=4, T=max(4,4)=4, slope=(10-1)/(4-1)=3.
    # Terms 1, -1, 4, 1 -> sum 5; /4 = 1.25.
    assert cutoff_value == pytest.approx(1.25)


def test_cutoff_is_all_emissions_when_source_never_completes():
    assert source_completion_cutoff([1, 2, 3], 10) == 3
    # P=3, R=4 -> T=4, slope=(10-1)/(3)=3; terms 1,-1,4 -> sum 4; /4=1.0.
    assert laal(10, 4, [1, 2, 10], cutoff=3) == pytest.approx(1.0)


def test_single_target_word():
    assert laal(5, 1, [2]) == pytest.approx(2.0)


def test_empty_target_is_undefined():
    assert laal(5, 0, []) is None


def test_over_generation_is_penalized_relative_to_average_lagging():
    source_len, target_len = 4, 12
    delays = [4] * target_len
    assert laal(source_len, target_len, delays) == pytest.approx(2.5)
    assert average_lagging(source_len, target_len, delays) == pytest.approx(26 / 12)
    assert laal(source_len, target_len, delays) > average_lagging(
        source_len, target_len, delays
    )


def test_invalid_arguments():
    with pytest.raises(ValueError):
        laal(5, 3, [1, 2], cutoff=3)
    with pytest.raises(ValueError):
        laal(-1, 0, [])
