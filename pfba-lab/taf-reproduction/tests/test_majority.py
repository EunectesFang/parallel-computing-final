"""TAF Eq. 6-7: exact-sequence majority and pi = Count(f)/n."""
import pytest

from taf.majority import exact_majority, extend_only


def test_majority_and_pi():
    hyps = [(1, 2)] * 8 + [(1, 3)] * 2
    maj = exact_majority(hyps)
    assert maj.hypothesis == (1, 2)
    assert maj.count == 8
    assert maj.n == 10
    assert maj.pi == pytest.approx(0.8)


def test_majority_tie_breaks_on_first_occurrence():
    maj = exact_majority([(1, 3), (1, 2), (1, 2), (1, 3)])
    assert maj.hypothesis == (1, 3)
    assert maj.count == 2
    assert maj.pi == pytest.approx(0.5)


def test_invalid_hypotheses_count_against_agreement_but_are_not_returned():
    # Six candidates extend (9,), four do not.
    hyps = [(9, 1)] * 6 + [(8, 1)] * 4
    maj = exact_majority(hyps, committed=(9,))
    assert maj.hypothesis == (9, 1)
    assert maj.count == 6
    assert maj.n == 10
    assert maj.pi == pytest.approx(0.6)


def test_empty_candidates_mean_zero_pi_and_committed_output():
    maj = exact_majority([], committed=(7, 8))
    assert maj.hypothesis == (7, 8)
    assert maj.count == 0
    assert maj.pi == 0.0


def test_extend_only_enforces_monotone_prefix():
    assert extend_only((1, 2), (1, 2, 3)) == (1, 2, 3)
    assert extend_only((1, 2), (1, 2)) == (1, 2)
    assert extend_only((1, 2), (1, 3)) == (1, 2)  # divergence: keep committed
    assert extend_only((1, 2), (0,)) == (1, 2)
