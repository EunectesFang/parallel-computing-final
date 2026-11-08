"""RALCP / relaxed longest-common-prefix invariants (paper Sec. 3.2, 4.2)."""
import pytest

from taf.ralcp import author_ralcp, relaxed_longest_common_prefix


# --- Author fidelity implementation (agents/utils.py:ralcp) ---------------


def test_author_ralcp_winner_can_change_between_depths():
    # Depth 1: "a" and "b" are tied at 2/4; first seen wins ("a").
    # Depth 2: "b y" has 2/4 and is the most common; it wins even though it
    # does not extend the depth-1 winner.  This is the author's behaviour.
    sel = author_ralcp(["a x", "a z", "b y", "b y"], threshold=0.5)
    assert sel.prefix == ("b", "y")
    assert sel.support == 2
    assert sel.total == 4


def test_author_ralcp_ties_are_broken_by_first_appearance():
    sel = author_ralcp(["a x", "a y", "b z", "b w"], threshold=0.5)
    assert sel.prefix == ("a",)
    assert sel.support == 2


def test_author_ralcp_has_no_exact_majority_prerequisite():
    # Every full sequence is unique (exact majority = 1/3), but all agree on
    # the first word, which clears the 0.5 relaxed threshold.
    sel = author_ralcp(["a b", "a c", "a d"], threshold=0.5)
    assert sel.prefix == ("a",)
    assert sel.support == 3
    assert sel.fraction == pytest.approx(1.0)


def test_author_ralcp_counts_ended_candidates_in_denominator():
    # The short candidate votes for its full sequence ("a") at every depth and
    # stays in the denominator.  At depth 2 support is only 1/3 < 0.7.
    sel = author_ralcp(["a b", "a c", "a"], threshold=0.7)
    assert sel.prefix == ("a",)
    assert sel.support == 3
    assert sel.total == 3


def test_author_ralcp_empty_candidates_are_counted():
    sel = author_ralcp(["a b", "", ""], threshold=0.5)
    assert sel.prefix == ()
    assert sel.extended is False
    assert sel.total == 3


def test_author_ralcp_char_mode():
    sel = author_ralcp(["你好世界", "你好朋友", "你好"], threshold=0.5, char=True)
    assert "".join(sel.prefix) == "你好"
    assert sel.support == 3


def test_author_ralcp_rejects_bad_threshold():
    with pytest.raises(ValueError):
        author_ralcp(["a"], threshold=1.5)


def test_standard_lcp_when_fraction_is_one():
    sel = relaxed_longest_common_prefix([(1, 2, 3, 4), (1, 2, 3, 5)], fraction=1.0)
    assert sel.prefix == (1, 2, 3)
    assert sel.support == 2
    assert sel.total == 2
    assert sel.extended is True


def test_relaxed_prefix_picks_longest_supported():
    candidates = [(1, 2, 3), (1, 2, 4), (1, 5, 6)]
    # (1, 2) has 2/3 support; (1,) has 3/3 support.
    sel = relaxed_longest_common_prefix(candidates, fraction=0.66)
    assert sel.prefix == (1, 2)
    assert sel.support == 2
    assert sel.fraction == pytest.approx(2 / 3)


def test_relaxed_prefix_falls_back_when_support_too_low():
    candidates = [(1, 2, 3), (1, 2, 4), (1, 5, 6)]
    sel = relaxed_longest_common_prefix(candidates, fraction=0.7)
    assert sel.prefix == (1,)
    assert sel.extended is True


def test_committed_prefix_is_never_rewritten():
    committed = (9,)
    candidates = [(9, 1), (8, 2)]  # second candidate is invalid (does not extend committed)
    sel = relaxed_longest_common_prefix(candidates, committed=committed, fraction=0.5)
    # Total denominator includes the invalid candidate, so support is 1/2.
    assert sel.prefix == (9, 1)
    assert sel.support == 1
    assert sel.total == 2

    sel_strict = relaxed_longest_common_prefix(candidates, committed=committed, fraction=0.6)
    assert sel_strict.prefix == committed
    assert sel_strict.extended is False


def test_no_candidates_returns_committed():
    sel = relaxed_longest_common_prefix([], committed=(4, 5), fraction=0.5)
    assert sel.prefix == (4, 5)
    assert sel.total == 0
    assert sel.extended is False


def test_tie_break_prefers_higher_support_then_order():
    # At length 3 two prefixes both have 2/4 = 0.5 support.
    candidates = [(1, 2, 3), (1, 2, 3), (1, 2, 4), (1, 2, 4)]
    sel = relaxed_longest_common_prefix(candidates, fraction=0.5)
    assert sel.prefix == (1, 2, 3)  # tied support, first appearance wins
    assert sel.support == 2


def test_invalid_fraction_rejected():
    with pytest.raises(ValueError):
        relaxed_longest_common_prefix([(1,)], fraction=1.5)
