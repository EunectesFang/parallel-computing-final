"""Relaxed longest common prefix (RALCP) over beam candidates.

Paper: Wang et al. (2024), used as the RALCP baseline and as the candidate
aggregation in RALCP+TAF (Ouyang et al., NAACL 2025, Sec. 3.2 and 4.2).

Two implementations live here:

* :func:`author_ralcp` is a line-for-line port of the authors'
  ``agents/utils.py:ralcp`` (NAACL 2025 release, commit 99e373a).  It is the
  fidelity implementation used by the ``ralcp`` and ``ralcp_taf`` policies.
  Candidates are decoded continuation strings; voting happens depth by depth
  over words (or characters).  The denominator is always the number of
  candidates, candidates that ended early keep voting for their full
  sequence at deeper depths, and ties at a depth go to the prefix first seen
  (Python ``Counter`` insertion order, exactly as ``max(cnt, key=cnt.get)``).
  The winning prefix at depth d need not extend the winner at depth d-1.

* :func:`relaxed_longest_common_prefix` is an alternative longest-supported
  formulation kept for direct invariants and comparison; it is not the
  fidelity path.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Sequence

Token = Hashable


@dataclass(frozen=True)
class PrefixSelection:
    prefix: tuple[Token, ...]
    support: int
    total: int
    fraction: float
    extended: bool

    def to_dict(self) -> dict:
        return {
            "prefix_len": len(self.prefix),
            "support": self.support,
            "total": self.total,
            "fraction": self.fraction,
            "extended": self.extended,
        }


def author_ralcp(
    candidates: Sequence[str],
    threshold: float = 0.5,
    char: bool = False,
) -> PrefixSelection:
    """Exact port of the authors' word/character RALCP.

    ``candidates`` are decoded MT continuations *after* the committed target
    prefix.  For each depth ``i`` (1-based units) count candidate prefixes.
    The most common prefix is accepted while its count is at least
    ``len(candidates) * threshold``; the loop stops at the first depth that
    fails.  The deepest accepted prefix is returned.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    total = len(candidates)
    if total == 0:
        return PrefixSelection((), 0, 0, 0.0, False)

    sequences = [list(c) if char else str(c).split() for c in candidates]
    max_len = max(len(seq) for seq in sequences)

    winner: tuple[Token, ...] = ()
    support = 0
    fraction = 0.0

    for depth in range(1, max_len + 1):
        counts = Counter(tuple(seq[:depth]) for seq in sequences)
        # Same tie rule as the authors: max() keeps the first key reaching the
        # maximum in Counter insertion order (first appearance among candidates).
        most_common = max(counts, key=counts.get)
        count = counts[most_common]
        if count >= total * threshold:
            winner = most_common
            support = count
            fraction = count / total
        else:
            break

    return PrefixSelection(
        prefix=winner,
        support=support,
        total=total,
        fraction=fraction,
        extended=bool(winner),
    )


def _matches_prefix(candidate: Sequence[Token], committed: tuple[Token, ...]) -> bool:
    return len(candidate) >= len(committed) and tuple(candidate[: len(committed)]) == committed


def relaxed_longest_common_prefix(
    candidates: Sequence[Sequence[Token]],
    committed: Sequence[Token] = (),
    fraction: float = 1.0,
) -> PrefixSelection:
    """Return the longest prefix with support >= ``fraction``.

    - ``fraction == 1.0`` gives the ordinary longest common prefix.
    - Candidates that do not start with ``committed`` are invalid: they never
      support an extension but remain in the denominator.
    - Ties at the same length are broken by higher support, then by first
      appearance in ``candidates``.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    committed_t = tuple(committed)
    total = len(candidates)
    if total == 0:
        return PrefixSelection(committed_t, 0, 0, 0.0, False)

    valid = [tuple(c) for c in candidates if _matches_prefix(c, committed_t)]
    if not valid:
        return PrefixSelection(committed_t, 0, total, 0.0, False)

    max_len = max(len(c) for c in valid)
    for length in range(max_len, len(committed_t), -1):
        prefixes = [c[:length] for c in valid if len(c) >= length]
        if not prefixes:
            continue
        counts = Counter(prefixes)
        # Preserve first-appearance order for deterministic tie-breaking.
        best_prefix = None
        best_support = -1
        seen = set()
        for prefix in prefixes:
            if prefix in seen:
                continue
            seen.add(prefix)
            support = counts[prefix]
            if support > best_support:
                best_support = support
                best_prefix = prefix
        if best_prefix is not None and best_support / total >= fraction:
            return PrefixSelection(
                prefix=best_prefix,
                support=best_support,
                total=total,
                fraction=best_support / total,
                extended=True,
            )

    # Nothing beyond the committed prefix reaches the threshold.  Every valid
    # candidate supports the committed prefix itself, including candidates
    # longer than it, so support at that depth is len(valid).
    return PrefixSelection(
        prefix=committed_t,
        support=len(valid),
        total=total,
        fraction=len(valid) / total,
        extended=False,
    )
