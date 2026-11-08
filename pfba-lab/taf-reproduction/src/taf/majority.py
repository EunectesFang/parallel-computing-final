"""TAF Eq. 6-7: exact-sequence majority vote and agreement rate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

Token = Hashable


@dataclass(frozen=True)
class MajorityResult:
    hypothesis: tuple[Token, ...]
    count: int
    n: int
    pi: float

    def to_dict(self) -> dict:
        return {
            "hypothesis_len": len(self.hypothesis),
            "count": self.count,
            "n": self.n,
            "pi": self.pi,
        }


def exact_majority(
    hypotheses: Sequence[Sequence[Token]],
    committed: Sequence[Token] = (),
) -> MajorityResult:
    """Majority over full target sequences, relative to a committed prefix.

    ``n`` in the paper is the number of LLM continuations, so the denominator
    is ``len(hypotheses)``.  Hypotheses that do not extend ``committed`` are
    invalid: they count against agreement but can never be returned.
    Ties are broken by first appearance.
    """
    committed_t = tuple(committed)
    n = len(hypotheses)
    if n == 0:
        return MajorityResult(committed_t, 0, 0, 0.0)

    counts: dict[tuple[Token, ...], int] = {}
    first_seen: dict[tuple[Token, ...], int] = {}
    for idx, hyp in enumerate(hypotheses):
        hyp_t = tuple(hyp)
        if len(hyp_t) < len(committed_t) or hyp_t[: len(committed_t)] != committed_t:
            continue
        counts[hyp_t] = counts.get(hyp_t, 0) + 1
        first_seen.setdefault(hyp_t, idx)

    if not counts:
        return MajorityResult(committed_t, 0, n, 0.0)

    best = max(counts, key=lambda h: (counts[h], -first_seen[h]))
    return MajorityResult(best, counts[best], n, counts[best] / n)


def extend_only(committed: Sequence[Token], proposal: Sequence[Token]) -> tuple[Token, ...]:
    """Return ``proposal`` if it extends ``committed``, else ``committed``.

    This is the hard invariant behind the forced committed prefix: a policy
    can append but never rewrite or delete target tokens.
    """
    committed_t = tuple(committed)
    proposal_t = tuple(proposal)
    if len(proposal_t) >= len(committed_t) and proposal_t[: len(committed_t)] == committed_t:
        return proposal_t
    return committed_t
