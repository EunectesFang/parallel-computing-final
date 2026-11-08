"""Length-Adaptive Average Lagging (LAAL) as implemented by SimulEval.

SimulEval's latency scorer (``simuleval/evaluator/scorers/latency_scorer.py``)
defines LAAL with two properties that matter for reproduction:

1. The normalization length is ``max(len(reference), len(prediction))``
   (``reference`` is the human target, not the system output).
2. Only emissions up to the first full-source emission batch are scored;
   later target tokens are cut off because the source is already complete.

Per-token terms are **not** clamped to zero.  With word-level delays
``g_i`` (the number of source units read when target word ``i`` was emitted),
0-based ``i``, source length ``S`` and normalization length ``T``:

    LAAL = (1/T) * sum_{i=0}^{P-1} (g_i - i * (S - 1) / (T - 1))

where ``P`` is the number of target tokens up to the cutoff.  For ``T == 1``
the slope is undefined and SimulEval returns the single delay.
"""
from __future__ import annotations

from statistics import mean
from typing import Sequence


def source_completion_cutoff(delays: Sequence[int], source_length: int) -> int:
    """Number of target emissions kept by SimulEval's cutoff.

    The cutoff is the *emission* (a whole write, possibly several tokens) at
    which the source first becomes fully consumed.  Tokens committed in that
    emission share the same ``read_pos`` delay, so the whole contiguous run is
    kept; tokens emitted later are dropped.  If the source is never emitted
    at full length, every emission is kept.

    This delay-only helper cannot distinguish two separate writes that happen
    to share the same delay (e.g. a full-source anticipation write followed by
    the fresh EOS write).  :func:`taf.runner.run_dataset` therefore computes
    the cutoff from step traces, where write boundaries are explicit; use this
    function only when per-write information is unavailable.
    """
    for index, delay in enumerate(delays):
        if delay >= source_length:
            end = index
            while end + 1 < len(delays) and delays[end + 1] == delay:
                end += 1
            return end + 1
    return len(delays)


def laal(
    source_length: int,
    reference_length: int,
    delays: Sequence[int],
    cutoff: int | None = None,
) -> float | None:
    """SimulEval LAAL in source units (word for De/En, char for Zh/Ja).

    ``reference_length`` is ignored as a ceiling: the effective target length
    is ``max(reference_length, kept_prediction_length)``.  ``cutoff`` limits
    ``delays`` to ``delays[:cutoff]``; by default every emission is used.
    Returns ``None`` only when there is nothing to normalize (``T == 0``).
    """
    if source_length < 0 or reference_length < 0:
        raise ValueError("lengths must be non-negative")
    if cutoff is None:
        cutoff = len(delays)
    if cutoff < 0 or cutoff > len(delays):
        raise ValueError(f"cutoff must be in [0, {len(delays)}], got {cutoff}")

    kept = [float(d) for d in delays[:cutoff]]
    target_length = max(reference_length, len(kept))
    if target_length == 0:
        return None
    if not kept:
        return 0.0
    if target_length == 1:
        return kept[0]

    slope = (source_length - 1) / (target_length - 1)
    return sum(delay - i * slope for i, delay in enumerate(kept)) / target_length


def laal_raw(
    source_length: int,
    reference_length: int,
    delays: Sequence[int],
    cutoff: int | None = None,
) -> float | None:
    """Compatibility alias.  SimulEval's formula itself is unclamped."""
    return laal(source_length, reference_length, delays, cutoff=cutoff)


def average_lagging(
    source_length: int,
    target_length: int,
    delays: Sequence[int],
) -> float | None:
    """Reference average lagging (Ma et al., 2019) for comparison only."""
    if source_length < 0 or target_length < 0:
        raise ValueError("lengths must be non-negative")
    if len(delays) != target_length:
        raise ValueError(f"expected {target_length} delays, got {len(delays)}")
    if target_length == 0:
        return None
    if target_length == 1:
        return float(delays[0])
    slope = source_length / target_length
    terms = [float(d) - i * slope for i, d in enumerate(delays)]
    return float(mean(terms))
