"""Immediate and terminal rewards for the cost-aware value controller.

The reward decomposes into a quality term and a latency term:

    quality_reward = weight_m * matched - weight_e * incorrect
    delay_penalty  = lambda_l * max(0, backlog) * max(0, source_advance)
    immediate      = quality_reward - delay_penalty

`matched` is the length of the common prefix between the words RALCP emitted
and the oracle translation; `incorrect` counts emitted words after the first
mismatch. `backlog` is how far the committed target lags the observed source,
and `source_advance` is how many source words arrived at this decision point.
A negative backlog or advance means nothing is waiting, so the penalty is zero.

`terminal_reward` is a deliberately cheap length-difference proxy used when a
sentence ends without an oracle word-level comparison. It is **not** COMET or
any learned metric: it only penalises a hypothesis whose word count differs
from the reference.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "RewardConfig",
    "quality_reward",
    "delay_penalty",
    "immediate_reward",
    "terminal_reward",
]


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights; defaults mirror `ControllerConfig`."""

    weight_m: float = 1.0
    weight_e: float = 2.0
    lambda_l: float = 0.05
    gamma: float = 0.99


def quality_reward(matched: int, incorrect: int, cfg: RewardConfig) -> float:
    """Matched-word reward minus a heavier penalty for wrong emitted words."""
    return float(cfg.weight_m) * int(matched) - float(cfg.weight_e) * int(incorrect)


def delay_penalty(backlog: float, source_advance: float, cfg: RewardConfig) -> float:
    """Latency cost for a source lead that the committed target has not closed.

    Both factors are clamped at zero: a non-positive backlog (target caught up
    or ahead) or a non-positive advance (no new source) means no delay cost.
    """
    return (
        float(cfg.lambda_l)
        * max(0.0, float(backlog))
        * max(0.0, float(source_advance))
    )


def immediate_reward(
    matched: int,
    incorrect: int,
    backlog: float,
    source_advance: float,
    cfg: RewardConfig,
) -> float:
    """Quality of one action minus its latency cost."""
    return quality_reward(matched, incorrect, cfg) - delay_penalty(
        backlog, source_advance, cfg
    )


def terminal_reward(
    hypothesis: str,
    reference: str,
    cfg: RewardConfig,
    *,
    backlog: float = 0.0,
    source_advance: float = 0.0,
) -> float:
    """Negative absolute word-count difference between hypothesis and reference.

    This is an intentionally simple terminal proxy, not COMET or BLEU. An
    optional delay term is available for a caller that wants the last state to
    price any remaining source lead; the defaults disable it so the historical
    length-only behaviour is unchanged. `cfg` is part of the shared contract so
    a future weighted terminal metric can be introduced without changing
    callers.
    """
    length_gap = -float(
        abs(len(hypothesis.split()) - len(reference.split()))
    )
    return length_gap - delay_penalty(backlog, source_advance, cfg)
