"""Post-candidate controller: add branches, wait, or stop after a write.

The main controller picks a `(branches, horizon)` write budget before any
candidate exists. This controller runs **after** the runtime has produced the
provisional RALCP result, so it can look at the vote agreement and at the
network's no-emit probability and decide whether the provisional pool is good
enough.

The controller only *proposes*. It never commits, never samples, and never
mutates the runtime. The caller is responsible for:

* `ADD_2`: sampling the extra branches through `ControlledRuntime.extend` and
  re-running RALCP over the **full** candidate pool, because the extra
  candidates can change which prefix wins the vote.
* `WAIT`: discarding the provisional result and leaving the committed prefix
  untouched.
* `STOP`: committing the provisional RALCP result as-is.

Thresholds are hand-picked heuristic defaults, **not** tuned on data. They are
module constants so a sweep or a learned post-controller can replace them
without touching the call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import ControllerConfig

__all__ = [
    "PostAction",
    "PostDecision",
    "PostController",
    "STOP_MARGIN_MIN",
    "STOP_NO_EMIT_MAX",
    "ADD_ENTROPY_MIN",
    "WAIT_NO_EMIT_MIN",
    "WAIT_BUDGET_S",
    "ADD_2_BRANCHES",
]

# A confident vote with a small chance of emitting nothing is good enough.
STOP_MARGIN_MIN = 0.30
STOP_NO_EMIT_MAX = 0.25

# Disagreement in the pool is worth two more candidates when the budget allows.
ADD_ENTROPY_MIN = 0.50

# A likely no-emit that already burned the budget is better retried later.
WAIT_NO_EMIT_MIN = 0.60
WAIT_BUDGET_S = 1.00

# Number of branches `ADD_2` requests; the caller caps it at `max_branches`.
ADD_2_BRANCHES = 2


class PostAction(Enum):
    """What the caller should do with the provisional write result."""

    STOP = "stop"
    ADD_2 = "add_2"
    WAIT = "wait"


@dataclass(frozen=True)
class PostDecision:
    """One proposal plus a short human-readable reason for logging."""

    action: PostAction
    reason: str


class PostController:
    """Heuristic post-candidate policy with a network-ready signature.

    `z_pre` is the state encoding that scored the write action. The default
    rule does not use it, but accepting it keeps the interface stable for a
    future learned head and for tests that pass a sentinel object.
    """

    def decide(
        self,
        z_pre,
        current_branches: int,
        horizon: int,
        vote_entropy: float,
        vote_margin: float,
        no_emit_prob: float,
        spent_s: float,
        cfg: ControllerConfig,
    ) -> PostDecision:
        """Propose STOP, ADD_2, or WAIT for one provisional write result.

        Rules are checked in this order and the first match wins:

        1. high `vote_margin` and low `no_emit_prob` -> STOP;
        2. high `vote_entropy` and room under `cfg.max_branches` -> ADD_2;
        3. high `no_emit_prob` and `spent_s` over budget -> WAIT;
        4. otherwise STOP, the conservative default.

        `horizon` is accepted for symmetry with the action that produced the
        provisional pool; `ADD_2` keeps that horizon and only grows branches.
        """
        del z_pre, horizon  # Reserved for a learned head / future horizon growth.

        branches = int(current_branches)
        max_branches = int(cfg.max_branches)
        margin = float(vote_margin)
        entropy = float(vote_entropy)
        no_emit = float(no_emit_prob)
        spent = float(spent_s)

        if margin >= STOP_MARGIN_MIN and no_emit <= STOP_NO_EMIT_MAX:
            return PostDecision(
                PostAction.STOP,
                f"confident vote (margin={margin:.2f}) with low no-emit risk",
            )

        if entropy >= ADD_ENTROPY_MIN and branches < max_branches:
            return PostDecision(
                PostAction.ADD_2,
                f"entropy={entropy:.2f} with branches {branches} < {max_branches}",
            )

        if no_emit >= WAIT_NO_EMIT_MIN and spent >= WAIT_BUDGET_S:
            return PostDecision(
                PostAction.WAIT,
                f"no-emit risk={no_emit:.2f} and spent={spent:.2f}s over budget",
            )

        return PostDecision(
            PostAction.STOP,
            "insufficient evidence to change the provisional result",
        )
