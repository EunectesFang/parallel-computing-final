"""Action grid for the cost-aware value controller.

The controller either READs (waits for more source) or WRITEs. A write action
is a `(branches, horizon)` pair: `branches` candidates are sampled and each is
truncated to at most `horizon` new tokens before translation and RALCP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "Action",
    "READ",
    "DEFAULT_ACTIONS",
    "EXTENDED_ACTIONS",
    "action_features",
]


@dataclass(frozen=True, order=True)
class Action:
    """One controller action: READ is `(0, 0)`; writes carry branches/horizon."""

    branches: int
    horizon: int

    @property
    def is_read(self) -> bool:
        return self.branches == 0 and self.horizon == 0

    @property
    def label(self) -> str:
        return "read" if self.is_read else f"b{self.branches}h{self.horizon}"


READ = Action(0, 0)

# 3 x 3 grid used by the default controller.
DEFAULT_ACTIONS: tuple[Action, ...] = tuple(
    Action(branches, horizon) for branches in (2, 4, 6) for horizon in (4, 6, 10)
)

# 5 x 5 grid for wider sweeps.
EXTENDED_ACTIONS: tuple[Action, ...] = tuple(
    Action(branches, horizon)
    for branches in (2, 4, 6, 8, 10)
    for horizon in (2, 4, 6, 8, 10)
)


def action_features(action: Action, threshold: float = 0.6) -> tuple[float, ...]:
    """Six numeric features for one action.

    The last slot is the fraction of the branch budget RALCP must match:
    ``ceil(threshold * branches) / branches``. READ returns six zeros.
    """
    if action.is_read:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    branches = action.branches
    horizon = action.horizon
    return (
        float(branches),
        float(horizon),
        float(branches * horizon),
        math.log(branches),
        math.log(horizon),
        math.ceil(threshold * branches) / branches,
    )
