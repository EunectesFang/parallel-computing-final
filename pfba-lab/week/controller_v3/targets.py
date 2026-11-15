"""Build `ActionTarget` training records from counterfactual outcomes.

`CounterfactualLabeler` produces an `ActionOutcome` for every action at one
decision point. This module turns one state's outcomes into the flat list of
`ActionTarget` rows the trainer consumes.

`value_target` is the **immediate** reward only. The trainer appends the
discounted one-step backup, so the same immediate value is used whether or not
the caller supplies `next_state`. READ receives no quality term (its
`matched` / `incorrect` counts are zero), so the only reward it can earn is the
negative delay penalty; that is the price of waiting.

Leakage boundary: this module reads *labelled outcomes*, never the source,
reference, or future words. It cannot see them because its inputs are the
already-computed `ActionOutcome` records.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from .action import READ, Action
from .config import ControllerConfig
from .features import StateInput
from .labeler import ActionOutcome
from .reward import RewardConfig, immediate_reward
from .trainer import ActionTarget

__all__ = ["build_action_targets", "derive_state_key"]


def derive_state_key(state: StateInput) -> str:
    """Return a deterministic, process-independent key for `state`.

    Python's built-in `hash()` is salted per process, so it must never be used
    for a value that has to survive a checkpoint or a separate training run.
    The key hashes the numeric feature tuple, both model ids, and the text id
    tuples with SHA-256. Float values are serialized through `repr`, which is
    the shortest round-tripping representation in Python 3.
    """
    payload = repr(
        (
            tuple(float(value) for value in state.numeric),
            int(state.predictor_id),
            int(state.translator_id),
            tuple(int(token) for token in state.src_ids),
            tuple(int(token) for token in state.tgt_ids),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reward_config(cfg: ControllerConfig) -> RewardConfig:
    """Project the controller config onto the reward weights it shares."""
    return RewardConfig(
        weight_m=cfg.weight_m,
        weight_e=cfg.weight_e,
        lambda_l=cfg.lambda_l,
        gamma=cfg.gamma,
    )


def build_action_targets(
    state: StateInput,
    outcomes: Sequence[ActionOutcome],
    cfg: ControllerConfig,
    *,
    next_state: StateInput | None = None,
    next_actions: Sequence[Action] = (),
    backlog: float = 0.0,
    source_advance: int = 1,
    state_key: str = "",
) -> list[ActionTarget]:
    """Convert one state's counterfactual outcomes into `ActionTarget` rows.

    Every outcome at a decision point shares the same `state` and `state_key`,
    so the trainer's pairwise ranking loss can group them. `next_state` is the
    successor state the caller observed after the decision (for READ, the
    post-READ state); the trainer bootstraps from it only when it is not
    ``None`` and the row is not terminal.

    `next_actions` defaults to ``()`` and is replaced by
    ``(READ, *cfg.actions)`` whenever `next_state` is supplied. Including READ
    in the backup is required: the one-step target must be allowed to value
    waiting, otherwise it systematically overvalues writing.

    An empty `state_key` is replaced by `derive_state_key(state)`. Production
    callers should still pass an explicit sentence/position key, because two
    distinct states with identical feature vectors would otherwise collapse
    into one ranking group.
    """
    key = state_key or derive_state_key(state)
    reward_cfg = _reward_config(cfg)

    if next_state is not None and not next_actions:
        next_actions = (READ, *cfg.actions)
    resolved_next_actions = tuple(next_actions)

    targets: list[ActionTarget] = []
    for outcome in outcomes:
        action = outcome.action
        value_target = immediate_reward(
            int(outcome.matched),
            int(outcome.incorrect),
            float(backlog),
            int(source_advance),
            reward_cfg,
        )

        # READ has no write head, so its auxiliary write targets are unused by
        # the trainer. They are still serialized with zero values so every row
        # has the same shape.
        if action.is_read:
            error_target = 0.0
            no_emit_target = 0.0
            emit_target = 0.0
        else:
            error_target = 1.0 if int(outcome.incorrect) > 0 else 0.0
            no_emit_target = 1.0 if bool(outcome.no_emit) else 0.0
            emit_target = float(outcome.emitted)

        targets.append(
            ActionTarget(
                state=state,
                action=action,
                value_target=float(value_target),
                error_target=error_target,
                no_emit_target=no_emit_target,
                emit_target=emit_target,
                state_key=key,
                next_state=next_state,
                next_actions=resolved_next_actions,
                terminal=False,
            )
        )
    return targets
