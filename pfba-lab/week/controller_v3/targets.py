"""Build `ActionTarget` training records from counterfactual outcomes.

`CounterfactualLabeler` produces an `ActionOutcome` for every action at one
decision point. This module turns one state's outcomes into the flat list of
`ActionTarget` rows the trainer consumes.

Successors are **per action**: `next_states[i]` pairs with `outcomes[i]`, because
READ and each write leave different committed prefixes and therefore different
successor feature vectors. The legacy `next_state` argument shares one
successor across every action and is kept only as an ablation that reproduces
the old action-independent Bellman target.

`value_target` is the immediate reward only; the trainer appends the discounted
one-step backup from the row's own successor. The delay term is evaluated on the
**post-action** backlog: one extra source word has arrived and the action has
committed (or failed to commit) words, so two writes with different `emitted`
values receive different delay penalties.

Leakage boundary: this module reads *labelled outcomes*, never the source,
reference, or future words. It cannot see them because its inputs are the
already-computed `ActionOutcome` records.
"""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

from .action import READ, Action
from .config import ControllerConfig
from .features import NUMERIC_FEATURES, StateInput
from .labeler import ActionOutcome
from .reward import RewardConfig, immediate_reward
from .trainer import ActionTarget

__all__ = ["build_action_targets", "derive_state_key"]

SRC_WORDS_INDEX = NUMERIC_FEATURES.index("src_words")
TGT_WORDS_INDEX = NUMERIC_FEATURES.index("tgt_words")


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
    next_states: Sequence[StateInput] | None = None,
    next_state: StateInput | None = None,
    next_actions: Sequence[Action] = (),
    src_words_after: int | None = None,
    tgt_words_after_per_outcome: Sequence[int] | None = None,
    terminal: bool = False,
    terminal_value: float | None = None,
    terminal_values: Sequence[float] | None = None,
    backlog: float = 0.0,
    source_advance: int = 1,
    state_key: str = "",
    rho: float = 1.0,
) -> list[ActionTarget]:
    """Convert one state's counterfactual outcomes into `ActionTarget` rows.

    Every outcome at a decision point shares the same `state` and `state_key`,
    so the trainer's pairwise ranking loss can group them.

    Successor selection:

    * `next_states` (preferred) is a sequence paired index-by-index with
      `outcomes`. READ and every write get the state that action actually
      reached.
    * `next_state` is the legacy ablation: one successor is shared by every
      action, which makes the Bellman target action-independent. It is only
      used when `next_states` is ``None``.
    * `terminal=True` forces every row to be terminal: `next_state=None`,
      `next_actions=()`, and `value_target=terminal_value` (no bootstrap).
      `terminal_values` may instead carry one value per outcome; each action
      reaches a different committed prefix, so its terminal target must be
      evaluated on that prefix rather than sharing the behavior action's.

    Delay:

    * With `src_words_after`, the post-action backlog is
      ``max(0, src_words_after - (current_tgt + emitted) / rho)``.
    * Without it, the caller's pre-action `backlog` is advanced by
      ``source_advance`` and reduced by the action's own emission
      (``emitted / rho``), which keeps the ablation action-dependent.

    `next_actions` defaults to ``(READ, *cfg.actions)`` for non-terminal rows
    with a successor: the backup must be allowed to value waiting. A row whose
    successor is missing (or a terminal row) advertises ``()`` because it cannot
    bootstrap. An empty `state_key` is replaced by `derive_state_key(state)` so
    it is never empty.
    """
    outcome_list = tuple(outcomes)
    key = state_key or derive_state_key(state)
    reward_cfg = _reward_config(cfg)
    resolved_rho = float(rho)
    if not math.isfinite(resolved_rho) or resolved_rho <= 0.0:
        resolved_rho = 1.0
    advance = int(source_advance)

    if terminal:
        successors: tuple[StateInput | None, ...] = tuple(None for _ in outcome_list)
        resolved_next_actions: tuple[Action, ...] = ()
    else:
        if next_states is not None:
            if len(next_states) != len(outcome_list):
                raise ValueError(
                    "next_states must pair one-to-one with outcomes: got "
                    f"{len(next_states)} successors for {len(outcome_list)} "
                    "outcomes"
                )
            successors = tuple(next_states)
        else:
            successors = tuple(next_state for _ in outcome_list)
        resolved_next_actions = (
            tuple(next_actions) if next_actions else (READ, *cfg.actions)
        )

    if (
        tgt_words_after_per_outcome is not None
        and len(tgt_words_after_per_outcome) != len(outcome_list)
    ):
        raise ValueError(
            "tgt_words_after_per_outcome must pair one-to-one with outcomes: "
            f"got {len(tgt_words_after_per_outcome)} values for "
            f"{len(outcome_list)} outcomes"
        )
    if terminal_values is not None and len(terminal_values) != len(outcome_list):
        raise ValueError(
            "terminal_values must pair one-to-one with outcomes: got "
            f"{len(terminal_values)} values for {len(outcome_list)} outcomes"
        )

    current_tgt = int(round(float(state.numeric[TGT_WORDS_INDEX])))

    targets: list[ActionTarget] = []
    for index, outcome in enumerate(outcome_list):
        action = outcome.action
        if terminal:
            if terminal_values is not None:
                value_target = float(terminal_values[index])
            else:
                value_target = (
                    float(terminal_value) if terminal_value is not None else 0.0
                )
        elif src_words_after is not None:
            if tgt_words_after_per_outcome is not None:
                words_after = int(tgt_words_after_per_outcome[index])
            else:
                words_after = current_tgt + int(outcome.emitted)
            backlog_after = max(
                0.0, float(src_words_after) - words_after / resolved_rho
            )
            value_target = immediate_reward(
                int(outcome.matched),
                int(outcome.incorrect),
                backlog_after,
                advance,
                reward_cfg,
            )
        else:
            # Legacy ablation: advance the pre-action backlog by the source
            # word and reduce it by this action's committed words.
            backlog_after = max(
                0.0,
                float(backlog) + advance - int(outcome.emitted) / resolved_rho,
            )
            value_target = immediate_reward(
                int(outcome.matched),
                int(outcome.incorrect),
                backlog_after,
                advance,
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

        # Advertise the backup action set only when this row actually has a
        # successor; a missing successor means no bootstrap, so no action set.
        row_next_actions = (
            resolved_next_actions if successors[index] is not None else ()
        )
        targets.append(
            ActionTarget(
                state=state,
                action=action,
                value_target=float(value_target),
                error_target=error_target,
                no_emit_target=no_emit_target,
                emit_target=emit_target,
                state_key=key,
                next_state=successors[index],
                next_actions=row_next_actions,
                terminal=bool(terminal),
            )
        )
    return targets
