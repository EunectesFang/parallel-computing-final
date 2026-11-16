"""Multi-task trainer for the CAVC value network.

`ActionTarget.value_target` is the immediate reward produced by the labeler.
`CAVCTrainer.compute_targets` turns it into a one-step Bellman target using a
frozen copy of the network and the injected cost model:

    y = r + gamma * max_a' [Q_target(s', a') - lambda_t * C(s', a')]

The bootstrap term is added only when `next_state` exists and `terminal` is
False. READ and write actions share the same Q scale.
"""

from __future__ import annotations

import copy
import math
import random
import time
import warnings
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .action import READ, Action
from .config import ControllerConfig
from .features import NUMERIC_FEATURES, StateInput
from .network import CAVCNetwork, WriteOutputs

__all__ = [
    "ActionTarget",
    "TrainConfig",
    "CAVCTrainer",
    "pairwise_ranking_loss",
]

SRC_WORDS_INDEX = NUMERIC_FEATURES.index("src_words")
TGT_WORDS_INDEX = NUMERIC_FEATURES.index("tgt_words")

MAX_RANK_PAIRS_PER_STATE = 32


@dataclass(frozen=True)
class ActionTarget:
    """One labelled `(state, action)` pair.

    `value_target` is the immediate reward; the trainer appends the discounted
    next-state value when `next_state` is present and `terminal` is False.
    """

    state: StateInput
    action: Action
    value_target: float
    error_target: float
    no_emit_target: float
    emit_target: float
    # Stable identity of `state` used to group actions for the ranking loss.
    # Production callers must always pass a non-empty key; the empty default is
    # for ad-hoc tests only, because empty keys intentionally disable ranking.
    state_key: str
    next_state: StateInput | None = None
    next_actions: tuple[Action, ...] = ()
    terminal: bool = False
    # CAVC-Q V4 three-component targets. ``None`` in V3 mode; the trainer uses
    # ``value_target`` (immediate reward) exactly as before.
    quality_target: float | None = None
    delay_target: float | None = None
    wall_target: float | None = None


@dataclass
class TrainConfig:
    """Optimisation knobs that are independent of the controller config."""

    epochs: int = 20
    batch_size: int = 128
    lr: float = 3e-4


def pairwise_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    state_keys: Sequence[str],
    epsilon: float,
    max_pairs: int = MAX_RANK_PAIRS_PER_STATE,
    write_mask: Sequence[bool] | None = None,
) -> torch.Tensor:
    """Pairwise logistic ranking loss over write actions of the same state.

    Only pairs whose target values differ by more than `epsilon` contribute.
    Pairs are grouped by `state_key` and capped at `max_pairs` per state. Rows
    with an empty `state_key` are skipped: an empty key carries no state
    identity, so pairing them would compare different states. When `write_mask`
    is supplied, only write-action rows may be paired; READ rows never enter the
    ranking loss. When no pair qualifies, the result is a differentiable zero so
    a state with only READ or only one write action adds no NaN and no spurious
    gradient.
    """
    if predictions.numel() == 0:
        return predictions.sum() * 0.0

    groups: dict[str, list[int]] = {}
    for index, key in enumerate(state_keys):
        if not key:
            continue
        if write_mask is not None and not bool(write_mask[index]):
            continue
        groups.setdefault(key, []).append(index)

    terms: list[torch.Tensor] = []
    for indices in groups.values():
        pairs: list[tuple[int, int]] = []
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                first, second = indices[left], indices[right]
                difference = float(targets[first] - targets[second])
                if abs(difference) <= epsilon:
                    continue
                if difference > 0:
                    pairs.append((first, second))
                else:
                    pairs.append((second, first))
        for high, low in pairs[:max_pairs]:
            terms.append(F.softplus(-(predictions[high] - predictions[low])))

    if not terms:
        return predictions.sum() * 0.0
    return torch.stack(terms).mean()


def _pad_ids(sequences: Sequence[Sequence[int]]) -> torch.Tensor:
    """Right-pad id sequences with PAD=0; `[N, 0]` when all are empty."""
    width = max((len(sequence) for sequence in sequences), default=0)
    if width == 0:
        return torch.zeros((len(sequences), 0), dtype=torch.long)
    padded = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        if sequence:
            padded[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return padded


class CAVCTrainer:
    """Trains `CAVCNetwork` with value, ranking, and write-side auxiliary losses."""

    def __init__(
        self,
        network: CAVCNetwork,
        cfg: ControllerConfig,
        cost_model=None,
    ) -> None:
        self.network = network
        self.cfg = cfg
        self.cost_model = cost_model
        self.target_network = copy.deepcopy(network)
        self.target_network.eval()
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)
        self.train_config = TrainConfig(
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
        )
        # Per-epoch component losses, exposed for callers that want a curve.
        self.history: list[dict[str, float]] = []
        # Warn at most once per trainer when a plain measured model is used
        # instead of a schedule-aware `PipelineCostModel`.
        self._warned_plain_cost = False

    # ------------------------------------------------------------------
    # Target computation
    # ------------------------------------------------------------------
    def compute_targets(self, targets: Sequence[ActionTarget]) -> list[float]:
        """Return one-step Bellman targets in the same order as `targets`.

        Each row bootstraps from *its own* `next_state`, so READ and two
        different writes from the same state receive different targets. Rows
        with no successor or a terminal row use `value_target` only.

        The backup is cached per unique ``(next_state, next_actions)`` within a
        call, and all unique successor states are encoded in one batch. Without
        this, a state's ten counterfactual action rows would each re-run the
        frozen network over the same successor.
        """
        self.target_network.eval()
        results: list[float] = []

        # Collect unique successors so they can be encoded in one batch.
        unique_states: list[StateInput] = []
        state_index: dict[StateInput, int] = {}
        for target in targets:
            if target.next_state is None or target.terminal:
                continue
            state = target.next_state
            if state not in state_index:
                state_index[state] = len(unique_states)
                unique_states.append(state)

        with torch.no_grad():
            encoded = (
                self._encode_states(self.target_network, unique_states)
                if unique_states
                else None
            )
            cache: dict[tuple[StateInput, tuple[Action, ...]], float] = {}
            for target in targets:
                if target.next_state is None or target.terminal:
                    results.append(float(target.value_target))
                    continue

                # READ is a first-class action, so the default backup set must
                # include it; omitting READ biases the target toward writing.
                actions = tuple(target.next_actions) or (READ, *self.cfg.actions)
                lookup = (target.next_state, actions)
                bonus = cache.get(lookup)
                if bonus is None:
                    row = state_index[target.next_state]
                    bonus = self._best_next_value_from_z(
                        encoded[row : row + 1], target.next_state, actions
                    )
                    cache[lookup] = bonus
                results.append(float(target.value_target) + self.cfg.gamma * bonus)
        return results

    def compute_component_targets(
        self, targets: Sequence[ActionTarget]
    ) -> list[tuple[float, float, float]]:
        """Double-DQN targets for the three V4 value components.

        The online network selects the next action by
        ``U = Qq - beta_lag*Ql - beta_wall*Qw``; the frozen target network
        evaluates that same action. Each component bootstraps independently,
        and no ``lambda_t * C`` term is subtracted (the wall head already
        carries time, so subtracting it again would double count).
        """
        self.target_network.eval()
        was_training = self.network.training
        # Double DQN selection must be deterministic: dropout would turn the
        # online argmax into noise.
        self.network.eval()
        results: list[tuple[float, float, float]] = []

        unique_states: list[StateInput] = []
        state_index: dict[StateInput, int] = {}
        for target in targets:
            if target.next_state is None or target.terminal:
                continue
            state = target.next_state
            if state not in state_index:
                state_index[state] = len(unique_states)
                unique_states.append(state)

        with torch.no_grad():
            online_encoded = (
                self._encode_states(self.network, unique_states)
                if unique_states
                else None
            )
            target_encoded = (
                self._encode_states(self.target_network, unique_states)
                if unique_states
                else None
            )
            cache: dict[tuple[StateInput, tuple[Action, ...]], tuple[float, float, float]] = {}
            for target in targets:
                immediate = (
                    float(target.quality_target)
                    if target.quality_target is not None
                    else float(target.value_target),
                    float(target.delay_target or 0.0),
                    float(target.wall_target or 0.0),
                )
                if target.next_state is None or target.terminal:
                    results.append(immediate)
                    continue
                actions = tuple(target.next_actions) or (READ, *self.cfg.actions)
                lookup = (target.next_state, actions)
                bonus = cache.get(lookup)
                if bonus is None:
                    row = state_index[target.next_state]
                    bonus = self._best_next_components_from_z(
                        online_encoded[row : row + 1],
                        target_encoded[row : row + 1],
                        actions,
                    )
                    cache[lookup] = bonus
                results.append(
                    (
                        immediate[0] + self.cfg.gamma * bonus[0],
                        immediate[1] + self.cfg.gamma_delay * bonus[1],
                        immediate[2] + self.cfg.gamma_wall * bonus[2],
                    )
                )
        if was_training:
            self.network.train()
        return results

    def _component_values(
        self,
        model: CAVCNetwork,
        z: torch.Tensor,
        actions: Sequence[Action],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Per-action `(quality, delay, wall)` values preserving `actions` order."""
        read_q, read_l, read_w = model.read_components(z)
        write_actions = [action for action in actions if not action.is_read]
        write_q = write_l = write_w = None
        if write_actions:
            outputs = model.write_values(z, write_actions)
            write_q, write_l, write_w = (
                outputs.quality[0],
                outputs.delay[0],
                outputs.wall[0],
            )
        column = {action: index for index, action in enumerate(write_actions)}
        quality: list[torch.Tensor] = []
        delay: list[torch.Tensor] = []
        wall: list[torch.Tensor] = []
        for action in actions:
            if action.is_read:
                quality.append(read_q[0])
                delay.append(read_l[0])
                wall.append(read_w[0])
            else:
                index = column[action]
                quality.append(write_q[index])
                delay.append(write_l[index])
                wall.append(write_w[index])
        return quality, delay, wall

    def _best_next_components_from_z(
        self,
        online_z: torch.Tensor,
        target_z: torch.Tensor,
        actions: Sequence[Action],
    ) -> tuple[float, float, float]:
        """Double DQN: online selects, target evaluates, three components."""
        if not actions:
            return 0.0, 0.0, 0.0
        online_q, online_l, online_w = self._component_values(
            self.network, online_z, actions
        )
        utilities = [
            float(q) - float(self.cfg.beta_lag) * float(l)
            - float(self.cfg.beta_wall) * float(w)
            for q, l, w in zip(online_q, online_l, online_w)
        ]
        best = max(range(len(utilities)), key=lambda index: utilities[index])
        target_q, target_l, target_w = self._component_values(
            self.target_network, target_z, actions
        )
        return (
            float(target_q[best]),
            float(target_l[best]),
            float(target_w[best]),
        )

    def _best_next_value(self, state: StateInput, actions: Sequence[Action]) -> float:
        """`max_a' [Q_target(s', a') - lambda_t * C(s', a')]` for one state."""
        if not actions:
            return 0.0
        z = self._encode_states(self.target_network, [state])
        return self._best_next_value_from_z(z, state, actions)

    def _best_next_value_from_z(
        self,
        z: torch.Tensor,
        state: StateInput,
        actions: Sequence[Action],
    ) -> float:
        """Backup value from an already-encoded one-row successor `z`."""
        if not actions:
            return 0.0
        values = self._q_values(self.target_network, z, actions)
        best = max(
            float(value) - self.cfg.lambda_t * self._cost(state, action)
            for value, action in zip(values, actions)
        )
        return best

    def _q_values(
        self,
        model: CAVCNetwork,
        z: torch.Tensor,
        actions: Sequence[Action],
    ) -> list[torch.Tensor]:
        """Per-action Q values for one encoded state, preserving `actions` order."""
        read_value = model.read_value(z)[0]
        write_actions = [action for action in actions if not action.is_read]
        write_values = None
        if write_actions:
            write_values = model.write_values(z, write_actions).value[0]
        column = {action: index for index, action in enumerate(write_actions)}

        values: list[torch.Tensor] = []
        for action in actions:
            if action.is_read:
                values.append(read_value)
            else:
                values.append(write_values[column[action]])
        return values

    def _cost(self, state: StateInput, action: Action) -> float:
        """Explicit runtime cost for `action`; zero when no model is injected.

        Training must price the Bellman backup the same way rollout does. The
        scheduled pipeline marginal wins when available: first from an
        estimate that carries ``marginal_wall_s``, then from a cost model with
        a ``marginal_wall`` method (priced from free devices). The serial
        ``wall_s`` is only a documented fallback for a plain measured model
        with no device scheduling; it is never presented as pipeline wall time.
        """
        if self.cost_model is None:
            return 0.0
        src_words = int(round(state.numeric[SRC_WORDS_INDEX]))
        tgt_words = int(round(state.numeric[TGT_WORDS_INDEX]))
        estimate = self.cost_model.predict(
            state.predictor_id,
            state.translator_id,
            action,
            src_words,
            tgt_words,
        )
        marginal = getattr(estimate, "marginal_wall_s", None)
        if marginal is None:
            marginal_wall = getattr(self.cost_model, "marginal_wall", None)
            if callable(marginal_wall):
                marginal = marginal_wall(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    state.predictor_id,
                    state.translator_id,
                    action,
                    src_words,
                    tgt_words,
                )
        if marginal is not None:
            return float(marginal)
        if not self._warned_plain_cost:
            warnings.warn(
                "CAVCTrainer was given a plain cost model without "
                "`marginal_wall`; the Bellman target falls back to the serial "
                "pred_s + mt_s diagnostic instead of pipeline wall time. Wrap "
                "the model in PipelineCostModel to price the same schedule "
                "rollout uses.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_plain_cost = True
        wall = getattr(estimate, "wall_s", estimate)
        return float(wall)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, targets: Sequence[ActionTarget]) -> dict:
        """Train on labelled pairs and return a summary of the final epoch."""
        torch.manual_seed(self.cfg.seed)
        self.history = []
        epochs = int(self.train_config.epochs)
        if not targets or epochs <= 0:
            return {
                "epochs": 0,
                "final_loss": 0.0,
                "value_loss": 0.0,
                "rank_loss": 0.0,
                "error_loss": 0.0,
                "no_emit_loss": 0.0,
                "emit_loss": 0.0,
                "n_samples": len(targets),
                # No epoch ran, so no Bellman target was computed.
                "target_seconds": 0.0,
                "target_calls": 0,
            }

        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.train_config.lr
        )
        self.network.train()
        self.target_network.eval()
        rng = random.Random(self.cfg.seed)
        sample_count = len(targets)
        update_every = max(1, int(self.cfg.target_update_every))
        target_seconds = 0.0
        target_calls = 0
        scales = self._component_scales(targets) if self.cfg.tri_value else None

        for epoch in range(epochs):
            # `compute_targets` encodes every unique successor in one batch and
            # caches the backup per `(state, actions)`, so this wall time is the
            # amortised cost of the whole epoch's Bellman targets.
            target_start = time.perf_counter()
            if self.cfg.tri_value:
                target_values = self.compute_component_targets(targets)
            else:
                target_values = self.compute_targets(targets)
            target_seconds += max(0.0, time.perf_counter() - target_start)
            target_calls += 1
            batches_of_indices = self._state_group_batches(targets, rng)

            epoch_totals = {
                "total": 0.0,
                "value": 0.0,
                "quality": 0.0,
                "delay": 0.0,
                "wall": 0.0,
                "pair": 0.0,
                "rank": 0.0,
                "error": 0.0,
                "no_emit": 0.0,
                "emit": 0.0,
            }
            batches = 0

            for indices in batches_of_indices:
                batch = [targets[index] for index in indices]
                batch_targets = [target_values[index] for index in indices]

                if self.cfg.tri_value:
                    loss, parts = self._batch_loss_v4(
                        batch, batch_targets, scales=scales
                    )
                else:
                    loss, parts = self._batch_loss(batch, batch_targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                if self.cfg.tri_value and float(self.cfg.target_tau) > 0.0:
                    self._soft_update(float(self.cfg.target_tau))

                epoch_totals["total"] += float(loss.detach())
                for name in (
                    "value",
                    "quality",
                    "delay",
                    "wall",
                    "pair",
                    "rank",
                    "error",
                    "no_emit",
                    "emit",
                ):
                    epoch_totals[name] += float(parts.get(name, 0.0))
                batches += 1

            for name in epoch_totals:
                epoch_totals[name] /= max(1, batches)
            self.history.append(epoch_totals)

            if not (self.cfg.tri_value and float(self.cfg.target_tau) > 0.0):
                if (epoch + 1) % update_every == 0:
                    self._sync_target()

        final = self.history[-1]
        summary = {
            # Report the epochs that actually ran rather than echoing the
            # configured value; the zero-epoch path returns 0 above.
            "epochs": len(self.history),
            "final_loss": final["total"],
            "value_loss": final["value"],
            "rank_loss": final["rank"],
            "error_loss": final["error"],
            "no_emit_loss": final["no_emit"],
            "emit_loss": final["emit"],
            "n_samples": sample_count,
            # Wall time spent building Bellman targets (unique-successor
            # batching + per-state cache), logged in the dev summary.
            "target_seconds": float(target_seconds),
            "target_calls": int(target_calls),
        }
        if self.cfg.tri_value:
            summary.update(
                {
                    "quality_loss": final.get("quality", 0.0),
                    "delay_loss": final.get("delay", 0.0),
                    "wall_loss": final.get("wall", 0.0),
                    "pair_loss": final.get("pair", 0.0),
                }
            )
        return summary

    def _state_group_batches(
        self,
        targets: Sequence[ActionTarget],
        rng: random.Random,
    ) -> list[list[int]]:
        """Shuffle whole state groups and assemble complete-group batches.

        The pairwise ranking loss only pairs rows that share a batch, and
        production targets are contiguous runs of ``1 + len(cfg.actions)`` rows
        per state. Shuffling individual rows and slicing fixed-size batches
        therefore leaves almost no same-state pairs (the loss is inert). This
        groups rows by `state_key` first, shuffles the groups, and fills each
        batch with whole groups only. A group larger than `batch_size` becomes
        its own variable-size batch so the ranking signal is never split.

        Rows with an empty `state_key` carry no state identity and are each
        their own group; the ranking loss skips empty keys anyway, but keeping
        them singletons avoids merging unrelated rows into one "group".
        """
        batch_size = max(1, int(self.train_config.batch_size))
        grouped: dict[str, list[int]] = {}
        groups: list[list[int]] = []
        for index, target in enumerate(targets):
            key = target.state_key
            if not key:
                groups.append([index])
            else:
                grouped.setdefault(key, []).append(index)
        groups.extend(grouped.values())
        rng.shuffle(groups)

        batches: list[list[int]] = []
        current: list[int] = []
        for group in groups:
            if current and len(current) + len(group) > batch_size:
                batches.append(current)
                current = []
            current.extend(group)
            if len(current) >= batch_size:
                batches.append(current)
                current = []
        if current:
            batches.append(current)
        return batches

    def _batch_loss(
        self,
        batch: Sequence[ActionTarget],
        target_values: Sequence[float],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the weighted multi-task loss for one batch."""
        states = [target.state for target in batch]
        actions = [target.action for target in batch]
        state_keys = [target.state_key for target in batch]

        # Encode each distinct state once so actions of the same state share a
        # representation; this keeps the pairwise ranking loss coherent under
        # dropout.
        unique_states: list[StateInput] = []
        state_to_index: dict[StateInput, int] = {}
        row_to_unique: list[int] = []
        for state in states:
            if state not in state_to_index:
                state_to_index[state] = len(unique_states)
                unique_states.append(state)
            row_to_unique.append(state_to_index[state])

        z = self._encode_states(self.network, unique_states)
        unique_index = torch.tensor(row_to_unique, dtype=torch.long)
        read_values = self.network.read_value(z)

        write_actions: list[Action] = []
        for action in actions:
            if not action.is_read and action not in write_actions:
                write_actions.append(action)

        write_outputs: WriteOutputs | None = None
        column: dict[Action, int] = {}
        gathered = read_values[unique_index]
        if write_actions:
            write_outputs = self.network.write_values(z, write_actions)
            column = {action: index for index, action in enumerate(write_actions)}
            action_columns = torch.tensor(
                [column.get(action, 0) for action in actions], dtype=torch.long
            )
            gathered = write_outputs.value[unique_index, action_columns]

        write_mask = torch.tensor(
            [not action.is_read for action in actions], dtype=torch.bool
        )
        q_values = torch.where(write_mask, gathered, read_values[unique_index])
        y = torch.tensor(target_values, dtype=torch.float32)

        value_loss = F.huber_loss(q_values, y, delta=self.cfg.huber_delta)
        rank_loss = pairwise_ranking_loss(
            q_values,
            y,
            state_keys,
            self.cfg.rank_epsilon,
            write_mask=write_mask.tolist(),
        )

        zero = q_values.new_zeros(())
        error_loss = zero
        no_emit_loss = zero
        emit_loss = zero

        write_rows = [
            index for index, action in enumerate(actions) if not action.is_read
        ]
        if write_outputs is not None and write_rows:
            rows = torch.tensor(write_rows, dtype=torch.long)
            columns = torch.tensor(
                [column[actions[index]] for index in write_rows],
                dtype=torch.long,
            )
            rows_in_states = unique_index[rows]

            error_logits = write_outputs.error_risk[rows_in_states, columns]
            error_targets = torch.tensor(
                [batch[index].error_target for index in write_rows],
                dtype=error_logits.dtype,
            )
            error_loss = F.binary_cross_entropy_with_logits(error_logits, error_targets)

            no_emit_logits = write_outputs.no_emit[rows_in_states, columns]
            no_emit_targets = torch.tensor(
                [batch[index].no_emit_target for index in write_rows],
                dtype=no_emit_logits.dtype,
            )
            no_emit_loss = F.binary_cross_entropy_with_logits(
                no_emit_logits, no_emit_targets
            )

            emit_values = write_outputs.expected_emit[rows_in_states, columns]
            emit_targets = torch.tensor(
                [batch[index].emit_target for index in write_rows],
                dtype=emit_values.dtype,
            )
            emit_loss = F.huber_loss(
                emit_values, emit_targets, delta=self.cfg.huber_delta
            )

        total = (
            self.cfg.alpha_q * value_loss
            + self.cfg.alpha_rank * rank_loss
            + self.cfg.alpha_error * error_loss
            + self.cfg.alpha_noemit * no_emit_loss
            + self.cfg.alpha_emit * emit_loss
        )
        parts = {
            "value": float(value_loss.detach()),
            "rank": float(rank_loss.detach()),
            "error": float(error_loss.detach()),
            "no_emit": float(no_emit_loss.detach()),
            "emit": float(emit_loss.detach()),
        }
        return total, parts

    def _component_scales(self, targets: Sequence[ActionTarget]) -> dict[str, float]:
        """Robust per-head scales from training targets (MAD-based)."""
        import statistics

        def scale(values: Sequence[float]) -> float:
            values = [float(value) for value in values if math.isfinite(float(value))]
            if not values:
                return 1.0
            center = statistics.median(values)
            deviation = statistics.median(abs(value - center) for value in values)
            return max(1.4826 * deviation, 1e-3)

        return {
            "quality": scale(
                [
                    target.quality_target
                    if target.quality_target is not None
                    else target.value_target
                    for target in targets
                ]
            ),
            "delay": scale([target.delay_target or 0.0 for target in targets]),
            "wall": scale([target.wall_target or 0.0 for target in targets]),
        }

    def _pairwise_advantage_loss(
        self,
        utilities: torch.Tensor,
        targets: torch.Tensor,
        state_keys: Sequence[str],
        actions: Sequence[Action],
    ) -> torch.Tensor:
        """Huber regression on `U(s,a) - U(s,b4h6)` within each state group."""
        baseline = Action(
            int(self.cfg.advantage_baseline[0]),
            int(self.cfg.advantage_baseline[1]),
        )
        groups: dict[str, list[int]] = {}
        for index, key in enumerate(state_keys):
            if key:
                groups.setdefault(key, []).append(index)
        predicted: list[torch.Tensor] = []
        expected: list[torch.Tensor] = []
        for indices in groups.values():
            anchor = next(
                (
                    index
                    for index in indices
                    if actions[index] == baseline
                ),
                None,
            )
            if anchor is None:
                continue
            for index in indices:
                if index == anchor:
                    continue
                predicted.append(utilities[index] - utilities[anchor])
                expected.append(targets[index] - targets[anchor])
        zero = utilities.new_zeros(())
        if not predicted:
            return zero
        return F.huber_loss(
            torch.stack(predicted),
            torch.stack(expected),
            delta=self.cfg.huber_delta,
        )

    def _batch_loss_v4(
        self,
        batch: Sequence[ActionTarget],
        target_values: Sequence[tuple[float, float, float]],
        *,
        scales: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Three-component value loss with paired advantage and ranking."""
        states = [target.state for target in batch]
        actions = [target.action for target in batch]
        state_keys = [target.state_key for target in batch]

        unique_states: list[StateInput] = []
        state_to_index: dict[StateInput, int] = {}
        row_to_unique: list[int] = []
        for state in states:
            if state not in state_to_index:
                state_to_index[state] = len(unique_states)
                unique_states.append(state)
            row_to_unique.append(state_to_index[state])

        z = self._encode_states(self.network, unique_states)
        unique_index = torch.tensor(row_to_unique, dtype=torch.long)
        read_q, read_l, read_w = self.network.read_components(z)

        write_actions: list[Action] = []
        for action in actions:
            if not action.is_read and action not in write_actions:
                write_actions.append(action)

        write_outputs: WriteOutputs | None = None
        column: dict[Action, int] = {}
        gathered_q = read_q[unique_index]
        gathered_l = read_l[unique_index]
        gathered_w = read_w[unique_index]
        if write_actions:
            write_outputs = self.network.write_values(z, write_actions)
            column = {action: index for index, action in enumerate(write_actions)}
            action_columns = torch.tensor(
                [column.get(action, 0) for action in actions], dtype=torch.long
            )
            gathered_q = write_outputs.quality[unique_index, action_columns]
            gathered_l = write_outputs.delay[unique_index, action_columns]
            gathered_w = write_outputs.wall[unique_index, action_columns]

        write_mask = torch.tensor(
            [not action.is_read for action in actions], dtype=torch.bool
        )
        quality = torch.where(write_mask, gathered_q, read_q[unique_index])
        delay = torch.where(write_mask, gathered_l, read_l[unique_index])
        wall = torch.where(write_mask, gathered_w, read_w[unique_index])
        utility = (
            quality
            - float(self.cfg.beta_lag) * delay
            - float(self.cfg.beta_wall) * wall
        )

        y_quality = torch.tensor(
            [float(value[0]) for value in target_values], dtype=torch.float32
        )
        y_delay = torch.tensor(
            [float(value[1]) for value in target_values], dtype=torch.float32
        )
        y_wall = torch.tensor(
            [float(value[2]) for value in target_values], dtype=torch.float32
        )
        y_utility = (
            y_quality
            - float(self.cfg.beta_lag) * y_delay
            - float(self.cfg.beta_wall) * y_wall
        )

        scales = scales or {"quality": 1.0, "delay": 1.0, "wall": 1.0}
        zero = utility.new_zeros(())
        quality_loss = F.huber_loss(
            (quality - y_quality) / float(scales["quality"]),
            torch.zeros_like(quality),
            delta=self.cfg.huber_delta,
        )
        delay_loss = F.huber_loss(
            (delay - y_delay) / float(scales["delay"]),
            torch.zeros_like(delay),
            delta=self.cfg.huber_delta,
        )
        wall_loss = F.huber_loss(
            (wall - y_wall) / float(scales["wall"]),
            torch.zeros_like(wall),
            delta=self.cfg.huber_delta,
        )
        value_loss = quality_loss + delay_loss + wall_loss
        pair_loss = self._pairwise_advantage_loss(
            utility, y_utility, state_keys, actions
        )
        rank_loss = pairwise_ranking_loss(
            utility,
            y_utility,
            state_keys,
            float(self.cfg.tri_rank_epsilon),
            write_mask=write_mask.tolist(),
        )

        error_loss = zero
        no_emit_loss = zero
        emit_loss = zero
        write_rows = [
            index for index, action in enumerate(actions) if not action.is_read
        ]
        if write_outputs is not None and write_rows:
            rows = torch.tensor(write_rows, dtype=torch.long)
            columns = torch.tensor(
                [column[actions[index]] for index in write_rows],
                dtype=torch.long,
            )
            rows_in_states = unique_index[rows]
            error_logits = write_outputs.error_risk[rows_in_states, columns]
            error_targets = torch.tensor(
                [batch[index].error_target for index in write_rows],
                dtype=error_logits.dtype,
            )
            error_loss = F.binary_cross_entropy_with_logits(
                error_logits, error_targets
            )
            no_emit_logits = write_outputs.no_emit[rows_in_states, columns]
            no_emit_targets = torch.tensor(
                [batch[index].no_emit_target for index in write_rows],
                dtype=no_emit_logits.dtype,
            )
            no_emit_loss = F.binary_cross_entropy_with_logits(
                no_emit_logits, no_emit_targets
            )
            emit_values = write_outputs.expected_emit[rows_in_states, columns]
            emit_targets = torch.tensor(
                [batch[index].emit_target for index in write_rows],
                dtype=emit_values.dtype,
            )
            emit_loss = F.huber_loss(
                emit_values, emit_targets, delta=self.cfg.huber_delta
            )

        total = (
            value_loss
            + float(self.cfg.pair_weight) * pair_loss
            + self.cfg.alpha_rank * rank_loss
            + self.cfg.alpha_error * error_loss
            + self.cfg.alpha_noemit * no_emit_loss
            + self.cfg.alpha_emit * emit_loss
        )
        return total, {
            "value": float(value_loss.detach()),
            "quality": float(quality_loss.detach()),
            "delay": float(delay_loss.detach()),
            "wall": float(wall_loss.detach()),
            "pair": float(pair_loss.detach()),
            "rank": float(rank_loss.detach()),
            "error": float(error_loss.detach()),
            "no_emit": float(no_emit_loss.detach()),
            "emit": float(emit_loss.detach()),
        }

    def _soft_update(self, tau: float) -> None:
        """Polyak-average the target network toward the online network."""
        tau = float(tau)
        with torch.no_grad():
            for target, online in zip(
                self.target_network.parameters(), self.network.parameters()
            ):
                target.mul_(1.0 - tau).add_(online, alpha=tau)
        self.target_network.eval()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _encode_states(
        self,
        model: CAVCNetwork,
        states: Sequence[StateInput],
    ) -> torch.Tensor:
        if not states:
            # Keep the batch dimension so callers can index into the result.
            return torch.zeros((0, self.cfg.value_hidden), dtype=torch.float32)
        numeric = torch.tensor([state.numeric for state in states], dtype=torch.float32)
        predictor_id = torch.tensor(
            [state.predictor_id for state in states], dtype=torch.long
        )
        translator_id = torch.tensor(
            [state.translator_id for state in states], dtype=torch.long
        )
        src_ids = _pad_ids([state.src_ids for state in states])
        tgt_ids = _pad_ids([state.tgt_ids for state in states])
        return model.encode(numeric, predictor_id, translator_id, src_ids, tgt_ids)

    def _sync_target(self) -> None:
        """Copy online parameters into the frozen target network."""
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)
