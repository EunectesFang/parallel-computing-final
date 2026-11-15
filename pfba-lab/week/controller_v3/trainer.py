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
import random
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

    # ------------------------------------------------------------------
    # Target computation
    # ------------------------------------------------------------------
    def compute_targets(self, targets: Sequence[ActionTarget]) -> list[float]:
        """Return one-step Bellman targets in the same order as `targets`."""
        self.target_network.eval()
        results: list[float] = []
        with torch.no_grad():
            for target in targets:
                if target.next_state is None or target.terminal:
                    results.append(float(target.value_target))
                    continue

                # READ is a first-class action, so the default backup set must
                # include it; omitting READ biases the target toward writing.
                actions = tuple(target.next_actions) or (READ, *self.cfg.actions)
                bonus = self._best_next_value(target.next_state, actions)
                results.append(float(target.value_target) + self.cfg.gamma * bonus)
        return results

    def _best_next_value(self, state: StateInput, actions: Sequence[Action]) -> float:
        """`max_a' [Q_target(s', a') - lambda_t * C(s', a')]` for one state."""
        if not actions:
            return 0.0
        z = self._encode_states(self.target_network, [state])
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
        """Explicit runtime cost for `action`; zero when no model is injected."""
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
            }

        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.train_config.lr
        )
        self.network.train()
        self.target_network.eval()
        rng = random.Random(self.cfg.seed)
        batch_size = max(1, int(self.train_config.batch_size))
        sample_count = len(targets)
        update_every = max(1, int(self.cfg.target_update_every))

        for epoch in range(epochs):
            target_values = self.compute_targets(targets)
            order = list(range(sample_count))
            rng.shuffle(order)

            epoch_totals = {
                "total": 0.0,
                "value": 0.0,
                "rank": 0.0,
                "error": 0.0,
                "no_emit": 0.0,
                "emit": 0.0,
            }
            batches = 0

            for start in range(0, sample_count, batch_size):
                indices = order[start : start + batch_size]
                batch = [targets[index] for index in indices]
                batch_targets = [target_values[index] for index in indices]

                loss, parts = self._batch_loss(batch, batch_targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()

                epoch_totals["total"] += float(loss.detach())
                for name in ("value", "rank", "error", "no_emit", "emit"):
                    epoch_totals[name] += float(parts[name])
                batches += 1

            for name in epoch_totals:
                epoch_totals[name] /= max(1, batches)
            self.history.append(epoch_totals)

            if (epoch + 1) % update_every == 0:
                self._sync_target()

        final = self.history[-1]
        return {
            "epochs": epochs,
            "final_loss": final["total"],
            "value_loss": final["value"],
            "rank_loss": final["rank"],
            "error_loss": final["error"],
            "no_emit_loss": final["no_emit"],
            "emit_loss": final["emit"],
            "n_samples": sample_count,
        }

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
