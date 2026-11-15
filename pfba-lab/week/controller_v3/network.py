"""Action-conditioned value network for the cost-aware controller.

The network encodes a state once, reads a scalar READ value from that
representation, and scores every write action through an action encoder. READ
and write values share one scale so the controller can compare them directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .action import Action, action_features
from .config import ControllerConfig
from .text_encoder import TextEncoder

__all__ = ["WriteOutputs", "CAVCNetwork"]

# Two model families (predictor and translator) share an 8-row embedding table.
MODEL_EMBEDDING_ROWS = 8


@dataclass
class WriteOutputs:
    """Per-write-action predictions for a batch of states."""

    value: torch.Tensor          # [B, A]
    error_risk: torch.Tensor     # [B, A] logits
    no_emit: torch.Tensor        # [B, A] logits
    expected_emit: torch.Tensor  # [B, A] non-negative softplus output


class CAVCNetwork(nn.Module):
    """State encoder with a READ head and action-conditioned write heads."""

    def __init__(
        self,
        cfg: ControllerConfig,
        src_vocab_size: int,
        tgt_vocab_size: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = cfg.value_hidden
        text_dim = 2 * cfg.text_hidden
        model_dim = cfg.model_emb

        self.numeric_encoder = nn.Sequential(
            nn.Linear(cfg.numeric_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.text_encoder = TextEncoder(
            src_vocab_size,
            tgt_vocab_size,
            emb=cfg.text_emb,
            hidden=cfg.text_hidden,
        )
        self.predictor_embedding = nn.Embedding(MODEL_EMBEDDING_ROWS, model_dim)
        self.translator_embedding = nn.Embedding(MODEL_EMBEDDING_ROWS, model_dim)

        fused_dim = hidden + text_dim + 2 * model_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, hidden),
            nn.SiLU(),
        )

        self.read_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        # A write action is described by six numbers and encoded into 16 units.
        self.action_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
        )
        self.write_head = nn.Sequential(
            nn.Linear(hidden + 16, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 32),
            nn.SiLU(),
        )
        self.value_out = nn.Linear(32, 1)
        self.error_out = nn.Linear(32, 1)
        self.no_emit_out = nn.Linear(32, 1)
        self.emit_out = nn.Linear(32, 1)

    def encode(
        self,
        numeric: torch.Tensor,
        predictor_id: torch.Tensor,
        translator_id: torch.Tensor,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return a `[B, value_hidden]` state representation.

        Text windows may be empty (`[B, 0]`); `TextEncoder` maps those rows to
        zero vectors. Model ids outside `[0, 7]` are clamped so a stray id can
        never index past the embedding table.
        """
        dtype = self.numeric_encoder[0].weight.dtype
        numeric = numeric.to(dtype=dtype)
        src_ids = src_ids.long()
        tgt_ids = tgt_ids.long()

        numeric_vector = self.numeric_encoder(numeric)
        text_vector = self.text_encoder(src_ids, tgt_ids)

        predictor_id = predictor_id.long().clamp(0, MODEL_EMBEDDING_ROWS - 1)
        translator_id = translator_id.long().clamp(0, MODEL_EMBEDDING_ROWS - 1)
        predictor_vector = self.predictor_embedding(predictor_id)
        translator_vector = self.translator_embedding(translator_id)

        fused = torch.cat(
            [numeric_vector, text_vector, predictor_vector, translator_vector],
            dim=-1,
        )
        return self.fusion(fused)

    def read_value(self, z: torch.Tensor) -> torch.Tensor:
        """Return the scalar READ value for each row of `z` as `[B]`."""
        return self.read_head(z).squeeze(-1)

    def write_values(self, z: torch.Tensor, actions: Sequence[Action]) -> WriteOutputs:
        """Score each write action for every row of `z`.

        `actions` may be empty, in which case every output has shape `[B, 0]`.
        """
        actions = list(actions)
        batch = z.size(0)
        if not actions:
            empty = z.new_zeros(batch, 0)
            return WriteOutputs(empty, empty, empty, empty)

        features = torch.tensor(
            [action_features(action, self.cfg.ralcp_threshold) for action in actions],
            dtype=z.dtype,
            device=z.device,
        )  # [A, 6]
        action_vector = self.action_encoder(features)  # [A, 16]

        repeated_z = z.unsqueeze(1).expand(batch, len(actions), z.size(-1))
        repeated_action = action_vector.unsqueeze(0).expand(batch, len(actions), -1)
        combined = torch.cat([repeated_z, repeated_action], dim=-1)
        hidden = self.write_head(combined.reshape(batch * len(actions), -1))
        hidden = hidden.reshape(batch, len(actions), -1)

        return WriteOutputs(
            value=self.value_out(hidden).squeeze(-1),
            error_risk=self.error_out(hidden).squeeze(-1),
            no_emit=self.no_emit_out(hidden).squeeze(-1),
            expected_emit=F.softplus(self.emit_out(hidden)).squeeze(-1),
        )

    def forward(
        self,
        numeric: torch.Tensor,
        predictor_id: torch.Tensor,
        translator_id: torch.Tensor,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        actions: Sequence[Action],
    ) -> tuple[torch.Tensor, WriteOutputs]:
        """Encode once, then return READ values and write-action outputs."""
        z = self.encode(numeric, predictor_id, translator_id, src_ids, tgt_ids)
        return self.read_value(z), self.write_values(z, actions)
