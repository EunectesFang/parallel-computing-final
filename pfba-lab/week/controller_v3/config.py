"""Frozen configuration shared by every controller_v3 module."""

from __future__ import annotations

from dataclasses import dataclass

from .action import DEFAULT_ACTIONS, Action

__all__ = ["ControllerConfig"]


@dataclass(frozen=True)
class ControllerConfig:
    """Hyper-parameters for features, network, training, and rollout.

    All defaults are immutable values so instances are safe to share. Action
    tuples are immutable, so `DEFAULT_ACTIONS` is a valid default object.
    """

    seed: int = 13
    min_start: int = 3
    arrival_gap: float = 0.3
    ralcp_threshold: float = 0.6
    gamma: float = 0.99
    lambda_l: float = 0.05
    lambda_t: float = 0.10
    weight_m: float = 1.0
    weight_e: float = 2.0
    advantage_baseline: tuple[int, int] = (4, 6)
    actions: tuple[Action, ...] = DEFAULT_ACTIONS
    numeric_dim: int = 34
    text_emb: int = 32
    text_hidden: int = 32
    text_window_src: int = 16
    text_window_tgt: int = 16
    model_emb: int = 8
    value_hidden: int = 64
    dropout: float = 0.10
    lr: float = 3e-4
    batch_size: int = 128
    epochs: int = 20
    huber_delta: float = 1.0
    rank_epsilon: float = 0.05
    alpha_q: float = 1.0
    alpha_rank: float = 0.5
    alpha_error: float = 0.5
    alpha_noemit: float = 0.5
    alpha_emit: float = 0.5
    target_update_every: int = 1
    max_branches: int = 10
    max_horizon: int = 10
