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
    numeric_dim: int = 22
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
    # CAVC-Q V4: three-component value decomposition. Defaults reproduce V3
    # exactly (single scalar Q, hard target sync, no paired-advantage loss).
    tri_value: bool = False
    beta_lag: float = 0.25
    beta_wall: float = 0.30
    lag_anchor: float = 3.0
    convex_kappa: float = 0.5
    double_dqn: bool = False
    target_tau: float = 0.0
    pair_weight: float = 0.0
    tri_rank_epsilon: float = 0.10
    normalize_value_loss: bool = True
    # V5 fixes: non-negative cost heads, per-head discount, EOS terminal cost,
    # and input standardisation using training-set statistics.
    gamma_delay: float = 1.0
    gamma_wall: float = 1.0
    terminal_eos_wall_s: float = 1.0
    feature_normalize: bool = True
    # V3.1 inference-time guards. Every default is the V3 behaviour, so an
    # existing checkpoint reproduces the original argmax exactly.
    #
    # READ is admitted only when its advantage over the best admitted WRITE
    # reaches ``read_threshold + lag/streak terms``. ``read_threshold_reads``
    # applies to consecutive READs; ``read_threshold_no_commit`` applies to
    # consecutive failed WRITEs (the two event sets are disjoint, so a state
    # does not pay twice for one wait).
    read_threshold: float = 0.0
    read_threshold_lag: float = 0.0
    read_threshold_reads: float = 0.0
    read_threshold_no_commit: float = 0.0
    read_threshold_lag_ref: float = 3.0
    read_threshold_lag_scale: float = 2.2
    # A WRITE costlier than the advantage baseline must beat it by this margin
    # of predicted net utility; near-ties keep the cheaper baseline.
    expensive_margin: float = 0.0
    # Predicted no-emit probability above which the best WRITE is vetoed in
    # favour of READ. 1.0 disables the veto. Prefer the soft penalty below for
    # production: a hard veto can suppress every WRITE at a state.
    no_emit_veto: float = 1.0
    # Soft no-emit penalty subtracted from a WRITE's net utility:
    # ``U(a) -= no_emit_penalty * P(no_emit | s, a)``. Zero disables it.
    no_emit_penalty: float = 0.0
