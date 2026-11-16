"""CAVC-Q V4: three-component targets, Double DQN, and wall-clock accounting."""

from __future__ import annotations

import math

import pytest
import torch

from controller_v3.action import READ, Action
from controller_v3.config import ControllerConfig
from controller_v3.features import NUMERIC_FEATURES, StateInput, advance_virtual_clock
from controller_v3.labeler import ActionOutcome
from controller_v3.network import CAVCNetwork
from controller_v3.targets import build_action_targets
from controller_v3.trainer import ActionTarget, CAVCTrainer


def _state(src_words: int = 9, tgt_words: int = 0) -> StateInput:
    numeric = [0.0] * len(NUMERIC_FEATURES)
    numeric[NUMERIC_FEATURES.index("src_words")] = float(src_words)
    numeric[NUMERIC_FEATURES.index("tgt_words")] = float(tgt_words)
    return StateInput(
        numeric=tuple(numeric),
        predictor_id=0,
        translator_id=0,
        src_ids=(),
        tgt_ids=(),
    )


def _outcome(action: Action, *, emitted: int = 0, matched: int = 0) -> ActionOutcome:
    return ActionOutcome(
        action=action,
        new_words=tuple("w" for _ in range(emitted)),
        matched=matched,
        incorrect=0,
        emitted=emitted,
        no_emit=emitted == 0,
        next_committed=tuple("w" for _ in range(emitted)),
        pred_s=0.0,
        mt_s=0.0,
        vote_top1=0.0,
        vote_margin=0.0,
        vote_entropy=0.0,
        marginal_wall_s=0.0,
    )


def test_read_wall_delta_is_zero_when_the_next_word_already_arrived():
    now_s = 10.0
    assert advance_virtual_clock(now_s, 3, 0.3, 0.0) == now_s


def test_read_wall_delta_waits_for_the_next_word():
    assert advance_virtual_clock(0.0, 3, 0.3, 0.0) == pytest.approx(0.6)
    assert advance_virtual_clock(0.0, 3, 0.3, 0.5) == pytest.approx(1.1)


def test_v4_delay_target_uses_the_convex_backlog_penalty():
    cfg = ControllerConfig(
        tri_value=True, lag_anchor=3.0, convex_kappa=0.5, weight_e=3.0
    )
    targets = build_action_targets(
        _state(),
        [_outcome(READ)],
        cfg,
        src_words_after=10,
        tgt_words_after_per_outcome=[0],
        wall_targets=[0.3],
        rho=1.0,
    )
    target = targets[0]
    assert target.quality_target == 0.0
    # backlog=10, excess=7, 10 + 0.5 * 7^2 = 34.5
    assert target.delay_target == pytest.approx(34.5)
    assert target.wall_target == pytest.approx(0.3)


def test_v4_linear_delay_target_matches_the_linear_backlog():
    cfg = ControllerConfig(
        tri_value=True, lag_anchor=3.0, convex_kappa=0.0, weight_e=3.0
    )
    target = build_action_targets(
        _state(),
        [_outcome(READ)],
        cfg,
        src_words_after=5,
        tgt_words_after_per_outcome=[0],
        wall_targets=[0.0],
        rho=1.0,
    )[0]
    assert target.delay_target == pytest.approx(5.0)


def test_v4_quality_target_separates_quality_from_lag():
    cfg = ControllerConfig(
        tri_value=True, weight_m=1.0, weight_e=3.0, convex_kappa=0.5
    )
    target = build_action_targets(
        _state(),
        [_outcome(Action(2, 4), emitted=2, matched=2)],
        cfg,
        src_words_after=4,
        tgt_words_after_per_outcome=[2],
        wall_targets=[0.7],
        rho=1.0,
    )[0]
    assert target.quality_target == pytest.approx(2.0)
    # backlog = 4 - 2 = 2, below the anchor, so the convex term is zero.
    assert target.delay_target == pytest.approx(2.0)


def test_v5_terminal_quality_is_immediate_and_wall_includes_eos():
    cfg = ControllerConfig(tri_value=True, terminal_eos_wall_s=0.8, weight_e=3.0)
    target = build_action_targets(
        _state(),
        [_outcome(Action(2, 4), emitted=2, matched=2)],
        cfg,
        src_words_after=10,
        tgt_words_after_per_outcome=[2],
        wall_targets=[0.5],
        terminal=True,
        state_key="terminal",
        rho=1.0,
    )[0]
    assert target.quality_target == pytest.approx(2.0)
    assert target.wall_target == pytest.approx(1.3)


def test_v5_cost_heads_are_non_negative():
    cfg = ControllerConfig(tri_value=True)
    network = CAVCNetwork(cfg, src_vocab_size=8, tgt_vocab_size=8)
    z = network.encode(
        torch.zeros(1, cfg.numeric_dim),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, 0, dtype=torch.long),
        torch.zeros(1, 0, dtype=torch.long),
    )
    _quality, delay, wall = network.read_components(z)
    assert float(delay.min()) >= 0.0
    assert float(wall.min()) >= 0.0
    outputs = network.write_values(z, [Action(2, 4), Action(6, 10)])
    assert float(outputs.delay.min()) >= 0.0
    assert float(outputs.wall.min()) >= 0.0


def test_three_component_network_shapes():
    cfg = ControllerConfig(tri_value=True)
    network = CAVCNetwork(cfg, src_vocab_size=8, tgt_vocab_size=8)
    batch = 2
    z = network.encode(
        torch.zeros(batch, cfg.numeric_dim),
        torch.zeros(batch, dtype=torch.long),
        torch.zeros(batch, dtype=torch.long),
        torch.zeros(batch, 0, dtype=torch.long),
        torch.zeros(batch, 0, dtype=torch.long),
    )
    quality, delay, wall = network.read_components(z)
    assert quality.shape == (batch,)
    assert delay.shape == (batch,)
    assert wall.shape == (batch,)
    outputs = network.write_values(z, [Action(2, 4), Action(4, 6)])
    assert outputs.quality.shape == (batch, 2)
    assert outputs.delay.shape == (batch, 2)
    assert outputs.wall.shape == (batch, 2)
    assert outputs.value is outputs.quality


def test_terminal_component_target_has_no_bootstrap_and_no_cost_recharge():
    cfg = ControllerConfig(tri_value=True, gamma=0.99, beta_lag=0.25, beta_wall=0.3)
    network = CAVCNetwork(cfg, src_vocab_size=8, tgt_vocab_size=8)
    trainer = CAVCTrainer(network, cfg, None)
    target = ActionTarget(
        state=_state(),
        action=READ,
        value_target=0.0,
        error_target=0.0,
        no_emit_target=0.0,
        emit_target=0.0,
        state_key="state",
        terminal=True,
        quality_target=1.0,
        delay_target=2.0,
        wall_target=0.5,
    )
    assert trainer.compute_component_targets([target]) == [(1.0, 2.0, 0.5)]


def test_nonterminal_component_target_bootstraps_all_three_heads():
    cfg = ControllerConfig(tri_value=True, gamma=0.99, beta_lag=0.25, beta_wall=0.3)
    network = CAVCNetwork(cfg, src_vocab_size=8, tgt_vocab_size=8)
    trainer = CAVCTrainer(network, cfg, None)
    target = ActionTarget(
        state=_state(),
        action=READ,
        value_target=0.0,
        error_target=0.0,
        no_emit_target=0.0,
        emit_target=0.0,
        state_key="state",
        next_state=_state(),
        next_actions=(READ, *cfg.actions),
        quality_target=1.0,
        delay_target=2.0,
        wall_target=0.3,
    )
    (quality, delay, wall), = trainer.compute_component_targets([target])
    assert all(math.isfinite(value) for value in (quality, delay, wall))
