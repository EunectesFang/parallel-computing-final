"""Tests for the multi-task trainer and one-step Bellman targets."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from controller_v3.action import READ, Action
from controller_v3.config import ControllerConfig
from controller_v3.features import NUMERIC_FEATURES, StateInput
from controller_v3.network import CAVCNetwork, WriteOutputs
from controller_v3.trainer import (
    ActionTarget,
    CAVCTrainer,
    TrainConfig,
    pairwise_ranking_loss,
)

ACTIONS = (Action(2, 4), Action(4, 6), Action(6, 10))
SRC_WORDS = NUMERIC_FEATURES.index("src_words")
TGT_WORDS = NUMERIC_FEATURES.index("tgt_words")


class FakeCostModel:
    """Deterministic cost model that records every query."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def predict(self, predictor_id, translator_id, action, src_words, tgt_words):
        self.calls.append((predictor_id, translator_id, action, src_words, tgt_words))
        wall = 0.25 * (1 + action.branches + action.horizon)
        return SimpleNamespace(pred_s=wall * 0.4, mt_s=wall * 0.6, wall_s=wall)


def make_state(index: int, src_words: int = 5, tgt_words: int = 3) -> StateInput:
    numeric = [0.0] * len(NUMERIC_FEATURES)
    numeric[SRC_WORDS] = float(src_words)
    numeric[TGT_WORDS] = float(tgt_words)
    numeric[1] = float(index)  # src_chars keeps otherwise identical states distinct
    return StateInput(
        numeric=tuple(numeric),
        predictor_id=index % 8,
        translator_id=(index + 1) % 8,
        src_ids=(2, 3),
        tgt_ids=(4,),
    )


def make_targets(*, terminal: bool = False, next_state: bool = True):
    """Three states x three write actions."""
    targets: list[ActionTarget] = []
    for state_index in range(3):
        state = make_state(state_index)
        successor = make_state(state_index, src_words=6, tgt_words=4) if next_state else None
        for action_index, action in enumerate(ACTIONS):
            targets.append(
                ActionTarget(
                    state=state,
                    action=action,
                    value_target=float(action_index) + 0.1 * state_index,
                    error_target=float(action_index % 2),
                    no_emit_target=float((action_index + 1) % 2),
                    emit_target=float(action_index + 1),
                    state_key=f"s{state_index}",
                    next_state=successor,
                    next_actions=ACTIONS,
                    terminal=terminal,
                )
            )
    return targets


def make_trainer(**cfg_overrides):
    cfg = replace(ControllerConfig(), **cfg_overrides)
    torch.manual_seed(cfg.seed)
    network = CAVCNetwork(cfg, src_vocab_size=30, tgt_vocab_size=30)
    return CAVCTrainer(network, cfg, FakeCostModel()), cfg, network


def test_terminal_target_is_the_immediate_reward_even_with_a_next_state():
    trainer, _, _ = make_trainer()
    targets = make_targets(terminal=True)
    assert trainer.compute_targets(targets) == [target.value_target for target in targets]


def test_missing_next_state_uses_the_immediate_reward():
    trainer, _, _ = make_trainer()
    targets = make_targets(next_state=False)
    assert trainer.compute_targets(targets) == [target.value_target for target in targets]


def test_non_terminal_target_adds_discounted_next_state_value():
    trainer, cfg, _ = make_trainer()
    targets = make_targets()
    computed = trainer.compute_targets(targets)

    for target, actual in zip(targets, computed):
        next_state = target.next_state
        assert next_state is not None
        with torch.no_grad():
            numeric = torch.tensor([next_state.numeric], dtype=torch.float32)
            predictor = torch.tensor([next_state.predictor_id])
            translator = torch.tensor([next_state.translator_id])
            src_ids = torch.tensor([next_state.src_ids], dtype=torch.long)
            tgt_ids = torch.tensor([next_state.tgt_ids], dtype=torch.long)
            z = trainer.target_network.encode(numeric, predictor, translator, src_ids, tgt_ids)
            read_value = float(trainer.target_network.read_value(z)[0])
            write_values = trainer.target_network.write_values(z, ACTIONS).value[0]

        best = max(
            float(write_values[index])
            - cfg.lambda_t * trainer.cost_model.predict(
                next_state.predictor_id,
                next_state.translator_id,
                action,
                int(next_state.numeric[SRC_WORDS]),
                int(next_state.numeric[TGT_WORDS]),
            ).wall_s
            for index, action in enumerate(ACTIONS)
        )
        expected = target.value_target + cfg.gamma * best
        assert actual == pytest.approx(expected, rel=1e-5, abs=1e-5)
        # READ is not in the candidate set here, but the head still exists.
        assert torch.isfinite(torch.tensor(read_value))


def _stub_heads(monkeypatch, trainer, *, read_value: float, write_value: float):
    """Force the target network's READ head above (or below) every write head."""

    def fake_read_value(z):
        return torch.full((z.size(0),), read_value)

    def fake_write_values(z, actions):
        batch, count = z.size(0), len(actions)
        zeros = torch.zeros(batch, count)
        return WriteOutputs(
            value=torch.full((batch, count), write_value),
            error_risk=zeros,
            no_emit=zeros,
            expected_emit=zeros + 0.5,
        )

    monkeypatch.setattr(trainer.target_network, "read_value", fake_read_value)
    monkeypatch.setattr(trainer.target_network, "write_values", fake_write_values)


def _bootstrap_target(next_actions=()):
    return ActionTarget(
        state=make_state(0),
        action=ACTIONS[0],
        value_target=0.0,
        error_target=0.0,
        no_emit_target=0.0,
        emit_target=0.0,
        state_key="s0",
        next_state=make_state(1, src_words=6, tgt_words=4),
        next_actions=next_actions,
        terminal=False,
    )


def test_default_bootstrap_candidates_include_read(monkeypatch):
    trainer, cfg, _ = make_trainer()
    _stub_heads(monkeypatch, trainer, read_value=10.0, write_value=-10.0)

    computed = trainer.compute_targets([_bootstrap_target()])

    # FakeCostModel charges 0.25 for READ; every write costs strictly more.
    expected = cfg.gamma * (10.0 - cfg.lambda_t * 0.25)
    assert computed[0] == pytest.approx(expected)


def test_explicit_next_actions_override_the_default_candidate_set(monkeypatch):
    trainer, cfg, _ = make_trainer()
    _stub_heads(monkeypatch, trainer, read_value=10.0, write_value=1.0)

    computed = trainer.compute_targets([_bootstrap_target(next_actions=(ACTIONS[0],))])

    # READ is available in the backup set but NOT when next_actions is supplied.
    write_cost = 0.25 * (1 + ACTIONS[0].branches + ACTIONS[0].horizon)
    expected = cfg.gamma * (1.0 - cfg.lambda_t * write_cost)
    assert computed[0] == pytest.approx(expected)


def test_encode_states_with_no_states_returns_an_empty_batch():
    trainer, cfg, network = make_trainer()
    encoded = trainer._encode_states(network, [])
    assert encoded.shape == (0, cfg.value_hidden)


def test_cost_model_is_queried_with_src_and_tgt_words():
    trainer, _, _ = make_trainer()
    targets = make_targets()
    trainer.compute_targets(targets)
    assert trainer.cost_model.calls
    for _, _, _, src_words, tgt_words in trainer.cost_model.calls:
        assert src_words == 6
        assert tgt_words == 4


def test_ranking_loss_ignores_pairs_within_epsilon():
    predictions = torch.tensor([0.3, 0.1], requires_grad=True)
    targets = torch.tensor([1.0, 1.02])
    loss = pairwise_ranking_loss(
        predictions, targets, ["s", "s"], epsilon=0.05
    )
    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert torch.isfinite(predictions.grad).all()


def test_ranking_loss_orders_pairs_beyond_epsilon():
    predictions = torch.tensor([0.0, 0.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.0])
    loss = pairwise_ranking_loss(predictions, targets, ["s", "s"], epsilon=0.05)
    # The pair (high index 0, low index 1) should push q0 above q1.
    loss.backward()
    assert predictions.grad[0] < 0
    assert predictions.grad[1] > 0


def test_ranking_loss_caps_pairs_per_state():
    count = 40
    predictions = torch.randn(count, requires_grad=True)
    targets = torch.arange(count, dtype=torch.float32)
    loss = pairwise_ranking_loss(
        predictions, targets, ["one-state"] * count, epsilon=0.05, max_pairs=32
    )
    # The first 32 pairs in nested order are (0, 1) ... (0, 32); with targets
    # increasing, each pair ranks the later index above index 0.
    expected_terms = [
        torch.nn.functional.softplus(-(predictions[j] - predictions[0]))
        for j in range(1, 33)
    ]
    assert loss.item() == pytest.approx(torch.stack(expected_terms).mean().item())


def test_ranking_loss_skips_empty_state_keys():
    predictions = torch.tensor([1.0, 0.0], requires_grad=True)
    targets = torch.tensor([0.0, 1.0])
    loss = pairwise_ranking_loss(predictions, targets, ["", ""], epsilon=0.05)
    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert torch.isfinite(predictions.grad).all()


def test_ranking_loss_skips_read_rows():
    predictions = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    # Only the READ row (index 1) differs by more than epsilon.
    targets = torch.tensor([0.0, 5.0, 0.01])
    loss = pairwise_ranking_loss(
        predictions,
        targets,
        ["s", "s", "s"],
        epsilon=0.05,
        write_mask=[True, False, True],
    )
    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert torch.isfinite(predictions.grad).all()


def test_fit_is_finite_when_state_keys_are_empty():
    trainer, _, _ = make_trainer(dropout=0.0)
    trainer.train_config = TrainConfig(epochs=3, batch_size=9, lr=0.01)
    targets = [
        replace(target, state_key="") for target in make_targets()
    ]
    result = trainer.fit(targets)
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
    assert result["rank_loss"] == 0.0


def test_fit_reduces_the_total_loss_over_thirty_epochs():
    trainer, cfg, network = make_trainer(target_update_every=1000, dropout=0.0, seed=3)
    trainer.train_config = TrainConfig(epochs=30, batch_size=9, lr=0.05)
    result = trainer.fit(make_targets())

    assert result["epochs"] == 30
    assert result["n_samples"] == 9
    assert set(result) == {
        "epochs",
        "final_loss",
        "value_loss",
        "rank_loss",
        "error_loss",
        "no_emit_loss",
        "emit_loss",
        "n_samples",
    }
    assert len(trainer.history) == 30
    for stats in trainer.history:
        assert all(torch.isfinite(torch.tensor(value)) for value in stats.values())
    assert trainer.history[-1]["total"] < trainer.history[0]["total"]
    assert result["final_loss"] == pytest.approx(trainer.history[-1]["total"])


def test_fit_is_finite_with_only_a_read_target():
    trainer, cfg, _ = make_trainer(dropout=0.0)
    trainer.train_config = TrainConfig(epochs=3, batch_size=4, lr=0.01)
    targets = [
        ActionTarget(
            state=make_state(0),
            action=READ,
            value_target=0.5,
            error_target=0.0,
            no_emit_target=0.0,
            emit_target=0.0,
            state_key="only-read",
            terminal=True,
        )
    ]
    result = trainer.fit(targets)
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
    assert result["rank_loss"] == 0.0
    assert result["error_loss"] == 0.0
    assert result["no_emit_loss"] == 0.0
    assert result["emit_loss"] == 0.0


def test_fit_is_finite_with_a_single_action_state():
    trainer, _, _ = make_trainer(dropout=0.0)
    trainer.train_config = TrainConfig(epochs=3, batch_size=4, lr=0.01)
    targets = [
        ActionTarget(
            state=make_state(0),
            action=ACTIONS[0],
            value_target=1.0,
            error_target=0.0,
            no_emit_target=1.0,
            emit_target=2.0,
            state_key="single",
            terminal=True,
        )
    ]
    result = trainer.fit(targets)
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
    assert result["rank_loss"] == 0.0


def test_target_network_is_frozen_and_updates_after_an_epoch():
    trainer, _, network = make_trainer(target_update_every=1, dropout=0.0)
    assert all(not parameter.requires_grad for parameter in trainer.target_network.parameters())
    before = [parameter.detach().clone() for parameter in trainer.target_network.parameters()]

    trainer.train_config = TrainConfig(epochs=1, batch_size=9, lr=0.05)
    trainer.fit(make_targets())

    after = [parameter.detach().clone() for parameter in trainer.target_network.parameters()]
    assert any(not torch.equal(old, new) for old, new in zip(before, after))
    for target_parameter, online_parameter in zip(
        trainer.target_network.parameters(), network.parameters()
    ):
        assert torch.equal(target_parameter, online_parameter)
