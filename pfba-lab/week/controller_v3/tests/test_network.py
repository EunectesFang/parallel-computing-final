"""Tests for the CAVC value network contract."""

import torch

from controller_v3.action import DEFAULT_ACTIONS
from controller_v3.config import ControllerConfig
from controller_v3.features import NUMERIC_FEATURES
from controller_v3.network import CAVCNetwork, WriteOutputs


def make_inputs(batch: int, src_len: int = 5, tgt_len: int = 4):
    """Small CPU batch with in-range model ids and non-empty text windows."""
    numeric = torch.randn(batch, len(NUMERIC_FEATURES))
    predictor_id = torch.arange(batch) % 8
    translator_id = torch.arange(batch) % 8
    src_ids = torch.randint(2, 20, (batch, src_len))
    tgt_ids = torch.randint(2, 20, (batch, tgt_len))
    return numeric, predictor_id, translator_id, src_ids, tgt_ids


def test_forward_shapes_for_batch_three_and_nine_actions():
    cfg = ControllerConfig()
    network = CAVCNetwork(cfg, src_vocab_size=30, tgt_vocab_size=30)
    read, write = network(*make_inputs(3), DEFAULT_ACTIONS)

    assert read.shape == (3,)
    assert isinstance(write, WriteOutputs)
    assert write.value.shape == (3, 9)
    for tensor in (write.value, write.error_risk, write.no_emit, write.expected_emit):
        assert tensor.shape == (3, 9)
        assert torch.isfinite(tensor).all()


def test_empty_text_ids_have_shape_batch_zero_length():
    cfg = ControllerConfig()
    network = CAVCNetwork(cfg, src_vocab_size=30, tgt_vocab_size=30)
    numeric = torch.randn(3, len(NUMERIC_FEATURES))
    predictor_id = torch.tensor([0, 1, 2])
    translator_id = torch.tensor([2, 1, 0])
    src_ids = torch.zeros((3, 0), dtype=torch.long)
    tgt_ids = torch.zeros((3, 0), dtype=torch.long)

    read, write = network(numeric, predictor_id, translator_id, src_ids, tgt_ids, DEFAULT_ACTIONS)
    assert read.shape == (3,)
    assert write.value.shape == (3, 9)
    assert torch.isfinite(read).all()
    assert torch.isfinite(write.value).all()


def test_one_empty_text_side_still_works():
    cfg = ControllerConfig()
    network = CAVCNetwork(cfg, src_vocab_size=30, tgt_vocab_size=30)
    numeric = torch.randn(2, len(NUMERIC_FEATURES))
    predictor_id = torch.tensor([0, 1])
    translator_id = torch.tensor([1, 0])
    src_ids = torch.zeros((2, 0), dtype=torch.long)
    tgt_ids = torch.randint(2, 20, (2, 4))

    read, write = network(numeric, predictor_id, translator_id, src_ids, tgt_ids, DEFAULT_ACTIONS)
    assert torch.isfinite(read).all()
    assert torch.isfinite(write.value).all()


def test_expected_emit_is_non_negative():
    cfg = ControllerConfig()
    network = CAVCNetwork(cfg, src_vocab_size=30, tgt_vocab_size=30)
    _, write = network(*make_inputs(4), DEFAULT_ACTIONS)
    assert (write.expected_emit >= 0).all()


def test_model_ids_outside_the_embedding_table_are_clamped():
    cfg = ControllerConfig()
    network = CAVCNetwork(cfg, src_vocab_size=30, tgt_vocab_size=30)
    network.eval()
    numeric = torch.randn(3, len(NUMERIC_FEATURES))
    src_ids = torch.randint(2, 20, (3, 4))
    tgt_ids = torch.randint(2, 20, (3, 3))

    out_of_range = torch.tensor([-5, 7, 99])
    explicitly_clamped = torch.tensor([0, 7, 7])
    z_raw = network.encode(numeric, out_of_range, out_of_range, src_ids, tgt_ids)
    z_clamped = network.encode(
        numeric, explicitly_clamped, explicitly_clamped, src_ids, tgt_ids
    )
    assert torch.allclose(z_raw, z_clamped)


def test_every_parameter_receives_a_gradient():
    cfg = ControllerConfig()
    network = CAVCNetwork(cfg, src_vocab_size=30, tgt_vocab_size=30)
    network.train()
    read, write = network(*make_inputs(3), DEFAULT_ACTIONS)
    loss = (
        read.pow(2).sum()
        + write.value.pow(2).sum()
        + write.error_risk.abs().sum()
        + write.no_emit.abs().sum()
        + write.expected_emit.sum()
    )
    loss.backward()

    for name, parameter in network.named_parameters():
        assert parameter.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"
