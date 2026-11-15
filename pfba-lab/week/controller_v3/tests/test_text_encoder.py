"""Tests for the vocabulary and the CPU text encoder."""

import torch

from controller_v3.text_encoder import TextEncoder, Vocab


def test_vocab_reserves_pad_and_unk():
    assert Vocab.PAD == 0
    assert Vocab.UNK == 1
    vocab = Vocab.from_texts(["hello world hello", "new words"])
    assert vocab.PAD == 0
    assert vocab.UNK == 1
    # Four unique tokens (hello, world, new, words) plus PAD and UNK.
    assert len(vocab) == 6
    assert vocab.encode(["hello"], 8) == (2,)


def test_vocab_encode_keeps_only_the_requested_window():
    vocab = Vocab.from_texts(["alpha beta gamma delta"])
    ids = vocab.encode(["alpha", "beta", "gamma", "delta"], 2)
    assert len(ids) == 2
    assert ids == vocab.encode(["gamma", "delta"], 2)
    assert ids == (4, 5)


def test_vocab_encode_maps_unknown_words_to_unk():
    vocab = Vocab.from_texts(["alpha beta"])
    ids = vocab.encode(["alpha", "missing", "beta"], 8)
    assert ids == (2, Vocab.UNK, 3)


def test_vocab_encode_empty_words_is_empty():
    vocab = Vocab.from_texts(["alpha beta"])
    assert vocab.encode([], 8) == ()


def test_text_encoder_forward_shapes():
    encoder = TextEncoder(src_vocab_size=20, tgt_vocab_size=20)
    src_ids = torch.tensor([[2, 3, 4], [5, 6, 0]])
    tgt_ids = torch.tensor([[7, 8, 9, 10], [11, 12, 0, 0]])
    output = encoder(src_ids, tgt_ids)
    assert output.shape == (2, 64)
    assert torch.isfinite(output).all()


def test_text_encoder_empty_sequences_return_zeros():
    encoder = TextEncoder(src_vocab_size=20, tgt_vocab_size=20)
    src_ids = torch.zeros((2, 0), dtype=torch.long)
    tgt_ids = torch.zeros((2, 0), dtype=torch.long)
    output = encoder(src_ids, tgt_ids)
    assert output.shape == (2, 64)
    assert torch.count_nonzero(output) == 0


def test_text_encoder_one_empty_side_still_works():
    encoder = TextEncoder(src_vocab_size=20, tgt_vocab_size=20)
    src_ids = torch.zeros((1, 0), dtype=torch.long)
    tgt_ids = torch.tensor([[2, 3, 4]])
    output = encoder(src_ids, tgt_ids)
    assert output.shape == (1, 64)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) > 0


def test_text_encoder_gradients_flow():
    encoder = TextEncoder(src_vocab_size=20, tgt_vocab_size=20)
    src_ids = torch.tensor([[2, 3, 4]])
    tgt_ids = torch.tensor([[5, 6, 7]])
    output = encoder(src_ids, tgt_ids)
    output.pow(2).sum().backward()
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"
