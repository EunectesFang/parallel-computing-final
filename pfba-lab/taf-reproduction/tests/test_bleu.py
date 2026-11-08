"""Internal 13a/exp BLEU invariants and tokenizer behavior."""
import pytest

from taf.bleu import compute_bleu, corpus_bleu, tokenize_13a


def test_identical_sentence_is_100():
    score = corpus_bleu(["the cat is on the mat"], ["the cat is on the mat"])
    assert score["score"] == pytest.approx(100.0)
    assert score["brevity_penalty"] == pytest.approx(1.0)
    assert score["total"] == [6, 5, 4, 3]


def test_disjoint_sentence_is_zero():
    assert corpus_bleu(["a b c d e"], ["v w x y z"])["score"] == pytest.approx(0.0)


def test_13a_tokenization():
    assert tokenize_13a("Hello, world!") == ["Hello", ",", "world", "!"]
    # Periods between digits are not split.
    assert tokenize_13a("pi is 3.14") == ["pi", "is", "3.14"]


def test_short_hypothesis_brevity_penalty():
    result = corpus_bleu(
        ["the cat sat on the mat"],
        ["the cat is sitting on the mat today"],
    )
    assert result["brevity_penalty"] < 1.0
    assert result["score"] < 100.0


def test_multiple_references_use_best_clipping():
    hyp = "the cat sat on the mat"
    refs = ["the cat sat on the mat", "a cat sat on a mat"]
    assert corpus_bleu([hyp], [refs])["score"] == pytest.approx(100.0)


def test_compute_bleu_wrapper_returns_score():
    result = compute_bleu(["the cat is on the mat"], [["the cat is on the mat"]])
    assert result["score"] == pytest.approx(100.0)
    assert "implementation" in result


def test_sacrebleu_scores_every_sentence_in_reference_stream():
    pytest.importorskip("sacrebleu")
    sentences = ["the blue bird can sing", "fresh green leaves grow slowly"]
    result = compute_bleu(sentences, sentences)
    assert result["hyp_len"] == result["ref_len"] == 10
