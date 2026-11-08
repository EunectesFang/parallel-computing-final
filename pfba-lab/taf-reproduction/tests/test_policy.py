"""Streaming policy invariants: majority threshold, forcing, RALCP, flush."""
import pytest

from taf.config import StreamingConfig
from taf.mocks import MockLLM, MockMT, make_vocab
from taf.policy import (
    AuthorRALCPTAFPolicy,
    RALCPPolicy,
    TAFPolicy,
    WaitKPolicy,
    build_policy,
)

MAPPING = {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}
VOCAB = make_vocab(["A", "B", "C", "D", "E"])
A, B, C, D, E = (VOCAB[x] for x in "ABCDE")


def taf_cfg(**kwargs):
    defaults = dict(policy="taf", n=10, horizon=10, top_k=10, tau=0.6,
                    cuda_sync=False, max_new_tokens=64)
    defaults.update(kwargs)
    return StreamingConfig(**defaults)


def test_taf_majority_writes_and_forces_committed_prefix():
    llm = MockLLM(table={
        "a": ["c"] * 8 + ["c e"] * 2,
        "a b": [""] * 10,
    })
    mt = MockMT(mapping=MAPPING, vocab=VOCAB,
                variant_fn=None)
    policy = TAFPolicy(llm, mt, taf_cfg())
    result = policy.run(["a", "b"])

    assert result.committed_ids == [A, C, B]
    assert result.text == "A C B"
    assert result.delays == [1, 1, 2]

    first, second = result.traces
    assert first.pi == pytest.approx(0.8)
    assert first.support == 8
    assert first.wrote is True
    assert first.committed_before == 0
    assert first.committed_after == 2
    assert second.pi == pytest.approx(1.0)
    assert second.wrote is True
    assert second.committed_after == 3

    # Monotone commits.
    for trace in result.traces:
        assert trace.committed_after >= trace.committed_before

    # The second step's MT calls receive the committed prefix.
    assert mt.calls[0]["committed_ids"] == []
    assert all(call["committed_ids"] == [A, C] for call in mt.calls[10:])
    assert len(mt.calls) == 20


def test_final_flush_finishes_when_threshold_never_met():
    llm = MockLLM(table={"a": ["c"] * 9 + ["d"]})
    mt = MockMT(mapping=MAPPING, vocab=VOCAB)
    policy = TAFPolicy(llm, mt, taf_cfg(tau=1.0))
    result = policy.run(["a"])

    assert result.traces[0].wrote is False
    assert result.traces[0].pi == pytest.approx(0.9)
    # Flush commits the majority tail at source end.
    assert result.committed_ids == [A, C]
    assert result.text == "A C"
    assert result.delays == [1, 1]
    assert result.metadata["flush_added_tokens"] == 2


def test_ralcp_relaxed_prefix_baseline():
    def variants(source, committed, num):
        if committed:
            return [[]]
        return [[A, B]] * 4 + [[A, C]] * 2

    mt = MockMT(mapping=MAPPING, vocab=VOCAB, variant_fn=variants)
    cfg = StreamingConfig(policy="ralcp", beam_width=6, ralcp_fraction=0.6,
                          cuda_sync=False, max_new_tokens=64)
    result = RALCPPolicy(mt, cfg).run(["a"])

    assert result.committed_ids == [A, B]
    assert result.text == "A B"
    trace = result.traces[0]
    assert trace.n_candidates == 6
    assert trace.support == 4
    assert trace.pi == pytest.approx(4 / 6)
    assert trace.wrote is True
    assert mt.calls[0]["num_beams"] == 6


def test_ralcp_taf_candidate_budget_is_40_for_n10():
    def variants(source, committed, num):
        if committed:
            return [[]]
        return [[A, B]] * num

    llm = MockLLM(table={"a": [""] * 10})
    mt = MockMT(mapping=MAPPING, vocab=VOCAB, variant_fn=variants)
    cfg = StreamingConfig(policy="ralcp+taf", n=10, beam_width=40,
                          tau=0.6, cuda_sync=False, max_new_tokens=64)
    result = build_policy(llm, mt, cfg).run(["a"])

    assert result.metadata["effective_beam_width"] == 4
    assert result.metadata["candidate_budget"] == 40
    assert result.traces[0].n_candidates == 40
    assert result.committed_ids == [A, B]
    # Anticipation calls use the n*b budget; the fresh full-source EOS decode
    # keeps the authors' per-call beam size but returns one sequence.
    assert all(call["num_beams"] == 4 for call in mt.calls)
    assert mt.calls[-1]["num_sequences"] == 1


# --- Author RALCP_TAF fidelity protocol -----------------------------------


def author_cfg(**kwargs):
    defaults = dict(policy="ralcp_taf", n=4, beam_width=4, ralcp_fraction=0.5,
                    tau=1.0, cuda_sync=False, max_new_tokens=64)
    defaults.update(kwargs)
    return StreamingConfig(**defaults)


def test_default_policy_is_named_author_protocol():
    cfg = StreamingConfig()
    assert cfg.policy == "ralcp_taf"
    policy = build_policy(object(), object(), cfg)
    assert isinstance(policy, AuthorRALCPTAFPolicy)


def test_author_ralcp_taf_votes_word_prefixes_without_exact_majority():
    # Four MT candidates all share "A" but have four unique full sequences:
    # exact majority is 1/4, far below tau=1.0.  The author protocol still
    # writes "A" because the word-prefix agreement is 4/4 >= 0.5.
    answers = iter([[[A, B]], [[A, C]], [[A, D]], [[A, E]]])

    def variants(source, committed, num):
        if source == "a c":
            return next(answers)
        if tuple(committed) == (A,):
            return [[B]]
        return [[]]

    llm = MockLLM(table={"a": ["c"] * 4})
    mt = MockMT(mapping=MAPPING, vocab=VOCAB, variant_fn=variants)
    result = build_policy(llm, mt, author_cfg()).run(["a", "b"])

    first = result.traces[0]
    assert first.wrote is True
    assert first.support == 4
    assert first.pi == pytest.approx(1.0)
    assert first.committed_after == 1
    # EOS then completes from the full source, keeping the committed prefix.
    assert result.committed_ids == [A, B]
    assert result.text == "A B"
    # Anticipation at read positions 1 and 2, then the EOS decode.
    assert len(result.traces) == 3
    assert result.traces[-1].detail["event"] == "eos_fresh_full_source"


def test_author_ralcp_taf_word_prefix_can_change_between_depths():
    # The author's ralcp picks the most common prefix independently at each
    # depth.  Depth 1 ties ("a" first), depth 2 is won by "b y".
    answers = iter(["a x", "a z", "b y", "b y"])
    word_ids = {"a": 10, "b": 11, "x": 12, "z": 13, "y": 14}

    def variants(source, committed, num):
        if source == "s p":
            return [[word_ids[w] for w in next(answers).split()]]
        return [[]]

    llm = MockLLM(table={"s": ["p"] * 4})
    mt = MockMT(mapping={}, vocab={**VOCAB, **word_ids}, variant_fn=variants)
    result = build_policy(llm, mt, author_cfg()).run(["s", "q"])

    assert mt.decode(result.committed_ids[:2]) == "b y"
    assert result.traces[0].detail["selected"] == "b y"
    assert result.traces[0].support == 2


def test_author_ralcp_taf_eos_is_a_fresh_full_source_decode():
    full_source_calls = {"n": 0}

    def variants(source, committed, num):
        # Both anticipation steps produce nothing; only the fresh EOS decode
        # (the fifth full-source call, after four n=4 anticipation calls)
        # returns a translation.
        if source == "a c":
            return [[]]
        full_source_calls["n"] += 1
        if full_source_calls["n"] <= 4:
            return [[]]
        return [[A, B]]

    llm = MockLLM(table={"a": ["c"] * 4})
    mt = MockMT(mapping=MAPPING, vocab=VOCAB, variant_fn=variants)
    result = build_policy(llm, mt, author_cfg()).run(["a", "b"])

    assert result.traces[0].wrote is False
    eos = mt.calls[-1]
    assert eos["source"] == "a b"
    assert eos["committed_ids"] == []
    assert eos["num_sequences"] == 1
    assert eos["num_beams"] == 1
    assert result.text == "A B"
    assert result.metadata["flush_added_tokens"] == 2


def test_waitk_writes_one_token_per_step_and_flushes():
    mt = MockMT(mapping=MAPPING, vocab=VOCAB)
    cfg = StreamingConfig(policy="waitk", wait_k=1, wait_stride=1,
                          cuda_sync=False, max_new_tokens=64)
    result = WaitKPolicy(mt, cfg).run(["a", "b"])

    assert result.committed_ids == [A, B]
    assert result.delays == [1, 2]
    assert [t.wrote for t in result.traces] == [True, True]
    assert [t.read_pos for t in result.traces] == [1, 2]


def test_invalid_configs_rejected():
    with pytest.raises(ValueError):
        StreamingConfig(tau=1.5)
    with pytest.raises(ValueError):
        StreamingConfig(n=0)
    with pytest.raises(ValueError):
        StreamingConfig(initial_read=0)
    with pytest.raises(ValueError):
        StreamingConfig(policy="nope")
