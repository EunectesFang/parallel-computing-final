"""End-to-end integration test for the CAVC v3 training and rollout path.

The dataset is three toy German/English sentences. Predictor, translator, and
tokenizer are deterministic fakes; everything runs on CPU with no downloads.
The test still exercises the real `FeatureBuilder`, `CAVCNetwork`,
`CAVCTrainer`, checkpointing, `CounterfactualLabeler`, and `run_rollout` code
paths.
"""

import importlib.util
import inspect
import json
import math
from dataclasses import dataclass, replace

import pytest
import torch

from controller_v3 import cli, evaluator
from controller_v3.action import READ, Action
from controller_v3.checkpoint import load_checkpoint, save_checkpoint
from controller_v3.config import ControllerConfig
from controller_v3.cost_model import MeasuredCostTable
from controller_v3.evaluator import summarize_traces
from controller_v3.features import NUMERIC_FEATURES, FeatureBuilder, StateInput
from controller_v3.labeler import ActionOutcome, CounterfactualLabeler
from controller_v3.network import CAVCNetwork
from controller_v3.reward import RewardConfig, immediate_reward
from controller_v3.rollout import run_rollout
from controller_v3.targets import build_action_targets, derive_state_key
from controller_v3.text_encoder import Vocab
from controller_v3.trainer import CAVCTrainer

SENTENCES = (
    ("s0", "hallo welt .", "hello world ."),
    ("s1", "der hund schläft .", "the dog sleeps ."),
    ("s2", "wir lernen heute .", "we learn today ."),
)

GLOSS = {
    "hallo": "hello",
    "welt": "world",
    "der": "the",
    "hund": "dog",
    "schläft": "sleeps",
    "wir": "we",
    "lernen": "learn",
    "heute": "today",
    ".": ".",
}

POOL = ("heute", "der", "lernen", ".", "hund", "wir", "welt", "schläft")

ACTIONS = (READ, Action(2, 4), Action(4, 6))
WRITE_ACTIONS = (Action(2, 4), Action(4, 6))

TRACE_KEYS = {
    "read_pos",
    "action",
    "observed_len",
    "committed_before",
    "committed_after",
    "new_words",
    "pred_s",
    "mt_s",
    "spent_s",
    "vote_top1",
    "vote_margin",
    "vote_entropy",
    "no_emit",
    "q_read",
    "q_write",
    "cost",
    "post_action",
}

NUMERIC_TRACE_KEYS = (
    "pred_s",
    "mt_s",
    "spent_s",
    "vote_top1",
    "vote_margin",
    "vote_entropy",
    "q_read",
    "q_write",
    "cost",
)


# ----------------------------------------------------------------------
# Fakes and harness
# ----------------------------------------------------------------------
class FakeTokenizer:
    """Word-level tokenizer over a shared `Vocab`; never stops on EOS."""

    eos_token_id = -1

    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        words = []
        for token in token_ids:
            token = int(token)
            if token == self.eos_token_id:
                break
            if 0 <= token < len(self.vocab.id_to_token):
                word = self.vocab.id_to_token[token]
                if word is not None:
                    words.append(word)
        return " ".join(words)

    def __call__(self, text: str, add_special_tokens: bool = False, **kwargs):
        del add_special_tokens, kwargs
        return {
            "input_ids": [
                self.vocab.token_to_id.get(word, self.vocab.UNK)
                for word in str(text).split()
            ]
        }


class FakePredictor:
    """Deterministic candidates for both labeling and rollout paths."""

    def __init__(self, vocab: Vocab, tokenizer: FakeTokenizer) -> None:
        self.vocab = vocab
        self.tok = tokenizer

    def _words(self, seed: int, index: int, length: int) -> list[str]:
        return [
            POOL[(int(seed) + int(index) + depth) % len(POOL)]
            for depth in range(max(1, int(length)))
        ]

    def sample_token_rows(
        self, observed: str, max_b: int, max_h: int, seed: int
    ) -> list[list[int]]:
        del observed
        return [
            [
                self.vocab.token_to_id.get(word, self.vocab.UNK)
                for word in self._words(seed, index, max_h)
            ]
            for index in range(int(max_b))
        ]

    def sample(self, observed: str, branches: int, horizon: int, seed: int) -> list[str]:
        del observed
        return [
            " ".join(self._words(seed, index, horizon))
            for index in range(max(1, int(branches)))
        ]


class FakeTranslator:
    """Whitespace-gloss translator; deterministic and side-effect free."""

    def translate_batch(self, sources, prefix, final: bool = False, **kwargs):
        del prefix, final, kwargs
        return [
            " ".join(GLOSS.get(word, word) for word in str(source).split())
            for source in sources
        ]


class FixedTimer:
    """Every timed stage costs a fixed number of seconds."""

    def __init__(self, seconds: float = 0.05) -> None:
        self.seconds = float(seconds)

    def run(self, name, function):
        del name
        value = function()
        return value, self.seconds


@dataclass(frozen=True)
class Sentence:
    sentence_id: str
    source: str
    reference: str
    split: str = "test"


def make_config(**overrides) -> ControllerConfig:
    defaults = dict(
        seed=7,
        epochs=5,
        batch_size=8,
        actions=WRITE_ACTIONS,
    )
    defaults.update(overrides)
    return ControllerConfig(**defaults)


def make_harness(**cfg_overrides):
    """Build the full CPU harness with a shared vocab and fresh network."""
    cfg = make_config(**cfg_overrides)
    torch.manual_seed(cfg.seed)

    source_texts = [source for _, source, _ in SENTENCES]
    target_texts = [reference for _, _, reference in SENTENCES]
    vocab = Vocab.from_texts([*source_texts, *target_texts, " ".join(POOL)])
    tokenizer = FakeTokenizer(vocab)
    predictor = FakePredictor(vocab, tokenizer)
    translator = FakeTranslator()
    timer = FixedTimer()
    builder = FeatureBuilder(length_ratio=1.0, src_vocab=vocab, tgt_vocab=vocab)
    labeler = CounterfactualLabeler(predictor, translator, cfg, timer=timer)
    network = CAVCNetwork(cfg, src_vocab_size=len(vocab), tgt_vocab_size=len(vocab))
    return {
        "cfg": cfg,
        "vocab": vocab,
        "predictor": predictor,
        "translator": translator,
        "timer": timer,
        "builder": builder,
        "labeler": labeler,
        "network": network,
    }


def make_dataset(harness):
    """Label three states x three actions and build `ActionTarget` rows."""
    cfg = harness["cfg"]
    builder = harness["builder"]
    labeler = harness["labeler"]

    targets = []
    states = []
    cost_records = []
    for index, (sentence_id, source, _reference) in enumerate(SENTENCES):
        words = source.split()
        observed = " ".join(words[:2])
        next_observed = " ".join(words[:3])
        committed: tuple[str, ...] = ()

        predictor_id = index % 2
        state = builder.build(
            observed, committed, predictor_id=predictor_id, translator_id=predictor_id
        )
        next_state = builder.build(
            next_observed,
            committed,
            predictor_id=predictor_id,
            translator_id=predictor_id,
        )
        outcomes = labeler.label_state(
            observed, committed, source, ACTIONS, seed=cfg.seed + index
        )
        targets.extend(
            build_action_targets(
                state,
                outcomes,
                cfg,
                next_state=next_state,
                backlog=float(index + 1),
                source_advance=1,
                state_key=f"{sentence_id}:2",
            )
        )
        states.append(state)
        for outcome in outcomes:
            if not outcome.action.is_read:
                cost_records.append(
                    (
                        predictor_id,
                        predictor_id,
                        outcome.action,
                        float(outcome.pred_s),
                        float(outcome.mt_s),
                    )
                )
    return targets, states, cost_records


def feature_stats(states):
    """Elementwise mean and population std; std defaults to 1.0 when zero."""
    count = float(len(states))
    width = len(states[0].numeric)
    mean = [
        sum(float(state.numeric[index]) for state in states) / count
        for index in range(width)
    ]
    std = []
    for index in range(width):
        variance = sum(
            (float(state.numeric[index]) - mean[index]) ** 2 for state in states
        ) / count
        std.append(math.sqrt(variance) if variance > 0.0 else 1.0)
    return mean, std


# ----------------------------------------------------------------------
# Leakage boundary
# ----------------------------------------------------------------------
def test_feature_builder_cannot_see_future_or_reference_inputs():
    signature = inspect.signature(FeatureBuilder.build)
    names = set(signature.parameters)
    for forbidden in ("source", "reference", "future", "full_source", "oracle"):
        assert forbidden not in names
    for parameter in signature.parameters.values():
        assert parameter.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    assert {"observed", "committed"} <= names


def test_state_input_has_no_source_or_reference_field():
    fields = set(StateInput.__dataclass_fields__)
    assert fields == {
        "numeric",
        "predictor_id",
        "translator_id",
        "src_ids",
        "tgt_ids",
    }


# ----------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------
def test_build_action_targets_uses_immediate_reward_and_read_delay():
    cfg = ControllerConfig()
    numeric = tuple([0.0] * len(NUMERIC_FEATURES))
    state = StateInput(numeric=numeric, predictor_id=0, translator_id=0)
    read = ActionOutcome(
        action=READ,
        new_words=(),
        matched=0,
        incorrect=0,
        emitted=0,
        no_emit=True,
        next_committed=(),
        pred_s=0.0,
        mt_s=0.0,
        vote_top1=0.0,
        vote_margin=0.0,
        vote_entropy=0.0,
    )
    write = ActionOutcome(
        action=Action(2, 4),
        new_words=("a", "b", "c", "d"),
        matched=3,
        incorrect=1,
        emitted=4,
        no_emit=False,
        next_committed=("a", "b", "c", "d"),
        pred_s=0.1,
        mt_s=0.2,
        vote_top1=0.75,
        vote_margin=0.5,
        vote_entropy=0.2,
    )
    reward_cfg = RewardConfig(
        weight_m=cfg.weight_m,
        weight_e=cfg.weight_e,
        lambda_l=cfg.lambda_l,
        gamma=cfg.gamma,
    )

    targets = build_action_targets(
        state,
        [read, write],
        cfg,
        backlog=2.0,
        source_advance=3,
        state_key="state-0",
    )
    assert [target.action for target in targets] == [READ, Action(2, 4)]
    assert all(target.state_key == "state-0" for target in targets)
    # Non-terminal rows always advertise the READ-inclusive backup set.
    assert targets[0].next_actions == (READ, *cfg.actions)
    # Post-action backlog without an explicit src_words_after: the pre-action
    # backlog (2) plus the new source word (3) minus this action's emission.
    assert targets[0].value_target == pytest.approx(
        immediate_reward(0, 0, 2.0 + 3 - 0, 3, reward_cfg)
    )
    assert targets[0].value_target == pytest.approx(-cfg.lambda_l * 5.0 * 3)
    assert targets[1].value_target == pytest.approx(
        immediate_reward(3, 1, 2.0 + 3 - 4, 3, reward_cfg)
    )
    assert targets[1].error_target == 1.0
    assert targets[1].no_emit_target == 0.0
    assert targets[1].emit_target == 4.0

    bootstrapped = build_action_targets(
        state,
        [read, write],
        cfg,
        next_state=state,
        backlog=2.0,
        source_advance=3,
        state_key="state-0",
    )
    assert all(
        target.next_actions == (READ, *cfg.actions) for target in bootstrapped
    )


def test_empty_state_key_is_derived_stably_and_without_python_hash():
    cfg = ControllerConfig()
    state = StateInput(
        numeric=tuple([0.5] * len(NUMERIC_FEATURES)),
        predictor_id=1,
        translator_id=2,
        src_ids=(3, 4),
        tgt_ids=(5,),
    )
    read = ActionOutcome(
        action=READ,
        new_words=(),
        matched=0,
        incorrect=0,
        emitted=0,
        no_emit=True,
        next_committed=(),
        pred_s=0.0,
        mt_s=0.0,
        vote_top1=0.0,
        vote_margin=0.0,
        vote_entropy=0.0,
    )
    first = build_action_targets(state, [read], cfg)[0]
    second = build_action_targets(state, [read], cfg)[0]
    assert first.state_key
    assert first.state_key == second.state_key == derive_state_key(state)
    # 64 hex characters: a SHA-256 digest, not Python's salted hash().
    assert len(first.state_key) == 64
    other = replace(state, predictor_id=3)
    assert derive_state_key(other) != first.state_key


# ----------------------------------------------------------------------
# Training, checkpoint round trip, rollout
# ----------------------------------------------------------------------
def test_integration_train_checkpoint_and_rollout(tmp_path):
    harness = make_harness()
    cfg = harness["cfg"]
    builder = harness["builder"]
    predictor = harness["predictor"]
    translator = harness["translator"]
    timer = harness["timer"]
    network = harness["network"]
    vocab = harness["vocab"]

    targets, states, cost_records = make_dataset(harness)
    assert len(targets) == 9
    assert len(states) == 3
    assert sum(target.action.is_read for target in targets) == 3
    assert all(target.state_key for target in targets)

    cost_model = MeasuredCostTable(cost_records)
    assert len(cost_model.records) == 6

    trainer = CAVCTrainer(network, cfg, cost_model)
    result = trainer.fit(targets)
    assert result["epochs"] == 5
    assert result["n_samples"] == 9
    for key in (
        "final_loss",
        "value_loss",
        "rank_loss",
        "error_loss",
        "no_emit_loss",
        "emit_loss",
    ):
        assert math.isfinite(result[key])

    mean, std = feature_stats(states)
    checkpoint_path = tmp_path / "controller.pt"
    saved = save_checkpoint(
        checkpoint_path,
        network,
        cfg,
        mean,
        std,
        cost_model,
        extra={"tag": "integration"},
    )
    assert checkpoint_path.is_file()
    assert saved["seed"] == cfg.seed
    assert len(saved["cost_records"]) == 6

    restored = CAVCNetwork(cfg, src_vocab_size=len(vocab), tgt_vocab_size=len(vocab))
    metadata = load_checkpoint(checkpoint_path, restored)
    assert metadata["seed"] == cfg.seed
    assert metadata["extra"]["tag"] == "integration"
    assert metadata["feature_mean"] == pytest.approx(mean)
    assert metadata["feature_std"] == pytest.approx(std)
    assert len(metadata["cost_records"]) == 6
    for original, loaded in zip(
        network.state_dict().values(), restored.state_dict().values()
    ):
        assert torch.equal(original, loaded)

    # A mismatched architecture must fail before any parameter is copied.
    wrong = CAVCNetwork(
        replace(cfg, value_hidden=32),
        src_vocab_size=len(vocab),
        tgt_vocab_size=len(vocab),
    )
    with pytest.raises(ValueError, match="architecture"):
        load_checkpoint(checkpoint_path, wrong)

    traces = []
    for sentence_id, source, reference in SENTENCES[1:3]:
        traces.append(
            run_rollout(
                Sentence(sentence_id, source, reference),
                predictor,
                translator,
                restored,
                builder,
                cost_model,
                cfg,
                timer=timer,
            )
        )
    assert len(traces) == 2

    for trace in traces:
        assert trace.sentence_id in {"s1", "s2"}
        assert trace.hypothesis
        assert math.isfinite(trace.completion_s)
        assert trace.laal is not None and math.isfinite(trace.laal)
        assert 0.0 <= trace.read_fraction <= 1.0
        assert set(trace.action_histogram) <= {"read", "b2h4", "b4h6"}
        for step in trace.steps:
            assert TRACE_KEYS <= set(step)
            assert math.isfinite(step["read_pos"])
            assert math.isfinite(step["committed_before"])
            assert math.isfinite(step["committed_after"])
            for key in NUMERIC_TRACE_KEYS:
                if step[key] is not None:
                    assert math.isfinite(step[key])

    summary = summarize_traces(traces)
    assert summary["n"] == 2
    assert math.isfinite(summary["completion_s"]["mean"])
    assert math.isfinite(summary["completion_s"]["p95"])
    assert 0.0 <= summary["read_fraction"] <= 1.0
    # Metric honesty: never invent a BLEU score when sacrebleu is absent.
    if importlib.util.find_spec("sacrebleu") is None:
        assert summary["bleu"]["status"] == "unavailable"
        assert "score" not in summary["bleu"]
    assert summary["comet"]["status"] == "skipped"


# ----------------------------------------------------------------------
# CLI acceptance path and metric honesty
# ----------------------------------------------------------------------
def test_cli_smoke_writes_summary_and_traces(tmp_path):
    exit_code = cli.main(["--smoke", "--output-dir", str(tmp_path)])
    assert exit_code == 0
    summary_path = tmp_path / "summary.json"
    traces_path = tmp_path / "traces.jsonl"
    assert summary_path.is_file()
    assert traces_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["mode"] == "smoke"
    assert summary["n_targets"] == 9
    assert summary["rollout"]["n"] == 2
    assert summary["rollout"]["bleu"] == cli.SMOKE_BLEU_STATUS
    assert summary["rollout"]["comet"] == cli.SMOKE_COMET_STATUS

    trace_lines = traces_path.read_text(encoding="utf-8").splitlines()
    assert len(trace_lines) == 2
    for line in trace_lines:
        record = json.loads(line)
        assert record["sentence_id"] in {"smoke-1", "smoke-2"}
        assert record["steps"]


def test_smoke_forces_unavailable_bleu_even_when_available(monkeypatch, tmp_path):
    """Pretend sacrebleu/BLEU is available; the smoke must still not call it."""
    calls = []

    def fake_corpus_bleu(hypotheses, references):
        calls.append((list(hypotheses), list(references)))
        return {"status": "ok", "score": 99.9}

    monkeypatch.setattr(evaluator, "corpus_bleu", fake_corpus_bleu)
    if evaluator.run_pfba is not None:
        monkeypatch.setattr(
            evaluator.run_pfba,
            "corpus_bleu",
            lambda *args, **kwargs: {"status": "ok", "score": 99.9},
        )
        monkeypatch.setattr(evaluator.run_pfba, "comet_score", fake_corpus_bleu)

    assert cli.main(["--smoke", "--output-dir", str(tmp_path)]) == 0
    assert calls == []

    summary = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["rollout"]["bleu"] == cli.SMOKE_BLEU_STATUS
    assert summary["rollout"]["bleu"]["status"] == "unavailable"
    assert "score" not in summary["rollout"]["bleu"]
    assert summary["rollout"]["comet"] == cli.SMOKE_COMET_STATUS


# ----------------------------------------------------------------------
# Checkpoint trust and version guards
# ----------------------------------------------------------------------
def test_checkpoint_rejects_unknown_format_version(tmp_path):
    cfg = ControllerConfig()
    network = CAVCNetwork(cfg, src_vocab_size=20, tgt_vocab_size=20)
    path = tmp_path / "controller.pt"
    save_checkpoint(
        path, network, cfg, [0.0] * cfg.numeric_dim, [1.0] * cfg.numeric_dim, None
    )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["format_version"] = 999
    torch.save(payload, path)

    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint(path, CAVCNetwork(cfg, src_vocab_size=20, tgt_vocab_size=20))


def test_checkpoint_rejects_unsafe_pickle_payload(tmp_path):
    cfg = ControllerConfig()
    network = CAVCNetwork(cfg, src_vocab_size=20, tgt_vocab_size=20)
    path = tmp_path / "unsafe.pt"
    save_checkpoint(
        path,
        network,
        cfg,
        [0.0] * cfg.numeric_dim,
        [1.0] * cfg.numeric_dim,
        None,
        extra={"bad": object()},
    )

    with pytest.raises(ValueError, match="trusted|weights_only"):
        load_checkpoint(path, CAVCNetwork(cfg, src_vocab_size=20, tgt_vocab_size=20))


def test_checkpoint_rejects_malformed_feature_stats(tmp_path):
    cfg = ControllerConfig()
    network = CAVCNetwork(cfg, src_vocab_size=20, tgt_vocab_size=20)
    path = tmp_path / "controller.pt"
    save_checkpoint(
        path, network, cfg, [0.0] * cfg.numeric_dim, [1.0] * cfg.numeric_dim, None
    )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["feature_mean"] = "not-a-list-of-numbers"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="feature_mean"):
        load_checkpoint(path, CAVCNetwork(cfg, src_vocab_size=20, tgt_vocab_size=20))
