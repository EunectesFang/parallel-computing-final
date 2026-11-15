"""CPU-only tests for the real-model Kaggle wiring.

Every adapter, tokenizer, and timer is a deterministic fake. Nothing here
downloads weights, reads Kaggle data, or uses a GPU. The tests still exercise
the real `CounterfactualLabeler`, `CAVCNetwork`, `CAVCTrainer`,
`MeasuredCostTable`, `run_rollout`, and checkpoint code paths.
"""

import hashlib
import importlib.util
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from controller_v3 import kaggle
from controller_v3.action import READ, Action
from controller_v3.checkpoint import load_checkpoint
from controller_v3.config import ControllerConfig
from controller_v3.cost_model import (
    CostEstimate,
    MeasuredCostTable,
    PipelineCostModel,
)
from controller_v3.features import NUMERIC_FEATURES, FeatureBuilder, StateInput
from controller_v3.labeler import ActionOutcome, CounterfactualLabeler
from controller_v3.network import CAVCNetwork, WriteOutputs
from controller_v3.rollout import run_fixed_rollout, run_rollout
from controller_v3.evaluator import risk_calibration
from controller_v3.runtime import ControlledRuntime, StepResult
from controller_v3.text_encoder import Vocab
from controller_v3.trainer import ActionTarget

# Long enough to have several decision positions past min_start=3.
SENTENCES = (
    ("s0", "hallo welt wir lernen heute .", "hello world we learn today ."),
    ("s1", "der hund schläft und wir lernen .", "the dog sleeps and we learn ."),
    ("s2", "heute lernen wir welt und der hund .", "today we learn world and the dog ."),
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
    "und": "and",
    ".": ".",
}

POOL = (
    "heute",
    "der",
    "lernen",
    ".",
    "hund",
    "wir",
    "welt",
    "schläft",
    "und",
)

WRITE_ACTIONS = (Action(2, 4), Action(4, 6))


# ----------------------------------------------------------------------
# Fakes
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
    """Deterministic word-pool candidates for the labeler and runtime."""

    def __init__(self, vocab: Vocab, tokenizer: FakeTokenizer) -> None:
        self.vocab = vocab
        self.tok = tokenizer
        self.sample_calls: list[tuple[str, int, int, int]] = []

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

    def sample(
        self, observed: str, branches: int, horizon: int, seed: int
    ) -> list[str]:
        self.sample_calls.append(
            (str(observed), int(branches), int(horizon), int(seed))
        )
        return [
            " ".join(self._words(seed, index, horizon))
            for index in range(max(1, int(branches)))
        ]


class FakeTranslator:
    """Whitespace gloss translator; deterministic and side-effect free."""

    def translate_batch(self, sources, prefix, final: bool = False, **kwargs):
        del prefix, final, kwargs
        return [
            " ".join(GLOSS.get(word, word) for word in str(source).split())
            for source in sources
        ]


class FixedTimer:
    """Every timed stage costs the same fixed number of seconds."""

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
    split: str = "dev"


def make_config(**overrides) -> ControllerConfig:
    defaults = dict(
        seed=7,
        epochs=3,
        batch_size=8,
        actions=WRITE_ACTIONS,
    )
    defaults.update(overrides)
    return ControllerConfig(**defaults)


def make_harness(**cfg_overrides):
    """Build fakes plus the real builder, network, and labeler."""
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
    network = CAVCNetwork(cfg, src_vocab_size=len(vocab), tgt_vocab_size=len(vocab))
    labeler = CounterfactualLabeler(predictor, translator, cfg, timer=timer)
    sentences = [
        Sentence(sentence_id, source, reference, split="dev")
        for sentence_id, source, reference in SENTENCES
    ]
    return {
        "cfg": cfg,
        "vocab": vocab,
        "predictor": predictor,
        "translator": translator,
        "timer": timer,
        "builder": builder,
        "network": network,
        "labeler": labeler,
        "sentences": sentences,
    }


def _frozen_builder_network(harness):
    """A caller-supplied frozen builder/network for no-checkpoint confirm tests."""
    cfg = harness["cfg"]
    vocab = harness["vocab"]
    return (
        FeatureBuilder(length_ratio=1.0, src_vocab=vocab, tgt_vocab=vocab),
        CAVCNetwork(cfg, src_vocab_size=len(vocab), tgt_vocab_size=len(vocab)),
    )


# ----------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------
def test_run_preflight_writes_artifacts_and_honest_metrics(tmp_path):
    harness = make_harness()
    cfg = harness["cfg"]

    summary = kaggle.run_preflight(
        limit=3,
        output_dir=str(tmp_path),
        label_states=3,
        rollout_sentences=2,
        seed=cfg.seed,
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=harness["sentences"],
        cfg=cfg,
        predictor_id=0,
        translator_id=0,
        timer=harness["timer"],
    )

    for name in (
        "summary.json",
        "traces.jsonl",
        "config.json",
        "cost_records.json",
        "feature_stats.json",
        "checkpoint.pt",
    ):
        assert (tmp_path / name).is_file(), name

    # Preflight writes the train/serve feature diagnostics even though it does
    # not run the fixed baseline or risk calibration.
    feature_payload = json.loads(
        (tmp_path / "feature_stats.json").read_text(encoding="utf-8")
    )
    assert feature_payload["candidate_mode"] == "per_candidate"
    assert feature_payload["training"]["n"] == summary["n_states"]
    assert set(feature_payload["serving"]["features"]) == set(NUMERIC_FEATURES)
    assert "warnings" in feature_payload["serving"]

    assert summary["mode"] == "preflight"
    assert summary["split"] == "dev"
    assert summary["n_states"] == 3
    assert summary["n_targets"] == 3 * (1 + len(WRITE_ACTIONS))
    assert summary["rollout"]["n"] == 2
    assert "git_commit" in summary

    # BLEU is delegated; without sacrebleu it must be unavailable and carry no
    # score. COMET is skipped because it would download weights.
    bleu = summary["rollout"]["bleu"]
    assert bleu["status"] in {"ok", "unavailable", "error"}
    if bleu["status"] != "ok":
        assert "score" not in bleu
    assert summary["rollout"]["comet"]["status"] == "skipped"

    trace_lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(trace_lines) == 2
    for line in trace_lines:
        record = json.loads(line)
        assert record["sentence_id"] in {"s0", "s1"}
        assert record["steps"]

    config_payload = json.loads(
        (tmp_path / "config.json").read_text(encoding="utf-8")
    )
    assert config_payload["config"]["actions"] == [
        [action.branches, action.horizon] for action in WRITE_ACTIONS
    ]

    cost_payload = json.loads(
        (tmp_path / "cost_records.json").read_text(encoding="utf-8")
    )
    assert cost_payload["records"]
    for record in cost_payload["records"]:
        predictor_id, translator_id, action_pair, pred_s, mt_s = record[:5]
        assert predictor_id == 0 and translator_id == 0
        assert pred_s >= 0.0 and mt_s >= 0.0
        assert action_pair[0] > 0 and action_pair[1] > 0
        # The optional sixth field is measured pipeline wall time.
        assert len(record) == 6 and record[5] >= 0.0
        assert record[5] <= pred_s + mt_s + 1e-9

    # The checkpoint restores into a freshly built network of the same shape.
    restored = CAVCNetwork(
        cfg,
        src_vocab_size=summary["vocab_size"],
        tgt_vocab_size=summary["vocab_size"],
    )
    metadata = load_checkpoint(tmp_path / "checkpoint.pt", restored)
    assert metadata["seed"] == cfg.seed
    assert metadata["extra"]["mode"] == "preflight"


# ----------------------------------------------------------------------
# Cost calibration
# ----------------------------------------------------------------------
def test_calibrate_cost_records_uses_per_candidate_seeds():
    harness = make_harness()
    cfg = harness["cfg"]
    predictor = harness["predictor"]

    records, summary = kaggle.calibrate_cost_records(
        predictor,
        harness["translator"],
        harness["sentences"],
        WRITE_ACTIONS,
        cfg,
        limit=1,
        states_per_sentence=1,
        predictor_id=1,
        translator_id=1,
        timer=harness["timer"],
    )

    # One record per action, all non-negative, keyed by the requested pair.
    assert len(records) == len(WRITE_ACTIONS)
    assert summary["n_states"] == 1
    assert summary["predictor_id"] == 1
    assert summary["translator_id"] == 1
    assert [record[2] for record in records] == list(WRITE_ACTIONS)
    for record in records:
        predictor_id, translator_id, _action, pred_s, mt_s = record[:5]
        assert predictor_id == 1 and translator_id == 1
        assert pred_s >= 0.0 and mt_s >= 0.0
        # Measured pipeline marginal is recorded separately; it is never
        # presented as the serial pred_s + mt_s sum.
        assert len(record) == 6
        assert record[5] >= 0.0
        assert record[5] == pytest.approx(pred_s + mt_s)

    # FixedTimer: the b2h4 action samples twice and translates once.
    assert records[0][3] == pytest.approx(2 * harness["timer"].seconds)
    assert records[0][4] == pytest.approx(harness["timer"].seconds)
    assert records[1][3] == pytest.approx(4 * harness["timer"].seconds)
    assert summary["per_action"]["b2h4"]["marginal_wall_s_median"] >= 0.0

    # Runtime path: candidate i uses seed + i with branches=1.
    calls = predictor.sample_calls
    assert len(calls) == 2 + 4
    assert all(branches == 1 for _observed, branches, _horizon, _seed in calls)
    first_seeds = [seed for _o, _b, _h, seed in calls[:2]]
    second_seeds = [seed for _o, _b, _h, seed in calls[2:]]
    assert first_seeds == [first_seeds[0], first_seeds[0] + 1]
    assert second_seeds == [second_seeds[0] + index for index in range(4)]
    assert [horizon for _o, _b, horizon, _s in calls] == [4, 4, 6, 6, 6, 6]


# ----------------------------------------------------------------------
# Confirmation guards
# ----------------------------------------------------------------------
def test_run_confirm_rejects_consumed_offset_without_override(tmp_path):
    harness = make_harness()
    with pytest.raises(ValueError, match="consumed|140"):
        kaggle.run_confirm(
            offset=139,
            output_dir=str(tmp_path),
            sentences=harness["sentences"],
            predictor=harness["predictor"],
            translator=harness["translator"],
            cfg=harness["cfg"],
        )


def test_run_confirm_accepts_allow_consumed_on_test_slice(tmp_path):
    harness = make_harness()
    cfg = harness["cfg"]
    test_sentences = [
        Sentence(f"test:{offset:04d}", source, reference, split="test")
        for offset, (_sid, source, reference) in enumerate(SENTENCES, start=140)
    ]

    builder, network = _frozen_builder_network(harness)
    summary = kaggle.run_confirm(
        offset=139,
        limit=2,
        output_dir=str(tmp_path),
        allow_consumed=True,
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=test_sentences,
        cfg=cfg,
        builder=builder,
        network=network,
        predictor_id=0,
        translator_id=0,
        timer=harness["timer"],
    )

    assert summary["mode"] == "confirm"
    assert summary["split"] == "test"
    assert summary["tuned"] is False
    assert summary["frozen_config"] is True
    assert summary["trained"] is False  # no checkpoint was supplied
    assert summary["n_traces"] == 2
    assert summary["rollout"]["comet"]["status"] == "skipped"
    bleu = summary["rollout"]["bleu"]
    assert bleu["status"] in {"ok", "unavailable", "error"}
    if bleu["status"] != "ok":
        assert "score" not in bleu
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "traces.jsonl").is_file()
    assert (tmp_path / "config.json").is_file()


def test_run_confirm_reuses_checkpoint_vocabulary(tmp_path):
    """A dev checkpoint must load even when test sentences add new words."""
    harness = make_harness()
    cfg = harness["cfg"]
    preflight_dir = tmp_path / "preflight"
    kaggle.run_preflight(
        limit=3,
        output_dir=str(preflight_dir),
        label_states=2,
        rollout_sentences=1,
        seed=cfg.seed,
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=harness["sentences"],
        cfg=cfg,
        predictor_id=0,
        translator_id=0,
        timer=harness["timer"],
    )
    # This test sentence introduces words the dev vocabulary never saw.
    test_sentence = Sentence(
        "test:0140",
        "guten morgen wir lernen heute .",
        "good morning we learn today .",
        split="test",
    )
    summary = kaggle.run_confirm(
        offset=140,
        limit=1,
        output_dir=str(tmp_path / "confirm"),
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=[test_sentence],
        checkpoint_path=preflight_dir / "checkpoint.pt",
        timer=harness["timer"],
    )
    assert summary["trained"] is True
    assert summary["n_traces"] == 1
    assert summary["seed"] == cfg.seed
    assert summary["checkpoint"] == str(preflight_dir / "checkpoint.pt")


def test_run_confirm_accepts_clean_offset_by_default(tmp_path):
    harness = make_harness()
    test_sentences = [
        Sentence(f"test:{offset:04d}", source, reference, split="test")
        for offset, (_sid, source, reference) in enumerate(SENTENCES, start=140)
    ]
    builder, network = _frozen_builder_network(harness)
    summary = kaggle.run_confirm(
        offset=140,
        limit=1,
        output_dir=str(tmp_path),
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=test_sentences,
        cfg=harness["cfg"],
        builder=builder,
        network=network,
        timer=harness["timer"],
    )
    assert summary["offset"] == 140
    assert summary["n_traces"] == 1
    assert summary["vocab_source"] == "supplied_frozen"


# ----------------------------------------------------------------------
# Labeling
# ----------------------------------------------------------------------
def test_label_sentence_labels_every_one_step_position_and_omits_candidates():
    harness = make_harness()
    cfg = harness["cfg"]
    sentence = harness["sentences"][0]
    row = kaggle.label_sentence(harness["labeler"], sentence, stride=1, cfg=cfg)

    positions = list(range(cfg.min_start, len(sentence.source.split())))
    assert [state["read_pos"] for state in row["states"]] == positions
    assert row["stride"] == 1
    assert row["label_format"] == kaggle.LABEL_FORMAT
    assert row["behavior"]["read_prob"] == pytest.approx(0.25)
    assert row["behavior"]["fixed_action"] == "b4h6"
    for index, state in enumerate(row["states"]):
        assert state["outcomes"][0]["action"] == READ.label
        assert state["oracle_words"] >= 0
        # Candidate text is omitted from the persisted row.
        assert "candidates" not in json.dumps(state)
        assert state["successor_observed"].split() == (
            state["observed"].split() + [sentence.source.split()[state["read_pos"]]]
        )
        # Every outcome persists compact successor inputs: the successor
        # observed prefix is shared, and each action stores its own committed
        # prefix and emitted count.
        before = state["committed_before"]
        for outcome in state["outcomes"]:
            assert "next_committed" in outcome
            assert outcome["next_committed"][: len(before)] == before
            assert outcome["emitted"] == len(outcome["next_committed"]) - len(before)
        # The committed prefix only ever grows by appended words.
        after = state["committed_after"]
        assert after[: len(before)] == before
        # Only the final state is terminal.
        assert state["terminal"] is (index == len(row["states"]) - 1)
        if state["terminal"]:
            assert state["terminal_value"] is not None


def test_label_sentence_raises_for_multi_step_stride():
    harness = make_harness()
    cfg = harness["cfg"]
    sentence = harness["sentences"][0]
    with pytest.raises(ValueError, match="one-step|stride"):
        kaggle.label_sentence(harness["labeler"], sentence, stride=2, cfg=cfg)


def test_label_sentence_refuses_test_and_holdout_sentences():
    harness = make_harness()
    cfg = harness["cfg"]
    with pytest.raises(ValueError, match="test"):
        kaggle.label_sentence(
            harness["labeler"],
            Sentence("test:0140", "hallo welt wir lernen heute .", "x"),
            1,
            cfg,
        )
    # dev:0009 hashes into holdout bucket 1.
    with pytest.raises(ValueError, match="holdout|bucket"):
        kaggle.label_sentence(
            harness["labeler"],
            Sentence("dev:0009", "hallo welt wir lernen heute .", "x"),
            1,
            cfg,
        )


# ----------------------------------------------------------------------
# Dev resume
# ----------------------------------------------------------------------
def test_run_dev_excludes_holdout_and_resumes_by_sentence_id(tmp_path):
    harness = make_harness()
    cfg = harness["cfg"]
    sentences = list(harness["sentences"]) + [
        Sentence(
            "dev:0009",
            "hallo welt wir lernen heute .",
            "hello world we learn today .",
            split="dev",
        )
    ]

    def run():
        return kaggle.run_dev(
            limit=0,
            stride=1,
            output_dir=str(tmp_path),
            max_hours=1.0,
            seed=cfg.seed,
            predictor=harness["predictor"],
            translator=harness["translator"],
            sentences=sentences,
            cfg=cfg,
            predictor_id=0,
            translator_id=0,
            timer=harness["timer"],
        )

    summary = run()
    labels_path = tmp_path / "labels.jsonl"
    rows = [
        json.loads(line)
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [row["sentence_id"] for row in rows]
    assert "dev:0009" not in ids  # holdout bucket 1 is never labelled
    assert set(ids) == {"s0", "s1", "s2"}
    assert summary["holdout_bucket"] == 1
    assert summary["n_label_rows"] == 3
    # Only dev:0009 is in the holdout bucket, so one rollout ran.
    assert summary["rollout"]["n"] == 1

    # A resumed session must not append duplicate per-sentence rows.
    before = len(rows)
    resumed = run()
    after = len(
        [
            line
            for line in labels_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )
    assert after == before
    assert resumed["newly_labeled"] == 0


def test_run_dev_refuses_labels_written_by_an_older_format(tmp_path):
    harness = make_harness()
    cfg = harness["cfg"]
    output_dir = tmp_path / "dev"
    output_dir.mkdir()
    stale_row = {
        "sentence_id": "s0",
        "split": "dev",
        "source": harness["sentences"][0].source,
        "stride": 4,
        # No `label_format`: this is the pre-fix multi-step schema.
        "states": [{"read_pos": 3, "observed": "hallo", "outcomes": []}],
    }
    (output_dir / "labels.jsonl").write_text(
        json.dumps(stale_row) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="label format|label_format"):
        kaggle.run_dev(
            limit=0,
            output_dir=str(output_dir),
            max_hours=1.0,
            seed=cfg.seed,
            predictor=harness["predictor"],
            translator=harness["translator"],
            sentences=harness["sentences"],
            cfg=cfg,
            predictor_id=0,
            translator_id=0,
            timer=harness["timer"],
        )


# ----------------------------------------------------------------------
# Kernel packaging
# ----------------------------------------------------------------------
def _load_prepare_module():
    path = Path(__file__).resolve().parents[3] / "prepare_cavc_kernel.py"
    spec = importlib.util.spec_from_file_location("prepare_cavc_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_cavc_kernel_bundles_expected_sources(tmp_path):
    module = _load_prepare_module()
    sources = module.bundle_sources()
    assert module.bundle_paths() == sorted(sources)

    controller_dir = Path(__file__).resolve().parents[1]
    expected_controller = {
        f"week/controller_v3/{path.name}"
        for path in controller_dir.glob("*.py")
    }
    assert expected_controller <= set(sources)
    assert "week/controller_v3/kaggle.py" in sources
    assert "week/controller_v3/__init__.py" in sources
    assert "week/cpu_embedding.py" in sources
    assert "paired/run_pfba.py" in sources
    assert not any("tests/" in relative for relative in sources)

    # The embedded layout is the one controller_v3 resolves through parents[2].
    script = module.build_entry_script(sources)
    compile(script, "kaggle_entry.py", "exec")
    assert "/tmp/pfba_cavc" in script
    assert "controller_v3.kaggle" in script
    assert "week/controller_v3" in script
    assert "paired/run_pfba.py" in script

    for relative, text in sources.items():
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert module.source_hashes(sources)[relative] == digest

    metadata = module.build_metadata()
    assert metadata["id"] == "eunectesqaq/pfba-cavc-v3-preflight"
    assert metadata["code_file"] == "kaggle_entry.py"
    assert metadata["kernel_type"] == "script"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["dataset_sources"] == []
    assert json.loads(json.dumps(metadata)) == metadata

    # Writing to a temp stage exercises the real output path without touching
    # the repository's generated kernel files.
    result = module.write_kernel(output_dir=tmp_path)
    assert Path(result["entry"]).is_file()
    assert Path(result["metadata"]).is_file()
    written = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
    assert written["id"] == metadata["id"]
    assert written["code_file"] == "kaggle_entry.py"


# ----------------------------------------------------------------------
# Dry-run
# ----------------------------------------------------------------------
def test_dry_run_prints_plan_without_loading_models(capsys, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("dry-run must not resolve adapters")

    monkeypatch.setattr(kaggle, "resolve_adapters", explode)
    assert kaggle.main(["--mode", "dry-run"]) == 0
    output = capsys.readouterr().out
    assert "dry run" in output
    plan = json.loads(output[output.index("{") :])
    assert plan["mode"] == "dry-run"
    assert plan["model_pair_ids"] == {"predictor_id": 0, "translator_id": 0}
    assert "test:0140" in plan["split_policy"]["confirm"]


def test_confirm_parser_rejects_consumed_offset_before_loading_models(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the offset guard must run before adapters load")

    monkeypatch.setattr(kaggle, "resolve_adapters", explode)
    with pytest.raises(ValueError, match="consumed|140"):
        kaggle.main(["--mode", "confirm", "--offset", "139"])


# ----------------------------------------------------------------------
# Fix round 1: model-pair ids, deadlines, vocab isolation, guards
# ----------------------------------------------------------------------
class RecordingBuilder(FeatureBuilder):
    """Real builder that records the model ids passed to `build`."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[int, int]] = []

    def build(self, observed, committed, **kwargs):
        self.calls.append(
            (
                int(kwargs.get("predictor_id", 0)),
                int(kwargs.get("translator_id", 0)),
            )
        )
        return super().build(observed, committed, **kwargs)


class RecordingNetwork(torch.nn.Module):
    """Fake value network that records the model-id tensors it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_predictor: list[int] = []
        self.seen_translator: list[int] = []

    def encode(self, numeric, predictor_id, translator_id, src_ids, tgt_ids):
        del src_ids, tgt_ids
        self.seen_predictor.extend(
            int(value) for value in predictor_id.flatten().tolist()
        )
        self.seen_translator.extend(
            int(value) for value in translator_id.flatten().tolist()
        )
        return torch.zeros((numeric.size(0), 4), dtype=torch.float32)

    def read_value(self, z):
        return torch.zeros(z.size(0), dtype=torch.float32)

    def write_values(self, z, actions):
        batch = z.size(0)
        count = len(actions)
        return WriteOutputs(
            value=torch.ones((batch, count), dtype=torch.float32),
            error_risk=torch.zeros((batch, count), dtype=torch.float32),
            no_emit=torch.zeros((batch, count), dtype=torch.float32),
            expected_emit=torch.ones((batch, count), dtype=torch.float32),
        )


class FakeRuntime:
    """Minimal runtime: a write emits nothing and EOS returns one word."""

    def execute(self, observed, committed, action, seed):
        del observed, action, seed
        return StepResult(
            candidates=(),
            selected="",
            committed_after=tuple(committed),
            pred_s=0.0,
            mt_s=0.0,
            spent_s=0.0,
            vote_top1=0.0,
            vote_margin=0.0,
            vote_entropy=0.0,
            no_emit=True,
        )

    def translate(self, sources, prefix, final=False):
        del sources, prefix, final
        return ["final words"], 0.0


class RecordingCostModel:
    """Cost model that records the ids used by the rollout lookup."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, Action]] = []

    def predict(
        self, predictor_id, translator_id, action, src_words, tgt_words
    ):
        del src_words, tgt_words
        self.calls.append((int(predictor_id), int(translator_id), action))
        return CostEstimate(0.1, 0.1)


def test_rollout_injects_model_pair_ids_into_scoring_and_cost():
    cfg = ControllerConfig(seed=7, actions=(Action(2, 4),))
    builder = RecordingBuilder()
    network = RecordingNetwork()
    cost_model = RecordingCostModel()
    sentence = Sentence(
        "s0", "hallo welt wir lernen heute .", "hello world we learn today ."
    )

    trace = run_rollout(
        sentence,
        None,
        None,
        network,
        builder,
        cost_model,
        cfg,
        runtime=FakeRuntime(),
        predictor_id=1,
        translator_id=1,
    )

    assert trace.sentence_id == "s0"
    # The feature builder saw the injected pair on every decision.
    assert builder.calls and all(pair == (1, 1) for pair in builder.calls)
    # Q scoring uses the same pair: the encoder received 1/1 tensors.
    assert network.seen_predictor and set(network.seen_predictor) == {1}
    assert network.seen_translator and set(network.seen_translator) == {1}
    # The measured cost lookup reads the vector's ids, so it also sees 1/1.
    assert cost_model.calls
    assert all(
        (predictor_id, translator_id) == (1, 1)
        for predictor_id, translator_id, _action in cost_model.calls
    )


@dataclass(frozen=True)
class StubLabeler:
    """Labeler stub with hand-set rewards for the trajectory test."""

    def oracle_words(self, source, committed):
        del source, committed
        return tuple("final words".split())

    def read_outcome(self, committed):
        return ActionOutcome(
            action=READ,
            new_words=(),
            matched=0,
            incorrect=0,
            emitted=0,
            no_emit=True,
            next_committed=tuple(committed),
            pred_s=0.0,
            mt_s=0.0,
            vote_top1=0.0,
            vote_margin=0.0,
            vote_entropy=0.0,
        )

    def label_state(self, observed, committed, source, actions, seed):
        del observed, source, seed
        specs = {
            Action(2, 4): (2, ("a", "b")),
            Action(2, 6): (2, ("x", "y", "z")),
            Action(4, 6): (1, ("c",)),
        }
        outcomes = []
        for action in actions:
            if action.is_read:
                outcomes.append(self.read_outcome(committed))
                continue
            matched, words = specs[action]
            outcomes.append(
                ActionOutcome(
                    action=action,
                    new_words=words,
                    matched=matched,
                    incorrect=0,
                    emitted=len(words),
                    no_emit=False,
                    next_committed=tuple(committed) + words,
                    pred_s=0.1,
                    mt_s=0.1,
                    vote_top1=0.75,
                    vote_margin=0.5,
                    vote_entropy=0.1,
                )
            )
        return outcomes


def test_label_sentence_behavior_policy_can_choose_read():
    cfg = ControllerConfig(
        seed=7,
        actions=(Action(2, 4), Action(2, 6), Action(4, 6)),
    )
    sentence = Sentence(
        "s0", "hallo welt wir lernen heute .", "hello world we learn today ."
    )
    row = kaggle.label_sentence(
        StubLabeler(), sentence, stride=1, cfg=cfg, read_prob=1.0
    )

    assert row["behavior"] == {"read_prob": 1.0, "fixed_action": "b4h6"}
    assert [state["chosen"] for state in row["states"]] == [
        READ.label for _ in row["states"]
    ]
    # READ keeps the committed prefix at every state.
    assert all(
        state["committed_before"] == state["committed_after"]
        for state in row["states"]
    )
    # The Bellman target still sees every write outcome; the behavior policy
    # only chooses which committed prefix becomes the next training state.
    assert {
        outcome["action"] for outcome in row["states"][0]["outcomes"]
    } == {"read", "b2h4", "b2h6", "b4h6"}


def test_label_sentence_behavior_policy_uses_the_fixed_write():
    cfg = ControllerConfig(
        seed=7,
        actions=(Action(2, 4), Action(2, 6), Action(4, 6)),
    )
    sentence = Sentence(
        "s0", "hallo welt wir lernen heute .", "hello world we learn today ."
    )
    row = kaggle.label_sentence(
        StubLabeler(), sentence, stride=1, cfg=cfg, read_prob=0.0
    )

    assert {state["chosen"] for state in row["states"]} == {"b4h6"}
    # StubLabeler's b4h6 emits ("c",), so the trajectory keeps growing.
    assert row["states"][0]["committed_after"] == ["c"]
    assert row["states"][1]["committed_before"] == ["c"]


def test_label_sentence_rewards_reflect_post_action_backlog():
    cfg = ControllerConfig(
        seed=7,
        actions=(Action(2, 4), Action(2, 6), Action(4, 6)),
    )
    sentence = Sentence(
        "s0", "hallo welt wir lernen heute .", "hello world we learn today ."
    )
    row = kaggle.label_sentence(
        StubLabeler(), sentence, stride=1, cfg=cfg, read_prob=0.0
    )
    rewards = {
        outcome["action"]: outcome["reward"]
        for outcome in row["states"][0]["outcomes"]
    }
    # b2h6 commits three words, b2h4 commits two, so b2h6 closes more of the
    # post-action backlog and earns a strictly better delay-adjusted reward.
    assert rewards["b2h6"] > rewards["b2h4"]
    # b4h6 matches only one word and emits one.
    assert rewards["b4h6"] < rewards["b2h4"]
    # READ emits nothing, so its post-action backlog is largest.
    assert rewards["read"] < rewards["b4h6"]


def test_run_dev_stops_gracefully_when_deadline_has_passed(tmp_path):
    harness = make_harness()
    cfg = harness["cfg"]

    def run(max_hours):
        return kaggle.run_dev(
            limit=0,
            stride=1,
            output_dir=str(tmp_path),
            max_hours=max_hours,
            seed=cfg.seed,
            predictor=harness["predictor"],
            translator=harness["translator"],
            sentences=harness["sentences"],
            cfg=cfg,
            predictor_id=0,
            translator_id=0,
            timer=harness["timer"],
        )

    first = run(max_hours=1.0)
    assert first["stopped_reason"] is None
    checkpoint = tmp_path / "checkpoint.pt"
    assert checkpoint.is_file()
    before = checkpoint.read_bytes()

    second = run(max_hours=0.0)
    assert second["stopped_reason"] == "deadline"
    assert second["trained"] is False
    assert second["training"]["status"] == "skipped"
    assert second["rollout"]["n"] == 0
    # No target was computed, so the logged target wall time is exactly zero.
    assert second["target_wall_s"] == 0.0
    assert second["training"]["target_seconds"] == 0.0
    # The trained checkpoint is kept rather than overwritten untrained.
    assert checkpoint.read_bytes() == before
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "traces.jsonl").is_file()


def test_run_dev_builds_vocab_from_train_buckets_only(tmp_path):
    harness = make_harness()
    cfg = harness["cfg"]
    holdout = Sentence(
        "dev:0009",
        "quark hallo welt wir lernen heute .",
        "quark hello world we learn today .",
        split="dev",
    )
    summary = kaggle.run_dev(
        limit=0,
        stride=1,
        output_dir=str(tmp_path),
        max_hours=1.0,
        seed=cfg.seed,
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=[*harness["sentences"], holdout],
        cfg=cfg,
        predictor_id=0,
        translator_id=0,
        timer=harness["timer"],
    )
    assert summary["rollout"]["n"] == 1  # the holdout sentence was rolled out
    payload = torch.load(
        tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    vocab = payload["extra"]["vocab"]
    assert "quark" not in vocab  # holdout-only word never enters the vocab
    assert "hallo" in vocab


def test_run_dev_and_preflight_reject_test_sentences(tmp_path):
    harness = make_harness()
    test_sentence = Sentence(
        "test:0140",
        "hallo welt wir lernen heute .",
        "hello world we learn today .",
        split="test",
    )
    common = dict(
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=[test_sentence, *harness["sentences"]],
        cfg=harness["cfg"],
        timer=harness["timer"],
    )
    with pytest.raises(ValueError, match="test"):
        kaggle.run_dev(
            limit=0,
            stride=1,
            output_dir=str(tmp_path / "dev"),
            max_hours=1.0,
            **common,
        )
    with pytest.raises(ValueError, match="test"):
        kaggle.run_preflight(
            limit=3,
            output_dir=str(tmp_path / "preflight"),
            **common,
        )


def test_read_jsonl_skips_and_repairs_a_truncated_tail(tmp_path):
    path = tmp_path / "labels.jsonl"
    complete = '{"sentence_id": "a"}\n{"sentence_id": "b"}\n'
    path.write_text(complete + '{"sentence_id": "c"', encoding="utf-8")

    rows = kaggle._read_jsonl(path)
    assert [row["sentence_id"] for row in rows] == ["a", "b"]

    kaggle._repair_jsonl_tail(path)
    repaired = path.read_text(encoding="utf-8")
    assert repaired == complete
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"sentence_id": "c"}\n')
    assert [row["sentence_id"] for row in kaggle._read_jsonl(path)] == [
        "a",
        "b",
        "c",
    ]


def test_reuse_cost_records_rejects_other_model_pair(tmp_path):
    path = tmp_path / "cost_records.json"
    kaggle._write_cost_records(
        path,
        [(0, 0, Action(2, 4), 0.1, 0.2)],
        {"predictor_id": 0, "translator_id": 0},
    )
    assert kaggle._reuse_cost_records(path, 0, 0) is not None
    # A table measured for another pair must raise, not silently recalibrate.
    with pytest.raises(ValueError, match="model pair|resolved pair"):
        kaggle._reuse_cost_records(path, 1, 1)


# ----------------------------------------------------------------------
# Fix round 2: never present untrained weights as trained
# ----------------------------------------------------------------------
def test_deadline_skip_without_prior_checkpoint_writes_no_checkpoint(tmp_path):
    harness = make_harness()
    cfg = harness["cfg"]

    def run(max_hours):
        return kaggle.run_dev(
            limit=0,
            stride=1,
            output_dir=str(tmp_path),
            max_hours=max_hours,
            seed=cfg.seed,
            predictor=harness["predictor"],
            translator=harness["translator"],
            sentences=harness["sentences"],
            cfg=cfg,
            predictor_id=0,
            translator_id=0,
            timer=harness["timer"],
        )

    run(max_hours=1.0)
    checkpoint = tmp_path / "checkpoint.pt"
    assert checkpoint.is_file()
    # Simulate labels surviving a killed session whose checkpoint was lost.
    checkpoint.unlink()

    summary = run(max_hours=0.0)
    assert summary["stopped_reason"] == "deadline"
    assert summary["trained"] is False
    assert summary["checkpoint"]["status"] == "not_written"
    assert summary["checkpoint"]["path"] is None
    assert summary["artifacts"]["checkpoint"] is None
    assert not checkpoint.exists()
    # Risk calibration is gated on the network actually being trained.
    assert summary["risk_calibration"]["status"] == "unavailable"
    assert summary["risk_calibration"]["reason"] == (
        "network was not trained in this run"
    )
    assert "brier" not in summary["risk_calibration"]["error_risk"]


def _feature_stats_payload():
    return {
        "n": 1,
        "features": {
            name: {"mean": 0.0, "std": 0.0, "nonzero": 0, "missing": 0}
            for name in NUMERIC_FEATURES
        },
    }


def _small_pair():
    return {
        "predictor": "small",
        "translator": "small",
        "predictor_id": 0,
        "translator_id": 0,
    }


def _write_checkpoint(path, harness, extra, cost_model=None):
    """Write a controller checkpoint with a caller-controlled `trained` flag."""
    cfg = harness["cfg"]
    kaggle.save_checkpoint(
        path,
        harness["network"],
        cfg,
        [0.0] * cfg.numeric_dim,
        [1.0] * cfg.numeric_dim,
        cost_model,
        extra={
            "vocab": list(harness["vocab"].id_to_token),
            "length_ratio": 1.0,
            "model_pair": _small_pair(),
            "feature_stats": _feature_stats_payload(),
            **extra,
        },
    )


def _confirm_test_sentences():
    return [
        Sentence(f"test:{offset:04d}", source, reference, split="test")
        for offset, (_sid, source, reference) in enumerate(SENTENCES, start=140)
    ]


@pytest.mark.parametrize(
    "extra",
    [{"trained": False}, {}],
    ids=["explicit-false", "missing-key"],
)
def test_confirm_refuses_untrained_checkpoint_without_override(tmp_path, extra):
    harness = make_harness()
    checkpoint = tmp_path / f"untrained-{len(extra)}.pt"
    _write_checkpoint(checkpoint, harness, extra)

    kwargs = dict(
        offset=140,
        limit=1,
        output_dir=str(tmp_path / "confirm"),
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=_confirm_test_sentences(),
        checkpoint_path=checkpoint,
        timer=harness["timer"],
    )
    with pytest.raises(ValueError, match="not trained|untrained"):
        kaggle.run_confirm(**kwargs)

    summary = kaggle.run_confirm(allow_untrained=True, **kwargs)
    assert summary["trained"] is False
    assert summary["allow_untrained"] is True
    assert summary["checkpoint"] == str(checkpoint)
    assert summary["n_traces"] == 1


def test_confirm_on_trained_checkpoint_reports_trained(tmp_path):
    harness = make_harness()
    cfg = harness["cfg"]
    preflight_dir = tmp_path / "preflight"
    kaggle.run_preflight(
        limit=3,
        output_dir=str(preflight_dir),
        label_states=2,
        rollout_sentences=1,
        seed=cfg.seed,
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=harness["sentences"],
        cfg=cfg,
        predictor_id=0,
        translator_id=0,
        timer=harness["timer"],
    )
    payload = torch.load(
        preflight_dir / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    assert payload["extra"]["trained"] is True

    summary = kaggle.run_confirm(
        offset=140,
        limit=1,
        output_dir=str(tmp_path / "confirm"),
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=_confirm_test_sentences(),
        checkpoint_path=preflight_dir / "checkpoint.pt",
        timer=harness["timer"],
    )
    assert summary["trained"] is True
    assert summary["allow_untrained"] is False
    assert summary["n_traces"] == 1


# ----------------------------------------------------------------------
# Task 9: cost/risk/fixed-baseline/diagnostics wiring
# ----------------------------------------------------------------------
def test_run_dev_writes_feature_stats_paired_and_risk_calibration(tmp_path):
    harness = make_harness()
    cfg = harness["cfg"]
    holdout = Sentence(
        "dev:0009",
        "hallo welt wir lernen heute .",
        "hello world we learn today .",
        split="dev",
    )
    summary = kaggle.run_dev(
        limit=0,
        stride=1,
        output_dir=str(tmp_path),
        max_hours=1.0,
        seed=cfg.seed,
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=[*harness["sentences"], holdout],
        cfg=cfg,
        predictor_id=0,
        translator_id=0,
        timer=harness["timer"],
    )

    assert summary["post_controller"] == "disabled"
    assert summary["candidate_mode"] == "per_candidate"
    assert summary["rollout"]["n"] == 1
    assert summary["rollout_fixed"]["n"] == 1
    assert summary["n_train_targets"] < summary["n_targets"]
    assert summary["n_risk_calibration_targets"] > 0
    # Bellman-target construction wall time is logged at the summary top level
    # and in the nested training block.
    assert summary["target_wall_s"] >= 0.0
    assert summary["training"]["target_calls"] > 0
    assert summary["target_wall_s"] == pytest.approx(
        summary["training"]["target_seconds"]
    )
    assert summary["risk_calibration"]["status"] == "ok"
    assert math.isfinite(summary["risk_calibration"]["error_risk"]["brier"])
    assert math.isfinite(summary["risk_calibration"]["no_emit"]["ece"])

    # The fixed baseline runs the same runtime path and the paired interval
    # file is written with honest per-metric statuses.
    paired = json.loads((tmp_path / "paired.json").read_text(encoding="utf-8"))
    assert paired["fixed"] == "(4,6)"
    assert set(paired["metrics"]) == {"bleu", "laal", "completion_s", "comet"}
    assert paired["metrics"]["comet"]["status"] == "skipped"
    assert paired["metrics"]["completion_s"]["status"] == "ok"
    assert paired["metrics"]["completion_s"]["n"] == 1
    assert (tmp_path / "traces_fixed.jsonl").is_file()

    feature_payload = json.loads(
        (tmp_path / "feature_stats.json").read_text(encoding="utf-8")
    )
    assert feature_payload["candidate_mode"] == "per_candidate"
    assert feature_payload["training"]["n"] == summary["n_states"]
    assert set(feature_payload["serving"]["features"]) == set(NUMERIC_FEATURES)
    assert "warnings" in feature_payload["serving"]

    # Checkpoint metadata carries the pair, vocabulary, and feature stats that
    # confirm validates against its resolved adapters.
    payload = torch.load(
        tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    extra = payload["extra"]
    assert extra["model_pair"]["predictor_id"] == 0
    assert extra["vocab"] and "hallo" in extra["vocab"]
    assert set(extra["feature_stats"]["features"]) == set(NUMERIC_FEATURES)
    assert extra["candidate_mode"] == "per_candidate"
    assert extra["post_controller"] == "disabled"


def test_run_preflight_records_post_controller_switch(tmp_path):
    harness = make_harness()
    cfg = harness["cfg"]

    def run(name, enabled):
        return kaggle.run_preflight(
            limit=3,
            output_dir=str(tmp_path / name),
            label_states=2,
            rollout_sentences=1,
            seed=cfg.seed,
            predictor=harness["predictor"],
            translator=harness["translator"],
            sentences=harness["sentences"],
            cfg=cfg,
            predictor_id=0,
            translator_id=0,
            timer=harness["timer"],
            post_controller=enabled,
        )

    disabled = run("off", False)
    enabled = run("on", True)
    assert disabled["post_controller"] == "disabled"
    assert enabled["post_controller"] == "enabled"
    assert any("training-only" in note for note in disabled["notes"])
    assert any("uncalibrated" in note for note in enabled["notes"])


def test_label_sentence_records_the_candidate_mode():
    harness = make_harness()
    cfg = harness["cfg"]
    sentence = harness["sentences"][0]
    row = kaggle.label_sentence(harness["labeler"], sentence, cfg=cfg, stride=1)
    assert row["candidate_mode"] == "per_candidate"

    batched = CounterfactualLabeler(
        harness["predictor"],
        harness["translator"],
        cfg,
        timer=harness["timer"],
        candidate_mode="batched",
    )
    batched_row = kaggle.label_sentence(batched, sentence, cfg=cfg, stride=1)
    assert batched_row["candidate_mode"] == "batched"


@pytest.mark.parametrize(
    "extra, match",
    [
        (
            {
                "trained": True,
                "model_pair": {
                    "predictor": "large",
                    "translator": "large",
                    "predictor_id": 1,
                    "translator_id": 1,
                },
            },
            "mismatch",
        ),
        ({"trained": True, "vocab": None}, "vocabulary"),
        ({"trained": True, "feature_stats": None}, "feature_stats"),
    ],
    ids=["pair-mismatch", "missing-vocab", "missing-feature-stats"],
)
def test_run_confirm_refuses_bad_checkpoint_metadata(tmp_path, extra, match):
    harness = make_harness()
    checkpoint = tmp_path / "bad.pt"
    _write_checkpoint(checkpoint, harness, extra)

    with pytest.raises(ValueError, match=match):
        kaggle.run_confirm(
            offset=140,
            limit=1,
            output_dir=str(tmp_path / "confirm"),
            predictor=harness["predictor"],
            translator=harness["translator"],
            sentences=_confirm_test_sentences(),
            checkpoint_path=checkpoint,
            timer=harness["timer"],
        )


def test_run_confirm_writes_fixed_baseline_and_paired_metrics(tmp_path):
    harness = make_harness()
    builder, network = _frozen_builder_network(harness)
    summary = kaggle.run_confirm(
        offset=140,
        limit=1,
        output_dir=str(tmp_path),
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=_confirm_test_sentences(),
        cfg=harness["cfg"],
        builder=builder,
        network=network,
        timer=harness["timer"],
    )

    assert summary["post_controller"] == "disabled"
    assert summary["rollout"]["n"] == 1
    assert summary["rollout_fixed"]["n"] == 1
    assert summary["risk_calibration"]["status"] == "unavailable"
    paired = json.loads((tmp_path / "paired.json").read_text(encoding="utf-8"))
    assert paired["metrics"]["comet"]["status"] == "skipped"
    assert paired["metrics"]["laal"]["status"] in {"ok", "unavailable"}
    assert (tmp_path / "traces_fixed.jsonl").is_file()


def test_run_dev_refuses_mixed_candidate_modes_in_labels(tmp_path):
    harness = make_harness()
    cfg = harness["cfg"]
    output_dir = tmp_path / "dev"
    output_dir.mkdir()
    row = {
        "sentence_id": "s0",
        "split": "dev",
        "source": harness["sentences"][0].source,
        "stride": 1,
        "label_format": kaggle.LABEL_FORMAT,
        "candidate_mode": "batched",
        "states": [{"read_pos": 3, "observed": "hallo", "outcomes": []}],
    }
    (output_dir / "labels.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="candidate_mode"):
        kaggle.run_dev(
            limit=0,
            output_dir=str(output_dir),
            max_hours=1.0,
            seed=cfg.seed,
            predictor=harness["predictor"],
            translator=harness["translator"],
            sentences=harness["sentences"],
            cfg=cfg,
            predictor_id=0,
            translator_id=0,
            timer=harness["timer"],
            candidate_mode="per_candidate",
        )


def test_run_confirm_refuses_to_build_a_test_vocabulary(tmp_path):
    harness = make_harness()
    with pytest.raises(ValueError, match="refusing to build a vocabulary"):
        kaggle.run_confirm(
            offset=140,
            limit=1,
            output_dir=str(tmp_path),
            predictor=harness["predictor"],
            translator=harness["translator"],
            sentences=_confirm_test_sentences(),
            cfg=harness["cfg"],
            timer=harness["timer"],
        )


# ----------------------------------------------------------------------
# Fix round 1: training/successor clocks use marginal wall time
# ----------------------------------------------------------------------
class ConstantStageCost:
    """Cost model whose stage estimates match a fixed timer exactly."""

    def __init__(self, pred_s: float = 0.2, mt_s: float = 0.05) -> None:
        self.pred_s = float(pred_s)
        self.mt_s = float(mt_s)

    def marginal_wall(
        self,
        t_pred_free,
        t_mt_free,
        release_s,
        dependency_s,
        predictor_id,
        translator_id,
        action,
        src_words,
        tgt_words,
    ):
        pred_start = max(t_pred_free, release_s)
        pred_end = pred_start + self.pred_s
        mt_start = max(t_mt_free, pred_end, dependency_s)
        mt_end = mt_start + self.mt_s
        baseline = max(t_pred_free, t_mt_free, release_s)
        return max(0.0, max(pred_end, mt_end) - baseline)

    def predict(self, predictor_id, translator_id, action, src_words, tgt_words):
        return CostEstimate(self.pred_s, self.mt_s)


def test_labeling_clock_bills_marginal_not_the_large_serial_sum():
    cfg = ControllerConfig(
        seed=7,
        actions=(Action(2, 4), Action(2, 6), Action(4, 6)),
    )
    sentence = Sentence(
        "s0", "hallo welt wir lernen heute .", "hello world we learn today ."
    )

    class TinyMarginalCost(ConstantStageCost):
        def marginal_wall(self, *args, **kwargs):
            return 0.01

        def predict(self, *args, **kwargs):
            # Large serial stage estimates: they must not drive the clock.
            return CostEstimate(10.0, 10.0)

    row = kaggle.label_sentence(
        StubLabeler(),
        sentence,
        stride=1,
        cfg=cfg,
        read_prob=0.0,
        cost_model=TinyMarginalCost(),
    )
    behavior = [
        next(
            outcome
            for outcome in state["outcomes"]
            if outcome["action"] == "b4h6"
        )
        for state in row["states"]
    ]
    assert all(outcome["marginal_wall_s"] == pytest.approx(0.01) for outcome in behavior)
    assert all(outcome["pred_s"] == 0.1 for outcome in behavior)
    # 0.6 s source arrival + 0.01 s marginal, not 0.6 + (0.1 + 0.1) serial.
    assert row["states"][1]["now_s"] == pytest.approx(0.61)
    assert row["states"][1]["now_s"] != pytest.approx(0.8)


def test_training_and_rollout_clocks_agree_for_the_same_actions():
    harness = make_harness()
    cfg = ControllerConfig(
        seed=7,
        actions=(Action(4, 6),),
        min_start=3,
        arrival_gap=0.3,
    )
    # Per-candidate (4,6): four 0.05 s samples + one 0.05 s translation.
    cost_model = PipelineCostModel(ConstantStageCost(pred_s=0.2, mt_s=0.05))
    sentence = harness["sentences"][0]

    row = kaggle.label_sentence(
        harness["labeler"],
        sentence,
        stride=1,
        cfg=cfg,
        builder=harness["builder"],
        read_prob=0.0,
        cost_model=cost_model,
    )
    behavior_marginals = [
        next(
            outcome["marginal_wall_s"]
            for outcome in state["outcomes"]
            if outcome["action"] == "b4h6"
        )
        for state in row["states"]
    ]

    trace = run_fixed_rollout(
        sentence,
        harness["predictor"],
        harness["translator"],
        harness["builder"],
        cost_model,
        cfg,
        timer=harness["timer"],
    )
    fixed_steps = [
        step for step in trace.steps if step["action"] == "b4h6"
    ]
    assert behavior_marginals == pytest.approx(
        [step["marginal_wall_s"] for step in fixed_steps]
    )

    # Absolute virtual clocks agree too: the labeling walk's pre-action clock
    # matches the fixed rollout's completion before the same write.
    completion = 0.0
    fixed_before = []
    for step in fixed_steps:
        fixed_before.append(completion)
        release = max(
            completion, (step["read_pos"] - 1) * cfg.arrival_gap
        )
        completion = release + step["marginal_wall_s"]
    assert [state["now_s"] for state in row["states"]] == pytest.approx(
        fixed_before
    )


def test_outcome_records_roundtrip_marginal_wall():
    record = {
        "action": "b4h6",
        "branches": 4,
        "horizon": 6,
        "pred_s": 10.0,
        "mt_s": 10.0,
        "marginal_wall_s": 0.01,
    }
    outcome = kaggle._outcome_from_record(record)
    assert outcome.marginal_wall_s == pytest.approx(0.01)
    assert kaggle._step_seconds(outcome) == pytest.approx(0.01)

    # A legacy record without the field keeps the documented serial fallback.
    legacy = kaggle._outcome_from_record(
        {"action": "b4h6", "branches": 4, "horizon": 6,
         "pred_s": 10.0, "mt_s": 10.0}
    )
    assert legacy.marginal_wall_s is None
    assert kaggle._step_seconds(legacy) == pytest.approx(20.0)


def test_labeling_without_a_cost_model_falls_back_to_serial_compute():
    cfg = ControllerConfig(
        seed=7,
        actions=(Action(2, 4), Action(2, 6), Action(4, 6)),
    )
    sentence = Sentence(
        "s0", "hallo welt wir lernen heute .", "hello world we learn today ."
    )
    row = kaggle.label_sentence(
        StubLabeler(), sentence, stride=1, cfg=cfg, read_prob=0.0
    )
    behavior = [
        next(
            outcome
            for outcome in state["outcomes"]
            if outcome["action"] == "b4h6"
        )
        for state in row["states"]
    ]
    assert all(outcome["marginal_wall_s"] is None for outcome in behavior)
    # StubLabeler: pred_s + mt_s = 0.2 per write, plus the 0.6 s arrival of the
    # first decision. This is the documented no-cost-model fallback only.
    assert row["states"][1]["now_s"] == pytest.approx(0.8)


# ----------------------------------------------------------------------
# Task 10: trained gating, deadline schema, calibration provenance,
# strict pair/vocab/cost-record checks, and hardening tests
# ----------------------------------------------------------------------
def _run_dev(harness, output_dir, **overrides):
    cfg = harness["cfg"]
    kwargs = dict(
        limit=0,
        stride=1,
        output_dir=str(output_dir),
        max_hours=1.0,
        seed=cfg.seed,
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=harness["sentences"],
        cfg=cfg,
        predictor_id=0,
        translator_id=0,
        timer=harness["timer"],
    )
    kwargs.update(overrides)
    return kaggle.run_dev(**kwargs)


def test_run_dev_marks_trained_false_when_fit_runs_no_epoch(tmp_path):
    harness = make_harness(epochs=0)
    summary = _run_dev(harness, tmp_path)

    # `fit` saw samples but ran zero optimisation epochs, so this is not a
    # trained network and no checkpoint may be written.
    assert summary["training"]["epochs"] == 0
    assert summary["training"]["n_samples"] > 0
    assert summary["trained"] is False
    assert summary["checkpoint"]["status"] == "not_written"
    assert summary["checkpoint"]["path"] is None
    assert summary["artifacts"]["checkpoint"] is None
    assert not (tmp_path / "checkpoint.pt").exists()
    assert summary["risk_calibration"]["status"] == "unavailable"
    assert summary["risk_calibration"]["reason"] == (
        "network was not trained in this run"
    )


def test_deadline_early_return_preserves_prior_artifacts_and_is_explicit(tmp_path):
    prior_summary = '{"prior": "summary"}\n'
    prior_traces = '{"prior": "traces"}\n'
    (tmp_path / "summary.json").write_text(prior_summary, encoding="utf-8")
    (tmp_path / "traces.jsonl").write_text(prior_traces, encoding="utf-8")

    harness = make_harness()
    summary = _run_dev(harness, tmp_path, max_hours=0.0)

    # The empty deadline return must not clobber the productive session's
    # non-empty artifacts; it goes to suffixed files instead.
    assert (tmp_path / "summary.json").read_text(encoding="utf-8") == prior_summary
    assert (tmp_path / "traces.jsonl").read_text(encoding="utf-8") == prior_traces
    assert (tmp_path / "summary.deadline.json").is_file()
    assert (tmp_path / "traces.deadline.jsonl").is_file()

    assert summary["stopped_reason"] == "deadline"
    assert summary["trained"] is False
    for key in ("risk_calibration", "rollout_fixed", "paired", "artifacts"):
        assert key in summary
        assert summary[key] is None
    assert summary["not_run"]["reason"]
    assert "risk_calibration" in summary["not_run"]["sections"]
    # The pointer must be inside the serialized file, not only the return value.
    written = json.loads(
        (tmp_path / "summary.deadline.json").read_text(encoding="utf-8")
    )
    assert written["deadline_artifacts"] == summary["deadline_artifacts"]
    assert written["deadline_artifacts"]["summary"].endswith(
        "summary.deadline.json"
    )


def test_dev_calibration_values_match_independent_holdout(tmp_path):
    harness = make_harness()
    cfg = harness["cfg"]
    summary = _run_dev(harness, tmp_path)
    assert summary["risk_calibration"]["status"] == "ok"

    # Reconstruct the same training targets from the persisted labels, split
    # them the same way, and recompute calibration on the checkpoint weights.
    rows = kaggle._read_jsonl(tmp_path / "labels.jsonl")
    train_sentences = [
        sentence
        for sentence in harness["sentences"]
        if kaggle.is_train(str(kaggle._field(sentence, "sentence_id", "")))
    ]
    builder, _, vocab = kaggle._components(cfg, train_sentences)
    targets, _states = kaggle._targets_from_rows(rows, builder, cfg, 0, 0)
    _train, holdout = kaggle._split_calibration_targets(targets)
    assert holdout

    restored = CAVCNetwork(
        cfg, src_vocab_size=len(vocab), tgt_vocab_size=len(vocab)
    )
    load_checkpoint(tmp_path / "checkpoint.pt", restored)
    independent = risk_calibration(restored, holdout)

    assert independent["status"] == "ok"
    assert independent["n"] == summary["risk_calibration"]["n"]
    assert independent["error_risk"]["brier"] == pytest.approx(
        summary["risk_calibration"]["error_risk"]["brier"]
    )
    assert independent["no_emit"]["ece"] == pytest.approx(
        summary["risk_calibration"]["no_emit"]["ece"]
    )


def _target(state_key, action=None):
    return ActionTarget(
        state=StateInput(
            numeric=tuple([0.0] * len(NUMERIC_FEATURES)),
            predictor_id=0,
            translator_id=0,
            src_ids=(2,),
            tgt_ids=(3,),
        ),
        action=action or Action(4, 6),
        value_target=0.0,
        error_target=1.0,
        no_emit_target=0.0,
        emit_target=1.0,
        state_key=state_key,
    )


def test_confirm_risk_calibration_is_dev_derived_and_never_test(tmp_path):
    harness = make_harness()
    builder, network = _frozen_builder_network(harness)
    kwargs = dict(
        offset=140,
        limit=1,
        output_dir=str(tmp_path / "confirm"),
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=_confirm_test_sentences(),
        cfg=harness["cfg"],
        builder=builder,
        network=network,
        timer=harness["timer"],
    )
    summary = kaggle.run_confirm(**kwargs)
    assert summary["risk_calibration"]["status"] == "unavailable"
    assert summary["risk_calibration_source"] == "none"
    assert "dev holdout" in summary["risk_calibration"]["reason"]
    assert any(
        "never calibrates on the test slice" in note
        for note in summary["notes"]
    )

    refused_dir = tmp_path / "refused"
    with pytest.raises(ValueError, match="test slice|dev holdout"):
        kaggle.run_confirm(
            calibration_targets=[_target("test:0140:3")],
            **{**kwargs, "output_dir": str(refused_dir)},
        )
    # The refusal happens before rollouts, so no evaluation artifact exists.
    assert not (refused_dir / "traces.jsonl").exists()
    assert not (refused_dir / "summary.json").exists()


def test_confirm_never_calibrates_a_network_it_cannot_prove_trained(tmp_path):
    harness = make_harness()
    builder, network = _frozen_builder_network(harness)
    # No checkpoint: the supplied frozen builder+network has no training
    # provenance, so calibration must stay unavailable even with targets.
    summary = kaggle.run_confirm(
        offset=140,
        limit=1,
        output_dir=str(tmp_path / "supplied"),
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=_confirm_test_sentences(),
        cfg=harness["cfg"],
        builder=builder,
        network=network,
        calibration_targets=[_target("dev:0000:3")],
        timer=harness["timer"],
    )
    assert summary["risk_calibration"]["status"] == "unavailable"
    assert "not a trained checkpoint" in summary["risk_calibration"]["reason"]
    assert summary["risk_calibration_source"] == "none"

    # An explicitly allowed untrained checkpoint is likewise uncalibratable.
    checkpoint = tmp_path / "untrained.pt"
    _write_checkpoint(checkpoint, harness, {"trained": False})
    untrained = kaggle.run_confirm(
        offset=140,
        limit=1,
        output_dir=str(tmp_path / "untrained"),
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=_confirm_test_sentences(),
        checkpoint_path=checkpoint,
        allow_untrained=True,
        calibration_targets=[_target("dev:0000:3")],
        timer=harness["timer"],
    )
    assert untrained["trained"] is False
    assert untrained["risk_calibration"]["status"] == "unavailable"
    assert untrained["risk_calibration_source"] == "none"


def test_confirm_labels_calibration_provenance_from_target_keys(tmp_path):
    harness = make_harness()
    cfg = harness["cfg"]
    preflight_dir = tmp_path / "preflight"
    kaggle.run_preflight(
        limit=3,
        output_dir=str(preflight_dir),
        label_states=2,
        rollout_sentences=1,
        seed=cfg.seed,
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=harness["sentences"],
        cfg=cfg,
        predictor_id=0,
        translator_id=0,
        timer=harness["timer"],
    )

    def confirm(name, key):
        return kaggle.run_confirm(
            offset=140,
            limit=1,
            output_dir=str(tmp_path / name),
            predictor=harness["predictor"],
            translator=harness["translator"],
            sentences=_confirm_test_sentences(),
            checkpoint_path=preflight_dir / "checkpoint.pt",
            calibration_targets=[_target(key)],
            timer=harness["timer"],
        )

    caller = confirm("caller", "s0:3")
    assert caller["risk_calibration"]["status"] == "ok"
    assert caller["risk_calibration_source"] == "caller_supplied"

    dev = confirm("dev", "dev:0000:3")
    assert dev["risk_calibration"]["status"] == "ok"
    assert dev["risk_calibration_source"] == "dev_holdout"


def test_confirm_refuses_cost_records_for_another_pair(tmp_path):
    harness = make_harness()
    checkpoint = tmp_path / "pair.pt"
    # Metadata declares pair (1, 1), but the stored measurements are (0, 0).
    _write_checkpoint(
        checkpoint,
        harness,
        {
            "trained": True,
            "model_pair": {
                "predictor": "small",
                "translator": "small",
                "predictor_id": 1,
                "translator_id": 1,
            },
        },
        cost_model=MeasuredCostTable(
            [(0, 0, Action(4, 6), 0.1, 0.2, 0.3)]
        ),
    )
    with pytest.raises(ValueError, match="cost record|model pair"):
        kaggle.run_confirm(
            offset=140,
            limit=1,
            output_dir=str(tmp_path / "confirm"),
            predictor=harness["predictor"],
            translator=harness["translator"],
            sentences=_confirm_test_sentences(),
            checkpoint_path=checkpoint,
            predictor_id=1,
            translator_id=1,
            timer=harness["timer"],
        )


def test_confirm_refuses_a_supplied_builder_with_a_different_vocab(tmp_path):
    harness = make_harness()
    cfg = harness["cfg"]
    preflight_dir = tmp_path / "preflight"
    kaggle.run_preflight(
        limit=3,
        output_dir=str(preflight_dir),
        label_states=2,
        rollout_sentences=1,
        seed=cfg.seed,
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=harness["sentences"],
        cfg=cfg,
        predictor_id=0,
        translator_id=0,
        timer=harness["timer"],
    )
    # A builder whose vocabulary contains an extra word cannot reproduce the
    # checkpoint's word ids, so confirmation must raise instead of remapping.
    wider = Vocab()
    for token in harness["vocab"].id_to_token:
        if token is not None:
            wider.add(token)
    wider.add("unseen-extra")
    bad_builder = FeatureBuilder(
        length_ratio=1.0, src_vocab=wider, tgt_vocab=wider
    )
    with pytest.raises(ValueError, match="vocabulary"):
        kaggle.run_confirm(
            offset=140,
            limit=1,
            output_dir=str(tmp_path / "confirm"),
            predictor=harness["predictor"],
            translator=harness["translator"],
            sentences=_confirm_test_sentences(),
            checkpoint_path=preflight_dir / "checkpoint.pt",
            builder=bad_builder,
            network=harness["network"],
            timer=harness["timer"],
        )


def test_confirm_resolves_pair_identity_from_adapters_not_call_names(tmp_path):
    harness = make_harness()
    cfg = harness["cfg"]
    preflight_dir = tmp_path / "preflight"
    kaggle.run_preflight(
        limit=3,
        output_dir=str(preflight_dir),
        label_states=2,
        rollout_sentences=1,
        seed=cfg.seed,
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=harness["sentences"],
        cfg=cfg,
        predictor_id=0,
        translator_id=0,
        timer=harness["timer"],
    )
    # The loaded adapters are the small pair even though the caller declares
    # "large"; identity must come from the adapters, so no mismatch is raised.
    harness["predictor"].model_id = kaggle.run_pfba.PREDICTORS["small"]
    harness["translator"].model_id = kaggle.run_pfba.TRANSLATORS["small"]
    summary = kaggle.run_confirm(
        offset=140,
        limit=1,
        output_dir=str(tmp_path / "confirm"),
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=_confirm_test_sentences(),
        checkpoint_path=preflight_dir / "checkpoint.pt",
        predictor_name="large",
        translator_name="large",
        timer=harness["timer"],
    )
    assert summary["model_pair"]["predictor"] == "small"
    assert summary["model_pair"]["translator"] == "small"
    assert summary["model_pair"]["predictor_id"] == 0
    assert summary["model_pair"]["translator_id"] == 0


def test_rollout_prices_actions_with_estimate_marginal():
    harness = make_harness()
    cfg = ControllerConfig(
        seed=7,
        actions=(Action(4, 6),),
        min_start=3,
        # Disable the advantage baseline so the high write value is selected
        # outright; either way the chosen action's cost is the marginal one.
        advantage_baseline=None,
    )
    network = RecordingNetwork()
    pipeline = PipelineCostModel(MeasuredCostTable())
    marginal_calls: list[tuple] = []

    def fake_marginal(*args, **kwargs):
        marginal_calls.append((args, kwargs))
        return 0.123

    # The rollout must ask the runtime to price the action through the
    # `PipelineCostModel.marginal_wall` schedule, not `predict().wall_s`.
    pipeline.marginal_wall = fake_marginal
    runtime = ControlledRuntime(
        harness["predictor"],
        harness["translator"],
        cfg,
        timer=harness["timer"],
        cost_model=pipeline,
    )
    trace = run_rollout(
        harness["sentences"][0],
        harness["predictor"],
        harness["translator"],
        network,
        harness["builder"],
        pipeline,
        cfg,
        runtime=runtime,
    )
    write_steps = [step for step in trace.steps if step["action"] == "b4h6"]
    assert write_steps
    assert marginal_calls
    for step in write_steps:
        assert step["cost"] == pytest.approx(0.123)


class _CommittingRuntime:
    """Execute emits one word and reports a high no-emit, over-budget write."""

    def __init__(self):
        self.t_pred_free = 0.0
        self.t_mt_free = 0.0

    def execute(self, observed, committed, action, seed):
        del observed, action, seed
        return StepResult(
            candidates=("x",),
            selected="x",
            committed_after=tuple(committed) + ("x",),
            pred_s=0.1,
            mt_s=0.1,
            spent_s=1.5,
            vote_top1=0.1,
            vote_margin=0.0,
            vote_entropy=0.0,
            no_emit=False,
        )

    def translate(self, sources, prefix, final=False):
        del sources, prefix, final
        return ["x"], 0.0


class _NoEmitNetwork(RecordingNetwork):
    """Write head with a high no-emit logit so the post controller WAITs."""

    def write_values(self, z, actions):
        outputs = super().write_values(z, actions)
        outputs.no_emit = torch.full_like(outputs.value, 5.0)
        return outputs


def test_post_controller_actually_changes_a_committed_step():
    harness = make_harness()
    cfg = ControllerConfig(seed=7, actions=(Action(2, 4),), min_start=3)
    sentence = harness["sentences"][0]

    def roll(post_controller):
        return run_rollout(
            sentence,
            harness["predictor"],
            harness["translator"],
            _NoEmitNetwork(),
            harness["builder"],
            None,
            cfg,
            runtime=_CommittingRuntime(),
            post_controller=post_controller,
        )

    without = roll(None)
    with_post = roll(kaggle.PostController())
    plain_step = next(
        step for step in without.steps if step["action"] == "b2h4"
    )
    waited_step = next(
        step for step in with_post.steps if step["action"] == "b2h4"
    )

    # Without the controller the provisional word is committed; WAIT discards
    # it, so the committed prefix really differs (not just a summary string).
    assert plain_step["committed_after"] > plain_step["committed_before"]
    assert waited_step["post_action"] == "wait"
    assert waited_step["committed_after"] == waited_step["committed_before"]
    assert waited_step["new_words"] == ()


def test_checkpoint_roundtrips_legacy_and_six_field_cost_records(tmp_path):
    harness = make_harness()
    cfg = harness["cfg"]
    model = MeasuredCostTable(
        [
            (0, 0, Action(2, 4), 0.1, 0.2),
            (0, 0, Action(4, 6), 0.3, 0.4, 0.5),
        ]
    )
    path = tmp_path / "records.pt"
    metadata = kaggle.save_checkpoint(
        path,
        harness["network"],
        cfg,
        [0.0] * cfg.numeric_dim,
        [1.0] * cfg.numeric_dim,
        model,
    )
    assert len(metadata["cost_records"][0]) == 5
    assert len(metadata["cost_records"][1]) == 6
    assert metadata["cost_records"][1][5] == pytest.approx(0.5)

    restored = CAVCNetwork(
        cfg,
        src_vocab_size=len(harness["vocab"]),
        tgt_vocab_size=len(harness["vocab"]),
    )
    loaded = load_checkpoint(path, restored)
    assert loaded["cost_records"][0][3:5] == pytest.approx((0.1, 0.2))
    assert loaded["cost_records"][1][3:] == pytest.approx((0.3, 0.4, 0.5))


def test_committed_kernel_bundle_byte_matches_generator():
    module = _load_prepare_module()
    expected_entry, expected_metadata = module.expected_outputs()
    committed_entry = (
        module.STAGE_DIR / "kaggle_entry.py"
    ).read_bytes().decode("utf-8")
    committed_metadata = (
        module.STAGE_DIR / "kernel-metadata.json"
    ).read_bytes().decode("utf-8")
    assert committed_entry == expected_entry
    assert committed_metadata == expected_metadata
    ok, problems = module.verify_kernel()
    assert ok, problems


def test_prepare_kernel_check_detects_stale_bundle(tmp_path, capsys):
    module = _load_prepare_module()
    # A clean tree (the committed bundle) passes the check.
    assert module.main(["--check", "--output-dir", str(module.STAGE_DIR)]) == 0
    assert "bundle up to date" in capsys.readouterr().out

    # Write a fresh copy, then mutate one generated file and confirm the
    # check fails with a non-zero exit status.
    module.write_kernel(output_dir=tmp_path)
    ok, problems = module.verify_kernel(tmp_path)
    assert ok, problems
    (tmp_path / "kaggle_entry.py").write_text("stale", encoding="utf-8")
    ok, problems = module.verify_kernel(tmp_path)
    assert not ok
    assert any("stale" in problem for problem in problems)
    assert module.main(["--check", "--output-dir", str(tmp_path)]) == 1


def test_risk_calibration_rejects_nonpositive_buckets():
    harness = make_harness()
    with pytest.raises(ValueError, match="buckets"):
        risk_calibration(harness["network"], [_target("s0:3")], buckets=0)


def test_feature_stats_list_form_validates_names():
    kaggle._validate_feature_stats_payload(
        {"n": 1, "features": list(NUMERIC_FEATURES)}, "payload"
    )
    with pytest.raises(ValueError, match="list form"):
        kaggle._validate_feature_stats_payload(
            {"n": 1, "features": list(reversed(NUMERIC_FEATURES))}, "payload"
        )


def test_stage_timer_defaults_to_synchronized_run_pfba_timer():
    harness = make_harness()
    timer = kaggle._stage_timer(
        kaggle.run_pfba,
        harness["predictor"],
        harness["translator"],
        None,
    )
    # Without an injected clock the real-adapters path must synchronize device
    # work rather than read an unsynchronized perf_counter.
    assert isinstance(timer, kaggle.run_pfba.StageTimer)
    injected = harness["timer"]
    assert (
        kaggle._stage_timer(
            kaggle.run_pfba,
            harness["predictor"],
            harness["translator"],
            injected,
        )
        is injected
    )


def test_stage_timer_syncs_each_distinct_adapter_device():
    class FakeStageTimer:
        def __init__(self, sync):
            self.sync = sync

    class FakePfba:
        StageTimer = FakeStageTimer

        def __init__(self):
            self.synced = []

        def _sync(self, device):
            self.synced.append(device)

    class Adapter:
        def __init__(self, device):
            self.device = device

    pfba = FakePfba()
    timer = kaggle._stage_timer(
        pfba, Adapter("cuda:0"), Adapter("cuda:1"), None
    )
    timer.sync()
    assert pfba.synced == ["cuda:0", "cuda:1"]

    same = FakePfba()
    timer = kaggle._stage_timer(
        same, Adapter("cuda:0"), Adapter("cuda:0"), None
    )
    timer.sync()
    assert same.synced == ["cuda:0"]

    # Adapters without `.device` degrade to a no-op sync; this is acceptable
    # for CPU/test adapters and is documented on the helper.
    no_device = FakePfba()
    timer = kaggle._stage_timer(no_device, object(), object(), None)
    timer.sync()
    assert no_device.synced == []


def test_plain_cost_model_warns_once_per_model():
    harness = make_harness()
    cfg = ControllerConfig(
        seed=7, actions=(Action(2, 4),), min_start=3, advantage_baseline=None
    )
    cost_model = RecordingCostModel()
    sentence = Sentence(
        "s0", "hallo welt wir lernen heute .", "hello world we learn today ."
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = run_rollout(
            sentence,
            None,
            None,
            RecordingNetwork(),
            harness["builder"],
            cost_model,
            cfg,
            runtime=FakeRuntime(),
        )
    assert any(step["action"] == "b2h4" for step in trace.steps)
    plain = [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
        and "plain cost model" in str(warning.message)
    ]
    assert len(plain) == 1


def test_per_candidate_downgrade_emits_a_warning():
    class NoSamplePredictor:
        """Has no `sample` method, so per-candidate cannot run."""

    cfg = ControllerConfig(seed=7)
    with pytest.warns(RuntimeWarning, match="downgrading"):
        labeler = CounterfactualLabeler(
            NoSamplePredictor(), object(), cfg, candidate_mode="per_candidate"
        )
    assert labeler.candidate_mode == "batched"


# ----------------------------------------------------------------------
# Fix wave: seeded init, oracle reuse, per-action terminal values, resume
# stamps, and cheap hardening
# ----------------------------------------------------------------------
def test_components_seeds_network_initialisation():
    harness = make_harness()
    torch.manual_seed(1)
    _builder_a, network_a, _vocab_a = kaggle._components(
        harness["cfg"], harness["sentences"]
    )
    torch.manual_seed(987654)
    _builder_b, network_b, _vocab_b = kaggle._components(
        harness["cfg"], harness["sentences"]
    )

    for (name_a, parameter_a), (name_b, parameter_b) in zip(
        network_a.named_parameters(), network_b.named_parameters()
    ):
        assert name_a == name_b
        assert torch.equal(parameter_a, parameter_b), name_a


def test_label_sentence_translates_the_oracle_once_per_nonterminal_state():
    harness = make_harness()
    cfg = ControllerConfig(seed=7, actions=(Action(2, 4),))
    sentence = harness["sentences"][0]
    oracle_calls: list[str] = []

    class MajorityPredictor:
        """Every candidate is identical, so RALCP commits both words."""

        def sample(self, observed, branches, horizon, seed):
            del observed, horizon, seed
            return ["der hund"] * max(1, int(branches))

    class CountingTranslator(FakeTranslator):
        def translate_batch(self, sources, prefix, final=False, **kwargs):
            if list(sources) == [sentence.source]:
                oracle_calls.append(prefix)
            return super().translate_batch(
                sources, prefix, final=final, **kwargs
            )

    labeler = CounterfactualLabeler(
        MajorityPredictor(), CountingTranslator(), cfg, timer=harness["timer"]
    )
    # Stop before the true last position so every labelled state is
    # non-terminal: terminal values additionally ask for one forced-prefix
    # oracle per distinct write prefix (see the terminal-value test).
    row = kaggle.label_sentence(
        labeler,
        sentence,
        stride=1,
        cfg=cfg,
        builder=harness["builder"],
        read_prob=0.0,
        max_states=2,
    )

    assert len(row["states"]) == 2
    assert not any(state["terminal"] for state in row["states"])
    # The committed prefix grows every state, so each state needs a distinct
    # oracle -- exactly one full-source translation per labelled state. Before
    # the fix, `label_sentence` re-requested it to count words, doubling this.
    assert row["states"][0]["committed_after"]
    assert len(oracle_calls) == len(row["states"])
    assert len(set(oracle_calls)) == len(oracle_calls)
    assert oracle_calls[0] == ""


def test_label_sentence_dedupes_read_in_the_action_grid():
    harness = make_harness()
    cfg = ControllerConfig(
        seed=7,
        actions=(READ, Action(2, 4), Action(4, 6)),
    )
    sentence = harness["sentences"][0]
    row = kaggle.label_sentence(
        StubLabeler(), sentence, stride=1, cfg=cfg, read_prob=0.0
    )

    for state in row["states"]:
        labels = [outcome["action"] for outcome in state["outcomes"]]
        assert labels.count(READ.label) == 1
        assert set(labels) == {READ.label, "b2h4", "b4h6"}


def test_label_sentence_warns_once_for_a_plain_cost_model():
    harness = make_harness()
    cfg = ControllerConfig(seed=7, actions=(Action(2, 4),))
    sentence = harness["sentences"][0]
    cost_model = RecordingCostModel()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            kaggle.label_sentence(
                StubLabeler(),
                sentence,
                stride=1,
                cfg=cfg,
                read_prob=0.0,
                cost_model=cost_model,
                max_states=1,
            )
    plain = [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
        and "plain cost model" in str(warning.message)
    ]
    assert len(plain) == 1


def test_run_dev_refuses_resume_labels_from_a_different_context(tmp_path):
    harness = make_harness()
    _run_dev(harness, tmp_path)
    labels_path = tmp_path / "labels.jsonl"
    rows = [
        json.loads(line)
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    # Every new-format row stamps the generation context.
    assert "predictor_id" in rows[0]
    assert "read_prob" in rows[0]
    assert "vocab_fingerprint" in rows[0]

    rows[0]["predictor_id"] = 1
    labels_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="predictor_id"):
        _run_dev(harness, tmp_path)

    rows[0]["predictor_id"] = 0
    rows[0]["vocab_fingerprint"] = "not-the-fingerprint"
    labels_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="vocab_fingerprint"):
        _run_dev(harness, tmp_path)


def test_confirm_refuses_a_supplied_builder_without_a_vocabulary(tmp_path):
    harness = make_harness()
    checkpoint = tmp_path / "trained.pt"
    _write_checkpoint(checkpoint, harness, {"trained": True})

    with pytest.raises(ValueError, match="vocabulary"):
        kaggle.run_confirm(
            offset=140,
            limit=1,
            output_dir=str(tmp_path / "confirm"),
            predictor=harness["predictor"],
            translator=harness["translator"],
            sentences=_confirm_test_sentences(),
            checkpoint_path=checkpoint,
            # No src_vocab/tgt_vocab: every text id would become UNK/PAD even
            # though `length_ratio` matches the checkpoint.
            builder=FeatureBuilder(length_ratio=1.0),
            timer=harness["timer"],
        )


def test_confirm_refuses_a_caller_cfg_with_a_changed_action_grid(tmp_path):
    harness = make_harness()
    checkpoint = tmp_path / "trained.pt"
    _write_checkpoint(checkpoint, harness, {"trained": True})
    changed = ControllerConfig(seed=harness["cfg"].seed, actions=(Action(2, 4),))

    with pytest.raises(ValueError, match="action grid"):
        kaggle.run_confirm(
            offset=140,
            limit=1,
            output_dir=str(tmp_path / "confirm"),
            predictor=harness["predictor"],
            translator=harness["translator"],
            sentences=_confirm_test_sentences(),
            checkpoint_path=checkpoint,
            cfg=changed,
            timer=harness["timer"],
        )


def test_repeated_deadline_return_preserves_an_existing_deadline_sibling(
    tmp_path,
):
    prior_summary = '{"prior": "summary"}\n'
    prior_deadline = '{"prior": "deadline"}\n'
    (tmp_path / "summary.json").write_text(prior_summary, encoding="utf-8")
    (tmp_path / "summary.deadline.json").write_text(
        prior_deadline, encoding="utf-8"
    )

    harness = make_harness()
    summary = _run_dev(harness, tmp_path, max_hours=0.0)

    assert (tmp_path / "summary.json").read_text(encoding="utf-8") == prior_summary
    assert (
        tmp_path / "summary.deadline.json"
    ).read_text(encoding="utf-8") == prior_deadline
    assert (tmp_path / "summary.deadline2.json").is_file()
    assert summary["deadline_artifacts"]["summary"].endswith(
        "summary.deadline2.json"
    )
