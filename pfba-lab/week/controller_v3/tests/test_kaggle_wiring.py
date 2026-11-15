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
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from controller_v3 import kaggle
from controller_v3.action import READ, Action
from controller_v3.checkpoint import load_checkpoint
from controller_v3.config import ControllerConfig
from controller_v3.cost_model import MeasuredCostTable
from controller_v3.features import FeatureBuilder
from controller_v3.labeler import CounterfactualLabeler
from controller_v3.network import CAVCNetwork
from controller_v3.text_encoder import Vocab

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
        "checkpoint.pt",
    ):
        assert (tmp_path / name).is_file(), name

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
    for predictor_id, translator_id, action_pair, pred_s, mt_s in cost_payload[
        "records"
    ]:
        assert predictor_id == 0 and translator_id == 0
        assert pred_s >= 0.0 and mt_s >= 0.0
        assert action_pair[0] > 0 and action_pair[1] > 0

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
    for predictor_id, translator_id, _action, pred_s, mt_s in records:
        assert predictor_id == 1 and translator_id == 1
        assert pred_s >= 0.0 and mt_s >= 0.0

    # FixedTimer: the b2h4 action samples twice and translates once.
    assert records[0][3] == pytest.approx(2 * harness["timer"].seconds)
    assert records[0][4] == pytest.approx(harness["timer"].seconds)
    assert records[1][3] == pytest.approx(4 * harness["timer"].seconds)

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

    summary = kaggle.run_confirm(
        offset=139,
        limit=2,
        output_dir=str(tmp_path),
        allow_consumed=True,
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=test_sentences,
        cfg=cfg,
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
    summary = kaggle.run_confirm(
        offset=140,
        limit=1,
        output_dir=str(tmp_path),
        predictor=harness["predictor"],
        translator=harness["translator"],
        sentences=test_sentences,
        cfg=harness["cfg"],
        timer=harness["timer"],
    )
    assert summary["offset"] == 140
    assert summary["n_traces"] == 1


# ----------------------------------------------------------------------
# Labeling
# ----------------------------------------------------------------------
def test_label_sentence_labels_only_kept_positions_and_omits_candidates():
    harness = make_harness()
    cfg = harness["cfg"]
    sentence = harness["sentences"][0]
    row = kaggle.label_sentence(harness["labeler"], sentence, stride=2, cfg=cfg)

    positions = list(range(cfg.min_start, len(sentence.source.split())))
    assert [state["read_pos"] for state in row["states"]] == positions[::2]
    for state in row["states"]:
        assert state["outcomes"][0]["action"] == READ.label
        assert state["oracle_words"] >= 0
        # Candidate text is omitted from the persisted row.
        assert "candidates" not in json.dumps(state)
        # The committed prefix only ever grows by appended words.
        before = state["committed_before"]
        after = state["committed_after"]
        assert after[: len(before)] == before


def test_label_sentence_refuses_test_and_holdout_sentences():
    harness = make_harness()
    cfg = harness["cfg"]
    with pytest.raises(ValueError, match="test"):
        kaggle.label_sentence(
            harness["labeler"],
            Sentence("test:0140", "hallo welt wir lernen heute .", "x"),
            2,
            cfg,
        )
    # dev:0009 hashes into holdout bucket 1.
    with pytest.raises(ValueError, match="holdout|bucket"):
        kaggle.label_sentence(
            harness["labeler"],
            Sentence("dev:0009", "hallo welt wir lernen heute .", "x"),
            2,
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
            stride=2,
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
