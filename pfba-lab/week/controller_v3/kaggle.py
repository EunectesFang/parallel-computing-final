"""Real-model Kaggle wiring for the CAVC v3 controller.

This module is the only place where `controller_v3` touches the real
`run_pfba` adapters (Qwen predictor, Marian/Tower translator). Importing it
loads no weights and downloads nothing: every model is resolved lazily inside
`resolve_adapters`, and every run function accepts injected adapters so the
CPU test suite can drive the whole pipeline with deterministic fakes.

Four entry points exist:

* ``run_preflight``: a wiring check on a few **dev** sentences. It calibrates
  the cost table, labels a few states, trains, saves a checkpoint, and rolls
  out a few sentences.
* ``run_dev``: labels dev train hash buckets 2-9, trains, and rolls out the dev
  holdout bucket 1. The labels file is appended one JSON row per sentence and
  flushed, so a Kaggle session that hits its wall-clock limit can resume by
  `sentence_id` without relabelling completed sentences.
* ``run_confirm``: frozen-config evaluation on the untouched **test** slice
  starting at ``test:0140``. It never trains and never tunes.
* ``--mode dry-run``: prints the plan without loading a model or a dataset.

Metric honesty
--------------
BLEU and COMET are delegated to `evaluator.summarize_traces`, which delegates
to `run_pfba`. A missing `sacrebleu` reports ``status="unavailable"`` and no
score; COMET is skipped because loading it downloads model weights. No score
is ever synthesised locally.

Label files
-----------
A labels row holds the sentence id, split, source, stride, and one entry per
kept decision position with ``read_pos``, ``observed``, ``committed_before``,
``committed_after``, ``oracle_words``, the chosen write action, and every
action outcome. Full candidate strings are deliberately omitted: they are
large, and only ``matched`` / ``incorrect`` / timing / vote statistics are
needed to rebuild `ActionTarget` rows. Training rows are rebuilt against an
online `FeatureBuilder`, so nothing in the labels file bypasses the leakage
boundary.

Model-pair ids
--------------
``MODEL_PAIR_IDS`` maps ``small -> 0`` and ``large -> 1``. These ids index the
measured cost table and the network's model embeddings. They are only
meaningful together with calibration records measured for that exact adapter
pair; a table measured for ``(0, 0)`` must not be used to price ``(1, 1)``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from .action import READ, Action
from .checkpoint import load_checkpoint, save_checkpoint
from .config import ControllerConfig
from .cost_model import MeasuredCostTable
from .evaluator import summarize_traces
from .features import FeatureBuilder, StateInput, estimate_length_ratio
from .labeler import ActionOutcome, CounterfactualLabeler
from .network import CAVCNetwork
from .reward import RewardConfig, immediate_reward
from .rollout import RolloutTrace, run_rollout
from .runtime import ControlledRuntime, run_pfba
from .targets import build_action_targets
from .text_encoder import Vocab
from .trainer import CAVCTrainer

__all__ = [
    "MODEL_PAIR_IDS",
    "HOLDOUT_BUCKET",
    "EXCLUDED_BUCKETS",
    "resolve_adapters",
    "calibrate_cost_records",
    "label_sentence",
    "run_preflight",
    "run_dev",
    "run_confirm",
    "main",
    "build_parser",
]

# Predictor/translator model-pair ids. Only meaningful with matching
# calibration records; see the module docstring.
MODEL_PAIR_IDS = {"small": 0, "large": 1}

# Dev hash buckets are identical to `week/week_run.py`: bucket 1 is the dev
# holdout and buckets 0-1 are excluded from training. The clean confirmation
# slice starts at `test:0140`.
HOLDOUT_BUCKET = 1
EXCLUDED_BUCKETS = frozenset({0, 1})
TEST_OFFSET_FLOOR = 140

DEFAULT_PREFLIGHT_LIMIT = 8
DEFAULT_CONFIRM_LIMIT = 64

# `test:0000` .. `test:0139` were consumed by earlier folds.
_CONSUMED_TEST_REASON = (
    "test:0000-test:0139 were consumed by earlier folds and must not be "
    "reported as a clean confirmation slice; pass allow_consumed=True only "
    "for a diagnostic rerun"
)


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _require_run_pfba():
    """Return the shared `run_pfba` module or raise a clear error."""
    if run_pfba is None:
        raise RuntimeError(
            "run_pfba was not found. controller_v3 expects "
            "pfba-lab/paired/run_pfba.py next to the week/ directory"
        )
    return run_pfba


def _field(obj, name: str, default):
    """Read a field from a dataclass-like object or a plain mapping."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _decision_positions(n_words: int, min_start: int) -> list[int]:
    """Decision positions from `min_start` to the penultimate source word."""
    if n_words <= int(min_start):
        return []
    return list(range(int(min_start), int(n_words)))


def _kept_positions(positions: Sequence[int], stride: int) -> set[int]:
    stride = int(stride)
    if stride < 1:
        raise ValueError("stride must be a positive integer")
    return {
        int(position)
        for index, position in enumerate(positions)
        if index % stride == 0
    }


def _step_seed(cfg: ControllerConfig, sentence_id: str, read_pos: int, split: str) -> int:
    """Deterministic per-sentence/read-position seed, shared with run_pfba."""
    if run_pfba is not None and hasattr(run_pfba, "derive_step_seed"):
        return int(
            run_pfba.derive_step_seed(cfg.seed, str(split), sentence_id, read_pos)
        )
    return int(cfg.seed) + int(read_pos)


def sentence_bucket(sentence_id: str) -> int:
    """Hash bucket identical to `week/week_run.py:hash_bucket`."""
    import hashlib

    digest = hashlib.sha256(f"13|{sentence_id}".encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big") % 10


def is_holdout(sentence_id: str) -> bool:
    """True for the dev holdout bucket."""
    return sentence_bucket(sentence_id) == HOLDOUT_BUCKET


def is_train(sentence_id: str) -> bool:
    """True for dev train buckets 2-9; buckets 0 and 1 are excluded."""
    return sentence_bucket(sentence_id) not in EXCLUDED_BUCKETS


def _reward_config(cfg: ControllerConfig) -> RewardConfig:
    return RewardConfig(
        weight_m=cfg.weight_m,
        weight_e=cfg.weight_e,
        lambda_l=cfg.lambda_l,
        gamma=cfg.gamma,
    )


def _median(values: Sequence[float]) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else 0.0


def _pair_id(explicit, adapter, names: Sequence[str] = ()) -> int:
    """Resolve a model-pair id from an explicit value, adapter attribute, or 0.

    Adapters may expose ``model_pair_id`` / ``predictor_id`` / ``translator_id``.
    A `QwenPredictor` exposes its Hugging Face `model_id`, so the known small
    and large model ids are mapped back to 0/1. Otherwise 0 is used; callers
    that care about the table key should pass it explicitly.
    """
    if explicit is not None:
        return int(explicit)
    for name in names:
        value = getattr(adapter, name, None)
        if value is not None:
            return int(value)
    model_id = getattr(adapter, "model_id", None)
    if run_pfba is not None and isinstance(model_id, str):
        for name, model in getattr(run_pfba, "PREDICTORS", {}).items():
            if model == model_id:
                return int(MODEL_PAIR_IDS.get(name, 0))
        for name, model in getattr(run_pfba, "TRANSLATORS", {}).items():
            if model == model_id:
                return int(MODEL_PAIR_IDS.get(name, 0))
    return 0


def _feature_stats(states: Sequence[StateInput]) -> tuple[list[float], list[float]]:
    """Elementwise mean and population std; std defaults to 1.0 when zero."""
    states = list(states)
    if not states:
        return [], []
    width = len(states[0].numeric)
    count = float(len(states))
    mean = [
        sum(float(state.numeric[index]) for state in states) / count
        for index in range(width)
    ]
    std: list[float] = []
    for index in range(width):
        variance = sum(
            (float(state.numeric[index]) - mean[index]) ** 2 for state in states
        ) / count
        std.append(math.sqrt(variance) if variance > 0.0 else 1.0)
    return mean, std


def _git_commit() -> str | None:
    """Return the current HEAD commit, or ``None`` outside a git checkout."""
    repo_root = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _config_payload(
    cfg: ControllerConfig, model_pair: dict, notes: Sequence[str] = ()
) -> dict:
    """JSON-friendly config view: actions and baseline become lists."""
    data = asdict(cfg)
    data["actions"] = [
        [int(action.branches), int(action.horizon)] for action in cfg.actions
    ]
    baseline = cfg.advantage_baseline
    data["advantage_baseline"] = (
        [int(baseline[0]), int(baseline[1])] if baseline else None
    )
    return {"config": data, "model_pair": dict(model_pair), "notes": list(notes)}


def _checkpoint_payload(path):
    """Load a trusted controller checkpoint payload with `weights_only=True`."""
    import torch

    return torch.load(Path(path), map_location="cpu", weights_only=True)


def _config_from_payload(payload, path) -> ControllerConfig:
    """Rebuild the frozen config stored in a checkpoint payload."""
    data = dict(payload.get("config") or {})
    if not data:
        raise ValueError(f"checkpoint {path} has no config")
    actions = tuple(
        Action(int(branches), int(horizon))
        for branches, horizon in data.pop("actions", [])
    )
    baseline = data.get("advantage_baseline")
    data["advantage_baseline"] = (
        tuple(int(value) for value in baseline) if baseline else None
    )
    data["actions"] = actions
    try:
        return ControllerConfig(**data)
    except TypeError as exc:
        raise ValueError(
            f"checkpoint {path} config does not match ControllerConfig: {exc}"
        ) from exc


def _frozen_vocab(
    payload,
) -> tuple[Vocab | None, float | None]:
    """Rebuild the training vocabulary and length ratio from a checkpoint.

    A checkpoint trained on dev sentences must be evaluated with the *same*
    word ids; rebuilding a vocabulary from the test sentences would change
    every text-encoder input. The vocabulary is stored in ``extra["vocab"]``
    by `_train_and_write`. Older checkpoints without it return ``(None, None)``
    and the caller falls back to the evaluation-sentence vocabulary.
    """
    extra = payload.get("extra") or {}
    tokens = extra.get("vocab")
    if not isinstance(tokens, (list, tuple)) or not tokens:
        return None, None
    words = list(tokens)
    # Vocab reserves PAD=0 and UNK=1, stored as two leading Nones.
    if len(words) >= 2 and words[0] is None and words[1] is None:
        words = words[2:]
    vocab = Vocab()
    for word in words:
        if word is not None:
            vocab.add(str(word))
    ratio = extra.get("length_ratio")
    return vocab, (float(ratio) if ratio is not None else None)


# ----------------------------------------------------------------------
# Adapters
# ----------------------------------------------------------------------
def _device_namespace(device):
    """Build the attributes `run_pfba.resolve_devices` reads."""
    if device is None:
        return SimpleNamespace(
            device="auto", predictor_device=None, translator_device=None
        )
    text = str(device)
    if text in ("auto", "cpu", "cuda"):
        return SimpleNamespace(
            device=text, predictor_device=None, translator_device=None
        )
    # A specific accelerator index pins both stages.
    return SimpleNamespace(
        device="auto", predictor_device=text, translator_device=text
    )


def _maybe_offload_large(predictor) -> None:
    """Optionally keep the 7B input embedding on CPU to save GPU memory.

    `cpu_embedding.py` lives at the `week/` root in the packaged kernel and at
    `pfba-lab/decoupled/` in the repository. It is optional: a missing import
    only skips the memory optimisation.
    """
    model = getattr(predictor, "model", None)
    if model is None:
        return
    try:
        from cpu_embedding import offload_input_embedding  # type: ignore
    except Exception:
        return
    try:
        offload_input_embedding(model)
    except Exception as exc:  # pragma: no cover - adapter dependent
        print(f"[cavc] embedding offload skipped: {exc}", flush=True)


def resolve_adapters(
    predictor_name: str = "small",
    translator_name: str = "small",
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    device=None,
):
    """Load the real predictor/translator pair and return the runtime config.

    Returns ``(predictor, translator, stream_cfg, predictor_device,
    translator_device)``. Model weights are downloaded/loaded here, so this is
    the only function that needs a GPU or network access.
    """
    pfba = _require_run_pfba()
    if predictor_name not in pfba.PREDICTORS:
        raise ValueError(f"unknown predictor {predictor_name!r}")
    if translator_name not in pfba.TRANSLATORS:
        raise ValueError(f"unknown translator {translator_name!r}")

    torch = pfba._torch()
    predictor_device, translator_device = pfba.resolve_devices(
        torch, _device_namespace(device), predictor_name, translator_name
    )
    stream_cfg = pfba.StreamConfig(
        predictor=str(predictor_name),
        translator=str(translator_name),
        local_files_only=bool(local_files_only),
        trust_remote_code=bool(trust_remote_code),
    )
    predictor = pfba.QwenPredictor(
        pfba.PREDICTORS[predictor_name],
        predictor_device,
        "auto",
        bool(local_files_only),
        bool(trust_remote_code),
    )
    predictor.load()
    if predictor_name == "large":
        _maybe_offload_large(predictor)
    translator = pfba.load_translator(
        translator_name,
        pfba.TRANSLATORS[translator_name],
        translator_device,
        stream_cfg,
    )
    print(
        f"[cavc] predictor {predictor_name}@{predictor_device} x "
        f"translator {translator_name}@{translator_device}",
        flush=True,
    )
    return predictor, translator, stream_cfg, predictor_device, translator_device


# ----------------------------------------------------------------------
# Cost calibration
# ----------------------------------------------------------------------
def calibrate_cost_records(
    predictor,
    translator,
    sentences,
    actions,
    cfg: ControllerConfig,
    limit: int = 8,
    states_per_sentence: int = 2,
    *,
    predictor_id: int | None = None,
    translator_id: int | None = None,
    timer=None,
):
    """Measure ``(predictor_id, translator_id, action, pred_s, mt_s)`` records.

    Every write action is measured through `ControlledRuntime.execute`, which
    is the exact per-candidate runtime path: candidate ``i`` is sampled with
    ``predictor.sample(observed, 1, horizon, seed + i)``. The same seed is used
    for every action at one state, so smaller budgets are prefixes of larger
    ones. READ, when present in `actions`, is recorded with zero time; the
    measured table ignores READ because waiting is free by definition.

    At most `limit` sentences and `states_per_sentence` decision positions per
    sentence are measured. Returns ``(records, summary)``.
    """
    cfg = cfg or ControllerConfig()
    limit = int(limit)
    states_per_sentence = int(states_per_sentence)
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if states_per_sentence <= 0:
        raise ValueError("states_per_sentence must be positive")

    resolved_pid = _pair_id(predictor_id, predictor, ("model_pair_id", "predictor_id"))
    resolved_tid = _pair_id(
        translator_id, translator, ("model_pair_id", "translator_id")
    )
    action_list = tuple(actions)
    runtime = ControlledRuntime(predictor, translator, cfg, timer=timer)

    per_action: dict[str, list[tuple[float, float]]] = {
        action.label: [] for action in action_list
    }
    records: list[tuple[int, int, Action, float, float]] = []
    used_sentences = 0
    used_states = 0
    sentence_list = list(sentences)
    if limit > 0:
        sentence_list = sentence_list[:limit]

    for sentence in sentence_list:
        sentence_id = str(_field(sentence, "sentence_id", ""))
        words = str(_field(sentence, "source", "")).split()
        positions = _decision_positions(len(words), cfg.min_start)
        if not positions:
            continue
        used_sentences += 1
        for read_pos in positions[:states_per_sentence]:
            observed = " ".join(words[:read_pos])
            committed: tuple[str, ...] = ()
            seed = _step_seed(cfg, sentence_id, read_pos, split="calibration")
            for action in action_list:
                if action.is_read:
                    pred_s = mt_s = 0.0
                else:
                    result = runtime.execute(observed, committed, action, seed)
                    pred_s = float(result.pred_s)
                    mt_s = float(result.mt_s)
                records.append(
                    (resolved_pid, resolved_tid, action, pred_s, mt_s)
                )
                per_action[action.label].append((pred_s, mt_s))
            used_states += 1

    summary = {
        "n_states": used_states,
        "n_sentences": used_sentences,
        "n_records": len(records),
        "predictor_id": resolved_pid,
        "translator_id": resolved_tid,
        "limit": limit,
        "states_per_sentence": states_per_sentence,
        "per_action": {
            label: {
                "n": len(values),
                "pred_s_median": _median([pred for pred, _mt in values]),
                "mt_s_median": _median([mt for _pred, mt in values]),
            }
            for label, values in per_action.items()
        },
        "notes": [
            "Ids are only meaningful with calibration records measured for "
            "the same adapter pair.",
            "READ is free and is ignored by MeasuredCostTable.",
        ],
    }
    return records, summary


# ----------------------------------------------------------------------
# Serialisation helpers
# ----------------------------------------------------------------------
def _serialize_cost_records(
    records: Sequence[tuple[int, int, Action, float, float]],
) -> list[list]:
    return [
        [
            int(predictor_id),
            int(translator_id),
            [int(action.branches), int(action.horizon)],
            float(pred_s),
            float(mt_s),
        ]
        for predictor_id, translator_id, action, pred_s, mt_s in records
    ]


def _deserialize_cost_records(raw) -> list[tuple[int, int, Action, float, float]]:
    records: list[tuple[int, int, Action, float, float]] = []
    for item in raw or []:
        predictor_id, translator_id, action_pair, pred_s, mt_s = item
        records.append(
            (
                int(predictor_id),
                int(translator_id),
                Action(int(action_pair[0]), int(action_pair[1])),
                float(pred_s),
                float(mt_s),
            )
        )
    return records


def _write_cost_records(
    path: Path,
    records: Sequence[tuple[int, int, Action, float, float]],
    summary: dict,
) -> None:
    _write_json(
        path,
        {
            "records": _serialize_cost_records(records),
            "summary": summary,
            "format": "records are [predictor_id, translator_id, "
            "[branches, horizon], pred_s, mt_s]",
        },
    )


def _load_cost_records(path: Path):
    """Load a previously written cost_records.json, or ``None`` when absent."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _deserialize_cost_records(payload.get("records")), payload.get(
        "summary", {}
    )


def _serialize_outcome(outcome: ActionOutcome, reward: float) -> dict:
    """One outcome with candidate text omitted to keep the labels file small."""
    return {
        "action": outcome.action.label,
        "branches": int(outcome.action.branches),
        "horizon": int(outcome.action.horizon),
        "matched": int(outcome.matched),
        "incorrect": int(outcome.incorrect),
        "emitted": int(outcome.emitted),
        "no_emit": bool(outcome.no_emit),
        "pred_s": float(outcome.pred_s),
        "mt_s": float(outcome.mt_s),
        "vote_top1": float(outcome.vote_top1),
        "vote_margin": float(outcome.vote_margin),
        "vote_entropy": float(outcome.vote_entropy),
        "reward": float(reward),
    }


def _outcome_from_record(record: dict) -> ActionOutcome:
    """Rebuild the minimal `ActionOutcome` needed by `build_action_targets`."""
    label = str(record.get("action", "read"))
    if label == READ.label:
        action = READ
    else:
        action = Action(int(record["branches"]), int(record["horizon"]))
    return ActionOutcome(
        action=action,
        new_words=(),
        matched=int(record.get("matched", 0)),
        incorrect=int(record.get("incorrect", 0)),
        emitted=int(record.get("emitted", 0)),
        no_emit=bool(record.get("no_emit", False)),
        next_committed=(),
        pred_s=float(record.get("pred_s", 0.0)),
        mt_s=float(record.get("mt_s", 0.0)),
        vote_top1=float(record.get("vote_top1", 0.0)),
        vote_margin=float(record.get("vote_margin", 0.0)),
        vote_entropy=float(record.get("vote_entropy", 0.0)),
        candidates=(),
    )


def _trace_record(trace: RolloutTrace) -> dict:
    return {
        "sentence_id": trace.sentence_id,
        "source": trace.source,
        "reference": trace.reference,
        "hypothesis": trace.hypothesis,
        "committed": list(trace.committed),
        "delays": list(trace.delays),
        "laal": trace.laal,
        "completion_s": trace.completion_s,
        "read_fraction": trace.read_fraction,
        "action_histogram": dict(trace.action_histogram),
        "steps": [dict(step) for step in trace.steps],
    }


def _write_traces(path: Path, traces: Sequence[RolloutTrace]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(
                json.dumps(_trace_record(trace), ensure_ascii=False, sort_keys=True)
                + "\n"
            )


def _rollout_summary(traces: Sequence[RolloutTrace]) -> dict:
    """Status-aware rollout summary without the (large) flattened step list.

    Steps stay in traces.jsonl; summary.json keeps the aggregate metrics only
    so a dev run with hundreds of sentences stays readable.
    """
    summary = dict(summarize_traces(list(traces), include_comet=False))
    summary.pop("steps", None)
    return summary


@dataclass(frozen=True)
class _SplitSentence:
    """A sentence with an explicit split, used for rollout seeding."""

    sentence_id: str
    source: str
    reference: str
    split: str = "dev"


def _with_split(sentences: Sequence, split: str) -> list[_SplitSentence]:
    return [
        _SplitSentence(
            sentence_id=str(_field(sentence, "sentence_id", "")),
            source=str(_field(sentence, "source", "")),
            reference=str(_field(sentence, "reference", "")),
            split=str(split),
        )
        for sentence in sentences
    ]


def _guard_no_test_sentences(sentences: Sequence, caller: str) -> None:
    for sentence in sentences:
        sentence_id = str(_field(sentence, "sentence_id", ""))
        split = str(_field(sentence, "split", "dev"))
        if sentence_id.startswith("test:") or split == "test":
            raise ValueError(
                f"{caller} must not use test sentences; got {sentence_id!r}"
            )


# ----------------------------------------------------------------------
# Labeling
# ----------------------------------------------------------------------
def _guard_label_sentence(sentence_id: str, split: str) -> None:
    if split == "test" or sentence_id.startswith("test:"):
        raise ValueError(
            f"label_sentence refuses test sentence {sentence_id!r}; labels are "
            "training-only"
        )
    if sentence_id.startswith("dev:") and is_holdout(sentence_id):
        raise ValueError(
            f"label_sentence refuses dev holdout sentence {sentence_id!r} "
            f"(bucket {HOLDOUT_BUCKET}); the holdout is evaluation-only"
        )


def _best_write(
    outcomes: Sequence[ActionOutcome],
    reward_cfg: RewardConfig,
    backlog: float,
    source_advance: int = 1,
) -> tuple[ActionOutcome | None, dict[Action, float]]:
    """Highest-immediate-reward write action; ties prefer smaller ``b * h``."""
    rewards: dict[Action, float] = {}
    writes: list[ActionOutcome] = []
    for outcome in outcomes:
        reward = immediate_reward(
            int(outcome.matched),
            int(outcome.incorrect),
            float(backlog),
            int(source_advance),
            reward_cfg,
        )
        rewards[outcome.action] = float(reward)
        if not outcome.action.is_read:
            writes.append(outcome)
    if not writes:
        return None, rewards

    def key(outcome: ActionOutcome):
        action = outcome.action
        return (
            -rewards[action],
            int(action.branches) * int(action.horizon),
            int(action.branches),
            int(action.horizon),
        )

    return min(writes, key=key), rewards


def label_sentence(
    labeler: CounterfactualLabeler,
    sentence,
    stride: int = 4,
    cfg: ControllerConfig | None = None,
) -> dict:
    """Label one sentence and return its JSON-serialisable labels row.

    The walk starts at `cfg.min_start` and ends at the penultimate word. Every
    ``stride``-th decision position is labelled; the others advance `read_pos`
    without a model call. At a kept position `label_state` scores every write
    action in `cfg.actions` plus READ, and the committed prefix advances by the
    write action with the highest immediate reward (ties prefer smaller
    ``branches * horizon``). READ by itself leaves the prefix unchanged, while
    a write's ``new_words`` are appended, exactly as the labelled trajectory in
    `week_run.label_split` does.

    Immediate rewards use ``backlog = max(0, read_pos - len(committed))``, the
    source/target lag at the decision point, and ``source_advance = 1`` because
    every decision consumes one source position. This mirrors the delay term
    used by `targets.build_action_targets`.

    `label_state` derives the oracle internally; this function recomputes the
    oracle word count so the persisted row records the length used by the
    training-only oracle path. That costs one extra translation call per
    labelled state and is deliberately confined to the offline label path.
    """
    cfg = cfg or ControllerConfig()
    stride = int(stride)
    if stride < 1:
        raise ValueError("stride must be a positive integer")

    sentence_id = str(_field(sentence, "sentence_id", ""))
    split = str(_field(sentence, "split", "dev"))
    _guard_label_sentence(sentence_id, split)
    source = str(_field(sentence, "source", ""))
    words = source.split()
    positions = _decision_positions(len(words), cfg.min_start)
    kept = _kept_positions(positions, stride)
    reward_cfg = _reward_config(cfg)

    committed: tuple[str, ...] = ()
    states: list[dict] = []
    for read_pos in positions:
        if read_pos not in kept:
            # Unkept positions still advance the source position. They do not
            # relabel, so the committed prefix stays where it was.
            continue
        observed = " ".join(words[:read_pos])
        seed = _step_seed(cfg, sentence_id, read_pos, split)
        outcomes = labeler.label_state(
            observed, committed, source, cfg.actions, seed
        )
        oracle = labeler.oracle_words(source, committed)
        read = labeler.read_outcome(committed)
        backlog = float(max(0, read_pos - len(committed)))
        all_outcomes = (read, *outcomes)
        best, rewards = _best_write(all_outcomes, reward_cfg, backlog)
        committed_after = (
            tuple(best.next_committed) if best is not None else committed
        )
        states.append(
            {
                "read_pos": int(read_pos),
                "observed": observed,
                "committed_before": list(committed),
                "committed_after": list(committed_after),
                "oracle_words": int(len(oracle)),
                "chosen": best.action.label if best is not None else READ.label,
                "chosen_reward": (
                    float(rewards[best.action]) if best is not None else None
                ),
                "outcomes": [
                    _serialize_outcome(outcome, rewards[outcome.action])
                    for outcome in all_outcomes
                ],
            }
        )
        committed = committed_after

    return {
        "sentence_id": sentence_id,
        "split": split,
        "source": source,
        "stride": stride,
        "states": states,
    }


# ----------------------------------------------------------------------
# Shared training pipeline
# ----------------------------------------------------------------------
def _components(cfg: ControllerConfig, sentences: Sequence):
    """Build the on-screen vocab, feature builder, and untrained network."""
    texts: list[str] = []
    pairs: list[tuple[str, str]] = []
    for sentence in sentences:
        source = str(_field(sentence, "source", ""))
        reference = str(_field(sentence, "reference", ""))
        texts.extend([source, reference])
        pairs.append((source, reference))
    vocab = Vocab.from_texts(texts)
    builder = FeatureBuilder(
        length_ratio=estimate_length_ratio(pairs),
        src_vocab=vocab,
        tgt_vocab=vocab,
    )
    network = CAVCNetwork(
        cfg, src_vocab_size=len(vocab), tgt_vocab_size=len(vocab)
    )
    return builder, network, vocab


def _targets_from_states(
    labeled_states: Sequence[tuple[dict, dict | None, str]],
    builder: FeatureBuilder,
    cfg: ControllerConfig,
    predictor_id: int,
    translator_id: int,
):
    """Rebuild `ActionTarget` rows from in-memory label states."""
    targets = []
    states: list[StateInput] = []
    for state_row, next_row, state_key in labeled_states:
        committed = tuple(state_row.get("committed_before") or ())
        state = builder.build(
            str(state_row.get("observed", "")),
            committed,
            predictor_id=predictor_id,
            translator_id=translator_id,
        )
        next_state = None
        if next_row is not None:
            next_state = builder.build(
                str(next_row.get("observed", "")),
                tuple(next_row.get("committed_before") or ()),
                predictor_id=predictor_id,
                translator_id=translator_id,
            )
        backlog = float(
            max(0, int(state_row.get("read_pos", 0)) - len(committed))
        )
        outcomes = [
            _outcome_from_record(record)
            for record in state_row.get("outcomes") or []
        ]
        targets.extend(
            build_action_targets(
                state,
                outcomes,
                cfg,
                next_state=next_state,
                backlog=backlog,
                source_advance=1,
                state_key=state_key,
            )
        )
        states.append(state)
    return targets, states


def _label_few_states(
    labeler: CounterfactualLabeler,
    sentences: Sequence,
    builder: FeatureBuilder,
    cfg: ControllerConfig,
    count: int,
    predictor_id: int,
    translator_id: int,
):
    """Label at most `count` consecutive decision states across sentences."""
    count = int(count)
    if count <= 0:
        raise ValueError("label_states must be positive")

    labeled: list[tuple[dict, dict | None, str]] = []
    for sentence in sentences:
        sentence_id = str(_field(sentence, "sentence_id", ""))
        split = str(_field(sentence, "split", "dev"))
        source = str(_field(sentence, "source", ""))
        words = source.split()
        positions = _decision_positions(len(words), cfg.min_start)
        if not positions:
            continue
        committed: tuple[str, ...] = ()
        reward_cfg = _reward_config(cfg)
        rows: list[dict] = []
        for read_pos in positions:
            observed = " ".join(words[:read_pos])
            seed = _step_seed(cfg, sentence_id, read_pos, split)
            outcomes = labeler.label_state(
                observed, committed, source, cfg.actions, seed
            )
            read = labeler.read_outcome(committed)
            backlog = float(max(0, read_pos - len(committed)))
            all_outcomes = (read, *outcomes)
            best, rewards = _best_write(all_outcomes, reward_cfg, backlog)
            committed_after = (
                tuple(best.next_committed) if best is not None else committed
            )
            rows.append(
                {
                    "read_pos": int(read_pos),
                    "observed": observed,
                    "committed_before": list(committed),
                    "committed_after": list(committed_after),
                    "oracle_words": len(
                        labeler.oracle_words(source, committed)
                    ),
                    "chosen": (
                        best.action.label if best is not None else READ.label
                    ),
                    "chosen_reward": (
                        float(rewards[best.action])
                        if best is not None
                        else None
                    ),
                    "outcomes": [
                        _serialize_outcome(outcome, rewards[outcome.action])
                        for outcome in all_outcomes
                    ],
                }
            )
            committed = committed_after
            if len(labeled) + len(rows) >= count:
                break
        for index, row in enumerate(rows):
            next_row = rows[index + 1] if index + 1 < len(rows) else None
            labeled.append((row, next_row, f"{sentence_id}:{row['read_pos']}"))
        if len(labeled) >= count:
            break

    labeled = labeled[:count]
    return _targets_from_states(
        labeled, builder, cfg, predictor_id, translator_id
    )


def _targets_from_rows(
    rows: Sequence[dict],
    builder: FeatureBuilder,
    cfg: ControllerConfig,
    predictor_id: int,
    translator_id: int,
):
    """Rebuild `ActionTarget` rows from persisted labels rows.

    The successor of a labelled state is the next kept state in the same
    sentence. For `stride=1` that is the exact next read position; for
    `stride>1` it is a multi-step successor discounted once by `cfg.gamma`,
    which is a documented simplification of the one-step backup.
    """
    labeled: list[tuple[dict, dict | None, str]] = []
    for row in rows:
        sentence_id = str(row.get("sentence_id", ""))
        states = row.get("states") or []
        for index, state_row in enumerate(states):
            next_row = states[index + 1] if index + 1 < len(states) else None
            labeled.append(
                (
                    state_row,
                    next_row,
                    f"{sentence_id}:{state_row.get('read_pos')}",
                )
            )
    return _targets_from_states(
        labeled, builder, cfg, predictor_id, translator_id
    )


def _train_and_write(
    *,
    mode: str,
    split: str,
    sentences: Sequence,
    train_sentences: Sequence,
    label_rows: Sequence[dict] | None,
    targets,
    states: Sequence[StateInput],
    network: CAVCNetwork,
    builder: FeatureBuilder,
    vocab: Vocab,
    cost_model: MeasuredCostTable,
    cost_summary: dict,
    cfg: ControllerConfig,
    predictor,
    translator,
    timer,
    output_dir: Path,
    rollout_sentences: Sequence,
    model_pair: dict,
    notes: Sequence[str],
    extra_summary: dict | None = None,
) -> dict:
    """Train, checkpoint, roll out, and write every artifact for a mode."""
    trainer = CAVCTrainer(network, cfg, cost_model)
    training = trainer.fit(targets)
    mean, std = _feature_stats(states)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    metadata = save_checkpoint(
        checkpoint_path,
        network,
        cfg,
        mean,
        std,
        cost_model,
        extra={
            "mode": mode,
            "split": split,
            "training": training,
            "n_targets": len(targets),
            # Frozen confirmation must rebuild the same word ids; store the
            # vocabulary and the training length ratio so a test sentence
            # cannot silently remap the text-encoder inputs.
            "vocab": list(vocab.id_to_token),
            "length_ratio": float(builder.rho),
        },
    )

    traces: list[RolloutTrace] = []
    for sentence in _with_split(rollout_sentences, split):
        traces.append(
            run_rollout(
                sentence,
                predictor,
                translator,
                network,
                builder,
                cost_model,
                cfg,
                timer=timer,
            )
        )
    rollout_summary = _rollout_summary(traces)

    config_path = output_dir / "config.json"
    cost_path = output_dir / "cost_records.json"
    traces_path = output_dir / "traces.jsonl"
    summary_path = output_dir / "summary.json"
    _write_json(
        config_path,
        _config_payload(
            cfg,
            model_pair,
            notes=[
                "Model-pair ids are only meaningful with calibration records "
                "measured for the same adapter pair."
            ],
        ),
    )
    _write_cost_records(cost_path, cost_model.records, cost_summary)
    _write_traces(traces_path, traces)

    summary = {
        "mode": mode,
        "split": split,
        "seed": int(cfg.seed),
        "git_commit": metadata.get("git_commit"),
        "model_pair": dict(model_pair),
        "vocab_size": len(vocab),
        "sentences_loaded": len(sentences),
        "label_sentences": len(train_sentences),
        "n_label_rows": len(label_rows) if label_rows is not None else None,
        "n_states": len(states),
        "n_targets": len(targets),
        "training": training,
        "checkpoint": {
            "path": str(checkpoint_path),
            "format_version": metadata["format_version"],
            "seed": metadata["seed"],
            "git_commit": metadata.get("git_commit"),
            "feature_mean_len": len(metadata["feature_mean"]),
            "feature_std_len": len(metadata["feature_std"]),
            "cost_records": len(metadata["cost_records"]),
        },
        "cost": cost_summary,
        "rollout": rollout_summary,
        "artifacts": {
            "summary": str(summary_path),
            "traces": str(traces_path),
            "config": str(config_path),
            "cost_records": str(cost_path),
            "checkpoint": str(checkpoint_path),
        },
        "notes": list(notes),
    }
    if extra_summary:
        summary.update(extra_summary)
    _write_json(summary_path, summary)

    bleu = rollout_summary.get("bleu", {})
    comet = rollout_summary.get("comet", {})
    print(
        f"[cavc] {mode} complete: states={len(states)} targets={len(targets)} "
        f"rollouts={len(traces)}",
        flush=True,
    )
    print(
        f"[cavc] BLEU={bleu.get('status', 'unknown')} "
        f"COMET={comet.get('status', 'unknown')}",
        flush=True,
    )
    print(f"[cavc] wrote {summary_path}", flush=True)
    return summary


# ----------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------
def run_preflight(
    limit: int = DEFAULT_PREFLIGHT_LIMIT,
    output_dir: str = "outputs/cavc_preflight",
    label_states: int = 3,
    rollout_sentences: int = 2,
    seed: int = 13,
    *,
    predictor=None,
    translator=None,
    sentences=None,
    cfg: ControllerConfig | None = None,
    predictor_id: int | None = None,
    translator_id: int | None = None,
    timer=None,
    predictor_name: str = "small",
    translator_name: str = "small",
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> dict:
    """Wire the real pipeline end to end on a few dev sentences.

    Loads **dev** sentences only, calibrates the cost table, labels a few
    decision states, trains, checkpoints, and rolls out a few sentences. The
    five required artifacts are written under `output_dir`; BLEU/COMET carry
    their real status.
    """
    pfba = _require_run_pfba()
    cfg = cfg or ControllerConfig(seed=int(seed))
    predictor_devices = (None, None)
    if predictor is None or translator is None:
        predictor, translator, _stream, pdev, tdev = resolve_adapters(
            predictor_name,
            translator_name,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        predictor_devices = (pdev, tdev)

    if sentences is None:
        sentences = pfba.load_split("dev", int(limit) if int(limit) > 0 else 0)
    sentences = list(sentences)
    if not sentences:
        raise RuntimeError("run_preflight loaded no dev sentences")
    _guard_no_test_sentences(sentences, "run_preflight")

    train_sentences = [
        sentence
        for sentence in sentences
        if is_train(str(_field(sentence, "sentence_id", "")))
    ]
    if not train_sentences:
        raise RuntimeError(
            "run_preflight has no dev train sentences (hash buckets 2-9)"
        )

    pid = _pair_id(predictor_id, predictor, ("model_pair_id", "predictor_id"))
    tid = _pair_id(
        translator_id, translator, ("model_pair_id", "translator_id")
    )
    builder, network, vocab = _components(cfg, sentences)
    labeler = CounterfactualLabeler(predictor, translator, cfg, timer=timer)

    cal_limit = min(8, max(1, int(limit))) if int(limit) > 0 else 8
    cost_records, cost_summary = calibrate_cost_records(
        predictor,
        translator,
        train_sentences,
        cfg.actions,
        cfg,
        limit=cal_limit,
        states_per_sentence=1,
        predictor_id=pid,
        translator_id=tid,
        timer=timer,
    )
    cost_model = MeasuredCostTable(cost_records)

    targets, states = _label_few_states(
        labeler,
        train_sentences,
        builder,
        cfg,
        label_states,
        pid,
        tid,
    )
    if not targets:
        raise RuntimeError(
            "run_preflight labelled no states; check the dev sentences and "
            "cfg.min_start"
        )

    model_pair = {
        "predictor": predictor_name,
        "translator": translator_name,
        "predictor_id": pid,
        "translator_id": tid,
        "predictor_device": predictor_devices[0],
        "translator_device": predictor_devices[1],
    }
    notes = [
        "Preflight only: a wiring check, not a quality result.",
        "Dev sentences only; no test sentence is loaded.",
        "BLEU/COMET report the status returned by run_pfba; no score is "
        "invented.",
    ]
    return _train_and_write(
        mode="preflight",
        split="dev",
        sentences=sentences,
        train_sentences=train_sentences,
        label_rows=None,
        targets=targets,
        states=states,
        network=network,
        builder=builder,
        vocab=vocab,
        cost_model=cost_model,
        cost_summary=cost_summary,
        cfg=cfg,
        predictor=predictor,
        translator=translator,
        timer=timer,
        output_dir=Path(output_dir),
        rollout_sentences=sentences[: max(0, int(rollout_sentences))],
        model_pair=model_pair,
        notes=notes,
        extra_summary={"preflight": True},
    )


def run_dev(
    limit: int = 0,
    stride: int = 4,
    output_dir: str = "outputs/cavc_dev",
    max_hours: float = 10.0,
    seed: int = 13,
    *,
    predictor=None,
    translator=None,
    sentences=None,
    cfg: ControllerConfig | None = None,
    predictor_id: int | None = None,
    translator_id: int | None = None,
    timer=None,
    predictor_name: str = "small",
    translator_name: str = "small",
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> dict:
    """Label dev train buckets 2-9, train, and roll out dev holdout bucket 1.

    Labels are appended to ``output_dir/labels.jsonl`` one row per sentence and
    flushed, so a session that hits `max_hours` can resume: existing
    `sentence_id` values are skipped and never relabelled. Holdout bucket 1 and
    every test sentence are excluded from labeling by construction.

    `max_hours` bounds the labeling and rollout loops. A single training call
    is not preempted mid-epoch; the checkpoint written at the end is complete.
    """
    pfba = _require_run_pfba()
    cfg = cfg or ControllerConfig(seed=int(seed))
    stride = int(stride)
    if stride < 1:
        raise ValueError("stride must be a positive integer")
    max_hours = float(max_hours)
    if max_hours < 0.0:
        raise ValueError("max_hours must be non-negative")
    deadline = time.perf_counter() + max_hours * 3600.0

    predictor_devices = (None, None)
    if predictor is None or translator is None:
        predictor, translator, _stream, pdev, tdev = resolve_adapters(
            predictor_name,
            translator_name,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        predictor_devices = (pdev, tdev)

    if sentences is None:
        sentences = pfba.load_split("dev", int(limit) if int(limit) > 0 else 0)
    sentences = list(sentences)
    if not sentences:
        raise RuntimeError("run_dev loaded no dev sentences")
    _guard_no_test_sentences(sentences, "run_dev")

    train_sentences = [
        sentence
        for sentence in sentences
        if is_train(str(_field(sentence, "sentence_id", "")))
    ]
    holdout_sentences = [
        sentence
        for sentence in sentences
        if is_holdout(str(_field(sentence, "sentence_id", "")))
        and not str(_field(sentence, "sentence_id", "")).startswith("test:")
    ]
    if not train_sentences:
        raise RuntimeError("run_dev has no dev train sentences (buckets 2-9)")

    pid = _pair_id(predictor_id, predictor, ("model_pair_id", "predictor_id"))
    tid = _pair_id(
        translator_id, translator, ("model_pair_id", "translator_id")
    )
    builder, network, vocab = _components(cfg, sentences)
    labeler = CounterfactualLabeler(predictor, translator, cfg, timer=timer)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.jsonl"
    done = {str(row.get("sentence_id", "")) for row in _read_jsonl(labels_path)}

    newly_labeled = 0
    stopped_for_deadline = False
    with labels_path.open("a", encoding="utf-8") as handle:
        for sentence in train_sentences:
            sentence_id = str(_field(sentence, "sentence_id", ""))
            if sentence_id in done:
                continue
            if time.perf_counter() > deadline:
                stopped_for_deadline = True
                break
            row = label_sentence(labeler, sentence, stride, cfg)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            done.add(sentence_id)
            newly_labeled += 1
            if newly_labeled % 25 == 0:
                print(
                    f"[cavc] labeled {newly_labeled} new dev sentences",
                    flush=True,
                )

    rows = [
        row
        for row in _read_jsonl(labels_path)
        if is_train(str(row.get("sentence_id", "")))
        and not str(row.get("sentence_id", "")).startswith("test:")
        and row.get("split") != "test"
        and row.get("states")
    ]
    _write_json(
        output_dir / "progress.json",
        {
            "mode": "dev",
            "stride": stride,
            "newly_labeled": newly_labeled,
            "total_labeled": len(rows),
            "train_sentences": len(train_sentences),
            "holdout_sentences": len(holdout_sentences),
            "stopped_for_deadline": stopped_for_deadline,
            "max_hours": max_hours,
        },
    )
    if not rows:
        raise RuntimeError(
            "run_dev has no labeled rows yet; rerun with more time to label "
            "at least one dev train sentence"
        )

    loaded_cost = _load_cost_records(output_dir / "cost_records.json")
    if loaded_cost is not None:
        cost_records, cost_summary = loaded_cost
        cost_summary = dict(cost_summary)
        cost_summary["reused_from_artifact"] = True
    else:
        cost_records, cost_summary = calibrate_cost_records(
            predictor,
            translator,
            train_sentences,
            cfg.actions,
            cfg,
            limit=min(8, max(1, len(train_sentences))),
            states_per_sentence=1,
            predictor_id=pid,
            translator_id=tid,
            timer=timer,
        )
    cost_model = MeasuredCostTable(cost_records)

    targets, states = _targets_from_rows(rows, builder, cfg, pid, tid)
    if not targets:
        raise RuntimeError("run_dev rebuilt no training targets from labels")

    rollout_pool: list = []
    for sentence in holdout_sentences:
        if time.perf_counter() > deadline:
            stopped_for_deadline = True
            break
        rollout_pool.append(sentence)

    model_pair = {
        "predictor": predictor_name,
        "translator": translator_name,
        "predictor_id": pid,
        "translator_id": tid,
        "predictor_device": predictor_devices[0],
        "translator_device": predictor_devices[1],
    }
    notes = [
        "Development run: train buckets 2-9, holdout bucket 1.",
        "Bucket 0 and bucket 1 are historically consumed; the holdout is an "
        "internal development estimate, not a pristine test result.",
        f"Test slice starts at test:{TEST_OFFSET_FLOOR:04d}; no test sentence "
        "is loaded here.",
    ]
    return _train_and_write(
        mode="dev",
        split="dev",
        sentences=sentences,
        train_sentences=train_sentences,
        label_rows=rows,
        targets=targets,
        states=states,
        network=network,
        builder=builder,
        vocab=vocab,
        cost_model=cost_model,
        cost_summary=cost_summary,
        cfg=cfg,
        predictor=predictor,
        translator=translator,
        timer=timer,
        output_dir=output_dir,
        rollout_sentences=rollout_pool,
        model_pair=model_pair,
        notes=notes,
        extra_summary={
            "stride": stride,
            "newly_labeled": newly_labeled,
            "stopped_for_deadline": stopped_for_deadline,
            "holdout_bucket": HOLDOUT_BUCKET,
            "excluded_buckets": sorted(EXCLUDED_BUCKETS),
        },
    )


def run_confirm(
    offset: int = TEST_OFFSET_FLOOR,
    limit: int = DEFAULT_CONFIRM_LIMIT,
    output_dir: str = "outputs/cavc_confirm",
    *,
    allow_consumed: bool = False,
    seed: int = 13,
    predictor=None,
    translator=None,
    sentences=None,
    cfg: ControllerConfig | None = None,
    network: CAVCNetwork | None = None,
    builder: FeatureBuilder | None = None,
    cost_model: MeasuredCostTable | None = None,
    checkpoint_path=None,
    predictor_id: int | None = None,
    translator_id: int | None = None,
    timer=None,
    predictor_name: str = "small",
    translator_name: str = "small",
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> dict:
    """Evaluate a frozen controller on the clean test slice; never tune.

    The clean slice starts at ``test:0140``. An earlier offset raises
    `ValueError` unless `allow_consumed=True` is passed explicitly. When
    `checkpoint_path` is supplied, its config, vocabulary, length ratio,
    measured cost records, and weights are reused; the stored vocabulary
    matters because rebuilding one from test sentences would change every
    text-encoder id. Without a checkpoint an untrained network is evaluated
    and the summary reports ``trained=False`` so the run cannot be mistaken
    for a real confirmation.
    """
    offset = int(offset)
    if offset < TEST_OFFSET_FLOOR and not allow_consumed:
        raise ValueError(_CONSUMED_TEST_REASON)

    pfba = _require_run_pfba()
    payload = None
    if checkpoint_path is not None:
        payload = _checkpoint_payload(checkpoint_path)
    if cfg is None:
        cfg = (
            _config_from_payload(payload, checkpoint_path)
            if payload is not None
            else ControllerConfig(seed=int(seed))
        )

    predictor_devices = (None, None)
    if predictor is None or translator is None:
        predictor, translator, _stream, pdev, tdev = resolve_adapters(
            predictor_name,
            translator_name,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        predictor_devices = (pdev, tdev)

    if sentences is None:
        all_test = list(pfba.load_split("test", 0))
        sentences = all_test[offset : offset + int(limit)]
    sentences = list(sentences)
    if int(limit) > 0:
        sentences = sentences[: int(limit)]
    if not sentences:
        raise RuntimeError(
            f"run_confirm has no test sentences at offset {offset}"
        )
    for sentence in sentences:
        if str(_field(sentence, "sentence_id", "")).startswith("dev:"):
            raise ValueError(
                "run_confirm uses the test split only; got a dev sentence "
                f"{_field(sentence, 'sentence_id', '')!r}"
            )

    pid = _pair_id(predictor_id, predictor, ("model_pair_id", "predictor_id"))
    tid = _pair_id(
        translator_id, translator, ("model_pair_id", "translator_id")
    )
    if builder is None or network is None:
        frozen_vocab, frozen_ratio = (
            _frozen_vocab(payload) if payload is not None else (None, None)
        )
        if frozen_vocab is not None:
            # A dev-trained checkpoint must see the same word ids at test
            # time, so the stored vocabulary and length ratio win over one
            # rebuilt from the test sentences.
            pairs = [
                (
                    str(_field(sentence, "source", "")),
                    str(_field(sentence, "reference", "")),
                )
                for sentence in sentences
            ]
            ratio = (
                frozen_ratio
                if frozen_ratio is not None
                else estimate_length_ratio(pairs)
            )
            builder = builder or FeatureBuilder(
                length_ratio=ratio,
                src_vocab=frozen_vocab,
                tgt_vocab=frozen_vocab,
            )
            network = network or CAVCNetwork(
                cfg,
                src_vocab_size=len(frozen_vocab),
                tgt_vocab_size=len(frozen_vocab),
            )
        else:
            built_builder, built_network, _built_vocab = _components(
                cfg, sentences
            )
            builder = builder or built_builder
            network = network or built_network

    loaded_metadata = None
    if checkpoint_path is not None:
        loaded_metadata = load_checkpoint(checkpoint_path, network)
        if cost_model is None and loaded_metadata.get("cost_records"):
            cost_model = MeasuredCostTable(loaded_metadata["cost_records"])
    if cost_model is None:
        cost_model = MeasuredCostTable()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = [
        run_rollout(
            sentence,
            predictor,
            translator,
            network,
            builder,
            cost_model,
            cfg,
            timer=timer,
        )
        for sentence in _with_split(sentences, "test")
    ]
    rollout_summary = _rollout_summary(traces)

    model_pair = {
        "predictor": predictor_name,
        "translator": translator_name,
        "predictor_id": pid,
        "translator_id": tid,
        "predictor_device": predictor_devices[0],
        "translator_device": predictor_devices[1],
    }
    config_path = output_dir / "config.json"
    traces_path = output_dir / "traces.jsonl"
    summary_path = output_dir / "summary.json"
    _write_json(
        config_path,
        _config_payload(
            cfg,
            model_pair,
            notes=["Frozen config; no training and no tuning on the test slice."],
        ),
    )
    _write_traces(traces_path, traces)
    summary = {
        "mode": "confirm",
        "split": "test",
        "offset": offset,
        "limit": int(limit),
        "allow_consumed": bool(allow_consumed),
        "tuned": False,
        "trained": checkpoint_path is not None,
        "frozen_config": True,
        "seed": int(cfg.seed),
        "git_commit": (
            loaded_metadata.get("git_commit")
            if loaded_metadata is not None
            else _git_commit()
        ),
        "model_pair": model_pair,
        "sentences": len(sentences),
        "n_traces": len(traces),
        "checkpoint": (
            str(checkpoint_path) if checkpoint_path is not None else None
        ),
        "rollout": rollout_summary,
        "artifacts": {
            "summary": str(summary_path),
            "traces": str(traces_path),
            "config": str(config_path),
            "checkpoint": (
                str(checkpoint_path) if checkpoint_path is not None else None
            ),
        },
        "notes": [
            f"Clean test slice starts at test:{TEST_OFFSET_FLOOR:04d}.",
            "No training and no threshold selection happens here.",
            "BLEU/COMET report the status returned by run_pfba.",
            (
                "No checkpoint was supplied, so an untrained network was "
                "rolled out; this is not a confirmation result."
                if checkpoint_path is None
                else "Weights and measured cost records loaded from the "
                "checkpoint."
            ),
        ],
    }
    _write_json(summary_path, summary)
    bleu = rollout_summary.get("bleu", {})
    comet = rollout_summary.get("comet", {})
    print(
        f"[cavc] confirm complete: n={len(traces)} "
        f"BLEU={bleu.get('status', 'unknown')} "
        f"COMET={comet.get('status', 'unknown')}",
        flush=True,
    )
    print(f"[cavc] wrote {summary_path}", flush=True)
    return summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m controller_v3.kaggle",
        description=(
            "Real-model CAVC v3 runs. preflight wires the pipeline on dev; "
            "dev labels buckets 2-9 and evaluates bucket 1; confirm is frozen "
            "on test:0140 onward; dry-run loads nothing."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "dev", "confirm", "dry-run"),
        required=True,
    )
    parser.add_argument(
        "--predictor", choices=("small", "large"), default="small"
    )
    parser.add_argument(
        "--translator", choices=("small", "large"), default="small"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="preflight: dev sentences (default 8); dev: loaded dev "
        "sentences (default 0 = all); confirm: test sentences (default 64)",
    )
    parser.add_argument("--label-states", type=int, default=3)
    parser.add_argument("--rollout-sentences", type=int, default=2)
    parser.add_argument("--offset", type=int, default=TEST_OFFSET_FLOOR)
    parser.add_argument("--output-dir", default="outputs/cavc")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-consumed", action="store_true")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--device",
        default=None,
        help="auto/cpu/cuda or a specific device such as cuda:0",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="confirm mode: trained checkpoint to evaluate",
    )
    return parser


def _dry_run_plan(args) -> dict:
    """Describe the run without loading a model or a dataset."""
    mode = args.mode
    plan = {
        "mode": mode,
        "predictor": args.predictor,
        "translator": args.translator,
        "model_pair_ids": {
            "predictor_id": MODEL_PAIR_IDS[args.predictor],
            "translator_id": MODEL_PAIR_IDS[args.translator],
        },
        "seed": int(args.seed),
        "output_dir": args.output_dir,
        "split_policy": {
            "preflight": "dev only, first --limit sentences",
            "dev": "label dev hash buckets 2-9; roll out holdout bucket 1",
            "confirm": f"test split from test:{TEST_OFFSET_FLOOR:04d}; frozen",
            "dry_run": "no data is loaded",
        },
        "expected_hours": {
            "preflight": "0.5-1",
            "dev": "4-6",
            "confirm": "1-2",
        },
        "notes": [
            "Model-pair ids are only meaningful with matching calibration "
            "records.",
            "No Kaggle submission or push is performed by this module.",
        ],
    }
    if mode == "preflight":
        limit = DEFAULT_PREFLIGHT_LIMIT if args.limit is None else args.limit
        plan["limit"] = int(limit)
        plan["label_states"] = int(args.label_states)
        plan["rollout_sentences"] = int(args.rollout_sentences)
    elif mode == "dev":
        plan["limit"] = int(args.limit or 0)
        plan["stride"] = int(args.stride)
        plan["max_hours"] = float(args.max_hours)
    elif mode == "confirm":
        limit = DEFAULT_CONFIRM_LIMIT if args.limit is None else args.limit
        plan["offset"] = int(args.offset)
        plan["limit"] = int(limit)
        plan["allow_consumed"] = bool(args.allow_consumed)
        plan["checkpoint"] = args.checkpoint
    return plan


def main(argv=None) -> int:
    """Parse arguments and dispatch one mode. Dry-run loads nothing."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "dry-run":
        print("controller_v3 kaggle dry run - no models or data are loaded")
        print(json.dumps(_dry_run_plan(args), indent=2, sort_keys=True))
        return 0

    if (
        args.mode == "confirm"
        and int(args.offset) < TEST_OFFSET_FLOOR
        and not args.allow_consumed
    ):
        raise ValueError(_CONSUMED_TEST_REASON)

    if args.mode == "preflight":
        limit = DEFAULT_PREFLIGHT_LIMIT if args.limit is None else args.limit
        predictor, translator, _stream, _pdev, _tdev = resolve_adapters(
            args.predictor,
            args.translator,
            local_files_only=args.local_files_only,
            device=args.device,
        )
        run_preflight(
            limit=int(limit),
            output_dir=args.output_dir,
            label_states=int(args.label_states),
            rollout_sentences=int(args.rollout_sentences),
            seed=int(args.seed),
            predictor=predictor,
            translator=translator,
            predictor_name=args.predictor,
            translator_name=args.translator,
        )
    elif args.mode == "dev":
        predictor, translator, _stream, _pdev, _tdev = resolve_adapters(
            args.predictor,
            args.translator,
            local_files_only=args.local_files_only,
            device=args.device,
        )
        run_dev(
            limit=int(args.limit or 0),
            stride=int(args.stride),
            output_dir=args.output_dir,
            max_hours=float(args.max_hours),
            seed=int(args.seed),
            predictor=predictor,
            translator=translator,
            predictor_name=args.predictor,
            translator_name=args.translator,
        )
    elif args.mode == "confirm":
        limit = DEFAULT_CONFIRM_LIMIT if args.limit is None else args.limit
        checkpoint = args.checkpoint
        if checkpoint is None:
            candidate = Path(args.output_dir) / "checkpoint.pt"
            if not candidate.is_file():
                raise FileNotFoundError(
                    "confirm mode needs a trained checkpoint; pass "
                    "--checkpoint or run --mode dev first"
                )
            checkpoint = str(candidate)
        predictor, translator, _stream, _pdev, _tdev = resolve_adapters(
            args.predictor,
            args.translator,
            local_files_only=args.local_files_only,
            device=args.device,
        )
        run_confirm(
            offset=int(args.offset),
            limit=int(limit),
            output_dir=args.output_dir,
            allow_consumed=bool(args.allow_consumed),
            seed=int(args.seed),
            predictor=predictor,
            translator=translator,
            checkpoint_path=checkpoint,
            predictor_name=args.predictor,
            translator_name=args.translator,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
