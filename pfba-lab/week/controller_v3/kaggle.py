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
A labels row holds the sentence id, split, source, ``label_format``, the
behavior policy (``read_prob`` and the fixed write action), and one entry per
decision position with ``read_pos``, ``observed``, ``successor_observed``,
``committed_before``/``committed_after``, ``oracle_words``, the behavior-chosen
action, the pre-action ``HistoryTracker`` snapshot, a ``terminal`` flag and
``terminal_value``, and every action outcome. Each outcome carries
``matched`` / ``incorrect`` / ``emitted`` / ``next_committed`` / timing / vote
statistics. Full candidate strings are deliberately omitted. Training rebuilds
each decision state through the stored tracker snapshot and each action's own
successor from its ``next_committed``, so training sees the same live History
features as serving without persisting a 34-float vector per action.
``stride`` is fixed at 1: a larger stride would skip source words while
applying a single bootstrapped discount, so `label_sentence` raises instead.
A ``labels.jsonl`` written by an older ``label_format`` is refused rather than
mixed with one-step records.

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
import random
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
from .features import (
    FeatureBuilder,
    History,
    HistoryTracker,
    RuntimeInfo,
    StateInput,
    advance_virtual_clock,
    estimate_length_ratio,
)
from .labeler import ActionOutcome, CounterfactualLabeler
from .network import CAVCNetwork
from .reward import RewardConfig, immediate_reward, terminal_reward
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

# Bump whenever the `labels.jsonl` state/outcome schema changes. A resumable
# dev run refuses a file written by an older schema instead of mixing a
# multi-step successor with the current one-step successor. Format 3 fixes the
# terminal proxy: the hypothesis must include the forced committed prefix,
# because `oracle_words` returns only its continuation.
LABEL_FORMAT = 3

# Trajectory-coverage policy: READ with probability `DEFAULT_READ_PROB`,
# otherwise the fixed write action. This is *not* the policy being learned; it
# only decides which committed prefixes appear as training states. The Bellman
# target still comes from every labeled action outcome.
DEFAULT_READ_PROB = 0.25
FIXED_BEHAVIOR_WRITE = Action(4, 6)

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
    """Read JSONL rows, tolerating one truncated trailing line.

    A hard kill during an append can leave the last line incomplete. That line
    is skipped so a resumed session is not blocked; every earlier malformed
    line is still a hard error because it would indicate a corrupt file.
    """
    if not path.is_file():
        return []
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows: list[dict] = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise ValueError(
                f"{path} has a malformed JSONL row at line {index + 1}"
            ) from None
    return rows


def _repair_jsonl_tail(path: Path) -> None:
    """Trim a partially written final line so appends stay well-formed.

    `labels.jsonl` is appended and flushed one sentence at a time. If the
    process is killed mid-write, the file may not end in a newline; truncating
    back to the last complete line lets the next session resume cleanly.
    """
    if not path.is_file():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n")
    path.write_bytes(data[: cut + 1] if cut >= 0 else b"")


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


def _reuse_cost_records(
    path: Path, predictor_id: int, translator_id: int
):
    """Load measured records only when they match the current adapter pair.

    A table measured for ``(0, 0)`` must never price ``(1, 1)``. When the
    recorded ids differ, the stale file is ignored and the caller recalibrates
    instead.
    """
    loaded = _load_cost_records(path)
    if loaded is None:
        return None
    records, summary = loaded
    if not records:
        return None
    recorded_pid = summary.get("predictor_id")
    recorded_tid = summary.get("translator_id")
    mismatch = (
        recorded_pid is not None and int(recorded_pid) != int(predictor_id)
    ) or (
        recorded_tid is not None and int(recorded_tid) != int(translator_id)
    )
    if mismatch:
        print(
            "[cavc] ignoring cost_records.json for model pair "
            f"{recorded_pid}/{recorded_tid}; current pair is "
            f"{predictor_id}/{translator_id}",
            flush=True,
        )
        return None
    return records, summary


def _serialize_outcome(outcome: ActionOutcome, reward: float) -> dict:
    """One outcome with candidate text omitted to keep the labels file small.

    `next_committed` and `emitted` are the successor inputs: training rebuilds
    each action's own successor from them, so READ and every write reach a
    different state. Full candidate strings are deliberately not persisted.
    """
    return {
        "action": outcome.action.label,
        "branches": int(outcome.action.branches),
        "horizon": int(outcome.action.horizon),
        "matched": int(outcome.matched),
        "incorrect": int(outcome.incorrect),
        "emitted": int(outcome.emitted),
        "no_emit": bool(outcome.no_emit),
        "next_committed": list(outcome.next_committed),
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
        next_committed=tuple(record.get("next_committed") or ()),
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
    if not sentence_id:
        raise ValueError(
            "label_sentence needs a non-empty sentence_id so the row can be "
            "resumed and never relabelled"
        )
    if split == "test" or sentence_id.startswith("test:"):
        raise ValueError(
            f"label_sentence refuses test sentence {sentence_id!r}; labels are "
            "training-only"
        )
    # Hash bucket 1 is the dev holdout. The check applies to every id, not
    # only `dev:`-prefixed ids, so a caller cannot relabel the holdout by
    # using a different id spelling.
    if is_holdout(sentence_id):
        raise ValueError(
            f"label_sentence refuses holdout sentence {sentence_id!r} "
            f"(hash bucket {HOLDOUT_BUCKET}); the holdout is evaluation-only"
        )


def _post_action_backlog(
    src_words_after: int,
    committed_before: Sequence[str],
    outcome: ActionOutcome,
    rho: float,
) -> float:
    """Backlog after one action: arrived source minus projected target."""
    words_after = len(committed_before) + int(outcome.emitted)
    return max(0.0, float(src_words_after) - words_after / float(rho))


def _action_reward(
    outcome: ActionOutcome,
    src_words_after: int,
    committed_before: Sequence[str],
    rho: float,
    reward_cfg: RewardConfig,
    source_advance: int = 1,
) -> float:
    """Immediate reward of one outcome under the post-action backlog."""
    return immediate_reward(
        int(outcome.matched),
        int(outcome.incorrect),
        _post_action_backlog(src_words_after, committed_before, outcome, rho),
        int(source_advance),
        reward_cfg,
    )


def _advance_tracker(
    tracker: HistoryTracker,
    outcome: ActionOutcome,
    next_now_s: float | None = None,
) -> HistoryTracker:
    """Return a clone of `tracker` advanced by `outcome`.

    Labeling and training both call this helper, so a successor rebuilt from a
    persisted state row is identical to the one built online. `now_s` is the
    virtual source-arrival clock of the successor decision.
    """
    successor = tracker.clone()
    now_s = successor.now_s if next_now_s is None else float(next_now_s)
    if outcome.action.is_read:
        successor.on_read(now_s=now_s)
    else:
        successor.on_write(outcome, now_s=now_s)
    return successor


def _terminal_value_proxy(
    labeler: CounterfactualLabeler,
    source: str,
    committed: Sequence[str],
    reference: str,
    reward_cfg: RewardConfig,
) -> float:
    """Length-gap proxy for a terminal state, including the forced prefix.

    `oracle_words` translates the full source with `committed` forced and
    returns only the continuation after that prefix. The final hypothesis is
    therefore ``committed + continuation``; using the continuation alone would
    make a longer correct prefix look worse. This remains a local word-count
    proxy, never BLEU or COMET.
    """
    committed_tuple = tuple(committed)
    continuation = labeler.oracle_words(source, committed_tuple)
    hypothesis = " ".join((*committed_tuple, *continuation))
    return float(terminal_reward(hypothesis, reference, reward_cfg))


def label_sentence(
    labeler: CounterfactualLabeler,
    sentence,
    stride: int = 1,
    cfg: ControllerConfig | None = None,
    *,
    builder: FeatureBuilder | None = None,
    predictor_id: int = 0,
    translator_id: int = 0,
    read_prob: float = DEFAULT_READ_PROB,
    max_states: int | None = None,
) -> dict:
    """Label one sentence with true one-step transitions.

    The walk starts at `cfg.min_start` and ends at the last source word. Every
    kept position is labelled, so a successor is exactly one source word later
    and one discount is correct. ``stride=1`` is the only supported value: a
    larger stride would jump several source words with a single bootstrapped
    discount, so it raises until a real multi-step target exists.

    At each state:

    * The state is built through `HistoryTracker`, the same implementation the
      rollout uses, so History features are live instead of all-zero.
    * `labeler.label_state` labels every write in `cfg.actions`; `read_outcome`
      supplies READ.
    * The compact successor inputs are persisted per action: READ keeps
      `committed` and advances `observed` by one word; a write stores its
      ``next_committed`` and advances `observed` by the same word.
      `_targets_from_rows` materializes one successor `StateInput` per action.
    * The behavior policy only chooses which committed prefix becomes the next
      training state: READ with probability `read_prob`, otherwise a fixed
      write. The fixed write is :data:`FIXED_BEHAVIOR_WRITE` when it is in
      `cfg.actions`, otherwise the first configured write action (an unreachable
      action must not be introduced). It is **not** an immediate-reward argmax,
      and the Bellman target still uses every labeled action outcome.

    The clock advances with :func:`features.advance_virtual_clock` using the
    behavior action's measured ``pred_s + mt_s``, matching the rollout cadence
    so `time_since_commit` is in the same range in training and serving.

    `max_states` stops after that many labeled positions. A stopped state is
    terminal only when it is the sentence's true last decision position.

    A terminal state's `terminal_value` is a documented proxy: the negative
    absolute word-count difference between ``committed + oracle continuation``
    and the reference. Including the forced prefix is essential; the
    continuation alone would make a more complete sentence look worse. It is
    not BLEU or COMET. Proxies such as this are deliberately local to training.
    """
    cfg = cfg or ControllerConfig()
    stride = int(stride)
    if stride < 1:
        raise ValueError("stride must be a positive integer")
    if stride != 1:
        raise ValueError(
            "label_sentence implements true one-step transitions only; "
            f"stride={stride} would skip source words while applying a single "
            "bootstrapped discount. Implement accumulated multi-step rewards "
            "before raising the stride."
        )
    read_prob = float(read_prob)
    if not 0.0 <= read_prob <= 1.0:
        raise ValueError("read_prob must be in [0, 1]")

    sentence_id = str(_field(sentence, "sentence_id", ""))
    split = str(_field(sentence, "split", "dev"))
    _guard_label_sentence(sentence_id, split)
    source = str(_field(sentence, "source", ""))
    reference = str(_field(sentence, "reference", ""))
    words = source.split()
    full_positions = _decision_positions(len(words), cfg.min_start)
    if max_states is not None:
        max_states = int(max_states)
        if max_states < 0:
            raise ValueError("max_states must be non-negative")
        positions = full_positions[:max_states]
    else:
        positions = full_positions
    builder = builder if builder is not None else FeatureBuilder()
    rho = float(builder.rho)
    reward_cfg = _reward_config(cfg)
    arrival_gap = float(cfg.arrival_gap)

    # The coverage policy may only execute an action that is actually labelled.
    writes = tuple(action for action in cfg.actions if not action.is_read)
    if FIXED_BEHAVIOR_WRITE in writes:
        behavior_write = FIXED_BEHAVIOR_WRITE
    elif writes:
        # Do not append (4, 6) to the grid: that would train an action the
        # controller never evaluates. Fall back to the first configured write.
        behavior_write = writes[0]
    else:
        raise ValueError(
            "label_sentence needs at least one write action in cfg.actions for "
            "the behavior policy"
        )
    label_actions = tuple(cfg.actions)

    tracker = HistoryTracker()
    committed: tuple[str, ...] = ()
    states: list[dict] = []
    # `index` indexes the (possibly truncated) positions, so this is the true
    # last decision position even when `max_states` stops the walk early.
    last_index = len(full_positions) - 1
    clock_s = 0.0

    for index, read_pos in enumerate(positions):
        observed = " ".join(words[:read_pos])
        successor_observed = " ".join(words[: read_pos + 1])
        src_words_after = len(successor_observed.split())

        # Build the decision state through the shared online tracker.
        tracker.set_clock(clock_s)
        tracker.build(
            builder, observed, committed, predictor_id, translator_id
        )

        seed = _step_seed(cfg, sentence_id, read_pos, split)
        write_outcomes = labeler.label_state(
            observed, committed, source, label_actions, seed
        )
        read = labeler.read_outcome(committed)
        all_outcomes = (read, *write_outcomes)
        by_action = {outcome.action: outcome for outcome in all_outcomes}
        rewards = {
            outcome.action: _action_reward(
                outcome, src_words_after, committed, rho, reward_cfg
            )
            for outcome in all_outcomes
        }

        # Deterministic per-state coverage draw; it decides the next training
        # state, not the Bellman target.
        rng = random.Random(_step_seed(cfg, sentence_id, read_pos, split))
        behavior_action = (
            READ if rng.random() < read_prob else behavior_write
        )
        behavior_outcome = by_action.get(behavior_action)
        if behavior_outcome is None:
            raise RuntimeError(
                f"behavior action {behavior_action.label} was not labelled at "
                f"{sentence_id}:{read_pos}; it must be in cfg.actions or the "
                "fixed behavior write"
            )

        terminal = index == last_index
        row = {
            "read_pos": int(read_pos),
            "observed": observed,
            "successor_observed": successor_observed,
            "committed_before": list(committed),
            "committed_after": list(behavior_outcome.next_committed),
            "oracle_words": int(len(labeler.oracle_words(source, committed))),
            "chosen": behavior_action.label,
            "chosen_reward": float(rewards[behavior_action]),
            "history": asdict(tracker.history),
            "runtime": asdict(tracker.runtime),
            "last_commit_s": float(tracker.last_commit_s),
            "now_s": clock_s,
            "terminal": bool(terminal),
            "terminal_value": None,
            "outcomes": [
                _serialize_outcome(outcome, rewards[outcome.action])
                for outcome in all_outcomes
            ],
        }

        if terminal:
            # EOS-flush proxy; `_terminal_value_proxy` appends the forced
            # committed prefix that `oracle_words` omits.
            row["terminal_value"] = _terminal_value_proxy(
                labeler,
                source,
                tuple(behavior_outcome.next_committed),
                reference,
                reward_cfg,
            )

        states.append(row)
        committed = tuple(behavior_outcome.next_committed)
        # Same clock recurrence as the rollout: settle the source arrival, then
        # bill the behavior action's measured compute (READ adds none).
        spent = float(behavior_outcome.pred_s) + float(behavior_outcome.mt_s)
        clock_s = advance_virtual_clock(
            clock_s, read_pos, arrival_gap, spent
        )
        tracker = _advance_tracker(tracker, behavior_outcome, clock_s)

    return {
        "sentence_id": sentence_id,
        "split": split,
        "source": source,
        "reference": reference,
        "stride": 1,
        "label_format": LABEL_FORMAT,
        "behavior": {
            "read_prob": float(read_prob),
            "fixed_action": behavior_write.label,
        },
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


def _tracker_from_row(state_row: dict) -> HistoryTracker:
    """Rebuild the pre-action tracker stored in one `states` entry."""
    history = History(**dict(state_row.get("history") or {}))
    runtime = RuntimeInfo(**dict(state_row.get("runtime") or {}))
    return HistoryTracker(
        history,
        runtime,
        last_commit_s=float(state_row.get("last_commit_s", 0.0)),
        now_s=float(state_row.get("now_s", 0.0)),
    )


def _targets_from_rows(
    rows: Sequence[dict],
    builder: FeatureBuilder,
    cfg: ControllerConfig,
    predictor_id: int,
    translator_id: int,
    *,
    limit: int | None = None,
):
    """Rebuild `ActionTarget` rows from persisted labels rows.

    Each state is rebuilt through the `HistoryTracker` snapshot saved in the
    row, so training states carry the same live History features as serving.
    Each outcome then rebuilds *its own* successor: READ keeps the committed
    prefix and advances one source word, while a write advances by its
    ``next_committed``. For ``stride=1`` the successor is exactly one source
    word later, so one `cfg.gamma` discount is correct.
    """
    targets = []
    states: list[StateInput] = []
    remaining = None if limit is None else int(limit)
    if remaining is not None and remaining < 0:
        raise ValueError("limit must be non-negative")

    for row in rows:
        sentence_id = str(row.get("sentence_id", ""))
        for state_row in row.get("states") or []:
            if remaining is not None and remaining <= 0:
                break
            tracker = _tracker_from_row(state_row)
            observed = str(state_row.get("observed", ""))
            committed = tuple(state_row.get("committed_before") or ())
            state = tracker.build(
                builder, observed, committed, predictor_id, translator_id
            )
            outcomes = [
                _outcome_from_record(record)
                for record in state_row.get("outcomes") or []
            ]
            successor_observed = str(
                state_row.get("successor_observed") or ""
            )
            # Per-action successor clock, using the same recurrence as the
            # rollout and the labeling walk. READ's stored pred_s/mt_s are 0.
            pre_now_s = float(state_row.get("now_s", tracker.now_s))
            read_pos = int(state_row.get("read_pos", 0))
            if "src_words_after" in state_row:
                src_words_after = int(state_row["src_words_after"])
            elif successor_observed:
                src_words_after = len(successor_observed.split())
            else:
                src_words_after = len(observed.split()) + 1
            key = f"{sentence_id}:{state_row.get('read_pos')}"
            terminal = bool(state_row.get("terminal", False))

            if terminal:
                targets.extend(
                    build_action_targets(
                        state,
                        outcomes,
                        cfg,
                        terminal=True,
                        terminal_value=float(
                            state_row.get("terminal_value") or 0.0
                        ),
                        src_words_after=src_words_after,
                        state_key=key,
                        rho=builder.rho,
                    )
                )
            else:
                next_states = []
                for outcome in outcomes:
                    spent = float(outcome.pred_s) + float(outcome.mt_s)
                    next_now_s = advance_virtual_clock(
                        pre_now_s, read_pos, cfg.arrival_gap, spent
                    )
                    next_states.append(
                        _advance_tracker(
                            tracker, outcome, next_now_s
                        ).build(
                            builder,
                            successor_observed,
                            outcome.next_committed,
                            predictor_id,
                            translator_id,
                        )
                    )
                targets.extend(
                    build_action_targets(
                        state,
                        outcomes,
                        cfg,
                        next_states=next_states,
                        src_words_after=src_words_after,
                        state_key=key,
                        rho=builder.rho,
                    )
                )
            states.append(state)
            if remaining is not None:
                remaining -= 1
        if remaining is not None and remaining <= 0:
            break

    return targets, states


def _label_few_states(
    labeler: CounterfactualLabeler,
    sentences: Sequence,
    builder: FeatureBuilder,
    cfg: ControllerConfig,
    count: int,
    predictor_id: int,
    translator_id: int,
    read_prob: float = DEFAULT_READ_PROB,
):
    """Label at most `count` one-step decision states across sentences."""
    count = int(count)
    if count <= 0:
        raise ValueError("label_states must be positive")

    rows: list[dict] = []
    labeled_states = 0
    for sentence in sentences:
        remaining = count - labeled_states
        row = label_sentence(
            labeler,
            sentence,
            stride=1,
            cfg=cfg,
            builder=builder,
            predictor_id=predictor_id,
            translator_id=translator_id,
            read_prob=read_prob,
            max_states=remaining,
        )
        rows.append(row)
        labeled_states += len(row.get("states") or [])
        if labeled_states >= count:
            break

    return _targets_from_rows(
        rows, builder, cfg, predictor_id, translator_id, limit=count
    )


def _skipped_training(n_samples: int) -> dict:
    """Training summary used when the wall-clock deadline halts a session."""
    return {
        "epochs": 0,
        "final_loss": 0.0,
        "value_loss": 0.0,
        "rank_loss": 0.0,
        "error_loss": 0.0,
        "no_emit_loss": 0.0,
        "emit_loss": 0.0,
        "n_samples": int(n_samples),
        "status": "skipped",
        "reason": "deadline",
    }


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
    deadline: float | None = None,
    initial_stopped_reason: str | None = None,
) -> dict:
    """Train, checkpoint, roll out, and write every artifact for a mode.

    `deadline` is an absolute `time.perf_counter()` value. It is checked before
    training and before each rollout sentence; when it has passed the run stops
    gracefully, sets ``stopped_reason="deadline"``, and still writes the
    partial summary/traces. An existing checkpoint is loaded instead of being
    overwritten by an untrained network. When no checkpoint exists yet, an
    untrained network is **never** written to disk: the summary records
    ``trained=False`` and ``checkpoint.status="not_written"``, so a later
    confirm cannot mistake a file for trained weights.

    `model_pair["predictor_id"]` / ``["translator_id"]`` are injected into
    every rollout feature vector so the network embedding and the cost lookup
    use the same adapter pair as training and calibration.
    """
    stopped_reason = initial_stopped_reason
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    mean, std = _feature_stats(states)

    trained = False
    if deadline is not None and time.perf_counter() >= deadline:
        training = _skipped_training(len(targets))
        stopped_reason = "deadline"
    else:
        trainer = CAVCTrainer(network, cfg, cost_model)
        training = trainer.fit(targets)
        trained = True

    training_extra = {
        "mode": mode,
        "split": split,
        "training": training,
        "n_targets": len(targets),
        "trained": trained,
        # Frozen confirmation must rebuild the same word ids; store the
        # vocabulary and the training length ratio so a test sentence cannot
        # silently remap the text-encoder inputs.
        "vocab": list(vocab.id_to_token),
        "length_ratio": float(builder.rho),
    }
    if trained:
        metadata = save_checkpoint(
            checkpoint_path, network, cfg, mean, std, cost_model, extra=training_extra
        )
        checkpoint_status = "written"
    elif checkpoint_path.is_file():
        # Keep the last trained checkpoint rather than overwriting it with an
        # untrained network when this session stopped before training.
        metadata = load_checkpoint(checkpoint_path, network)
        checkpoint_status = "kept_existing"
    else:
        # Never persist an untrained network. A checkpoint file is a promise
        # that its weights were trained; confirm relies on that promise.
        metadata = None
        checkpoint_status = "not_written"

    predictor_id = int(model_pair.get("predictor_id", 0))
    translator_id = int(model_pair.get("translator_id", 0))
    traces: list[RolloutTrace] = []
    for sentence in _with_split(rollout_sentences, split):
        if deadline is not None and time.perf_counter() >= deadline:
            stopped_reason = "deadline"
            break
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
                predictor_id=predictor_id,
                translator_id=translator_id,
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
        "git_commit": (
            metadata.get("git_commit")
            if metadata is not None
            else _git_commit()
        ),
        "model_pair": dict(model_pair),
        "vocab_size": len(vocab),
        "sentences_loaded": len(sentences),
        "label_sentences": len(train_sentences),
        "n_label_rows": len(label_rows) if label_rows is not None else None,
        "n_states": len(states),
        "n_targets": len(targets),
        "training": training,
        "trained": trained,
        "stopped_reason": stopped_reason,
        "checkpoint": (
            {
                "path": str(checkpoint_path),
                "status": checkpoint_status,
                "format_version": metadata["format_version"],
                "seed": metadata["seed"],
                "git_commit": metadata.get("git_commit"),
                "feature_mean_len": len(metadata["feature_mean"]),
                "feature_std_len": len(metadata["feature_std"]),
                "cost_records": len(metadata["cost_records"]),
            }
            if metadata is not None
            else {
                "path": None,
                "status": checkpoint_status,
                "reason": stopped_reason,
            }
        ),
        "cost": cost_summary,
        "rollout": rollout_summary,
        "artifacts": {
            "summary": str(summary_path),
            "traces": str(traces_path),
            "config": str(config_path),
            "cost_records": str(cost_path),
            "checkpoint": (
                str(checkpoint_path) if metadata is not None else None
            ),
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
    if checkpoint_status == "not_written":
        print(
            "[cavc] no checkpoint written: training was skipped before the "
            "deadline",
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
    read_prob: float = DEFAULT_READ_PROB,
) -> dict:
    """Wire the real pipeline end to end on a few dev sentences.

    Loads **dev** sentences only, calibrates the cost table, labels a few
    one-step decision states, trains, checkpoints, and rolls out a few
    sentences. The five required artifacts are written under `output_dir`;
    BLEU/COMET carry their real status.
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
        read_prob=read_prob,
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
        extra_summary={
            "preflight": True,
            "label_format": LABEL_FORMAT,
            "behavior": {
                "read_prob": float(read_prob),
                "fixed_action": FIXED_BEHAVIOR_WRITE.label,
            },
        },
    )


def run_dev(
    limit: int = 0,
    stride: int = 1,
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
    read_prob: float = DEFAULT_READ_PROB,
) -> dict:
    """Label dev train buckets 2-9, train, and roll out dev holdout bucket 1.

    Labels are appended to ``output_dir/labels.jsonl`` one row per sentence and
    flushed, so a session that hits `max_hours` can resume: existing
    `sentence_id` values are skipped and never relabelled, and a truncated
    trailing line is trimmed before appending. The vocabulary and length ratio
    are built from train buckets 2-9 only, so holdout bucket 1 contributes no
    vocabulary entries. Holdout bucket 1 and every test sentence are excluded
    from labeling by construction.

    `max_hours` bounds labeling, cost calibration, training, and the rollout
    loop. When the deadline passes, the run stops gracefully, writes the
    partial ``summary.json``/``traces.jsonl`` with
    ``stopped_reason="deadline"``, and keeps the last trained checkpoint.
    A training call is not preempted mid-epoch; it is skipped if the deadline
    has already passed when training would start.
    """
    pfba = _require_run_pfba()
    cfg = cfg or ControllerConfig(seed=int(seed))
    stride = int(stride)
    if stride != 1:
        raise ValueError(
            "run_dev only supports stride=1 true one-step labels; a multi-step "
            "discounted target is not implemented"
        )
    read_prob = float(read_prob)
    if not 0.0 <= read_prob <= 1.0:
        raise ValueError("read_prob must be in [0, 1]")
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
    # Train-only vocabulary: holdout bucket 1 must not contribute entries, or
    # its words would leak into the frozen text encoder.
    builder, network, vocab = _components(cfg, train_sentences)
    labeler = CounterfactualLabeler(predictor, translator, cfg, timer=timer)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.jsonl"
    _repair_jsonl_tail(labels_path)
    existing_rows = _read_jsonl(labels_path)
    stale = [
        row
        for row in existing_rows
        if int(row.get("label_format", 0)) != LABEL_FORMAT
    ]
    if stale:
        raise ValueError(
            f"{labels_path} was written by an older label format "
            f"(expected label_format={LABEL_FORMAT}). Move it aside so the "
            "stride=1 one-step labels are regenerated instead of mixing "
            "multi-step successors with one-step ones."
        )
    done = {str(row.get("sentence_id", "")) for row in existing_rows}

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
            row = label_sentence(
                labeler,
                sentence,
                1,
                cfg,
                builder=builder,
                predictor_id=pid,
                translator_id=tid,
                read_prob=read_prob,
            )
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
            "label_format": LABEL_FORMAT,
            "read_prob": float(read_prob),
            "fixed_action": FIXED_BEHAVIOR_WRITE.label,
            "newly_labeled": newly_labeled,
            "total_labeled": len(rows),
            "train_sentences": len(train_sentences),
            "holdout_sentences": len(holdout_sentences),
            "stopped_for_deadline": stopped_for_deadline,
            "max_hours": max_hours,
        },
    )
    if not rows:
        if time.perf_counter() >= deadline:
            summary = {
                "mode": "dev",
                "split": "dev",
                "stopped_reason": "deadline",
                "trained": False,
                "n_label_rows": 0,
                "n_targets": 0,
                "label_format": LABEL_FORMAT,
                "behavior": {
                    "read_prob": float(read_prob),
                    "fixed_action": FIXED_BEHAVIOR_WRITE.label,
                },
                "newly_labeled": newly_labeled,
                "train_sentences": len(train_sentences),
                "holdout_sentences": len(holdout_sentences),
                "max_hours": max_hours,
                "notes": ["No label row was written before the deadline."],
            }
            _write_json(output_dir / "summary.json", summary)
            _write_traces(output_dir / "traces.jsonl", [])
            return summary
        raise RuntimeError(
            "run_dev has no labeled rows yet; rerun with more time to label "
            "at least one dev train sentence"
        )

    stopped_reason = "deadline" if stopped_for_deadline else None
    loaded_cost = _reuse_cost_records(
        output_dir / "cost_records.json", pid, tid
    )
    if loaded_cost is not None:
        cost_records, cost_summary = loaded_cost
        cost_summary = dict(cost_summary)
        cost_summary["reused_from_artifact"] = True
    elif time.perf_counter() >= deadline:
        print(
            "[cavc] deadline reached before cost calibration; using an empty "
            "cost table",
            flush=True,
        )
        cost_records = []
        cost_summary = {
            "status": "skipped",
            "reason": "deadline",
            "n_records": 0,
            "predictor_id": pid,
            "translator_id": tid,
        }
        stopped_reason = "deadline"
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
        rollout_sentences=holdout_sentences,
        model_pair=model_pair,
        notes=notes,
        deadline=deadline,
        initial_stopped_reason=stopped_reason,
        extra_summary={
            "stride": stride,
            "label_format": LABEL_FORMAT,
            "behavior": {
                "read_prob": float(read_prob),
                "fixed_action": FIXED_BEHAVIOR_WRITE.label,
            },
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
    allow_untrained: bool = False,
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
    text-encoder id.

    A supplied checkpoint must carry ``extra["trained"] is True``. If it is
    false or missing, `ValueError` is raised unless `allow_untrained=True` is
    passed explicitly; either way the summary reports ``trained=False`` rather
    than claiming the weights were trained. Without a checkpoint an untrained
    network is evaluated and the summary reports ``trained=False`` so the run
    cannot be mistaken for a real confirmation.
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
    checkpoint_trained = False
    if checkpoint_path is not None:
        loaded_metadata = load_checkpoint(checkpoint_path, network)
        checkpoint_trained = bool(
            (loaded_metadata.get("extra") or {}).get("trained", False)
        )
        if not checkpoint_trained and not allow_untrained:
            raise ValueError(
                f"checkpoint {checkpoint_path} was not trained "
                "(extra['trained'] is false or missing). Refusing to present "
                "untrained weights as a confirmation result; pass "
                "allow_untrained=True only for a diagnostic run."
            )
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
            predictor_id=pid,
            translator_id=tid,
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
        "allow_untrained": bool(allow_untrained),
        "tuned": False,
        "trained": checkpoint_trained,
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
                "No trained checkpoint is in use, so untrained weights were "
                "rolled out; this is not a confirmation result."
                if not checkpoint_trained
                else "Trained weights and measured cost records loaded from "
                "the checkpoint."
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
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="confirm mode: evaluate a checkpoint whose extra['trained'] is "
        "false (diagnostic only)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="decision stride; only 1 (true one-step labels) is supported",
    )
    parser.add_argument(
        "--read-prob",
        type=float,
        default=DEFAULT_READ_PROB,
        help="behavior-policy READ probability for trajectory coverage",
    )
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
        plan["read_prob"] = float(args.read_prob)
    elif mode == "dev":
        plan["limit"] = int(args.limit or 0)
        plan["stride"] = int(args.stride)
        plan["read_prob"] = float(args.read_prob)
        plan["max_hours"] = float(args.max_hours)
    elif mode == "confirm":
        limit = DEFAULT_CONFIRM_LIMIT if args.limit is None else args.limit
        plan["offset"] = int(args.offset)
        plan["limit"] = int(limit)
        plan["allow_consumed"] = bool(args.allow_consumed)
        plan["allow_untrained"] = bool(args.allow_untrained)
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
            read_prob=float(args.read_prob),
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
            read_prob=float(args.read_prob),
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
            allow_untrained=bool(args.allow_untrained),
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
