"""Closed-loop A/B/C/D evaluation of retrained checkpoints.

The retraining path is offline: it never loads the predictor or translator.
This module is the other half: it loads the real small adapter pair once and
rolls out each variant on unseen dev sentences (``dev:0200+`` by default). It
does **not** label, train, or tune anything.

Variants are paired against the fixed ``(4,6)`` baseline on the exact same
sentences and runtime path:

* ``A``: the original checkpoint, no guards;
* ``B``: the original checkpoint, V3.1 guards;
* ``C``: a retrained checkpoint, no guards;
* ``D``: a retrained checkpoint, V3.1 guards.

The guard configuration is the medium setting used in the offline diagnostics,
so the closed-loop run answers whether the offline action shift survives real
sampling, translation, and RALCP.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from .action import Action
from .checkpoint import _deserialize_cost_records, load_checkpoint
from .config import ControllerConfig
from .cost_model import MeasuredCostTable
from .evaluator import summarize_traces
from .features import FeatureBuilder
from .network import CAVCNetwork
from .rollout import run_fixed_rollout, run_rollout
from . import kaggle

__all__ = [
    "chrf_stats",
    "guard_config",
    "read_run_stats",
    "run_evaluation",
    "select_sentences",
    "service_time_stats",
]


def guard_config(cfg: ControllerConfig) -> ControllerConfig:
    """The medium-strength V3.1 guard used for variants B and D.

    The no-emit head uses a **soft** penalty, not a hard veto: the risk head is
    calibrated but not discriminative enough to justify suppressing every
    WRITE at a state. The hard veto remains available as an explicit ablation
    by overriding ``no_emit_veto`` on the returned config.
    """
    return dataclasses.replace(
        cfg,
        read_threshold=0.10,
        read_threshold_lag=0.15,
        read_threshold_reads=0.15,
        read_threshold_no_commit=0.10,
        expensive_margin=0.25,
        no_emit_veto=1.0,
        no_emit_penalty=0.20,
    )


def select_sentences(sentences, *, offset: int, limit: int):
    """Slice the unseen dev suffix and refuse a test sentence."""
    offset = int(offset)
    limit = int(limit)
    if offset < 0 or limit < 0:
        raise ValueError("offset and limit must be non-negative")
    materialized = list(sentences)
    selected = (
        materialized[offset : offset + limit]
        if limit > 0
        else materialized[offset:]
    )
    if not selected:
        raise RuntimeError(
            f"no evaluation sentences at offset {offset} limit {limit}"
        )
    for sentence in selected:
        sentence_id = str(kaggle._field(sentence, "sentence_id", ""))
        if sentence_id.startswith("test:"):
            raise ValueError(
                "closed-loop evaluation must not consume the test split; "
                f"got {sentence_id!r}"
            )
    return selected


def read_run_stats(traces) -> dict:
    """Consecutive READ runs and the read fraction over real rollout steps."""
    runs: list[int] = []
    current = 0
    reads = 0
    steps = 0
    for trace in traces:
        for step in trace.steps:
            steps += 1
            if str(step.get("action")) == "read":
                current += 1
                reads += 1
            else:
                if current:
                    runs.append(current)
                current = 0
        if current:
            runs.append(current)
            current = 0
    return {
        "n_reads": reads,
        "n_steps": steps,
        "read_fraction": reads / steps if steps else None,
        "n_runs": len(runs),
        "mean_run": (sum(runs) / len(runs)) if runs else None,
        "max_run": max(runs) if runs else 0,
    }


def service_time_stats(traces) -> dict:
    """Measured predictor/translator service time, overall and per action."""
    per_action: dict[str, dict[str, float]] = {}
    total_pred = 0.0
    total_mt = 0.0
    for trace in traces:
        for step in trace.steps:
            pred_s = float(step.get("pred_s", 0.0) or 0.0)
            mt_s = float(step.get("mt_s", 0.0) or 0.0)
            total_pred += pred_s
            total_mt += mt_s
            label = str(step.get("action", "unknown"))
            bucket = per_action.setdefault(
                label, {"pred_s": 0.0, "mt_s": 0.0, "count": 0.0}
            )
            bucket["pred_s"] += pred_s
            bucket["mt_s"] += mt_s
            bucket["count"] += 1.0
    return {
        "pred_s": total_pred,
        "mt_s": total_mt,
        "total_s": total_pred + total_mt,
        "per_action": per_action,
    }


def chrf_stats(traces) -> dict:
    """Corpus chrF when sacrebleu is installed; explicit status otherwise."""
    try:
        from sacrebleu.metrics import CHRF
    except Exception as exc:  # pragma: no cover - kernel installs sacrebleu
        return {"status": "unavailable", "reason": str(exc)}
    if not traces:
        return {"status": "unavailable", "reason": "no traces"}
    hypotheses = [trace.hypothesis for trace in traces]
    references = [[trace.reference for trace in traces]]
    try:
        score = CHRF().corpus_score(hypotheses, references).score
    except Exception as exc:  # pragma: no cover - defensive
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    return {"status": "ok", "score": float(score)}


def _resolve_checkpoint(path, *, search_root: Path | None = None) -> Path:
    """Resolve a checkpoint path against Kaggle's current input layout.

    Kaggle used to mount an attached dataset at ``/kaggle/input/<slug>`` but
    newer runtimes mount it at ``/kaggle/input/datasets/<owner>/<slug>``.
    A literal path therefore breaks when the layout changes, so fall back to
    locating the file by name under the input root.
    """
    checkpoint = Path(path)
    if checkpoint.is_file():
        return checkpoint
    root = Path("/kaggle/input") if search_root is None else Path(search_root)
    candidates = (
        sorted(
            candidate
            for candidate in root.rglob(checkpoint.name)
            if candidate.is_file()
        )
        if root.is_dir()
        else []
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        same_parent = [
            candidate
            for candidate in candidates
            if candidate.parent.name == checkpoint.parent.name
        ]
        if len(same_parent) == 1:
            return same_parent[0]
    available = (
        sorted(item.name for item in root.glob("*")) if root.is_dir() else []
    )
    raise FileNotFoundError(
        f"evaluation checkpoint {checkpoint} is missing; mounted files: "
        f"{available}; matches by name: {[str(item) for item in candidates]}"
    )


def _policy_checkpoints(
    old_checkpoint,
    retrained: dict[str, str],
    *,
    include_baseline: bool,
) -> dict[str, str]:
    """Return the ordered `{policy_name: checkpoint}` map for an eval run.

    ``include_baseline=False`` drops the `--old-checkpoint` policy so a
    retrained-only kernel never pays for the A/B baseline again.
    """
    policies: dict[str, str] = {}
    if include_baseline:
        policies["v3_baseline"] = old_checkpoint
    policies.update(dict(retrained))
    return policies


def _load_policy(path, predictor_id: int, translator_id: int):
    checkpoint = _resolve_checkpoint(path)
    payload = kaggle._checkpoint_payload(checkpoint)
    cfg = kaggle._config_from_payload(payload, str(checkpoint))
    vocab, ratio = kaggle._frozen_vocab(payload)
    if vocab is None:
        raise ValueError(f"checkpoint {checkpoint} has no stored vocabulary")
    builder = FeatureBuilder(
        length_ratio=float(ratio if ratio is not None else 1.0),
        src_vocab=vocab,
        tgt_vocab=vocab,
    )
    network = CAVCNetwork(cfg, len(vocab), len(vocab))
    load_checkpoint(checkpoint, network)
    mean = payload.get("feature_mean") or []
    std = payload.get("feature_std") or []
    if mean and std:
        network.set_feature_stats(mean, std)
    network.eval()
    records = _deserialize_cost_records(payload.get("cost_records", []))
    topology = str((payload.get("extra") or {}).get("cost_topology", "serial"))
    cost_model = kaggle._cost_model_for_topology(
        MeasuredCostTable(records), topology
    )
    return cfg, builder, network, cost_model


def run_evaluation(
    *,
    old_checkpoint,
    retrained: dict[str, str],
    offset: int = 200,
    limit: int = 60,
    output_dir="outputs/cavc_eval",
    include_comet: bool = False,
    predictor_name: str = "small",
    translator_name: str = "small",
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    seed: int = 13,
    include_fixed: bool = True,
    include_baseline: bool = True,
) -> dict:
    """Roll out the requested policies on unseen dev sentences.

    ``include_fixed=False`` skips the fixed `(4,6)` rollout (and its paired
    metrics), for a retrained-only kernel that will be paired offline against a
    previously saved `fixed.traces.jsonl`. ``include_baseline=False`` skips the
    `--old-checkpoint` policy. Defaults preserve the full A/B/C/D behavior.
    """
    pfba = kaggle._require_run_pfba()
    predictor, translator, _stream, pdev, tdev = kaggle.resolve_adapters(
        predictor_name,
        translator_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    timer = kaggle._stage_timer(pfba, predictor, translator, None)
    predictor_id = int(getattr(predictor, "model_pair_id", 0))
    translator_id = int(getattr(translator, "model_pair_id", 0))

    sentences = select_sentences(
        pfba.load_split("dev", 0), offset=offset, limit=limit
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    old_cfg, old_builder, _old_network, old_cost = _load_policy(
        old_checkpoint, predictor_id, translator_id
    )

    if include_fixed:
        fixed_traces = [
            run_fixed_rollout(
                sentence,
                predictor,
                translator,
                old_builder,
                old_cost,
                old_cfg,
                predictor_id=predictor_id,
                translator_id=translator_id,
                timer=timer,
            )
            for sentence in sentences
        ]
        fixed_summary = summarize_traces(
            fixed_traces, include_comet=include_comet
        )
        fixed_summary["read_runs"] = read_run_stats(fixed_traces)
        fixed_summary["service_time"] = service_time_stats(fixed_traces)
        fixed_summary["chrf"] = chrf_stats(fixed_traces)
        kaggle._write_traces(output_dir / "fixed.traces.jsonl", fixed_traces)
    else:
        fixed_traces = []
        fixed_summary = {
            "status": "skipped",
            "reason": "include_fixed=False",
        }

    variants: dict[str, dict] = {}
    # The supplied checkpoint is the **corrected** V3 baseline (A/B), not the
    # historical pre-serial-cost checkpoint. The report keeps those separate.
    policies = _policy_checkpoints(
        old_checkpoint, retrained, include_baseline=include_baseline
    )
    for policy_name, checkpoint_path in policies.items():
        cfg, builder, network, cost_model = _load_policy(
            checkpoint_path, predictor_id, translator_id
        )
        for guarded in (False, True):
            name = f"{policy_name}{'_guard' if guarded else ''}"
            variant_cfg = guard_config(cfg) if guarded else cfg
            traces = [
                run_rollout(
                    sentence,
                    predictor,
                    translator,
                    network,
                    builder,
                    cost_model,
                    variant_cfg,
                    timer=timer,
                    predictor_id=predictor_id,
                    translator_id=translator_id,
                )
                for sentence in sentences
            ]
            summary = summarize_traces(traces, include_comet=include_comet)
            paired = (
                kaggle._paired_metrics(
                    traces, fixed_traces, include_comet=include_comet
                )
                if include_fixed
                else {
                    "status": "skipped",
                    "reason": "include_fixed=False",
                }
            )
            kaggle._write_traces(output_dir / f"{name}.traces.jsonl", traces)
            variants[name] = {
                "checkpoint": str(checkpoint_path),
                "guarded": bool(guarded),
                "summary": summary,
                "read_runs": read_run_stats(traces),
                "service_time": service_time_stats(traces),
                "chrf": chrf_stats(traces),
                "paired_vs_fixed": paired,
            }
            print(
                f"[cavc-eval] {name}: "
                f"BLEU={summary.get('bleu', {}).get('score')} "
                f"LAAL={summary.get('laal')} "
                f"completion={summary.get('completion_s', {}).get('mean')} "
                f"READ={summary.get('read_fraction')}",
                flush=True,
            )

    summary = {
        "mode": "evaluate",
        "offset": int(offset),
        "limit": int(limit),
        "n_sentences": len(sentences),
        "model_pair": {
            "predictor": predictor_name,
            "translator": translator_name,
            "predictor_id": predictor_id,
            "translator_id": translator_id,
        },
        "cost_topology": "serial",
        "include_fixed": bool(include_fixed),
        "include_baseline": bool(include_baseline),
        "baseline_checkpoint": str(old_checkpoint),
        "guard": kaggle._config_payload(guard_config(old_cfg), {})["config"],
        "fixed": fixed_summary,
        "variants": variants,
        "notes": [
            "Unseen dev:0200+ sentences; no labels or tuning in this run.",
            "Every variant uses the same adapter pair, runtime, sampling seeds, "
            "RALCP gate, and timing path as the fixed baseline.",
            "A/B use the corrected V3 baseline checkpoint (serial cost, "
            "decoupled History); the historical pre-correction checkpoint is "
            "reported separately and is not this baseline.",
            "C/D use the latency-focused retrained checkpoint; the balanced "
            "checkpoint is the lower-cost E/ablation reference.",
            "Guarded variants use the medium V3.1 READ threshold, expensive-"
            "action margin, and a soft no-emit penalty (no hard veto).",
            "Sixty sentences is a feasibility check, not a final quality claim; "
            "per-sentence traces and paired intervals are retained.",
        ],
    }
    kaggle._write_json(output_dir / "summary.json", summary)
    return summary
