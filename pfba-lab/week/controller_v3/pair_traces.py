"""Pair a retrained-only eval trace file against a saved fixed baseline.

V5 retrained-only kernels deliberately skip the fixed ``(4,6)`` rollout. Pair
their traces with the ``fixed.traces.jsonl`` produced earlier on the exact same
sentences and runtime, and compute the paired bootstrap offline.

Usage::

    python -m controller_v3.pair_traces \
        --candidate v5.traces.jsonl \
        --fixed fixed.traces.jsonl \
        --output-dir out
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from .evaluator import MetricUnavailableError, paired_bootstrap

__all__ = ["load_traces", "pair_trace_files", "main"]


def load_traces(path) -> list[SimpleNamespace]:
    """Load one JSON trace per line into attribute-accessible records."""
    rows: list[SimpleNamespace] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(SimpleNamespace(**json.loads(line)))
    return rows


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _paired_values(
    candidate: Sequence[SimpleNamespace],
    fixed: Sequence[SimpleNamespace],
    metric: str,
) -> list[float]:
    fixed_by_id = {row.sentence_id: row for row in fixed}
    values: list[float] = []
    for row in candidate:
        other = fixed_by_id.get(row.sentence_id)
        if other is None:
            continue
        left = getattr(row, metric, None)
        right = getattr(other, metric, None)
        if left is None or right is None:
            continue
        values.append(float(left) - float(right))
    return values


def _bootstrap(values: Sequence[float], *, samples: int, seed: int) -> dict:
    if not values:
        return {"point": None, "low": None, "high": None, "n": 0}
    n = len(values)
    rng = random.Random(int(seed))
    means = [
        sum(rng.choice(values) for _ in range(n)) / n
        for _ in range(max(1, int(samples)))
    ]
    return {
        "point": sum(values) / n,
        "low": _percentile(means, 0.025),
        "high": _percentile(means, 0.975),
        "n": n,
    }


def _sentence_chrf(row: SimpleNamespace, cache: dict) -> float:
    from sacrebleu.metrics import CHRF

    key = (row.hypothesis, row.reference)
    if key not in cache:
        metric = CHRF()
        try:
            score = metric.sentence_score(row.hypothesis, [row.reference]).score
        except AttributeError:  # pragma: no cover - older sacrebleu
            score = metric.corpus_score(
                [row.hypothesis], [[row.reference]]
            ).score
        cache[key] = float(score)
    return cache[key]


def _chrf_paired(
    candidate: Sequence[SimpleNamespace],
    fixed: Sequence[SimpleNamespace],
    *,
    samples: int,
    seed: int,
) -> dict:
    try:
        fixed_by_id = {row.sentence_id: row for row in fixed}
        cache: dict = {}
        differences = [
            _sentence_chrf(row, cache) - _sentence_chrf(fixed_by_id[row.sentence_id], cache)
            for row in candidate
            if row.sentence_id in fixed_by_id
        ]
    except Exception as exc:  # sacrebleu missing or scorer error
        return {
            "metric": "chrf",
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "point": None,
            "low": None,
            "high": None,
            "n": 0,
        }
    return {"metric": "chrf", "status": "ok", **_bootstrap(differences, samples=samples, seed=seed)}


def pair_trace_files(
    candidate_path,
    fixed_path,
    *,
    samples: int = 2000,
    seed: int = 13,
) -> dict:
    """Pair ``candidate - fixed`` per sentence and bootstrap every metric."""
    candidate = load_traces(candidate_path)
    fixed = load_traces(fixed_path)
    metrics: dict[str, dict] = {}

    for metric in ("bleu", "laal", "completion_s"):
        try:
            result = paired_bootstrap(
                candidate, fixed, metric, samples=samples, seed=seed
            )
            metrics[metric] = {"metric": metric, "status": "ok", **result}
        except MetricUnavailableError as exc:
            metrics[metric] = {
                "metric": metric,
                "status": "unavailable",
                "reason": str(exc),
                "point": None,
                "low": None,
                "high": None,
                "n": 0,
            }

    metrics["chrf"] = _chrf_paired(
        candidate, fixed, samples=samples, seed=seed
    )
    paired_ids = {
        row.sentence_id for row in candidate
    } & {row.sentence_id for row in fixed}
    return {
        "candidate": str(candidate_path),
        "fixed": str(fixed_path),
        "n_pairs": len(paired_ids),
        "metrics": metrics,
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# Paired traces: candidate - fixed",
        "",
        f"- candidate: `{payload['candidate']}`",
        f"- fixed: `{payload['fixed']}`",
        f"- paired sentences: {payload['n_pairs']}",
        "",
        "| metric | point | 95% CI | n | status |",
        "|---|---:|---|---:|---|",
    ]
    for name, result in payload["metrics"].items():
        point = result.get("point")
        low = result.get("low")
        high = result.get("high")
        if point is None:
            cell = "—"
        else:
            cell = f"{point:+.4f}"
        interval = (
            "—"
            if low is None or high is None
            else f"[{low:+.4f}, {high:+.4f}]"
        )
        lines.append(
            f"| {name} | {cell} | {interval} | {result.get('n', 0)} | "
            f"{result.get('status', 'ok')} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--output-dir", default="outputs/cavc_pair")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    payload = pair_trace_files(
        args.candidate,
        args.fixed,
        samples=args.samples,
        seed=args.seed,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (output / "paired.md").write_text(_markdown(payload), encoding="utf-8")
    print(f"wrote {output / 'paired.json'}")
    print(f"wrote {output / 'paired.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
