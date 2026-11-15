"""Trace summaries and sentence-paired bootstrap tests for rollouts.

BLEU and COMET are delegated to `run_pfba` so the controller report uses the
same implementations and the same status shape as the baselines. Neither metric
is ever invented locally: when `sacrebleu` or `comet` is missing the delegated
result carries ``status`` ``"unavailable"`` / ``"error"`` and no score.

COMET is opt-in in `summarize_traces` because loading the model downloads
weights; BLEU is cheap and always attempted.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Sequence

from .runtime import run_pfba

__all__ = [
    "MetricUnavailableError",
    "corpus_bleu",
    "comet_score",
    "summarize_traces",
    "paired_bootstrap",
    "DEFAULT_COMET_MODEL",
]

DEFAULT_COMET_MODEL = "Unbabel/wmt22-comet-da"

# Metrics `paired_bootstrap` knows how to pair.
_NUMERIC_METRICS = ("completion_s", "read_fraction")


class MetricUnavailableError(RuntimeError):
    """Raised when a paired bootstrap needs a metric that cannot be computed."""


def _paired_path() -> str:
    """Directory of `run_pfba.py`, kept for callers that re-import it."""
    return str(Path(__file__).resolve().parents[2] / "paired")


def corpus_bleu(hypotheses, references) -> dict:
    """Delegate to `run_pfba.corpus_bleu`; never fabricate a score."""
    if run_pfba is None:
        return {
            "status": "unavailable",
            "reason": f"run_pfba module not found under {_paired_path()}",
        }
    try:
        return run_pfba.corpus_bleu(list(hypotheses), list(references))
    except Exception as exc:  # pragma: no cover - defensive around optional deps
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def comet_score(
    hypotheses,
    sources,
    references,
    model_name: str = DEFAULT_COMET_MODEL,
    device: str = "cpu",
) -> dict:
    """Delegate to `run_pfba.comet_score`, including its download/cache path."""
    if run_pfba is None:
        return {
            "status": "unavailable",
            "reason": f"run_pfba module not found under {_paired_path()}",
        }
    try:
        return run_pfba.comet_score(
            list(hypotheses), list(sources), list(references), model_name, device
        )
    except Exception as exc:  # pragma: no cover - defensive around optional deps
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    """Linear-interpolated percentile; ``None`` for an empty sequence."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(quantile))) * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _mean(values: Sequence[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize_traces(
    traces: Sequence,
    *,
    include_comet: bool = False,
    comet_model: str = DEFAULT_COMET_MODEL,
    comet_device: str = "cpu",
) -> dict:
    """Aggregate a list of `RolloutTrace` objects into a report-ready dict.

    Returns ``n``, status-aware ``bleu`` / ``comet`` dicts, mean LAAL over
    non-``None`` traces, mean and P95 virtual completion, the aggregate read
    fraction and action histogram, and a flattened per-step record list. COMET
    runs only when `include_comet` is true, because it downloads a model.
    """
    traces = list(traces)
    n = len(traces)
    hypotheses = [trace.hypothesis for trace in traces]
    sources = [trace.source for trace in traces]
    references = [trace.reference for trace in traces]

    laal_values = [float(trace.laal) for trace in traces if trace.laal is not None]
    completion_values = [float(trace.completion_s) for trace in traces]

    histogram: dict[str, int] = {}
    for trace in traces:
        for action, count in dict(trace.action_histogram).items():
            histogram[str(action)] = histogram.get(str(action), 0) + int(count)
    decisions = sum(histogram.values())
    reads = histogram.get("read", 0)
    read_fraction = reads / decisions if decisions else 0.0

    steps: list[dict] = []
    for trace in traces:
        for step in trace.steps:
            record = dict(step)
            record["sentence_id"] = trace.sentence_id
            steps.append(record)

    if include_comet:
        comet = comet_score(
            hypotheses, sources, references, comet_model, comet_device
        )
    else:
        comet = {
            "status": "skipped",
            "reason": "COMET is opt-in because loading the model downloads weights",
        }

    return {
        "n": n,
        "bleu": corpus_bleu(hypotheses, references),
        "comet": comet,
        "laal": _mean(laal_values),
        "completion_s": {
            "mean": _mean(completion_values),
            "p95": _percentile(completion_values, 0.95),
        },
        "read_fraction": float(read_fraction),
        "action_histogram": histogram,
        "steps": steps,
    }


def _first_by_sentence(traces: Sequence) -> dict[str, object]:
    """Keep the first occurrence of each `sentence_id` in list order."""
    first: dict[str, object] = {}
    for trace in traces:
        first.setdefault(trace.sentence_id, trace)
    return first


def _pair_by_sentence(traces_a: Sequence, traces_b: Sequence):
    """Pair traces by `sentence_id`, keeping the first occurrence on each side.

    Pairs follow `traces_a` order and only include ids present in both lists.
    Later duplicates of the same id in either arm are ignored, so both arms use
    the same occurrence policy.
    """
    by_id = _first_by_sentence(traces_b)
    pairs = []
    for trace in _first_by_sentence(traces_a).values():
        sentence_id = trace.sentence_id
        if sentence_id in by_id:
            pairs.append((trace, by_id[sentence_id]))
    return pairs


def _sentence_bleu(trace, cache: dict[tuple[str, str, str], float]) -> float:
    # Key on the text too: the same sentence id appears once in each arm and
    # the two hypotheses normally differ.
    key = (trace.sentence_id, trace.hypothesis, trace.reference)
    if key in cache:
        return cache[key]
    result = corpus_bleu([trace.hypothesis], [trace.reference])
    if result.get("status") != "ok":
        raise MetricUnavailableError(
            "per-sentence BLEU is unavailable: "
            f"{result.get('reason', result.get('status', 'unknown reason'))}"
        )
    score = float(result["score"])
    cache[key] = score
    return score


def paired_bootstrap(
    traces_a: Sequence,
    traces_b: Sequence,
    metric: str,
    samples: int = 2000,
    seed: int = 13,
) -> dict:
    """Sentence-paired bootstrap of ``metric(a) - metric(b)``.

    Supported metrics are ``"laal"`` (``None`` values are skipped),
    ``"bleu"`` (per-sentence `corpus_bleu`; raises `MetricUnavailableError` when
    BLEU is unavailable), ``"completion_s"``, and ``"read_fraction"``.

    Returns the point estimate (mean paired difference), the 2.5/97.5 percentile
    bootstrap interval, and the number of paired sentences actually used. When
    no sentence pair survives, the point and interval are ``None``.
    """
    trace_pairs = _pair_by_sentence(traces_a, traces_b)

    if metric == "laal":
        differences = [
            float(a.laal) - float(b.laal)
            for a, b in trace_pairs
            if a.laal is not None and b.laal is not None
        ]
    elif metric == "bleu":
        cache: dict[tuple[str, str, str], float] = {}
        differences = [
            _sentence_bleu(a, cache) - _sentence_bleu(b, cache)
            for a, b in trace_pairs
        ]
    elif metric in _NUMERIC_METRICS:
        differences = [
            float(getattr(a, metric)) - float(getattr(b, metric))
            for a, b in trace_pairs
        ]
    else:
        raise ValueError(
            f"paired_bootstrap does not know metric {metric!r}; use 'laal', "
            "'bleu', 'completion_s', or 'read_fraction'"
        )

    n = len(differences)
    if n == 0:
        return {"metric": metric, "point": None, "low": None, "high": None, "n": 0}

    point = sum(differences) / n
    rng = random.Random(int(seed))
    resample_count = max(1, int(samples))
    means = [
        sum(rng.choice(differences) for _ in range(n)) / n
        for _ in range(resample_count)
    ]

    return {
        "metric": metric,
        "point": float(point),
        "low": _percentile(means, 0.025),
        "high": _percentile(means, 0.975),
        "n": n,
    }
