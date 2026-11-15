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
    "risk_calibration",
    "DEFAULT_COMET_MODEL",
]

DEFAULT_COMET_MODEL = "Unbabel/wmt22-comet-da"

# Metrics `paired_bootstrap` knows how to pair.
_NUMERIC_METRICS = ("completion_s", "read_fraction")


def _sigmoid(value: float) -> float:
    """Numerically safe logistic for the network's risk logits."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _bucket_calibration(
    probabilities: Sequence[float],
    labels: Sequence[float],
    buckets: int,
) -> tuple[float | None, list[dict]]:
    """Brier-free reliability table: ECE-style bucket error and its buckets.

    Each bucket reports ``lo``/``hi``, ``count``, ``mean_predicted``,
    ``mean_label``, and ``abs_gap``. The returned scalar is
    ``sum(count_b / n * abs(mean_predicted_b - mean_label_b))``, i.e. the
    expected calibration error of the reliability diagram. It is diagnostic
    only: the underlying sigmoid is not calibrated and must never be used as a
    hard safety threshold.
    """
    usable = [
        (float(probability), float(label))
        for probability, label in zip(probabilities, labels)
        if math.isfinite(float(probability)) and math.isfinite(float(label))
    ]
    if not usable:
        return None, []
    count = len(usable)
    rows: list[dict] = []
    error = 0.0
    for index in range(int(buckets)):
        low = index / buckets
        high = (index + 1) / buckets
        members = [
            (probability, label)
            for probability, label in usable
            if (low <= probability < high)
            or (index == buckets - 1 and probability == 1.0)
        ]
        if not members:
            rows.append(
                {
                    "lo": float(low),
                    "hi": float(high),
                    "count": 0,
                    "mean_predicted": None,
                    "mean_label": None,
                    "abs_gap": None,
                }
            )
            continue
        mean_predicted = sum(item[0] for item in members) / len(members)
        mean_label = sum(item[1] for item in members) / len(members)
        gap = abs(mean_predicted - mean_label)
        error += gap * len(members) / count
        rows.append(
            {
                "lo": float(low),
                "hi": float(high),
                "count": len(members),
                "mean_predicted": float(mean_predicted),
                "mean_label": float(mean_label),
                "abs_gap": float(gap),
            }
        )
    return float(error), rows


def _brier(probabilities: Sequence[float], labels: Sequence[float]) -> float | None:
    pairs = [
        (float(probability), float(label))
        for probability, label in zip(probabilities, labels)
        if math.isfinite(float(probability)) and math.isfinite(float(label))
    ]
    if not pairs:
        return None
    return sum((probability - label) ** 2 for probability, label in pairs) / len(pairs)


def risk_calibration(
    network,
    targets: Sequence,
    *,
    buckets: int = 10,
) -> dict:
    """Calibrate the network's `error_risk` / `no_emit` heads on targets.

    `targets` must be held-out `ActionTarget` rows (the caller is responsible
    for not calibrating on rows that trained the network). READ rows are
    skipped because READ has no write head. Every write target contributes one
    predicted probability from the network's logistic head and one 0/1 label
    (`error_target` / `no_emit_target`).

    Returns Brier scores, ECE-style reliability-bucket errors, the reliability
    tables, and base rates for both heads. An empty write set reports
    ``status="unavailable"`` with no invented score. The probabilities come
    straight from the network's sigmoid: they are recorded for diagnosis and
    must never be used as an uncalibrated hard safety threshold.
    """
    import torch

    if int(buckets) <= 0:
        raise ValueError(
            f"buckets must be positive; got {buckets!r}"
        )

    rows = [
        target
        for target in targets
        if not getattr(target.action, "is_read", False)
    ]
    if not rows:
        return {
            "status": "unavailable",
            "reason": "no held-out write targets",
            "n": 0,
            "error_risk": {},
            "no_emit": {},
        }

    states = [target.state for target in rows]
    numeric = torch.tensor(
        [list(state.numeric) for state in states], dtype=torch.float32
    )
    predictor_ids = torch.tensor(
        [int(state.predictor_id) for state in states], dtype=torch.long
    )
    translator_ids = torch.tensor(
        [int(state.translator_id) for state in states], dtype=torch.long
    )
    def _padded(sequences) -> torch.Tensor:
        """Right-pad variable-length id tuples so they fit one batch tensor."""
        width = max((len(sequence) for sequence in sequences), default=0)
        if width == 0:
            return torch.zeros((len(sequences), 0), dtype=torch.long)
        return torch.tensor(
            [
                list(sequence) + [0] * (width - len(sequence))
                for sequence in sequences
            ],
            dtype=torch.long,
        )

    src_ids = _padded([state.src_ids for state in states])
    tgt_ids = _padded([state.tgt_ids for state in states])

    actions = sorted({target.action for target in rows})
    action_index = {action: index for index, action in enumerate(actions)}

    can_eval = callable(getattr(network, "eval", None))
    was_training = getattr(network, "training", None)
    if can_eval:
        network.eval()
    try:
        with torch.no_grad():
            z = network.encode(
                numeric, predictor_ids, translator_ids, src_ids, tgt_ids
            )
            outputs = network.write_values(z, actions)
            error_logits = outputs.error_risk
            no_emit_logits = outputs.no_emit
            error_probabilities: list[float] = []
            no_emit_probabilities: list[float] = []
            error_labels: list[float] = []
            no_emit_labels: list[float] = []
            for index, target in enumerate(rows):
                column = action_index[target.action]
                error_probabilities.append(
                    _sigmoid(float(error_logits[index][column]))
                )
                no_emit_probabilities.append(
                    _sigmoid(float(no_emit_logits[index][column]))
                )
                error_labels.append(float(target.error_target))
                no_emit_labels.append(float(target.no_emit_target))
    finally:
        if (
            can_eval
            and was_training is not None
            and callable(getattr(network, "train", None))
        ):
            network.train(bool(was_training))

    error_ece, error_buckets = _bucket_calibration(
        error_probabilities, error_labels, buckets
    )
    no_emit_ece, no_emit_buckets = _bucket_calibration(
        no_emit_probabilities, no_emit_labels, buckets
    )

    def _head(probabilities, labels, ece, bucket_rows) -> dict:
        return {
            "brier": _brier(probabilities, labels),
            "ece": ece,
            "buckets": bucket_rows,
            "base_rate": (
                sum(labels) / len(labels) if labels else None
            ),
            "mean_predicted": (
                sum(probabilities) / len(probabilities)
                if probabilities
                else None
            ),
        }

    return {
        "status": "ok",
        "n": len(rows),
        "buckets": int(buckets),
        "error_risk": _head(
            error_probabilities, error_labels, error_ece, error_buckets
        ),
        "no_emit": _head(
            no_emit_probabilities,
            no_emit_labels,
            no_emit_ece,
            no_emit_buckets,
        ),
        "threshold_note": (
            "Predicted probabilities come from an uncalibrated sigmoid. Use "
            "them as diagnostics or soft inputs only, never as a hard safety "
            "threshold."
        ),
    }


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
    include_bleu: bool = True,
    comet_model: str = DEFAULT_COMET_MODEL,
    comet_device: str = "cpu",
) -> dict:
    """Aggregate a list of `RolloutTrace` objects into a report-ready dict.

    Returns ``n``, status-aware ``bleu`` / ``comet`` dicts, mean LAAL over
    non-``None`` traces, mean and P95 virtual completion, the aggregate read
    fraction and action histogram, and a flattened per-step record list. COMET
    runs only when `include_comet` is true, because it downloads a model.
    `include_bleu=False` skips the BLEU call entirely; callers that know their
    hypotheses are synthetic use it so a locally installed `sacrebleu` cannot
    produce a real-looking score for meaningless data.
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

    if include_bleu:
        bleu = corpus_bleu(hypotheses, references)
    else:
        bleu = {"status": "unavailable", "reason": "BLEU disabled by caller"}

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
        "bleu": bleu,
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
    *,
    comet_model: str = DEFAULT_COMET_MODEL,
    comet_device: str = "cpu",
) -> dict:
    """Sentence-paired bootstrap of ``metric(a) - metric(b)``.

    Supported metrics are ``"laal"`` (``None`` values are skipped),
    ``"bleu"`` (per-sentence `corpus_bleu`; raises `MetricUnavailableError` when
    BLEU is unavailable), ``"comet"`` (per-sentence COMET through the delegated
    `comet_score`; raises when COMET is unavailable or still downloading),
    ``"completion_s"``, and ``"read_fraction"``.

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
    elif metric == "comet":
        arm_a = [(a.hypothesis, a.source, a.reference) for a, _b in trace_pairs]
        arm_b = [(b.hypothesis, b.source, b.reference) for _a, b in trace_pairs]
        result_a = comet_score(
            [row[0] for row in arm_a],
            [row[1] for row in arm_a],
            [row[2] for row in arm_a],
            comet_model,
            comet_device,
        )
        result_b = comet_score(
            [row[0] for row in arm_b],
            [row[1] for row in arm_b],
            [row[2] for row in arm_b],
            comet_model,
            comet_device,
        )
        for label, result in (("a", result_a), ("b", result_b)):
            if result.get("status") != "ok":
                raise MetricUnavailableError(
                    f"COMET is unavailable for arm {label}: "
                    f"{result.get('reason', result.get('status', 'unknown'))}"
                )
        differences = [
            float(score_a) - float(score_b)
            for score_a, score_b in zip(
                result_a.get("sentence_scores", []),
                result_b.get("sentence_scores", []),
            )
        ]
    elif metric in _NUMERIC_METRICS:
        differences = [
            float(getattr(a, metric)) - float(getattr(b, metric))
            for a, b in trace_pairs
        ]
    else:
        raise ValueError(
            f"paired_bootstrap does not know metric {metric!r}; use 'laal', "
            "'bleu', 'comet', 'completion_s', or 'read_fraction'"
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
