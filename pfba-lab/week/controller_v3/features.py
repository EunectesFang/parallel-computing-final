"""Online state features for the controller.

Strict leakage boundary: `FeatureBuilder.build` sees only the observed source
prefix (`observed`), the committed target prefix (`committed`), history
statistics already produced, runtime/cost information, and optional
confidence values. It never sees the full source, source length, reference,
oracle continuation, or future words. Training labels are built in
`labeler.py`, which is a separate path.

`NUMERIC_FEATURES` is the authoritative order of the 34 numeric values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "HISTORY_FEATURES",
    "SURFACE_FEATURES",
    "CONFIDENCE_FEATURES",
    "MASK_FEATURES",
    "NUMERIC_FEATURES",
    "History",
    "RuntimeInfo",
    "OptionalConfidence",
    "FeatureVector",
    "StateInput",
    "FeatureBuilder",
    "estimate_length_ratio",
]

HISTORY_FEATURES = (
    "last_B",
    "last_H",
    "last_vote_top1",
    "last_vote_margin",
    "last_vote_entropy",
    "last_commit_count",
    "last_empty",
)

SURFACE_FEATURES = (
    "src_words",
    "src_chars",
    "tgt_words",
    "tgt_chars",
    "length_ratio",
    "normalized_lag",
    "words_since_commit",
    "no_emit_streak",
    "last_B",
    "last_H",
    "last_vote_top1",
    "last_vote_margin",
    "last_vote_entropy",
    "last_commit_count",
    "last_empty",
    "src_end_punct",
    "tgt_end_punct",
    "last_capital",
    "subordinate_marker",
    "current_clause_len",
    "pred_cost_ref",
    "mt_cost_ref",
    "pred_queue_work",
    "mt_queue_work",
    "time_since_commit",
    "arrival_slack",
)

CONFIDENCE_FEATURES = (
    "mt_next_max_prob",
    "mt_next_entropy",
    "mt_prefix_logprob",
    "pred_next_entropy",
)

MASK_FEATURES = tuple(f"{name}_available" for name in CONFIDENCE_FEATURES)

NUMERIC_FEATURES = SURFACE_FEATURES + CONFIDENCE_FEATURES + MASK_FEATURES  # 34

# Window used when a vocabulary is supplied. It matches
# `ControllerConfig.text_window_src` / `text_window_tgt`.
TEXT_WINDOW = 16

# Characters treated as sentence punctuation. Mirrors the set used by the
# earlier `week_run.py` feature code, with a few continental extras.
PUNCTUATION = frozenset(".,;:!?\"'()[]{}—–…»«") | frozenset("，。！？；：、")

# Punctuation that closes a clause for `current_clause_len` and
# `subordinate_marker`; brackets and quotes are deliberately excluded.
CLAUSE_PUNCTUATION = frozenset(".,;:!?…") | frozenset("。！？；：、")

# German subordinating conjunctions that introduce a dependent clause.
# Articles (der/die/das/den/dem) are deliberately excluded: they are not
# clause markers.
SUBORDINATE_MARKERS = frozenset(
    {
        "dass",
        "weil",
        "ob",
        "wenn",
        "obwohl",
        "während",
        "nachdem",
        "bevor",
        "sobald",
        "falls",
        "da",
        "damit",
        "sodass",
    }
)

_STRIP_CHARS = "".join(PUNCTUATION)


@dataclass(frozen=True)
class History:
    """Statistics produced by earlier steps of the same sentence.

    `last_branches` / `last_horizon` fill the `last_B` / `last_H` feature slots.
    """

    last_branches: int = 0
    last_horizon: int = 0
    last_vote_top1: float = 0.0
    last_vote_margin: float = 0.0
    last_vote_entropy: float = 0.0
    last_commit_count: int = 0
    last_empty: bool = False
    words_since_commit: int = 0
    no_emit_streak: int = 0


@dataclass(frozen=True)
class RuntimeInfo:
    """Runtime and cost information for the current decision point."""

    pred_cost_ref: float = 0.0
    mt_cost_ref: float = 0.0
    pred_queue_work: float = 0.0
    mt_queue_work: float = 0.0
    time_since_commit: float = 0.0
    arrival_slack: float = 0.0


@dataclass(frozen=True)
class OptionalConfidence:
    """Confidence signals; `available` masks which entries are trustworthy."""

    values: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    available: tuple[bool, bool, bool, bool] = (False, False, False, False)


@dataclass(frozen=True)
class FeatureVector:
    """One state: 34 numeric values plus model ids and optional text ids."""

    numeric: tuple[float, ...]
    predictor_id: int
    translator_id: int
    src_ids: tuple[int, ...]
    tgt_ids: tuple[int, ...]


@dataclass(frozen=True)
class StateInput:
    """Serializable subset of a state used by the trainer and checkpoints."""

    numeric: tuple[float, ...]
    predictor_id: int
    translator_id: int
    src_ids: tuple[int, ...] = ()
    tgt_ids: tuple[int, ...] = ()


def _source_words(observed: str) -> list[str]:
    return observed.split()


def _current_clause_words(source_words: Sequence[str]) -> list[str]:
    """Words after the most recent clause-closing token.

    A token that itself ends with clause punctuation closes its own clause, so
    the returned list is empty when the prefix ends on a clause boundary.
    """
    clause: list[str] = []
    for word in source_words:
        clause.append(word)
        if word[-1:] in CLAUSE_PUNCTUATION:
            clause = []
    return clause


def _normalize_marker_word(word: str) -> str:
    return word.lower().strip(_STRIP_CHARS)


class FeatureBuilder:
    """Builds the 34-value online state vector.

    `length_ratio` is the training-estimated source/target ratio `rho`. It is
    used only for `normalized_lag`; the `length_ratio` feature is always the
    current committed-vs-observed ratio.
    """

    def __init__(self, length_ratio: float = 1.0, src_vocab=None, tgt_vocab=None):
        rho = float(length_ratio)
        if not math.isfinite(rho) or rho <= 0.0:
            rho = 1.0
        self.rho = rho
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def build(
        self,
        observed: str,
        committed: Sequence[str],
        *,
        history: History = History(),
        runtime: RuntimeInfo = RuntimeInfo(),
        confidence: OptionalConfidence = OptionalConfidence(),
        predictor_id: int = 0,
        translator_id: int = 0,
    ) -> FeatureVector:
        if len(confidence.values) != 4 or len(confidence.available) != 4:
            raise ValueError("OptionalConfidence values and available must have 4 entries")

        source_words = _source_words(observed)
        committed_words = tuple(committed)
        src_words = len(source_words)
        tgt_words = len(committed_words)

        clause_words = _current_clause_words(source_words)
        marker_hit = any(
            _normalize_marker_word(word) in SUBORDINATE_MARKERS
            for word in clause_words
        )
        last_source_word = source_words[-1] if source_words else ""
        target_text = " ".join(committed_words)

        values: dict[str, float] = {
            "src_words": float(src_words),
            "src_chars": float(len(observed)),
            "tgt_words": float(tgt_words),
            "tgt_chars": float(len(target_text)),
            "length_ratio": tgt_words / max(1, src_words),
            "normalized_lag": max(0.0, src_words - tgt_words / self.rho),
            "words_since_commit": float(history.words_since_commit),
            "no_emit_streak": float(history.no_emit_streak),
            "last_B": float(history.last_branches),
            "last_H": float(history.last_horizon),
            "last_vote_top1": float(history.last_vote_top1),
            "last_vote_margin": float(history.last_vote_margin),
            "last_vote_entropy": float(history.last_vote_entropy),
            "last_commit_count": float(history.last_commit_count),
            "last_empty": float(history.last_empty),
            "src_end_punct": float(bool(observed) and observed[-1] in PUNCTUATION),
            "tgt_end_punct": float(bool(target_text) and target_text[-1] in PUNCTUATION),
            "last_capital": float(bool(last_source_word[:1].isupper())),
            "subordinate_marker": float(marker_hit),
            "current_clause_len": float(len(clause_words)),
            "pred_cost_ref": float(runtime.pred_cost_ref),
            "mt_cost_ref": float(runtime.mt_cost_ref),
            "pred_queue_work": float(runtime.pred_queue_work),
            "mt_queue_work": float(runtime.mt_queue_work),
            "time_since_commit": float(runtime.time_since_commit),
            "arrival_slack": float(runtime.arrival_slack),
        }

        for name, value, available in zip(
            CONFIDENCE_FEATURES, confidence.values, confidence.available
        ):
            values[name] = float(value) if available else 0.0
            values[f"{name}_available"] = 1.0 if available else 0.0

        numeric = tuple(float(values[name]) for name in NUMERIC_FEATURES)

        src_ids: tuple[int, ...] = ()
        if self.src_vocab is not None:
            src_ids = tuple(self.src_vocab.encode(source_words, TEXT_WINDOW))
        tgt_ids: tuple[int, ...] = ()
        if self.tgt_vocab is not None:
            tgt_ids = tuple(self.tgt_vocab.encode(committed_words, TEXT_WINDOW))

        return FeatureVector(
            numeric=numeric,
            predictor_id=int(predictor_id),
            translator_id=int(translator_id),
            src_ids=src_ids,
            tgt_ids=tgt_ids,
        )


def estimate_length_ratio(samples: Sequence[tuple[str, str]]) -> float:
    """Mean target/source word ratio over pairs where both sides are non-empty.

    Returns 1.0 when no usable pair exists and clamps the result to
    `[0.5, 2.0]` so a few extreme sentences cannot distort the lag feature.
    """
    ratios: list[float] = []
    for source, target in samples:
        src_words = len(source.split())
        tgt_words = len(target.split())
        if src_words == 0 or tgt_words == 0:
            continue
        ratios.append(tgt_words / src_words)
    if not ratios:
        return 1.0
    mean = sum(ratios) / len(ratios)
    return min(2.0, max(0.5, mean))
