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
from dataclasses import asdict, dataclass, replace
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
    "HistoryTracker",
    "advance_virtual_clock",
    "estimate_length_ratio",
    "feature_diagnostics",
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


class HistoryTracker:
    """The single state-update implementation shared by labeling and rollout.

    One tracker holds the `History` and `RuntimeInfo` for the *current* decision
    point. Call `on_read` or `on_write` once per decision to advance it to the
    successor, then `build` the online `FeatureVector` for that successor. Using
    the same class on the offline label path and the online rollout path is what
    makes the History features identical in training and serving; previously
    training used a default all-zero `History` while serving used live values.

    `last_commit_s` and `now_s` are the virtual clock used for the
    `time_since_commit` runtime feature. `now_s` is the clock at the decision
    (or at the just-completed action when `on_*` is called); a committed write
    resets `last_commit_s` to `now_s`.

    READ leaves the vote statistics untouched because no vote happened; it
    zeroes the branch/horizon slots as the brief requires and increments both
    the words-since-commit and no-emit streak counters.
    """

    def __init__(
        self,
        history: History | None = None,
        runtime: RuntimeInfo | None = None,
        last_commit_s: float = 0.0,
        now_s: float = 0.0,
    ) -> None:
        self.history = history if history is not None else History()
        self.runtime = runtime if runtime is not None else RuntimeInfo()
        self.last_commit_s = float(last_commit_s)
        self.now_s = float(now_s)

    def clone(self) -> "HistoryTracker":
        """Return an independent copy so one transition cannot affect another."""
        return HistoryTracker(
            history=self.history,
            runtime=self.runtime,
            last_commit_s=self.last_commit_s,
            now_s=self.now_s,
        )

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------
    def on_read(self, now_s: float | None = None) -> History:
        """Advance over a READ: one more observed source word, no commit."""
        if now_s is not None:
            self.now_s = float(now_s)
        previous = self.history
        self.history = History(
            last_branches=0,
            last_horizon=0,
            last_vote_top1=previous.last_vote_top1,
            last_vote_margin=previous.last_vote_margin,
            last_vote_entropy=previous.last_vote_entropy,
            last_commit_count=0,
            last_empty=True,
            words_since_commit=previous.words_since_commit + 1,
            no_emit_streak=previous.no_emit_streak + 1,
        )
        self._refresh_runtime()
        return self.history

    def on_write(self, outcome, now_s: float | None = None) -> History:
        """Advance over a WRITE using an `ActionOutcome`-like object.

        The object must expose ``action``, ``emitted``, ``no_emit``,
        ``vote_top1``, ``vote_margin``, and ``vote_entropy``. A commit
        (``emitted > 0``) resets both lag counters and the commit clock; a
        no-emit write increments them, exactly like READ.
        """
        if now_s is not None:
            self.now_s = float(now_s)
        action = outcome.action
        emitted = int(outcome.emitted)
        no_emit = bool(outcome.no_emit)
        previous = self.history
        if emitted > 0:
            words_since_commit = 0
            no_emit_streak = 0
        else:
            words_since_commit = previous.words_since_commit + 1
            no_emit_streak = previous.no_emit_streak + 1
        self.history = History(
            last_branches=int(action.branches),
            last_horizon=int(action.horizon),
            last_vote_top1=float(outcome.vote_top1),
            last_vote_margin=float(outcome.vote_margin),
            last_vote_entropy=float(outcome.vote_entropy),
            last_commit_count=emitted,
            last_empty=no_emit,
            words_since_commit=words_since_commit,
            no_emit_streak=no_emit_streak,
        )
        if emitted > 0:
            self.last_commit_s = self.now_s
        self._refresh_runtime()
        return self.history

    def set_clock(self, now_s: float) -> RuntimeInfo:
        """Move the virtual clock without changing history or commit time."""
        self.now_s = float(now_s)
        return self._refresh_runtime()

    def _refresh_runtime(self) -> RuntimeInfo:
        self.runtime = replace(
            self.runtime,
            time_since_commit=max(0.0, self.now_s - self.last_commit_s),
        )
        return self.runtime

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def build(
        self,
        builder: FeatureBuilder,
        observed: str,
        committed: Sequence[str],
        predictor_id: int = 0,
        translator_id: int = 0,
    ) -> FeatureVector:
        """Build the current state through the shared online `FeatureBuilder`."""
        return builder.build(
            observed,
            committed,
            history=self.history,
            runtime=self.runtime,
            predictor_id=int(predictor_id),
            translator_id=int(translator_id),
        )

    def snapshot(self) -> dict:
        """Serializable copy of the tracker state for labels and diagnostics."""
        return {
            "history": asdict(self.history),
            "runtime": asdict(self.runtime),
            "last_commit_s": float(self.last_commit_s),
            "now_s": float(self.now_s),
        }


def advance_virtual_clock(
    now_s: float,
    read_pos: int,
    arrival_gap: float,
    spent_s: float = 0.0,
) -> float:
    """Completion time after one decision, shared by labeling and rollout.

    Both paths advance the clock the same way: the clock is at least the source
    arrival time of the just-read word (``(read_pos - 1) * gap`` in the rollout
    cadence) and then advances by the measured compute spent on the action.
    Using one helper keeps the `time_since_commit` feature in the same range in
    training and serving; READ passes ``spent_s=0`` and only waits for arrival.
    """
    arrived = (int(read_pos) - 1) * float(arrival_gap)
    return max(float(now_s), arrived) + max(0.0, float(spent_s))


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


def _numeric_rows(vectors: Sequence) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    for vector in vectors:
        numeric = getattr(vector, "numeric", None)
        if numeric is None:
            raise TypeError("feature_diagnostics needs objects with .numeric")
        values = tuple(float(value) for value in numeric)
        if len(values) != len(NUMERIC_FEATURES):
            raise ValueError(
                f"feature vector has {len(values)} values but "
                f"NUMERIC_FEATURES has {len(NUMERIC_FEATURES)}"
            )
        rows.append(values)
    return rows


def _feature_stats(rows: Sequence[tuple[float, ...]]) -> dict[str, dict]:
    """Per-feature mean/std/non-zero/missing over finite values only."""
    count = len(rows)
    result: dict[str, dict] = {}
    for index, name in enumerate(NUMERIC_FEATURES):
        finite = [row[index] for row in rows if math.isfinite(row[index])]
        missing = count - len(finite)
        if finite:
            mean = sum(finite) / len(finite)
            variance = sum((value - mean) ** 2 for value in finite) / len(finite)
            std = math.sqrt(variance)
            nonzero = sum(1 for value in finite if value != 0.0)
        else:
            mean = None
            std = None
            nonzero = 0
        result[name] = {
            "mean": mean,
            "std": std,
            "nonzero": int(nonzero),
            "missing": int(missing),
        }
    return result


def feature_diagnostics(
    vectors: Sequence,
    reference: Sequence | None = None,
) -> dict:
    """Per-feature statistics plus a train/serve mean shift.

    `vectors` is the serving (or training) set and `reference`, when supplied,
    is the training set. Every feature reports ``mean``, ``std``, ``nonzero``
    (count of finite non-zero values), and ``missing`` (non-finite values).
    With a reference, each feature also reports ``train_mean``, ``serve_mean``,
    ``mean_diff`` (serving minus training), and
    ``constant_train_varies_serving``.

    That last flag is the silent-skew warning: a feature that never moved in
    training but moves in serving means the served distribution is outside the
    one the network saw. The returned top-level ``warnings`` list names every
    such feature and ``constant_train_varies_serving`` collects them.
    """
    rows = _numeric_rows(vectors)
    serving = _feature_stats(rows)
    payload: dict = {
        "n": len(rows),
        "reference_n": 0,
        "features": serving,
        "warnings": [],
        "constant_train_varies_serving": [],
    }
    if reference is None:
        return payload

    reference_rows = _numeric_rows(reference)
    training = _feature_stats(reference_rows)
    payload["reference_n"] = len(reference_rows)
    warnings: list[str] = []
    flagged: list[str] = []
    for name in NUMERIC_FEATURES:
        train = training[name]
        serve = serving[name]
        train_mean = train["mean"]
        serve_mean = serve["mean"]
        mean_diff = (
            None
            if train_mean is None or serve_mean is None
            else float(serve_mean) - float(train_mean)
        )
        train_constant = train["std"] == 0.0 and train["mean"] is not None
        serve_varies = serve["std"] not in (None, 0.0)
        varies_from_train = (
            train_constant
            and serve_mean is not None
            and abs(float(serve_mean) - float(train_mean)) > 1e-12
        )
        flag = bool(train_constant and (serve_varies or varies_from_train))
        payload["features"][name] = {
            **serve,
            "train_mean": train_mean,
            "train_std": train["std"],
            "serve_mean": serve_mean,
            "mean_diff": mean_diff,
            "constant_train_varies_serving": flag,
        }
        if flag:
            flagged.append(name)
            warnings.append(
                f"feature {name!r} is constant in training "
                f"(mean={train_mean}, std=0) but varies in serving"
            )
    payload["warnings"] = warnings
    payload["constant_train_varies_serving"] = flagged
    return payload
