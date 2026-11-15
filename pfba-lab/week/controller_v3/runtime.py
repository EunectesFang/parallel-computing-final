"""Controlled execution runtime for candidate sampling, translation, and RALCP.

The runtime wraps the same predictor / translator adapters used by
`week_run.py`:

* ``predictor.sample(prefix, branches, horizon, seed) -> list[str]``
* ``translator.translate_batch(sources, prefix, final=False, ...) -> list[str]``

`run_pfba` is imported from `pfba-lab/paired` when it can be found so the
exact upstream RALCP gate and the translator-call helper are shared instead of
re-implemented. Both have local fallbacks so the runtime still works in a
stripped checkout.
"""

from __future__ import annotations

import inspect
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .action import Action
from .config import ControllerConfig

__all__ = ["StepResult", "ControlledRuntime", "vote_statistics"]


def _load_run_pfba():
    """Import `run_pfba` from `pfba-lab/paired`, or return ``None``.

    The brief's relative path (``parents[1] / "paired"``) is tried alongside
    the real location (``parents[2] / "paired"``) so the module resolves both
    when this file is run in place and when the repository layout shifts.
    """
    roots = (
        Path(__file__).resolve().parents[2] / "paired",
        Path(__file__).resolve().parents[1] / "paired",
    )
    for root in roots:
        if not (root / "run_pfba.py").is_file():
            continue
        location = str(root)
        if location not in sys.path:
            sys.path.insert(0, location)
        try:
            import run_pfba  # type: ignore[import-not-found]
        except Exception:
            continue
        return run_pfba
    return None


run_pfba = _load_run_pfba()


def _local_ralcp(candidates: Sequence[str], threshold: float = 0.6) -> str:
    """Local port of ``upstream_ralcp`` used when `run_pfba` is unavailable."""
    if not candidates:
        return ""
    words = [candidate.split(" ") for candidate in candidates]
    translation = ""
    for index in range(max(len(word_list) for word_list in words)):
        counts = Counter(" ".join(word_list[: index + 1]) for word_list in words)
        best = max(counts, key=counts.get)
        if counts[best] < len(candidates) * threshold:
            break
        translation = best
    return translation


def vote_statistics(
    candidates: Sequence[str], threshold: float = 0.6
) -> tuple[float, float, float]:
    """``(top1, margin, entropy)`` over word prefixes at the decisive depth.

    The decisive depth is the first word depth where the modal prefix count
    falls below ``threshold * candidate_count``. When the threshold never
    fails, the decisive depth is the maximum depth reached by the pool:
    RALCP iterates to the longest candidate, and ended candidates contribute
    their full prefix at every later depth. Support is the modal prefix count
    divided by the original candidate count. Entropy is Shannon entropy in
    nats over the support distribution.

    Candidates are tokenized with ``text.split(" ")``, exactly as
    `_local_ralcp` and upstream `run_pfba.upstream_ralcp` do, so repeated,
    leading, or trailing spaces produce the same depths and support that RALCP
    voted on.
    """
    original = len(candidates)
    if original == 0:
        return 0.0, 0.0, 0.0

    words = [candidate.split(" ") for candidate in candidates]
    max_depth = max((len(word_list) for word_list in words), default=0)
    decisive_depth = max_depth
    for depth in range(1, max_depth + 1):
        counts = Counter(" ".join(word_list[:depth]) for word_list in words)
        if max(counts.values()) < original * threshold:
            decisive_depth = depth
            break

    counts = Counter(
        " ".join(word_list[:decisive_depth]) for word_list in words
    )
    ordered = sorted(counts.values(), reverse=True)
    top1 = ordered[0] / original
    second = ordered[1] / original if len(ordered) > 1 else 0.0
    entropy = 0.0
    for count in ordered:
        probability = count / original
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return float(top1), float(top1 - second), float(entropy)


@dataclass(frozen=True)
class StepResult:
    """Everything one write action produced.

    `candidates` are the translated strings that RALCP voted over, not the
    predictor's source-side continuations. `committed_after` is a fresh tuple;
    the caller's `committed` sequence is never mutated.

    `pred_s` and `mt_s` are the measured stage durations. `spent_s` is the
    serial compute billed to the decision (for `extend`, cumulative). The two
    must never be added and reported as pipeline wall time.

    `marginal_wall_s` is the incremental pipeline wall time from the runtime's
    device schedule: the finish time of prediction and translation minus the
    moment both devices and the action release were free. It is ``None`` when a
    caller-supplied fake runtime does not provide it; rollout can then fall
    back to `spent_s` as a documented approximation.
    """

    candidates: tuple[str, ...]
    selected: str
    committed_after: tuple[str, ...]
    pred_s: float
    mt_s: float
    spent_s: float
    vote_top1: float
    vote_margin: float
    vote_entropy: float
    no_emit: bool
    marginal_wall_s: float | None = None


class ControlledRuntime:
    """Executes controller actions with per-candidate sampling seeds.

    A write action samples each candidate with its own ``predictor.sample``
    call (``branches=1``), using seeds ``seed + 0 .. seed + branches - 1``.
    This is deliberately B small calls instead of one
    ``num_return_sequences=B`` call: candidate ``i`` then keeps the same
    sample when the budget grows from B=2 to B=4, so budget comparisons measure
    the controller rather than a shifted sampling stream. The trade-off is
    that per-candidate calls can be slower than one batched call; the matched
    baseline for any controller experiment must use this same path.

    The runtime tracks two virtual devices, `t_pred_free` and `t_mt_free`, the
    absolute times at which the prediction and translation devices finish their
    last job. Each `execute`/`extend` schedules prediction at
    ``max(t_pred_free, release_s)`` and translation at
    ``max(t_mt_free, pred_end, dependency_s)``, then reports
    `StepResult.marginal_wall_s` as the finish time minus
    ``max(t_pred_free, t_mt_free, release_s)``. Work that overlaps with an
    already-busy device is therefore not charged twice, and
    ``pred_s + mt_s`` is never treated as pipeline wall time.

    `cost_model` may be a `PipelineCostModel`. `estimate_marginal` exposes the
    model-priced marginal for an action that has not run yet, using the current
    device schedule; this is what rollout uses to score actions.

    `timer` may be a `run_pfba.StageTimer` (which device-synchronizes around
    each stage). Without a timer, durations come from `time.perf_counter`.
    """

    def __init__(self, predictor, translator, cfg, timer=None, cost_model=None) -> None:
        self.predictor = predictor
        self.translator = translator
        self.cfg = cfg if cfg is not None else ControllerConfig()
        self.timer = timer
        self.cost_model = cost_model
        self.t_pred_free = 0.0
        self.t_mt_free = 0.0

    def reset_timing(self) -> None:
        """Move both virtual devices back to time zero."""
        self.t_pred_free = 0.0
        self.t_mt_free = 0.0

    def _schedule_marginal(
        self,
        pred_s: float,
        mt_s: float,
        release_s: float,
        dependency_s: float,
    ) -> float:
        """Schedule one measured prediction+translation and return marginal wall.

        Both devices and the action may be blocked by prior work (`t_*_free`),
        the action release (`release_s`), or the translation's dependency
        (`dependency_s`). The marginal wall is the finish time minus the first
        moment all three constraints allow work to start, so overlapping with
        prior work is not double-billed.
        """
        release = max(0.0, float(release_s))
        pred_start = max(float(self.t_pred_free), release)
        pred_end = pred_start + max(0.0, float(pred_s))
        mt_start = max(float(self.t_mt_free), pred_end, float(dependency_s))
        mt_end = mt_start + max(0.0, float(mt_s))
        baseline = max(float(self.t_pred_free), float(self.t_mt_free), release)
        self.t_pred_free = pred_end
        self.t_mt_free = mt_end
        return max(0.0, max(pred_end, mt_end) - baseline)

    def estimate_marginal(
        self,
        predictor_id: int,
        translator_id: int,
        action: Action,
        src_words: int,
        tgt_words: int,
        *,
        release_s: float = 0.0,
        dependency_s: float = 0.0,
    ) -> float | None:
        """Model-priced marginal wall for an action, or ``None``.

        Returns ``None`` when no `PipelineCostModel` was supplied or the model
        has no ``marginal_wall`` method. The release argument is folded into
        the device free times so the result matches this runtime's scheduling.
        """
        marginal = getattr(self.cost_model, "marginal_wall", None)
        if not callable(marginal):
            return None
        release = max(0.0, float(release_s))
        return float(
            marginal(
                max(float(self.t_pred_free), release),
                max(float(self.t_mt_free), release),
                release,
                float(dependency_s),
                int(predictor_id),
                int(translator_id),
                action,
                int(src_words),
                int(tgt_words),
            )
        )

    # ------------------------------------------------------------------
    # Stage primitives
    # ------------------------------------------------------------------
    def _timed(self, name: str, function):
        runner = getattr(self.timer, "run", None)
        if callable(runner):
            return runner(name, function)
        start = time.perf_counter()
        value = function()
        return value, max(0.0, time.perf_counter() - start)

    def predict(self, observed: str, action: Action, seed: int):
        """Sample ``action.branches`` candidates, one call and seed each."""
        if action.is_read:
            # READ samples nothing. `execute` rejects READ explicitly; this
            # keeps a direct call from looping zero times with no explanation.
            return [], 0.0

        continuations: list[str] = []
        seconds = 0.0
        for index in range(int(action.branches)):
            batch, elapsed = self._timed(
                "pred",
                lambda sample_seed=int(seed) + index: self.predictor.sample(
                    observed, 1, int(action.horizon), sample_seed
                ),
            )
            continuations.extend(batch)
            seconds += elapsed
        return continuations, seconds

    def translate(self, requests: Sequence[str], prefix: str, final: bool = False):
        """Translate `requests` with `prefix` forced; return outputs and time."""
        sources = list(requests)
        if not sources:
            return [], 0.0
        outputs, seconds = self._timed(
            "mt", lambda: self._translate_call(sources, prefix, final)
        )
        return list(outputs), seconds

    def _translate_call(self, sources: Sequence[str], prefix: str, final: bool):
        if run_pfba is not None and hasattr(run_pfba, "call_translate_batch"):
            return run_pfba.call_translate_batch(
                self.translator, sources, prefix, final=final
            )

        method = self.translator.translate_batch
        if not final:
            return method(sources, prefix)
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_final = "final" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_final:
            return method(sources, prefix, final=final)
        return method(sources, prefix)

    def ralcp(self, candidates: Sequence[str]) -> str:
        """Run the exact upstream RALCP gate over translated candidates."""
        pool = list(candidates)
        if not pool:
            return ""
        threshold = self.cfg.ralcp_threshold
        if run_pfba is not None and hasattr(run_pfba, "upstream_ralcp"):
            return run_pfba.upstream_ralcp(pool, threshold)
        return _local_ralcp(pool, threshold)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def execute(
        self,
        observed: str,
        committed: Sequence[str],
        action: Action,
        seed: int,
        release_s: float = 0.0,
        dependency_s: float = 0.0,
    ) -> StepResult:
        """Sample, translate, and RALCP one write action.

        `release_s` is the earliest moment the action may start (usually the
        decision's virtual clock) and `dependency_s` the earliest moment the
        translation may start (usually the source-arrival time). Both default
        to zero for a cold runtime.
        """
        self._require_write(action)
        committed_tuple = tuple(committed)
        continuations, pred_s = self.predict(observed, action, seed)
        requests = [
            self._request_text(observed, continuation)
            for continuation in continuations
        ] or [observed]
        outputs, mt_s = self.translate(requests, " ".join(committed_tuple))
        marginal = self._schedule_marginal(pred_s, mt_s, release_s, dependency_s)
        return self._build_result(
            committed_tuple,
            outputs,
            pred_s,
            mt_s,
            pred_s + mt_s,
            marginal,
        )

    def extend(
        self,
        observed: str,
        committed: Sequence[str],
        current_candidates: Sequence[str],
        extra_branches: int,
        horizon: int,
        seed: int,
        spent_s: float = 0.0,
        release_s: float = 0.0,
        dependency_s: float = 0.0,
    ) -> StepResult:
        """Add candidates to an existing pool without re-translating it.

        New candidates use seeds ``seed + current_branches + index`` so they
        line up exactly with what a single larger `execute` would have drawn.
        Only the new requests are translated; RALCP then votes over the full
        pool. `committed` and `current_candidates` are not mutated. Raises
        `ValueError` unless both `extra_branches` and `horizon` are positive.

        `marginal_wall_s` covers only this extension; the caller adds the
        provisional execution's marginal when it bills the decision. `spent_s`
        stays cumulative so discarded work is still charged once.
        """
        committed_tuple = tuple(committed)
        pool = list(current_candidates)
        extra = int(extra_branches)
        if extra <= 0:
            raise ValueError("extra_branches must be positive")
        if int(horizon) <= 0:
            raise ValueError("extend needs a positive horizon")

        pred_s = 0.0
        mt_s = 0.0
        base_seed = int(seed) + len(pool)
        continuations: list[str] = []
        for index in range(extra):
            batch, elapsed = self._timed(
                "pred",
                lambda sample_seed=base_seed + index: self.predictor.sample(
                    observed, 1, int(horizon), sample_seed
                ),
            )
            continuations.extend(batch)
            pred_s += elapsed

        requests = [
            self._request_text(observed, continuation)
            for continuation in continuations
        ]
        if requests:
            outputs, mt_s = self.translate(requests, " ".join(committed_tuple))
            pool.extend(outputs)

        marginal = self._schedule_marginal(pred_s, mt_s, release_s, dependency_s)
        return self._build_result(
            committed_tuple,
            pool,
            pred_s,
            mt_s,
            float(spent_s) + pred_s + mt_s,
            marginal,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _require_write(action: Action) -> None:
        if action.is_read:
            raise ValueError(
                "ControlledRuntime.execute cannot run READ: READ performs no "
                "prediction or translation; branch on action.is_read instead"
            )

    @staticmethod
    def _request_text(observed: str, continuation: str) -> str:
        continuation = continuation.strip()
        return observed if not continuation else f"{observed} {continuation}"

    def _build_result(
        self,
        committed: Sequence[str],
        pool: Sequence[str],
        pred_s: float,
        mt_s: float,
        spent_s: float,
        marginal_wall_s: float,
    ) -> StepResult:
        candidates = tuple(pool)
        selected = self.ralcp(candidates)
        selected_words = tuple(selected.split())
        top1, margin, entropy = vote_statistics(
            candidates, self.cfg.ralcp_threshold
        )
        return StepResult(
            candidates=candidates,
            selected=selected,
            committed_after=tuple(committed) + selected_words,
            pred_s=float(pred_s),
            mt_s=float(mt_s),
            spent_s=float(spent_s),
            vote_top1=top1,
            vote_margin=margin,
            vote_entropy=entropy,
            no_emit=not selected_words,
            marginal_wall_s=float(marginal_wall_s),
        )
