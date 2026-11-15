"""Runtime cost models for prediction and translation.

Three models share the `CostModel` protocol:

* `MeasuredCostTable` answers from measured medians and degrades gracefully to
  the nearest measured action for the same model pair, then to a global
  median, then to a small floor.
* `PowerLawCostModel` fits ``pred = a * b^r_b * h^r_h`` and
  ``mt = c * b^s_b * h^s_h`` by log-linear least squares and falls back to the
  measured table when a model pair has fewer than four usable records.
* `PipelineCostModel` prices the incremental wall time of one prediction plus
  one translation on two devices that may already be busy.

Every model treats READ as free: the trainer's Bellman backup queries READ for
every non-terminal target, and READ performs neither prediction nor
translation.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from .action import Action

__all__ = [
    "MIN_COST_S",
    "CostEstimate",
    "CostModel",
    "MeasuredCostTable",
    "PowerLawCostModel",
    "PipelineCostModel",
]

# Smallest non-zero cost used when no measurement exists at all.
MIN_COST_S = 1e-3

# A full-rank log-linear fit needs three coefficients; the contract asks for a
# little more evidence, hence four records.
MIN_POWER_LAW_RECORDS = 4


@dataclass(frozen=True)
class CostEstimate:
    """Prediction, translation, and their serial sum in seconds.

    ``wall_s`` is always recomputed as ``pred_s + mt_s``. It is a **serial
    compute diagnostic**, not pipeline wall time: two devices can overlap, and
    the runtime/scheduler uses `PipelineCostModel.marginal_wall` (or the
    runtime's measured marginal) for wall-time pricing. Keeping the serial sum
    on this diagnostic field would otherwise let it leak into the reward or
    target code as if it were a pipeline cost.
    """

    pred_s: float
    mt_s: float
    wall_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "wall_s", float(self.pred_s) + float(self.mt_s))


class CostModel(Protocol):
    """Structural interface consumed by the trainer and runtime."""

    def predict(
        self,
        predictor_id: int,
        translator_id: int,
        action: Action,
        src_words: int,
        tgt_words: int,
    ) -> CostEstimate: ...


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


class MeasuredCostTable:
    """Median measured costs keyed by ``(predictor_id, translator_id, action)``.

    Records are ``(predictor_id, translator_id, action, pred_s, mt_s)`` or the
    newer ``(predictor_id, translator_id, action, pred_s, mt_s,
    marginal_wall_s)`` tuples. The optional sixth value is measured pipeline
    wall time; it is preserved on `.records` for audit/checkpointing but does
    not change the ``(pred_s, mt_s)`` medians this table returns, because the
    runtime and trainer price the two stages separately.

    READ records are ignored: READ is always free, and a stored zero-cost row
    could otherwise win the nearest-action fallback for a write action.
    `src_words` / `tgt_words` are accepted by `predict` for interface parity
    but do not enter the lookup: the table is already conditioned on the model
    pair and action.
    """

    def __init__(
        self,
        records: Sequence[
            tuple[int, int, Action, float, float]
            | tuple[int, int, Action, float, float, float]
        ] = (),
    ) -> None:
        self.records: list[tuple] = []
        self._table: dict[tuple[int, int, Action], tuple[float, float]] = {}
        self._by_pair: dict[tuple[int, int], dict[Action, tuple[float, float]]] = {}
        self._global: tuple[float, float] | None = None
        self.fit(records)

    @staticmethod
    def unpack_record(record) -> tuple:
        """Normalise one record and validate it has five or six fields.

        A legacy five-field record keeps its length; no marginal wall time is
        invented from ``pred_s + mt_s`` because that serial sum is not the
        pipeline wall time.
        """
        values = tuple(record)
        if len(values) not in (5, 6):
            raise ValueError(
                "cost records must be (predictor_id, translator_id, action, "
                "pred_s, mt_s) or the same with marginal_wall_s appended"
            )
        predictor_id, translator_id, action, pred_s, mt_s = values[:5]
        if not isinstance(action, Action):
            raise ValueError("cost record action must be an Action")
        head = (
            int(predictor_id),
            int(translator_id),
            action,
            float(pred_s),
            float(mt_s),
        )
        if len(values) == 6:
            return (*head, float(values[5]))
        return head

    def fit(self, records: Iterable) -> "MeasuredCostTable":
        """Replace the table with the medians of `records`; return ``self``."""
        grouped: dict[tuple[int, int, Action], tuple[list[float], list[float]]] = {}
        all_pred: list[float] = []
        all_mt: list[float] = []

        self.records = []
        for record in records:
            normalised = self.unpack_record(record)
            predictor_id, translator_id, action, pred, mt = normalised[:5]
            # READ is free by definition. Keeping a zero-cost READ row would
            # let the nearest-action fallback price a write action at zero, so
            # READ records never enter the table or the global median.
            if action.is_read:
                continue
            key = (int(predictor_id), int(translator_id), action)
            preds, mts = grouped.setdefault(key, ([], []))
            preds.append(pred)
            mts.append(mt)
            all_pred.append(pred)
            all_mt.append(mt)
            self.records.append(normalised)

        self._table = {
            key: (_median(preds), _median(mts))
            for key, (preds, mts) in grouped.items()
        }
        self._by_pair = defaultdict(dict)
        for (predictor_id, translator_id, action), value in self._table.items():
            self._by_pair[(predictor_id, translator_id)][action] = value
        self._by_pair = dict(self._by_pair)
        self._global = (
            (_median(all_pred), _median(all_mt)) if all_pred else None
        )
        return self

    def predict(
        self,
        predictor_id: int,
        translator_id: int,
        action: Action,
        src_words: int,
        tgt_words: int,
    ) -> CostEstimate:
        """Exact median, then nearest action for the pair, then global median."""
        if action.is_read:
            return CostEstimate(0.0, 0.0)

        pair = (int(predictor_id), int(translator_id))
        exact = self._table.get((pair[0], pair[1], action))
        if exact is not None:
            return CostEstimate(exact[0], exact[1])

        measured = self._by_pair.get(pair)
        if measured:
            nearest = min(
                measured,
                key=lambda candidate: (
                    abs(candidate.branches - action.branches)
                    + abs(candidate.horizon - action.horizon),
                    candidate.branches,
                    candidate.horizon,
                ),
            )
            pred, mt = measured[nearest]
            return CostEstimate(pred, mt)

        if self._global is not None:
            pred, mt = self._global
            return CostEstimate(max(pred, MIN_COST_S), max(mt, MIN_COST_S))
        return CostEstimate(MIN_COST_S, MIN_COST_S)


def _solve_3x3(
    matrix: list[list[float]], vector: list[float]
) -> tuple[float, float, float] | None:
    """Gaussian elimination with partial pivoting; ``None`` when singular."""
    a = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(a[row][column]))
        if abs(a[pivot][column]) < 1e-12:
            return None
        a[column], a[pivot] = a[pivot], a[column]
        diagonal = a[column][column]
        for row in range(column + 1, 3):
            factor = a[row][column] / diagonal
            if factor == 0.0:
                continue
            for item in range(column, 4):
                a[row][item] -= factor * a[column][item]
    solution = [0.0, 0.0, 0.0]
    for row in (2, 1, 0):
        total = a[row][3]
        for column in range(row + 1, 3):
            total -= a[row][column] * solution[column]
        solution[row] = total / a[row][row]
    return solution[0], solution[1], solution[2]


def _fit_log_linear(
    rows: Sequence[tuple[int, int, float]],
) -> tuple[float, float, float] | None:
    """Least-squares ``log value = log a + r_b log b + r_h log h``.

    Returns ``(log a, r_b, r_h)`` with exponents clamped to ``[0, 1]``, or
    ``None`` when the system is singular.
    """
    xtx = [[0.0, 0.0, 0.0] for _ in range(3)]
    xty = [0.0, 0.0, 0.0]
    for branches, horizon, value in rows:
        if branches <= 0 or horizon <= 0 or value <= 0.0:
            return None
        features = (1.0, math.log(branches), math.log(horizon))
        target = math.log(value)
        for row in range(3):
            xty[row] += features[row] * target
            for column in range(3):
                xtx[row][column] += features[row] * features[column]

    solution = _solve_3x3(xtx, xty)
    if solution is None or not all(math.isfinite(value) for value in solution):
        return None
    log_a, exponent_b, exponent_h = solution
    return (
        log_a,
        min(1.0, max(0.0, exponent_b)),
        min(1.0, max(0.0, exponent_h)),
    )


class PowerLawCostModel:
    """Log-linear power-law fit with a measured-table fallback.

    The fit is per model pair: ``(predictor_id, translator_id)``. A pair with
    fewer than `MIN_POWER_LAW_RECORDS` usable records (positive prediction and
    translation times) uses the measured table instead. Fitted exponents are
    clamped to ``[0, 1]`` so an extrapolated cost cannot decrease with more
    branches or a longer horizon.
    """

    def __init__(
        self,
        records: Sequence[tuple[int, int, Action, float, float]] = (),
    ) -> None:
        # pair -> (log_a, r_b, r_h, log_c, s_b, s_h)
        self.coefficients: dict[
            tuple[int, int], tuple[float, float, float, float, float, float]
        ] = {}
        self.table = MeasuredCostTable()
        self.fit(records)

    def fit(self, records: Iterable) -> "PowerLawCostModel":
        """Refit every model pair from `records`; return ``self``."""
        # Preserve an optional sixth ``marginal_wall_s`` field while the fit
        # itself uses only branches, horizon, and the two stage durations.
        materialized = [
            MeasuredCostTable.unpack_record(record) for record in records
        ]
        self.table = MeasuredCostTable(materialized)
        grouped: dict[tuple[int, int], list[tuple[Action, float, float]]] = defaultdict(list)
        for record in materialized:
            predictor_id, translator_id, action, pred_s, mt_s = record[:5]
            grouped[(predictor_id, translator_id)].append((action, pred_s, mt_s))

        self.coefficients = {}
        for pair, rows in grouped.items():
            usable = [
                (action, pred_s, mt_s)
                for action, pred_s, mt_s in rows
                if action.branches > 0
                and action.horizon > 0
                and pred_s > 0.0
                and mt_s > 0.0
            ]
            if len(usable) < MIN_POWER_LAW_RECORDS:
                continue
            pred_fit = _fit_log_linear(
                [(action.branches, action.horizon, pred_s) for action, pred_s, _ in usable]
            )
            mt_fit = _fit_log_linear(
                [(action.branches, action.horizon, mt_s) for action, _, mt_s in usable]
            )
            if pred_fit is None or mt_fit is None:
                continue
            self.coefficients[pair] = pred_fit + mt_fit
        return self

    def exponents(
        self, predictor_id: int, translator_id: int
    ) -> tuple[float, float, float, float] | None:
        """``(r_b, r_h, s_b, s_h)`` for a fitted pair, else ``None``."""
        coefficients = self.coefficients.get((int(predictor_id), int(translator_id)))
        if coefficients is None:
            return None
        _, r_b, r_h, _, s_b, s_h = coefficients
        return r_b, r_h, s_b, s_h

    def predict(
        self,
        predictor_id: int,
        translator_id: int,
        action: Action,
        src_words: int,
        tgt_words: int,
    ) -> CostEstimate:
        """Power-law prediction, or the measured-table estimate as fallback."""
        if action.is_read:
            return CostEstimate(0.0, 0.0)

        coefficients = self.coefficients.get((int(predictor_id), int(translator_id)))
        if coefficients is None:
            return self.table.predict(
                predictor_id, translator_id, action, src_words, tgt_words
            )

        log_a, r_b, r_h, log_c, s_b, s_h = coefficients
        # READ is handled above; a defensive max(1, .) keeps a malformed
        # b=0/h=0 write from taking log(0).
        log_branches = math.log(max(1, int(action.branches)))
        log_horizon = math.log(max(1, int(action.horizon)))
        pred_s = math.exp(log_a + r_b * log_branches + r_h * log_horizon)
        mt_s = math.exp(log_c + s_b * log_branches + s_h * log_horizon)
        if not (math.isfinite(pred_s) and math.isfinite(mt_s)):
            return self.table.predict(
                predictor_id, translator_id, action, src_words, tgt_words
            )
        return CostEstimate(pred_s, mt_s)


class PipelineCostModel:
    """Prices one action against devices that may already be busy.

    `marginal_wall` simulates a prediction that starts at
    ``max(t_pred_free, release_s)`` and a translation that starts at
    ``max(t_mt_free, pred_end, dependency_s)``. The incremental cost is the
    finish time minus the moment both devices were free, so work that overlaps
    with an already-busy device is not charged twice.

    `arrival_gap` is retained for callers that derive ``release_s`` and
    ``dependency_s`` from a fixed source cadence; `marginal_wall` takes those
    two times explicitly.
    """

    def __init__(self, base: CostModel, arrival_gap: float = 0.3) -> None:
        self.base = base
        self.arrival_gap = float(arrival_gap)

    def predict(
        self,
        predictor_id: int,
        translator_id: int,
        action: Action,
        src_words: int,
        tgt_words: int,
    ) -> CostEstimate:
        """Delegate the per-stage estimate to the base model."""
        return self.base.predict(
            predictor_id, translator_id, action, src_words, tgt_words
        )

    def marginal_wall(
        self,
        t_pred_free: float,
        t_mt_free: float,
        release_s: float,
        dependency_s: float,
        predictor_id: int,
        translator_id: int,
        action: Action,
        src_words: int,
        tgt_words: int,
    ) -> float:
        """Wall time added by running `action` now, never negative."""
        estimate = self.base.predict(
            predictor_id, translator_id, action, src_words, tgt_words
        )
        pred_s = float(estimate.pred_s)
        mt_s = float(estimate.mt_s)

        pred_start = max(float(t_pred_free), float(release_s))
        pred_end = pred_start + pred_s
        mt_start = max(float(t_mt_free), pred_end, float(dependency_s))
        mt_end = mt_start + mt_s

        baseline = max(float(t_pred_free), float(t_mt_free))
        return max(0.0, max(mt_end, pred_end) - baseline)
