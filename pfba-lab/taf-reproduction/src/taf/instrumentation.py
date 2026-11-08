"""Wall-clock instrumentation for streaming steps.

Paper Table 1 reports average wall-clock per step with and without TAF.  This
module measures the same quantity and records stage breakdowns so the PFBA
parent project can separate LLM prediction, MT decode, and policy overhead.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


def _cuda_sync() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class Stopwatch:
    """Accumulate named stage durations in milliseconds."""

    def __init__(self, cuda_sync: bool = False) -> None:
        self.cuda_sync = cuda_sync
        self._totals: dict[str, float] = {}

    def sync(self) -> None:
        if self.cuda_sync:
            _cuda_sync()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._totals[name] = self._totals.get(name, 0.0) + elapsed_ms

    def add(self, name: str, milliseconds: float) -> None:
        self._totals[name] = self._totals.get(name, 0.0) + float(milliseconds)

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 6) for k, v in self._totals.items()}


def summarize_timings(traces, extra_stages: dict[str, float] | None = None) -> dict:
    """Aggregate per-step timing dictionaries into means and totals."""
    stages = ("llm_sample", "mt_decode", "aggregate", "total")
    sums = {k: 0.0 for k in stages}
    for trace in traces:
        for key in stages:
            sums[key] += float(trace.timings_ms.get(key, 0.0))
    if extra_stages:
        for key, value in extra_stages.items():
            sums[key] = float(value)

    n = len(traces)
    out = {"n_steps": n}
    for key in stages:
        out[f"sum_{key}_ms"] = round(sums[key], 6)
        out[f"avg_{key}_ms"] = round(sums[key] / n, 6) if n else 0.0
    if extra_stages:
        for key, value in extra_stages.items():
            out[key] = round(float(value), 6)
    return out
