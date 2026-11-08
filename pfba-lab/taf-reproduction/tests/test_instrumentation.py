"""Clock instrumentation invariants."""
import json
from types import SimpleNamespace

from taf.instrumentation import Stopwatch, summarize_timings


def test_stopwatch_records_stages_and_accumulates():
    sw = Stopwatch(cuda_sync=False)
    with sw.stage("mt_decode"):
        pass
    with sw.stage("mt_decode"):
        pass
    sw.add("aggregate", 2.0)
    data = sw.as_dict()
    assert set(data) == {"mt_decode", "aggregate"}
    assert data["mt_decode"] >= 0.0
    assert data["aggregate"] == 2.0


def test_summarize_timings_means_and_sums():
    traces = [
        SimpleNamespace(timings_ms={"llm_sample": 10, "mt_decode": 20, "aggregate": 1, "total": 31}),
        SimpleNamespace(timings_ms={"llm_sample": 30, "mt_decode": 40, "aggregate": 3, "total": 73}),
    ]
    summary = summarize_timings(traces, {"flush_ms": 5.0})
    assert summary["n_steps"] == 2
    assert summary["sum_total_ms"] == 104.0
    assert summary["avg_llm_sample_ms"] == 20.0
    assert summary["avg_mt_decode_ms"] == 30.0
    assert summary["flush_ms"] == 5.0
    json.dumps(summary)  # must be JSON serializable
