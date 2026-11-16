"""Tests for the closed-loop A/B/C/D evaluation helper."""

import json
from types import SimpleNamespace

import pytest

from controller_v3.config import ControllerConfig
from controller_v3.evaluate import (
    _policy_checkpoints,
    _resolve_checkpoint,
    guard_config,
    select_sentences,
)
from controller_v3.pair_traces import pair_trace_files


def test_policy_checkpoints_drops_the_baseline_when_requested():
    retrained = {"v5": "v5.pt"}
    assert _policy_checkpoints(
        "old.pt", retrained, include_baseline=True
    ) == {"v3_baseline": "old.pt", "v5": "v5.pt"}
    assert _policy_checkpoints(
        "old.pt", retrained, include_baseline=False
    ) == {"v5": "v5.pt"}


def test_pair_trace_files_reports_all_metrics_and_pairs(tmp_path):
    candidate = tmp_path / "candidate.traces.jsonl"
    fixed = tmp_path / "fixed.traces.jsonl"
    candidate_rows = [
        {"sentence_id": f"dev:{index:04d}", "hypothesis": "a b c",
         "reference": "a b c", "laal": 2.0 + index, "completion_s": 5.0 + index}
        for index in range(3)
    ]
    fixed_rows = [
        {"sentence_id": f"dev:{index:04d}", "hypothesis": "a b c",
         "reference": "a b c", "laal": 2.0, "completion_s": 7.0}
        for index in range(3)
    ]
    candidate.write_text(
        "\n".join(json.dumps(row) for row in candidate_rows) + "\n",
        encoding="utf-8",
    )
    fixed.write_text(
        "\n".join(json.dumps(row) for row in fixed_rows) + "\n",
        encoding="utf-8",
    )

    payload = pair_trace_files(candidate, fixed, samples=200)
    assert payload["n_pairs"] == 3
    assert set(payload["metrics"]) == {
        "bleu", "laal", "completion_s", "chrf"
    }
    # LAAL: (0, 1, 2) / 3 = 1.0; completion: (-2, -1, 0) / 3 = -1.0
    assert payload["metrics"]["laal"]["point"] == pytest.approx(1.0)
    assert payload["metrics"]["completion_s"]["point"] == pytest.approx(-1.0)
    # sacrebleu may be absent locally; the keys must still exist.
    assert payload["metrics"]["bleu"]["status"] in {"ok", "unavailable"}
    assert payload["metrics"]["chrf"]["status"] in {"ok", "unavailable"}


def test_resolve_checkpoint_accepts_kaggle_datasets_mount_layout(tmp_path):
    mounted = tmp_path / "datasets" / "owner" / "cavc-retrain-checkpoints"
    mounted.mkdir(parents=True)
    target = mounted / "v3_baseline.pt"
    target.write_bytes(b"checkpoint")

    legacy = tmp_path / "cavc-retrain-checkpoints" / "v3_baseline.pt"
    assert _resolve_checkpoint(legacy, search_root=tmp_path) == target


def test_resolve_checkpoint_prefers_the_existing_literal_path(tmp_path):
    literal = tmp_path / "cavc-retrain-checkpoints" / "v3_baseline.pt"
    literal.parent.mkdir(parents=True)
    literal.write_bytes(b"checkpoint")
    other = tmp_path / "datasets" / "owner" / "cavc-retrain-checkpoints"
    other.mkdir(parents=True)
    (other / "v3_baseline.pt").write_bytes(b"stale")

    assert _resolve_checkpoint(literal, search_root=tmp_path) == literal


def test_resolve_checkpoint_lists_available_files_when_missing(tmp_path):
    (tmp_path / "other.pt").write_bytes(b"checkpoint")
    with pytest.raises(FileNotFoundError, match="other.pt"):
        _resolve_checkpoint(
            tmp_path / "cavc-retrain-checkpoints" / "v3_baseline.pt",
            search_root=tmp_path,
        )


def test_select_sentences_skips_the_consumed_prefix():
    sentences = [
        SimpleNamespace(sentence_id=f"dev:{index:04d}") for index in range(300)
    ]
    selected = select_sentences(sentences, offset=200, limit=3)
    assert [sentence.sentence_id for sentence in selected] == [
        "dev:0200",
        "dev:0201",
        "dev:0202",
    ]
    with pytest.raises(ValueError, match="test:"):
        select_sentences(
            [SimpleNamespace(sentence_id="test:0140")], offset=0, limit=1
        )


def test_guard_config_enables_all_guards_without_mutating_the_base():
    base = ControllerConfig()
    guarded = guard_config(base)
    assert guarded.read_threshold > 0.0
    assert guarded.read_threshold_lag > 0.0
    assert guarded.expensive_margin > 0.0
    # The guard uses a soft no-emit penalty, not a hard veto; the hard veto is
    # an explicit ablation.
    assert guarded.no_emit_veto == 1.0
    assert guarded.no_emit_penalty > 0.0
    # The base config keeps V3 defaults, so variant A stays reproducible.
    assert base.read_threshold == 0.0
    assert base.expensive_margin == 0.0
    assert base.no_emit_veto == 1.0
    assert base.no_emit_penalty == 0.0


def test_read_run_and_service_time_stats_use_real_steps():
    from types import SimpleNamespace

    from controller_v3.evaluate import read_run_stats, service_time_stats

    traces = [
        SimpleNamespace(
            steps=(
                {"action": "read", "pred_s": 0.0, "mt_s": 0.0},
                {"action": "read", "pred_s": 0.0, "mt_s": 0.0},
                {"action": "b4h6", "pred_s": 1.0, "mt_s": 0.5},
                {"action": "read", "pred_s": 0.0, "mt_s": 0.0},
            )
        )
    ]
    runs = read_run_stats(traces)
    assert runs["n_reads"] == 3
    assert runs["n_steps"] == 4
    assert runs["read_fraction"] == 0.75
    assert runs["n_runs"] == 2
    assert runs["mean_run"] == 1.5
    assert runs["max_run"] == 2

    service = service_time_stats(traces)
    assert service["pred_s"] == 1.0
    assert service["mt_s"] == 0.5
    assert service["total_s"] == 1.5
    assert service["per_action"]["b4h6"]["count"] == 1.0
