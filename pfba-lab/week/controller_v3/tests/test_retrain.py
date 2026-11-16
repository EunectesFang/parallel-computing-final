"""Tests for offline retraining from persisted labels."""

from controller_v3.action import READ, Action
from controller_v3.retrain import apply_serial_costs, replay_histories


def _row():
    return {
        "sentence_id": "dev:0000",
        "states": [
            {
                "chosen": "b4h6",
                "history": {},
                "outcomes": [
                    {"action": "b4h6", "emitted": 2},
                    {"action": "read", "emitted": 0},
                ],
            },
            {
                "chosen": "read",
                "history": {},
                "outcomes": [
                    {"action": "b4h6", "emitted": 0},
                    {"action": "read", "emitted": 0},
                ],
            },
            {
                "chosen": "b6h10",
                "history": {},
                "outcomes": [
                    {"action": "b6h10", "emitted": 0},
                    {"action": "read", "emitted": 0},
                ],
            },
            {
                "chosen": "b2h4",
                "history": {},
                "outcomes": [
                    {"action": "b2h4", "emitted": 1},
                    {"action": "read", "emitted": 0},
                ],
            },
        ],
    }


def test_replay_histories_uses_pre_action_state_and_failed_write_streak():
    rows = replay_histories([_row()])
    states = rows[0]["states"]
    assert [
        (state["history"]["words_since_commit"], state["history"]["no_emit_streak"])
        for state in states
    ] == [(0, 0), (0, 0), (1, 0), (2, 1)]


def test_apply_serial_costs_replaces_scheduled_marginal_with_stage_sum():
    row = {
        "states": [
            {
                "outcomes": [
                    {
                        "action": "b4h6",
                        "pred_s": 0.5,
                        "mt_s": 0.25,
                        "marginal_wall_s": 0.1,
                    },
                    {
                        "action": "read",
                        "pred_s": 0.0,
                        "mt_s": 0.0,
                        "marginal_wall_s": 0.0,
                    },
                ]
            }
        ]
    }
    apply_serial_costs([row], overhead_s=0.05)
    outcomes = row["states"][0]["outcomes"]
    assert outcomes[0]["marginal_wall_s"] == 0.8
    assert outcomes[1]["marginal_wall_s"] == 0.0


def test_auc_and_behavior_read_run_stats():
    import pytest

    from controller_v3.retrain import _auc, behavior_read_run_stats

    assert _auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert _auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(0.0)
    assert _auc([0.1], [1]) is None

    row = {
        "states": [
            {"chosen": "read"},
            {"chosen": "read"},
            {"chosen": "b4h6"},
        ]
    }
    stats = behavior_read_run_stats([row])
    assert stats["n_read_runs"] == 1
    assert stats["mean_run"] == 2.0
    assert stats["max_run"] == 2
    assert stats["read_fraction"] == pytest.approx(2 / 3)
