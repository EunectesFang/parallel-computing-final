"""Tests for the post-candidate controller's heuristic rules.

The controller only proposes; these tests pin the STOP / ADD_2 / WAIT
thresholds and the conservative default. No runtime, model, or torch is
involved.
"""

from controller_v3.action import Action
from controller_v3.config import ControllerConfig
from controller_v3.post_controller import (
    ADD_ENTROPY_MIN,
    STOP_MARGIN_MIN,
    STOP_NO_EMIT_MAX,
    WAIT_BUDGET_S,
    WAIT_NO_EMIT_MIN,
    PostAction,
    PostController,
)


def decide(**overrides):
    values = dict(
        z_pre=object(),
        current_branches=2,
        horizon=4,
        vote_entropy=0.0,
        vote_margin=0.0,
        no_emit_prob=0.0,
        spent_s=0.0,
        cfg=ControllerConfig(actions=(Action(2, 4),)),
    )
    values.update(overrides)
    return PostController().decide(**values)


def test_confident_margin_and_low_no_emit_risk_stops():
    decision = decide(
        vote_margin=STOP_MARGIN_MIN + 0.1,
        no_emit_prob=STOP_NO_EMIT_MAX - 0.1,
        vote_entropy=0.0,
    )

    assert decision.action is PostAction.STOP
    assert "margin" in decision.reason


def test_high_entropy_with_room_under_max_branches_adds_two():
    decision = decide(
        vote_entropy=ADD_ENTROPY_MIN + 0.1,
        vote_margin=0.0,
        no_emit_prob=0.1,
        current_branches=2,
        cfg=ControllerConfig(actions=(Action(2, 4),), max_branches=10),
    )

    assert decision.action is PostAction.ADD_2
    assert "entropy" in decision.reason


def test_high_entropy_at_max_branches_does_not_add():
    decision = decide(
        vote_entropy=ADD_ENTROPY_MIN + 0.1,
        vote_margin=0.0,
        no_emit_prob=0.1,
        current_branches=10,
        cfg=ControllerConfig(actions=(Action(2, 4),), max_branches=10),
    )

    assert decision.action is PostAction.STOP


def test_high_no_emit_risk_and_overspent_budget_waits():
    decision = decide(
        vote_entropy=0.0,
        vote_margin=0.0,
        no_emit_prob=WAIT_NO_EMIT_MIN + 0.1,
        spent_s=WAIT_BUDGET_S + 1.0,
    )

    assert decision.action is PostAction.WAIT
    assert "no-emit" in decision.reason


def test_high_no_emit_risk_without_spending_budget_stops_conservatively():
    decision = decide(
        vote_entropy=0.0,
        vote_margin=0.0,
        no_emit_prob=WAIT_NO_EMIT_MIN + 0.1,
        spent_s=WAIT_BUDGET_S - 0.5,
    )

    assert decision.action is PostAction.STOP
    assert "insufficient evidence" in decision.reason


def test_ambiguous_low_signal_case_defaults_to_stop():
    decision = decide()

    assert decision.action is PostAction.STOP
    assert "insufficient evidence" in decision.reason


def test_z_pre_and_horizon_are_accepted_and_ignored():
    sentinel = object()
    decision = decide(z_pre=sentinel, horizon=99)

    assert decision.action is PostAction.STOP
