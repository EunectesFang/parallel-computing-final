"""Tests for the action grid and its six-dimensional feature encoding."""

import math

import pytest

from controller_v3.action import (
    DEFAULT_ACTIONS,
    EXTENDED_ACTIONS,
    READ,
    Action,
    action_features,
)


def test_read_action_label_and_flag():
    assert READ == Action(0, 0)
    assert READ.is_read is True
    assert READ.label == "read"


def test_write_action_label_and_flag():
    action = Action(2, 4)
    assert action.is_read is False
    assert action.label == "b2h4"


def test_default_action_grid_has_nine_entries():
    assert len(DEFAULT_ACTIONS) == 9
    assert DEFAULT_ACTIONS == tuple(
        Action(branches, horizon)
        for branches in (2, 4, 6)
        for horizon in (4, 6, 10)
    )
    assert len(set(DEFAULT_ACTIONS)) == 9


def test_extended_action_grid_has_twenty_five_entries():
    assert len(EXTENDED_ACTIONS) == 25
    assert EXTENDED_ACTIONS == tuple(
        Action(branches, horizon)
        for branches in (2, 4, 6, 8, 10)
        for horizon in (2, 4, 6, 8, 10)
    )
    assert len(set(EXTENDED_ACTIONS)) == 25


def test_action_features_exact_formula_for_four_by_six():
    features = action_features(Action(4, 6))
    assert len(features) == 6
    expected = (
        4.0,
        6.0,
        24.0,
        math.log(4),
        math.log(6),
        0.75,  # ceil(0.6 * 4) / 4 == 3 / 4
    )
    assert features == expected


def test_action_features_custom_threshold_changes_only_the_last_slot():
    default = action_features(Action(6, 4))
    custom = action_features(Action(6, 4), threshold=0.5)
    assert default[:5] == custom[:5]
    assert default[5] == pytest.approx(4 / 6)  # ceil(0.6 * 6) / 6 == 4 / 6
    assert custom[5] == 0.5  # ceil(0.5 * 6) / 6 == 0.5


def test_action_features_for_read_are_six_zeros():
    assert action_features(READ) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_actions_are_hashable_and_ordered():
    mapping = {Action(2, 4): "small", Action(4, 6): "default"}
    assert mapping[Action(2, 4)] == "small"
    ordered = sorted([Action(4, 6), Action(2, 4), Action(2, 6)])
    assert ordered == [Action(2, 4), Action(2, 6), Action(4, 6)]


@pytest.mark.parametrize("action", DEFAULT_ACTIONS)
def test_all_default_actions_produce_six_floats(action):
    features = action_features(action)
    assert len(features) == 6
    assert all(isinstance(value, float) for value in features)
