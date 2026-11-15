"""Regression tests for the modeling-core fixes.

These pin the defects the fix wave is about:

* every action gets its own successor, so the Bellman backup is
  action-conditioned;
* READ keeps the committed prefix and only advances the observed source;
* training states replay the live `History` rather than an all-zero default;
* delay is computed from the post-action backlog, so `emitted` matters;
* the last state is terminal and never bootstraps;
* `compute_targets` caches per unique ``(successor, actions)`` and encodes
  unique successors in one batch;
* `fit` reports the epochs it actually ran.
"""

from dataclasses import dataclass

import pytest
import torch

from controller_v3 import kaggle
from controller_v3.action import READ, Action
from controller_v3.config import ControllerConfig
from controller_v3.features import (
    NUMERIC_FEATURES,
    FeatureBuilder,
    History,
    HistoryTracker,
    RuntimeInfo,
    StateInput,
)
from controller_v3.labeler import ActionOutcome
from controller_v3.network import CAVCNetwork, WriteOutputs
from controller_v3.trainer import ActionTarget, CAVCTrainer, TrainConfig

SRC_WORDS = NUMERIC_FEATURES.index("src_words")
TGT_WORDS = NUMERIC_FEATURES.index("tgt_words")
SRC_CHARS = NUMERIC_FEATURES.index("src_chars")

WRITES = (Action(2, 4), Action(4, 6))


def make_state(index: int, src_words: int = 5, tgt_words: int = 3) -> StateInput:
    numeric = [0.0] * len(NUMERIC_FEATURES)
    numeric[SRC_WORDS] = float(src_words)
    numeric[TGT_WORDS] = float(tgt_words)
    numeric[SRC_CHARS] = float(index)
    return StateInput(
        numeric=tuple(numeric),
        predictor_id=index % 4,
        translator_id=(index + 1) % 4,
        src_ids=(2,),
        tgt_ids=(3,),
    )


def outcome(action: Action, words: tuple[str, ...], committed=()) -> ActionOutcome:
    return ActionOutcome(
        action=action,
        new_words=words,
        matched=len(words),
        incorrect=0,
        emitted=len(words),
        no_emit=not words,
        next_committed=tuple(committed) + words,
        pred_s=0.1,
        mt_s=0.2,
        vote_top1=0.75,
        vote_margin=0.5,
        vote_entropy=0.1,
    )


@dataclass(frozen=True)
class FakeSentence:
    sentence_id: str
    source: str
    reference: str
    split: str = "dev"


class FakeLabeler:
    """Two writes with different emissions plus a READ."""

    def oracle_words(self, source, committed):
        del source, committed
        return tuple("oracle final".split())

    def read_outcome(self, committed):
        return ActionOutcome(
            action=READ,
            new_words=(),
            matched=0,
            incorrect=0,
            emitted=0,
            no_emit=True,
            next_committed=tuple(committed),
            pred_s=0.0,
            mt_s=0.0,
            vote_top1=0.0,
            vote_margin=0.0,
            vote_entropy=0.0,
        )

    def label_state(self, observed, committed, source, actions, seed):
        del observed, source, seed
        emissions = {
            Action(2, 4): ("alpha", "beta"),
            Action(4, 6): ("gamma",),
        }
        return [
            outcome(action, emissions[action], tuple(committed))
            for action in actions
        ]


def make_cfg(**overrides) -> ControllerConfig:
    defaults = dict(seed=7, actions=WRITES)
    defaults.update(overrides)
    return ControllerConfig(**defaults)


def make_trainer(**cfg_overrides):
    cfg = make_cfg(**cfg_overrides)
    torch.manual_seed(cfg.seed)
    network = CAVCNetwork(cfg, src_vocab_size=30, tgt_vocab_size=30)
    return CAVCTrainer(network, cfg, None), cfg, network


# ----------------------------------------------------------------------
# Per-action successors
# ----------------------------------------------------------------------
def test_read_keeps_committed_and_advances_observed_while_writes_advance_committed():
    cfg = make_cfg()
    sentence = FakeSentence("s0", "a b c d e f", "x y z")
    builder = FeatureBuilder()
    row = kaggle.label_sentence(
        FakeLabeler(), sentence, stride=1, cfg=cfg, builder=builder,
        read_prob=0.0,
    )

    state_row = row["states"][0]
    assert state_row["successor_observed"] == "a b c d"  # read_pos=3 -> 4 words
    assert state_row["observed"] == "a b c"

    targets, states = kaggle._targets_from_rows(
        [row], builder, cfg, predictor_id=0, translator_id=0
    )
    first_state = states[0]
    by_action = {
        target.action: target
        for target in targets
        if target.state == first_state
    }

    committed_before = len(state_row["committed_before"])
    observed_words = len(state_row["observed"].split())

    # Every successor advances the observed source by exactly one word.
    for target in by_action.values():
        assert target.next_state is not None
        assert int(target.next_state.numeric[SRC_WORDS]) == observed_words + 1

    # READ keeps the committed prefix.
    read_next = by_action[READ].next_state
    assert int(read_next.numeric[TGT_WORDS]) == committed_before

    # The two writes reach different successors, and their emitted counts are
    # reflected in the committed length.
    write_two = by_action[Action(2, 4)].next_state
    write_one = by_action[Action(4, 6)].next_state
    assert write_two != write_one
    assert int(write_two.numeric[TGT_WORDS]) == committed_before + 2
    assert int(write_one.numeric[TGT_WORDS]) == committed_before + 1


# ----------------------------------------------------------------------
# READ can win the backup
# ----------------------------------------------------------------------
def _stub_heads(monkeypatch, trainer, *, read_value: float, write_value: float):
    def fake_read(z):
        return torch.full((z.size(0),), read_value)

    def fake_write(z, actions):
        batch, count = z.size(0), len(actions)
        zeros = torch.zeros(batch, count)
        return WriteOutputs(
            value=torch.full((batch, count), write_value),
            error_risk=zeros,
            no_emit=zeros,
            expected_emit=zeros + 0.5,
        )

    monkeypatch.setattr(trainer.target_network, "read_value", fake_read)
    monkeypatch.setattr(trainer.target_network, "write_values", fake_write)


def test_read_wins_the_backup_when_its_q_is_higher(monkeypatch):
    trainer, cfg, _network = make_trainer()
    _stub_heads(monkeypatch, trainer, read_value=10.0, write_value=-10.0)
    target = ActionTarget(
        state=make_state(0),
        action=WRITES[0],
        value_target=1.0,
        error_target=0.0,
        no_emit_target=0.0,
        emit_target=0.0,
        state_key="s0",
        next_state=make_state(1),
        next_actions=(),
        terminal=False,
    )

    computed = trainer.compute_targets([target])[0]

    # The default backup set is (READ, *cfg.actions); with no cost model READ
    # costs nothing, so its Q dominates.
    assert computed == pytest.approx(1.0 + cfg.gamma * 10.0)


def test_behavior_policy_can_produce_read_transitions():
    cfg = make_cfg()
    sentence = FakeSentence("s0", "a b c d e f", "x y z")
    row = kaggle.label_sentence(
        FakeLabeler(), sentence, stride=1, cfg=cfg, read_prob=1.0
    )
    assert {state["chosen"] for state in row["states"]} == {READ.label}
    assert all(
        state["committed_before"] == state["committed_after"]
        for state in row["states"]
    )


# ----------------------------------------------------------------------
# History replay
# ----------------------------------------------------------------------
def test_offline_history_replay_equals_online_construction():
    builder = FeatureBuilder(length_ratio=1.4)
    tracker = HistoryTracker()
    online: list[tuple[tuple[float, ...], History, RuntimeInfo, float, float]] = []

    steps = [
        ("a b c", (), "read"),
        ("a b c d", (), outcome(Action(2, 4), ("x", "y"))),
        ("a b c d e", ("x", "y"), outcome(Action(4, 6), ())),
        ("a b c d e f", ("x", "y"), "read"),
    ]
    for step_index, (observed, committed, event) in enumerate(steps):
        now_s = float(step_index)
        tracker.set_clock(now_s)
        vector = tracker.build(builder, observed, committed)
        online.append(
            (
                vector.numeric,
                tracker.history,
                tracker.runtime,
                tracker.last_commit_s,
                tracker.now_s,
            )
        )
        if event == "read":
            tracker.on_read(now_s=now_s)
        else:
            tracker.on_write(event, now_s=now_s)

    # Rebuild each state only from the persisted snapshot, exactly as the
    # labels JSONL path does.
    for (expected_numeric, history, runtime, last_commit_s, now_s), (
        observed,
        committed,
        _event,
    ) in zip(online, steps):
        replay = HistoryTracker(
            History(**{field: getattr(history, field) for field in history.__dataclass_fields__}),
            RuntimeInfo(**{field: getattr(runtime, field) for field in runtime.__dataclass_fields__}),
            last_commit_s=last_commit_s,
            now_s=now_s,
        )
        assert replay.build(builder, observed, committed).numeric == pytest.approx(
            expected_numeric
        )

    # Sanity: live history is not the all-zero default the old trainer used.
    assert online[1][0] != builder.build("a b c d", ()).numeric


def test_history_tracker_read_and_write_counters():
    tracker = HistoryTracker()
    tracker.on_read(now_s=1.0)
    assert tracker.history.last_empty is True
    assert tracker.history.last_branches == 0
    assert tracker.history.words_since_commit == 1

    tracker.on_write(outcome(Action(2, 4), ("x", "y")), now_s=2.0)
    assert tracker.history.last_commit_count == 2
    assert tracker.history.last_empty is False
    assert tracker.history.words_since_commit == 0
    assert tracker.history.last_branches == 2
    assert tracker.last_commit_s == 2.0
    assert tracker.runtime.time_since_commit == 0.0

    tracker.on_write(outcome(Action(4, 6), ()), now_s=3.0)
    assert tracker.history.last_commit_count == 0
    assert tracker.history.last_empty is True
    assert tracker.history.words_since_commit == 1
    assert tracker.runtime.time_since_commit == pytest.approx(1.0)


# ----------------------------------------------------------------------
# compute_targets: per-target successors, caching, batching
# ----------------------------------------------------------------------
def _bootstrap_target(successor: StateInput, value: float, actions=()) -> ActionTarget:
    return ActionTarget(
        state=make_state(0),
        action=WRITES[0],
        value_target=value,
        error_target=0.0,
        no_emit_target=0.0,
        emit_target=0.0,
        state_key="shared",
        next_state=successor,
        next_actions=actions,
        terminal=False,
    )


def test_compute_targets_uses_each_targets_own_successor(monkeypatch):
    trainer, cfg, _network = make_trainer()
    successors = [make_state(1), make_state(2), make_state(3)]
    seen: list[StateInput] = []

    def fake_best(z, state, actions):
        del z, actions
        seen.append(state)
        return float(state.numeric[SRC_CHARS])  # distinct per successor

    monkeypatch.setattr(trainer, "_best_next_value_from_z", fake_best)
    targets = [
        _bootstrap_target(successor, float(index))
        for index, successor in enumerate(successors)
    ]

    computed = trainer.compute_targets(targets)

    expected = [
        float(index) + cfg.gamma * float(successor.numeric[SRC_CHARS])
        for index, successor in enumerate(successors)
    ]
    assert computed == pytest.approx(expected)
    assert seen == successors


def test_compute_targets_caches_per_unique_successor_and_action_set(monkeypatch):
    trainer, cfg, _network = make_trainer()
    shared_a = make_state(1)
    shared_b = make_state(2)
    encode_calls = {"count": 0}
    bonus_calls = {"count": 0}
    original_encode = trainer._encode_states

    def counting_encode(model, states):
        encode_calls["count"] += 1
        return original_encode(model, states)

    def fake_best(z, state, actions):
        del z, actions
        bonus_calls["count"] += 1
        return float(state.numeric[SRC_CHARS])

    monkeypatch.setattr(trainer, "_encode_states", counting_encode)
    monkeypatch.setattr(trainer, "_best_next_value_from_z", fake_best)

    targets = [
        _bootstrap_target(shared_a, 0.0),
        _bootstrap_target(shared_a, 0.0),
        _bootstrap_target(shared_b, 0.0),
        _bootstrap_target(shared_b, 0.0),
    ]
    computed = trainer.compute_targets(targets)

    # One batch encode for both unique successors, one backup evaluation per
    # unique (successor, actions).
    assert encode_calls["count"] == 1
    assert bonus_calls["count"] == 2
    assert computed[0] == computed[1]
    assert computed[2] == computed[3]
    assert computed[0] != computed[2]


# ----------------------------------------------------------------------
# Stride and terminal
# ----------------------------------------------------------------------
def test_stride_one_is_a_true_one_step_transition_and_larger_stride_raises():
    cfg = make_cfg()
    sentence = FakeSentence("s0", "a b c d e f", "x y z")
    row = kaggle.label_sentence(
        FakeLabeler(), sentence, stride=1, cfg=cfg, read_prob=1.0
    )
    first = row["states"][0]
    assert first["successor_observed"].split() == first["observed"].split() + [
        "d"
    ]

    with pytest.raises(ValueError, match="one-step|stride"):
        kaggle.label_sentence(
            FakeLabeler(), sentence, stride=2, cfg=cfg
        )


def _build_terminal_targets(state, successor, cfg):
    from controller_v3.targets import build_action_targets

    return build_action_targets(
        state,
        [outcome(READ, ()), outcome(WRITES[0], ("x",))],
        cfg,
        next_states=[successor, successor],
        terminal=True,
        terminal_value=7.5,
        state_key="terminal",
    )


def test_terminal_target_has_no_bootstrap_and_uses_the_terminal_value():
    cfg = make_cfg()
    targets = _build_terminal_targets(make_state(0), make_state(1), cfg)

    assert all(target.terminal for target in targets)
    assert all(target.next_state is None for target in targets)
    assert all(target.next_actions == () for target in targets)
    assert [target.value_target for target in targets] == [7.5] * len(targets)

    trainer, _, _ = make_trainer()
    assert trainer.compute_targets(targets) == [7.5] * len(targets)


# ----------------------------------------------------------------------
# Delay depends on emitted
# ----------------------------------------------------------------------
def test_delay_changes_when_emitted_changes_the_post_action_backlog():
    from controller_v3.targets import build_action_targets

    cfg = make_cfg()
    state = make_state(0, src_words=4, tgt_words=0)
    # Equal quality (matched=0) so the only difference is the post-action
    # backlog; the write commits four words and closes most of it.
    no_emit = ActionOutcome(
        action=WRITES[0],
        new_words=(),
        matched=0,
        incorrect=0,
        emitted=0,
        no_emit=True,
        next_committed=(),
        pred_s=0.0,
        mt_s=0.0,
        vote_top1=0.0,
        vote_margin=0.0,
        vote_entropy=0.0,
    )
    commits = ActionOutcome(
        action=Action(2, 4),
        new_words=("a", "b", "c", "d"),
        matched=0,
        incorrect=0,
        emitted=4,
        no_emit=False,
        next_committed=("a", "b", "c", "d"),
        pred_s=0.0,
        mt_s=0.0,
        vote_top1=0.0,
        vote_margin=0.0,
        vote_entropy=0.0,
    )

    targets = build_action_targets(
        state,
        [no_emit, commits],
        cfg,
        next_states=[make_state(1), make_state(2)],
        src_words_after=5,
        state_key="delay",
        rho=1.0,
    )
    waits, writes = targets

    # The no-emit action leaves the full 5-word backlog; the write leaves 1.
    assert waits.value_target == pytest.approx(-cfg.lambda_l * 5.0)
    assert writes.value_target == pytest.approx(-cfg.lambda_l * 1.0)
    assert writes.value_target > waits.value_target


# ----------------------------------------------------------------------
# Zero-epoch fit
# ----------------------------------------------------------------------
def test_zero_epoch_fit_reports_zero_epochs():
    trainer, _, _ = make_trainer()
    trainer.train_config = TrainConfig(epochs=0, batch_size=8, lr=0.01)
    targets = [
        ActionTarget(
            state=make_state(0),
            action=WRITES[0],
            value_target=1.0,
            error_target=0.0,
            no_emit_target=0.0,
            emit_target=0.0,
            state_key="s0",
            terminal=True,
        )
    ]
    result = trainer.fit(targets)
    assert result["epochs"] == 0
    assert result["n_samples"] == 1
    assert result["final_loss"] == 0.0

    empty = trainer.fit([])
    assert empty["epochs"] == 0
    assert empty["n_samples"] == 0
