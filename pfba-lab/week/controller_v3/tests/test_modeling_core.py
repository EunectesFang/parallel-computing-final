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
    advance_virtual_clock,
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


class PrefixOracleLabeler:
    """Oracle continuation that depends on the forced committed prefix."""

    def oracle_words(self, source, committed):
        del source
        committed = tuple(committed)
        if committed == ("the", "cat"):
            return ("sits",)
        if committed == ("the", "cat", "sits"):
            return ()
        return ()


class PrefixTerminalLabeler(PrefixOracleLabeler):
    """Two writes with different correct prefixes and prefix-dependent oracles."""

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

    def oracle_words(self, source, committed):
        del source
        committed = tuple(committed)
        return {
            (): ("a", "b", "c"),
            # The shorter correct prefix continues with an extra word, so its
            # terminal length gap is worse even though the prefix is right.
            ("a",): ("b", "c", "d"),
            ("a", "b"): ("c",),
        }.get(committed, ())

    def label_state(self, observed, committed, source, actions, seed):
        del observed, source, seed
        emissions = {
            Action(2, 4): ("a",),
            Action(4, 6): ("a", "b"),
        }
        outcomes = []
        for action in actions:
            if action.is_read:
                outcomes.append(self.read_outcome(committed))
            else:
                outcomes.append(outcome(action, emissions[action], tuple(committed)))
        return outcomes


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

    # History must also be action-conditioned: READ reports no action, while
    # each write reports its own B/H. A shared behavior History would make all
    # three successors identical here.
    def feature(state, name):
        return state.numeric[NUMERIC_FEATURES.index(name)]

    assert feature(read_next, "last_B") == 0.0
    assert feature(read_next, "last_H") == 0.0
    assert feature(read_next, "last_empty") == 1.0
    assert feature(write_two, "last_B") == 2.0
    assert feature(write_two, "last_H") == 4.0
    assert feature(write_two, "last_empty") == 0.0
    assert feature(write_one, "last_B") == 4.0
    assert feature(write_one, "last_H") == 6.0
    assert feature(write_one, "last_empty") == 0.0


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
    captured: list[tuple[StateInput, torch.Tensor]] = []
    original_encode = trainer._encode_states

    def fake_best(z, state, actions):
        del actions
        seen.append(state)
        # Keep the row so the test can prove the batch slice is this
        # successor's encoding, not just any row.
        captured.append((state, z.clone()))
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
    for state, z in captured:
        expected_z = original_encode(trainer.target_network, [state])
        # Batch kernels can differ from a one-row encode at the 1e-8 level.
        assert torch.allclose(z, expected_z, atol=1e-6)


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


def test_terminal_proxy_includes_the_forced_committed_prefix():
    labeler = PrefixOracleLabeler()
    reward_cfg = kaggle._reward_config(make_cfg())
    reference = "the cat sits"

    short_prefix = kaggle._terminal_value_proxy(
        labeler, "src", ("the", "cat"), reference, reward_cfg
    )
    long_prefix = kaggle._terminal_value_proxy(
        labeler, "src", ("the", "cat", "sits"), reference, reward_cfg
    )

    # `oracle_words` returns only "sits", so the hypothesis must be
    # "the cat sits", not "sits". Both prefixes describe the same correct
    # sentence and score 0 rather than getting worse with a longer prefix.
    assert short_prefix == 0.0
    assert long_prefix == 0.0
    assert long_prefix >= short_prefix

    partial = kaggle._terminal_value_proxy(
        labeler, "src", ("the", "cat"), "the cat sits down", reward_cfg
    )
    assert partial == -1.0


def test_terminal_targets_are_conditioned_on_each_actions_prefix():
    cfg = make_cfg(
        min_start=3, actions=(Action(2, 4), Action(4, 6))
    )
    # One decision position (read_pos=3), so it is also the terminal state.
    sentence = FakeSentence("s0", "a b c d", "a b c")
    builder = FeatureBuilder()
    row = kaggle.label_sentence(
        PrefixTerminalLabeler(),
        sentence,
        stride=1,
        cfg=cfg,
        builder=builder,
        read_prob=0.0,
    )

    assert len(row["states"]) == 1
    state_row = row["states"][0]
    assert state_row["terminal"] is True
    serialized = {
        outcome["action"]: outcome["terminal_value"]
        for outcome in state_row["outcomes"]
    }
    # READ reuses the state oracle; each write uses its own committed prefix.
    assert serialized[READ.label] == 0.0
    assert serialized["b2h4"] == -1.0
    assert serialized["b4h6"] == 0.0

    targets, _states = kaggle._targets_from_rows(
        [row], builder, cfg, predictor_id=0, translator_id=0
    )
    values = {target.action: target.value_target for target in targets}
    assert values[READ] == 0.0
    assert values[Action(2, 4)] == -1.0
    assert values[Action(4, 6)] == 0.0
    # Two actions with different correct prefixes get different terminal
    # targets; the longer correct prefix does not lower the value.
    assert values[Action(2, 4)] != values[Action(4, 6)]
    assert values[Action(4, 6)] >= values[Action(2, 4)]


def test_max_states_stops_labeling_early_without_false_terminal():
    cfg = make_cfg()
    # Eight source words -> five decision positions (3..7). Stop at three.
    sentence = FakeSentence("s0", "a b c d e f g h", "x y z")
    row = kaggle.label_sentence(
        FakeLabeler(),
        sentence,
        stride=1,
        cfg=cfg,
        read_prob=0.0,
        max_states=3,
    )

    assert len(row["states"]) == 3
    assert [state["read_pos"] for state in row["states"]] == [3, 4, 5]
    # The stop position is not the true last decision position (7), so it must
    # keep its successor and must not pretend to be terminal.
    assert all(state["terminal"] is False for state in row["states"])
    assert all(state["terminal_value"] is None for state in row["states"])


def test_behavior_write_falls_back_to_the_first_configured_action():
    cfg = make_cfg(actions=(Action(2, 4),))
    sentence = FakeSentence("s0", "a b c d e f", "x y z")
    row = kaggle.label_sentence(
        FakeLabeler(), sentence, stride=1, cfg=cfg, read_prob=0.0
    )

    # (4,6) is absent from the grid, so the policy uses the first write rather
    # than appending an action the controller never evaluates.
    assert row["behavior"]["fixed_action"] == "b2h4"
    assert {
        outcome["action"] for outcome in row["states"][0]["outcomes"]
    } == {"read", "b2h4"}


def test_labeling_clock_matches_the_rollout_recurrence():
    cfg = make_cfg()
    sentence = FakeSentence("s0", "a b c d e f", "x y z")
    # read_prob=0.0 makes every behavior action the fixed write, whose
    # FakeLabeler timing is pred_s=0.1 + mt_s=0.2.
    row = kaggle.label_sentence(
        FakeLabeler(), sentence, stride=1, cfg=cfg, read_prob=0.0
    )

    # The exact recurrence rollout uses: settle the source arrival, then bill
    # the action's measured compute.
    expected = []
    clock = 0.0
    for read_pos in (3, 4, 5):
        expected.append(clock)
        clock = max(clock, (read_pos - 1) * cfg.arrival_gap) + 0.3

    assert [state["now_s"] for state in row["states"]] == pytest.approx(expected)
    # Every behavior write commits, so time_since_commit resets in both paths.
    assert all(
        state["runtime"]["time_since_commit"] == pytest.approx(0.0)
        for state in row["states"][1:]
    )
    # The shared helper is the same function rollout calls.
    assert advance_virtual_clock(0.0, 3, cfg.arrival_gap, 0.3) == pytest.approx(
        0.9
    )


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
