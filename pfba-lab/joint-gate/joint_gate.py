#!/usr/bin/env python3
"""Cheap joint READ / (B,h) gate for source-streaming TAF (small Qwen + Marian).

The gate is an offline-trained, online-cheap decision model.  At each source
position the online policy sees only features derivable from the *observed*
source prefix and the *already committed* target prefix:

    observed length/chars/punctuation, committed target words/chars, and
    optionally one cheap next-token entropy/max-probability probe.

It never sees the full source, the future, the sample source length, an oracle
translation, or work done by actions it did not take.

Offline labels are produced separately.  For a state (sentence, read position,
committed target prefix) every action in the grid is executed on the *same*
observed source and *same* forced target prefix: source futures are sampled,
Qwen futures plus the observed source are translated by Marian with the prefix
forced, and the exact upstream RALCP rule selects new words.  Those new words
are compared with the prefix continuation of the offline full-TRUE-source
translation.  Matched and incorrect committed words are recorded separately,
together with synchronized prediction + MT + aggregate time.  The oracle is
only used to create labels; it is never part of online features.

The action grid is B in {2,4} x h in {2,6}.  Learned policies and baselines:
joint (all four), B-only (h=6), h-only (B=4), and fixed (4,6).

This module is self-contained when ``run_pfba.py`` is placed next to it.  It
also falls back to ``../paired/run_pfba.py`` for in-repo development.  When the
real adapters are used, small Qwen is the predictor and Marian is the
translator.  Nothing here submits to a cloud service.

Usage (after the parent completes the phase-1 baselines; training is not run by
this module's tests)::

    D:/MINICONDA/envs/py312/python.exe joint_gate.py --dev-limit 10 \
        --test-limit 10 --metrics bleu

Tests use fakes only::

    D:/MINICONDA/envs/py312/python.exe -m unittest tests.test_joint_gate -v
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import string
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Upstream run_pfba loading (adapters, data, RALCP, timing, metrics)
# ---------------------------------------------------------------------------

_PFBA = None
_PFBA_TRIED = False


def _load_pfba(required: bool = False):
    """Load run_pfba.py from beside this file, else ../paired/run_pfba.py."""
    global _PFBA, _PFBA_TRIED
    if not _PFBA_TRIED:
        _PFBA_TRIED = True
        here = Path(__file__).resolve().parent
        candidates = (here / "run_pfba.py", here.parent / "paired" / "run_pfba.py")
        for path in candidates:
            if path.is_file():
                spec = importlib.util.spec_from_file_location("run_pfba_adjacent", path)
                module = importlib.util.module_from_spec(spec)
                sys.modules.setdefault("run_pfba_adjacent", module)
                spec.loader.exec_module(module)
                _PFBA = module
                break
    if _PFBA is None and required:
        raise RuntimeError(
            "run_pfba.py is required for real model adapters and official data; "
            "place it next to joint_gate.py (or in ../paired/run_pfba.py)")
    return _PFBA


# ---------------------------------------------------------------------------
# Actions and deterministic helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class Action:
    branches: int
    horizon: int

    @property
    def is_read(self) -> bool:
        return self.branches == 0 and self.horizon == 0

    @property
    def label(self) -> str:
        return "read" if self.is_read else f"b{self.branches}h{self.horizon}"

    @classmethod
    def from_label(cls, label: str) -> "Action":
        if label == "read":
            return READ
        if not (label.startswith("b") and "h" in label):
            raise ValueError(f"bad action label: {label!r}")
        branches, horizon = label[1:].split("h", 1)
        return cls(int(branches), int(horizon))


READ = Action(0, 0)
ACTION_GRID: tuple[Action, ...] = (
    Action(2, 2), Action(2, 6), Action(4, 2), Action(4, 6))
B_ONLY_ACTIONS: tuple[Action, ...] = (Action(2, 6), Action(4, 6))
H_ONLY_ACTIONS: tuple[Action, ...] = (Action(4, 2), Action(4, 6))
FIXED_BASELINE = Action(4, 6)
FIXED_DEV_REFERENCE = Action(2, 6)

_PUNCT = set(string.punctuation) | set("，。！？；：、…—–«»„“”‘’（）【】《》")


def upstream_ralcp(candidates: Sequence[str], threshold: float = 0.6) -> str:
    """Exact upstream agents/utils.py::ralcp; use the adjacent copy if present."""
    pfba = _load_pfba()
    if pfba is not None:
        return pfba.upstream_ralcp(candidates, threshold)
    # Faithful fallback for standalone CPU tests.
    if not candidates:
        return ""
    words, translation = [c.split(" ") for c in candidates], ""
    for i in range(max(len(w) for w in words)):
        counts = Counter(" ".join(w[:i + 1]) for w in words)
        best = max(counts, key=counts.get)
        if counts[best] < len(candidates) * threshold:
            break
        translation = best
    return translation


def derive_state_seed(base_seed: int, split: str, sentence_id: str,
                      read_pos: int) -> int:
    """Same deterministic per-sentence/word seed as run_pfba modes."""
    pfba = _load_pfba()
    if pfba is not None:
        return pfba.derive_step_seed(base_seed, split, sentence_id, read_pos)
    payload = f"{int(base_seed)}|{split}|{sentence_id}|{int(read_pos)}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def arrived_words(elapsed_s: float, gap_s: float, source_len: int) -> int:
    pfba = _load_pfba()
    if pfba is not None:
        return pfba.arrived_words(elapsed_s, gap_s, source_len)
    if source_len <= 0:
        return 0
    return source_len if gap_s <= 0 else min(source_len, int(elapsed_s / gap_s + 1e-9) + 1)


def laal_score(delays: Sequence[int], source_len: int, reference_len: int,
               prediction_len: int) -> float | None:
    pfba = _load_pfba()
    if pfba is not None:
        return pfba.laal_score(delays, source_len, reference_len, prediction_len)
    used = []
    for delay in delays:
        used.append(float(delay))
        if delay >= source_len:
            break
    target_len = max(int(reference_len), int(prediction_len))
    if not used or target_len <= 0:
        return None
    slope = float(source_len) / target_len
    return sum(d - i * slope for i, d in enumerate(used)) / len(used)


class NullTimer:
    def __init__(self):
        self.seconds: dict[str, float] = {}

    def run(self, name, fn):
        return fn(), 0.0


@dataclass(frozen=True)
class Sentence:
    sentence_id: str
    source: str
    reference: str


# ---------------------------------------------------------------------------
# Online features (only observed source prefix + committed target prefix)
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Observed-surface features.  The signature deliberately has no source or future."""

    def __init__(self, entropy_fn: Callable[[str], tuple[float, float]] | None = None):
        self.entropy_fn = entropy_fn

    def feature_names(self) -> tuple[str, ...]:
        names = [
            "observed_len", "observed_chars", "observed_punct",
            "observed_punct_ratio", "observed_ends_punct",
            "committed_words", "committed_chars",
        ]
        if self.entropy_fn is not None:
            names.extend(("entropy", "max_prob"))
        return tuple(names)

    def extract(self, observed: str, committed: Sequence[str],
                timer: Any | None = None) -> tuple[dict[str, float], float]:
        def compute() -> dict[str, float]:
            normalized = " ".join(observed.split())
            punct = sum(1 for ch in normalized if ch in _PUNCT)
            words = normalized.split()
            committed_text = " ".join(committed)
            features = {
                "observed_len": float(len(words)),
                "observed_chars": float(len(normalized)),
                "observed_punct": float(punct),
                "observed_punct_ratio": float(punct / max(1, len(normalized))),
                "observed_ends_punct": float(bool(normalized) and normalized[-1] in _PUNCT),
                "committed_words": float(len(committed)),
                "committed_chars": float(len(committed_text)),
            }
            if self.entropy_fn is not None:
                entropy, max_prob = self.entropy_fn(observed)
                features["entropy"] = float(entropy)
                features["max_prob"] = float(max_prob)
            return features

        if timer is None:
            start = time.perf_counter()
            features = compute()
            return features, max(0.0, time.perf_counter() - start)
        return timer.run("features", compute)


# ---------------------------------------------------------------------------
# Offline action execution and oracle-based labels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionOutcome:
    action: Action
    new_words: tuple[str, ...]
    n_candidates: int
    predict_s: float
    translate_s: float
    aggregate_s: float
    total_s: float


def execute_action(predictor, translator, observed: str, committed: Sequence[str],
                   action: Action, seed: int, ralcp_threshold: float = 0.6,
                   timer: Any | None = None) -> ActionOutcome:
    """Execute one action on the actual observed source and forced target prefix."""
    timer = timer or NullTimer()
    if action.is_read:
        return ActionOutcome(action, (), 0, 0.0, 0.0, 0.0, 0.0)
    prefix = " ".join(committed)
    continuations, predict_s = timer.run(
        "predict", lambda: predictor.sample(observed, action.branches, action.horizon, seed))
    futures = [observed if not c.strip() else f"{observed} {c.strip()}"
               for c in (continuations or [])]
    requests = futures or [observed]
    outs, translate_s = timer.run(
        "translate", lambda: translator.translate_batch(list(requests), prefix))
    selected, aggregate_s = timer.run(
        "aggregate", lambda: upstream_ralcp(list(outs), ralcp_threshold))
    new_words = tuple(selected.split()) if selected else ()
    return ActionOutcome(
        action=action, new_words=new_words, n_candidates=len(requests),
        predict_s=predict_s, translate_s=translate_s, aggregate_s=aggregate_s,
        total_s=predict_s + translate_s + aggregate_s)


def oracle_continuation(translator, source: str, committed: Sequence[str],
                        timer: Any | None = None) -> tuple[tuple[str, ...], float]:
    """Offline-only: full-TRUE-source translation with the same forced prefix."""
    timer = timer or NullTimer()
    prefix = " ".join(committed)
    outs, seconds = timer.run(
        "oracle_translate", lambda: translator.translate_batch([source], prefix))
    return tuple((outs[0] if outs else "").split()), seconds


def common_prefix_length(left: Sequence[str], right: Sequence[str]) -> int:
    matched = 0
    for a, b in zip(left, right):
        if a != b:
            break
        matched += 1
    return matched


def score_commit(new_words: Sequence[str], oracle_words: Sequence[str],
                 incorrect_penalty: float = 1.0) -> tuple[int, int, int, float]:
    """Return (matched, incorrect, success, utility) for one committed suffix."""
    matched = common_prefix_length(new_words, oracle_words)
    incorrect = len(new_words) - matched
    success = int(matched > 0)
    utility = float(matched) - float(incorrect_penalty) * float(incorrect)
    return matched, incorrect, success, utility


@dataclass(frozen=True)
class State:
    sentence_id: str
    split: str
    read_pos: int
    observed: str
    committed: tuple[str, ...]
    features: dict[str, float]
    feature_s: float


@dataclass(frozen=True)
class TrainingRow:
    sentence_id: str
    split: str
    read_pos: int
    observed: str
    committed: tuple[str, ...]
    features: dict[str, float]
    feature_s: float
    action: str
    new_words: tuple[str, ...]
    matched: int
    incorrect: int
    success: int
    utility: float
    predict_s: float
    translate_s: float
    total_s: float


def label_state(state: State, source: str, predictor, translator,
                actions: Sequence[Action] = ACTION_GRID, seed: int = 13,
                ralcp_threshold: float = 0.6, timer: Any | None = None,
                incorrect_penalty: float = 1.0) -> list[TrainingRow]:
    """Label all actions for one state; the oracle is computed once per state."""
    timer = timer or NullTimer()
    oracle_words, _ = oracle_continuation(translator, source, state.committed, timer)
    rows: list[TrainingRow] = []
    action_seed = derive_state_seed(seed, state.split, state.sentence_id, state.read_pos)
    for action in actions:
        outcome = execute_action(
            predictor, translator, state.observed, state.committed, action,
            action_seed, ralcp_threshold, timer)
        matched, incorrect, success, utility = score_commit(
            outcome.new_words, oracle_words, incorrect_penalty)
        rows.append(TrainingRow(
            sentence_id=state.sentence_id, split=state.split,
            read_pos=state.read_pos, observed=state.observed,
            committed=state.committed, features=dict(state.features),
            feature_s=float(state.feature_s), action=action.label,
            new_words=outcome.new_words, matched=matched, incorrect=incorrect,
            success=success, utility=utility,
            predict_s=outcome.predict_s, translate_s=outcome.translate_s,
            total_s=outcome.total_s))
    return rows


def label_states(states: Sequence[State], sentences_by_id: dict[str, Sentence],
                 predictor, translator, actions: Sequence[Action] = ACTION_GRID,
                 seed: int = 13, ralcp_threshold: float = 0.6,
                 timer: Any | None = None,
                 incorrect_penalty: float = 1.0) -> list[TrainingRow]:
    rows: list[TrainingRow] = []
    for state in states:
        sentence = sentences_by_id[state.sentence_id]
        rows.extend(label_state(
            state, sentence.source, predictor, translator, actions, seed,
            ralcp_threshold, timer, incorrect_penalty))
    return rows


# ---------------------------------------------------------------------------
# Deterministic behaviour policies / state collection
# ---------------------------------------------------------------------------

class FixedBehavior:
    def __init__(self, action: Action):
        self.action = action
        self.name = f"fixed_{action.label}"

    def decide(self, features: dict[str, float]) -> Action:
        return self.action


class PeriodicBehavior:
    """READ unless the observed word count is a multiple of ``every``."""

    def __init__(self, action: Action, every: int = 3):
        if every < 1:
            raise ValueError("every must be >= 1")
        self.action = action
        self.every = every
        self.name = f"periodic{every}_{action.label}"

    def decide(self, features: dict[str, float]) -> Action:
        observed_len = int(features["observed_len"])
        return self.action if observed_len and observed_len % self.every == 0 else READ


def _dedupe_states(states: Sequence[State]) -> list[State]:
    seen, unique = set(), []
    for state in states:
        key = (state.sentence_id, state.read_pos, state.committed)
        if key not in seen:
            seen.add(key)
            unique.append(state)
    return unique


def _limit_per_sentence(states: Sequence[State], limit: int) -> list[State]:
    if limit <= 0:
        return list(states)
    grouped: dict[str, list[State]] = defaultdict(list)
    for state in states:
        grouped[state.sentence_id].append(state)
    kept: list[State] = []
    for sentence_id in sorted(grouped):
        group = sorted(grouped[sentence_id], key=lambda s: (s.read_pos, s.committed))
        if len(group) <= limit:
            kept.extend(group)
            continue
        if limit == 1:
            kept.append(group[0])
            continue
        indices = sorted({int(i * (len(group) - 1) / (limit - 1) + 0.5)
                          for i in range(limit)})
        kept.extend(group[i] for i in indices)
    return kept


def collect_states(sentences: Sequence[Sentence], split: str, predictor,
                   translator, behaviors: Sequence[Any], extractor: FeatureExtractor,
                   cfg: "GateConfig", timer: Any | None = None,
                   max_states_per_sentence: int | None = None) -> list[State]:
    """Deterministic behaviour-policy rollouts, collecting pre-action states."""
    timer = timer or NullTimer()
    limit = (cfg.max_states_per_sentence if max_states_per_sentence is None
             else int(max_states_per_sentence))
    collected: list[State] = []
    for sentence in sentences:
        words = sentence.source.split()
        n = len(words)
        for behavior in behaviors:
            committed: list[str] = []
            read_pos = min(cfg.min_start, n)
            while read_pos < n:
                observed = " ".join(words[:read_pos])
                features, feature_s = extractor.extract(observed, tuple(committed), timer)
                collected.append(State(
                    sentence_id=sentence.sentence_id, split=split,
                    read_pos=read_pos, observed=observed,
                    committed=tuple(committed), features=features,
                    feature_s=feature_s))
                action = behavior.decide(features)
                if not action.is_read:
                    action_seed = derive_state_seed(
                        cfg.seed, split, sentence.sentence_id, read_pos)
                    outcome = execute_action(
                        predictor, translator, observed, tuple(committed), action,
                        action_seed, cfg.ralcp_threshold, timer)
                    committed.extend(outcome.new_words)
                read_pos += 1
    states = _limit_per_sentence(_dedupe_states(collected), limit)
    if cfg.max_states and len(states) > cfg.max_states:
        states = states[: cfg.max_states]
    return states


def split_dev_sentences(sentences: Sequence[Sentence], tune_fraction: float = 0.2,
                        seed: int = 13) -> dict[str, str]:
    """Sentence-disjoint deterministic dev split into train and tune."""
    ids = sorted(s.sentence_id for s in sentences)
    if len(ids) < 2:
        raise ValueError("need at least two dev sentences for a train/tune split")
    rng = np.random.RandomState(int(seed))
    order = list(ids)
    rng.shuffle(order)
    n_tune = int(round(len(order) * float(tune_fraction)))
    n_tune = max(1, min(len(order) - 1, n_tune))
    tune = set(order[:n_tune])
    return {sid: ("tune" if sid in tune else "train") for sid in ids}


# ---------------------------------------------------------------------------
# Estimators and joint gate
# ---------------------------------------------------------------------------

class ConstantProbability:
    """Small-sample fallback; keeps calibrated probability semantics."""

    def __init__(self, probability: float):
        self.probability = float(probability)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = np.full((len(X), 1), self.probability, dtype=float)
        return np.concatenate([1.0 - p, p], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@dataclass
class ActionEstimator:
    success: Any
    utility: Any
    cost: Any


@dataclass
class ActionConditionedConfidence:
    shared: Any
    action: Action

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        action_features = np.tile([self.action.branches, self.action.horizon], (len(features), 1))
        return self.shared.predict_proba(np.column_stack((features, action_features)))


@dataclass
class ThresholdSelection:
    threshold: float
    tune_reward: float
    compute_fraction: float
    n_states: int


@dataclass
class GateModel:
    actions: tuple[Action, ...]
    estimators: dict[Action, ActionEstimator]
    feature_names: tuple[str, ...]
    threshold: float
    cost_weight: float
    incorrect_penalty: float
    seed: int
    selection: ThresholdSelection | None = None

    def _matrix(self, features: dict[str, float]) -> np.ndarray:
        return np.array([[float(features.get(name, 0.0))
                          for name in self.feature_names]], dtype=float)

    def score_actions(self, features: dict[str, float]) -> dict[Action, float]:
        x = self._matrix(features)
        scores: dict[Action, float] = {}
        for action in self.actions:
            estimator = self.estimators[action]
            probability = float(estimator.success.predict_proba(x)[0, 1])
            conditioned = np.concatenate((np.repeat(x, 2, axis=0), [[0.0], [1.0]]), axis=1)
            failure_utility, success_utility = estimator.utility.predict(conditioned)
            utility = (1.0 - probability) * failure_utility + probability * success_utility
            cost = max(0.0, float(estimator.cost.predict(x)[0]))
            scores[action] = float(utility) - self.cost_weight * cost
        return scores

    def decide(self, features: dict[str, float]) -> Action:
        scores = self.score_actions(features)
        best = max(self.actions, key=lambda action: scores[action])
        return best if scores[best] >= self.threshold else READ


def _feature_names(rows: Sequence[TrainingRow]) -> tuple[str, ...]:
    names: set[str] = set()
    for row in rows:
        names.update(row.features)
    if not names:
        raise ValueError("no features found in training rows")
    return tuple(sorted(names))


def _fit_success(X: np.ndarray, y: np.ndarray, sentence_ids: Sequence[str], seed: int,
                 neural: bool = False):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = y.astype(int)
    if len(y) < 4 or len(np.unique(y)) < 2:
        return ConstantProbability(float(np.mean(y)) if len(y) else 0.0)
    min_class = int(min(np.sum(y == 0), np.sum(y == 1)))
    classifier = (MLPClassifier(hidden_layer_sizes=(12,), activation="tanh", solver="lbfgs", alpha=2.0,
                                max_iter=800, random_state=seed)
                  if neural else LogisticRegression(max_iter=2000, random_state=seed))
    base = make_pipeline(StandardScaler(), classifier)
    for folds in range(min(3, min_class, len(set(sentence_ids))), 1, -1):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y, sentence_ids))
        if all(len(np.unique(y[train])) == len(np.unique(y[calibration])) == 2
               for train, calibration in splits):
            model = CalibratedClassifierCV(base, method="sigmoid", cv=splits)
            model.fit(X, y)
            return model
    return ConstantProbability(float(np.mean(y)))


def _fit_regressor(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X, y.astype(float))
    return model


def validate_sentence_disjoint(*row_groups: Sequence[TrainingRow]) -> None:
    seen: dict[str, int] = {}
    for index, rows in enumerate(row_groups):
        ids = {row.sentence_id for row in rows}
        for sentence_id in ids:
            if sentence_id in seen and seen[sentence_id] != index:
                raise ValueError(f"sentence {sentence_id!r} appears in multiple splits")
            seen[sentence_id] = index


def fit_gate_model(train_rows: Sequence[TrainingRow], tune_rows: Sequence[TrainingRow],
                   actions: Sequence[Action] = ACTION_GRID, seed: int = 13,
                   cost_weight: float = 0.1, incorrect_penalty: float = 1.0,
                   *, neural: bool = False) -> GateModel:
    if any(row.split != "train" for row in train_rows):
        raise ValueError("fit rows must have split='train'")
    if any(row.split != "tune" for row in tune_rows):
        raise ValueError("tune rows must have split='tune'")
    if not train_rows or not tune_rows:
        raise ValueError("train and tune rows are both required")
    validate_sentence_disjoint(train_rows, tune_rows)

    actions = tuple(actions)
    names = _feature_names(train_rows)
    shared = None
    if neural:
        action_features = {action.label: (action.branches, action.horizon) for action in actions}
        rows = [row for row in train_rows if row.action in action_features]
        X = np.array([[*(float(row.features.get(name, 0.0)) for name in names),
                       *action_features[row.action]] for row in rows], dtype=float)
        shared = _fit_success(X, np.array([row.success for row in rows]),
                              [row.sentence_id for row in rows], seed, neural=True)
    estimators: dict[Action, ActionEstimator] = {}
    for action in actions:
        rows = [row for row in train_rows if row.action == action.label]
        if not rows:
            raise ValueError(f"no training rows for action {action.label}")
        X = np.array([[float(row.features.get(name, 0.0)) for name in names]
                      for row in rows], dtype=float)
        targets = np.array([row.success for row in rows])
        success = (ActionConditionedConfidence(shared, action) if shared is not None
                   else _fit_success(X, targets, [row.sentence_id for row in rows], seed))
        utility = _fit_regressor(np.column_stack((X, targets)), np.array([row.utility for row in rows]))
        cost = _fit_regressor(X, np.array([row.total_s for row in rows]))
        estimators[action] = ActionEstimator(success, utility, cost)

    model = GateModel(
        actions=actions, estimators=estimators, feature_names=names,
        threshold=float("-inf"), cost_weight=float(cost_weight),
        incorrect_penalty=float(incorrect_penalty), seed=int(seed))
    selection = choose_threshold(model, tune_rows)
    model.threshold = selection.threshold
    model.selection = selection
    return model


def _state_groups(rows: Sequence[TrainingRow]) -> dict[tuple[str, int, tuple[str, ...]], list[TrainingRow]]:
    grouped: dict[tuple[str, int, tuple[str, ...]], list[TrainingRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.sentence_id, row.read_pos, row.committed)].append(row)
    return grouped


def choose_threshold(model: GateModel, tune_rows: Sequence[TrainingRow]) -> ThresholdSelection:
    """Choose the READ/compute score threshold using tune labels only."""
    if not tune_rows:
        raise ValueError("tune rows are required")
    if any(row.split != "tune" for row in tune_rows):
        raise ValueError("threshold selection is tune-only")
    wanted = {action.label for action in model.actions}
    grouped = _state_groups([row for row in tune_rows if row.action in wanted])
    if not grouped:
        raise ValueError("no tune rows for the model's actions")

    states = []
    for rows in grouped.values():
        by_action = {row.action: row for row in rows if row.action in wanted}
        if not by_action:
            continue
        scores = model.score_actions(rows[0].features)
        available = tuple(action for action in model.actions if action.label in by_action)
        states.append((scores, by_action, available))
    if not states:
        raise ValueError("no usable tune states")

    all_scores = [score for scores, _, _ in states for score in scores.values()]
    candidates = sorted(set(all_scores + [min(all_scores) - 1.0, max(all_scores) + 1.0]))

    def evaluate(threshold: float) -> tuple[float, float]:
        reward, compute = 0.0, 0
        for scores, by_action, available in states:
            best = max(available, key=lambda action: scores[action])
            if scores[best] < threshold:
                continue
            row = by_action[best.label]
            reward += row.utility - model.cost_weight * row.total_s
            compute += 1
        return reward / len(states), compute / len(states)

    best_threshold, best_reward, best_fraction = candidates[0], float("-inf"), 0.0
    for threshold in candidates:
        reward, fraction = evaluate(threshold)
        if reward > best_reward or (reward == best_reward and threshold > best_threshold):
            best_threshold, best_reward, best_fraction = threshold, reward, fraction
    return ThresholdSelection(
        threshold=float(best_threshold), tune_reward=float(best_reward),
        compute_fraction=float(best_fraction), n_states=len(states))


def evaluate_offline(model: GateModel, rows: Sequence[TrainingRow],
                     expected_split: str = "test") -> dict[str, Any]:
    """Held-out offline scoring; never used to choose the threshold."""
    if not rows:
        raise ValueError("rows are required")
    if any(row.split != expected_split for row in rows):
        raise ValueError(f"offline evaluation requires split={expected_split!r} rows")
    wanted = {action.label for action in model.actions}
    scored_rows = [row for row in rows if row.action in wanted]
    grouped = _state_groups(scored_rows)
    by_label = {action.label: estimator for action, estimator in model.estimators.items()}
    squared_errors = [
        (float(by_label[row.action].success.predict_proba(model._matrix(row.features))[0, 1])
         - row.success) ** 2
        for row in scored_rows
    ]
    histogram: Counter[str] = Counter()
    rewards, matched, incorrect, costs, success = [], [], [], [], []
    compute = 0
    for key in sorted(grouped):
        state_rows = grouped[key]
        by_action = {row.action: row for row in state_rows if row.action in wanted}
        if not by_action:
            continue
        decision = model.decide(state_rows[0].features)
        if decision is READ or decision.label not in by_action:
            histogram["read"] += 1
            rewards.append(0.0)
            matched.append(0)
            incorrect.append(0)
            costs.append(0.0)
            continue
        row = by_action[decision.label]
        histogram[decision.label] += 1
        rewards.append(row.utility - model.cost_weight * row.total_s)
        matched.append(row.matched)
        incorrect.append(row.incorrect)
        costs.append(row.total_s)
        success.append(row.success)
        compute += 1
    n = len(rewards)
    return {
        "n_states": n,
        "compute_rate": compute / n if n else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_matched": float(np.mean(matched)) if matched else 0.0,
        "mean_incorrect": float(np.mean(incorrect)) if incorrect else 0.0,
        "mean_cost_s": float(np.mean(costs)) if costs else 0.0,
        "success_rate": float(np.mean(success)) if success else 0.0,
        "brier_score": float(np.mean(squared_errors)) if squared_errors else None,
        "brier_rows": len(squared_errors),
        "action_histogram": dict(sorted(histogram.items())),
        "threshold": float(model.threshold),
    }


# ---------------------------------------------------------------------------
# Online rollout policies
# ---------------------------------------------------------------------------

class FixedActionPolicy:
    def __init__(self, action: Action):
        self.action = action
        self.name = f"fixed_{action.branches}_{action.horizon}"

    def decide(self, features: dict[str, float]) -> Action:
        return self.action


class LearnedGatePolicy:
    def __init__(self, model: GateModel, name: str = "learned"):
        self.model = model
        self.name = name

    def decide(self, features: dict[str, float]) -> Action:
        return self.model.decide(features)


@dataclass
class OnlineStep:
    read_pos: int
    action: str
    observed_len: int
    committed_after: int
    new_words: tuple[str, ...]
    feature_s: float
    predict_s: float
    translate_s: float
    total_s: float
    delay: int


@dataclass
class RolloutTrace:
    sentence_id: str
    split: str
    policy: str
    source: str
    reference: str
    hypothesis: str
    committed: tuple[str, ...]
    delays: tuple[int, ...]
    steps: tuple[OnlineStep, ...]
    laal: float | None
    completion_s: float
    inference_s: float
    feature_s: float
    predict_s: float
    translate_s: float
    first_output_s: float | None
    action_histogram: dict[str, int]


@dataclass
class GateConfig:
    seed: int = 13
    min_start: int = 3
    arrival_gap: float = 0.3
    ralcp_threshold: float = 0.6
    incorrect_penalty: float = 1.0
    cost_weight: float = 0.1
    tune_fraction: float = 0.2
    max_states_per_sentence: int = 4
    max_states: int = 0
    entropy_features: bool = False
    neural: bool = False
    device: str = "cpu"


def run_online_rollout(sentence: Sentence, predictor, translator, policy,
                       extractor: FeatureExtractor, cfg: GateConfig,
                       split: str = "test", timer: Any | None = None) -> RolloutTrace:
    """Full online rollout; the policy only ever receives feature dictionaries."""
    timer = timer or NullTimer()
    words = sentence.source.split()
    n = len(words)
    committed: list[str] = []
    delays: list[int] = []
    steps: list[OnlineStep] = []
    histogram: Counter[str] = Counter()
    virtual_s = 0.0
    first_output_s = None
    feature_s = predict_s = translate_s = aggregate_s = 0.0

    if n == 0:
        return RolloutTrace(
            sentence.sentence_id, split, policy.name, sentence.source,
            sentence.reference, "", (), (), (), None, 0.0, 0.0, 0.0,
            0.0, 0.0, None, {})

    read_pos = min(cfg.min_start, n)
    while read_pos < n:
        observed = " ".join(words[:read_pos])
        if isinstance(policy, FixedActionPolicy):
            action, elapsed = policy.action, 0.0
        else:
            features, elapsed = extractor.extract(observed, tuple(committed), timer)
            action, decision_s = timer.run("decision", lambda: policy.decide(features))
            elapsed += decision_s
        feature_s += elapsed
        virtual_s = max(virtual_s, (read_pos - 1) * cfg.arrival_gap) + elapsed
        if action.is_read:
            histogram["read"] += 1
            delay = arrived_words(virtual_s, cfg.arrival_gap, n)
            steps.append(OnlineStep(
                read_pos, "read", len(observed.split()), len(committed), (),
                elapsed, 0.0, 0.0, elapsed, delay))
        else:
            action_seed = derive_state_seed(
                cfg.seed, split, sentence.sentence_id, read_pos)
            outcome = execute_action(
                predictor, translator, observed, tuple(committed), action,
                action_seed, cfg.ralcp_threshold, timer)
            predict_s += outcome.predict_s
            translate_s += outcome.translate_s
            aggregate_s += outcome.aggregate_s
            virtual_s += outcome.total_s
            delay = arrived_words(virtual_s, cfg.arrival_gap, n)
            if outcome.new_words and first_output_s is None:
                first_output_s = virtual_s
            committed.extend(outcome.new_words)
            delays.extend([read_pos] * len(outcome.new_words))
            histogram[action.label] += 1
            steps.append(OnlineStep(
                read_pos, action.label, len(observed.split()), len(committed),
                outcome.new_words, elapsed, outcome.predict_s,
                outcome.translate_s, outcome.total_s, delay))
        read_pos += 1

    # EOS: the true full source is translated once with the forced prefix.
    prefix = " ".join(committed)
    virtual_s = max(virtual_s, (n - 1) * cfg.arrival_gap)
    outs, elapsed = timer.run(
        "translate", lambda: translator.translate_batch([sentence.source], prefix))
    translate_s += elapsed
    virtual_s += elapsed
    new_words = tuple(outs[0].split()) if outs else ()
    delay = arrived_words(virtual_s, cfg.arrival_gap, n)
    if new_words and first_output_s is None:
        first_output_s = virtual_s
    committed.extend(new_words)
    delays.extend([delay] * len(new_words))
    histogram["eos"] += 1
    steps.append(OnlineStep(
        n, "eos", n, len(committed), new_words, 0.0, 0.0, elapsed,
        elapsed, delay))

    return RolloutTrace(
        sentence_id=sentence.sentence_id, split=split, policy=policy.name,
        source=sentence.source, reference=sentence.reference,
        hypothesis=" ".join(committed), committed=tuple(committed),
        delays=tuple(delays), steps=tuple(steps),
        laal=laal_score(delays, n, len(sentence.reference.split()), len(committed)),
        completion_s=virtual_s,
        inference_s=feature_s + predict_s + translate_s + aggregate_s,
        feature_s=feature_s, predict_s=predict_s, translate_s=translate_s,
        first_output_s=first_output_s,
        action_histogram=dict(sorted(histogram.items())))


# ---------------------------------------------------------------------------
# Metrics work using full-source data
# ---------------------------------------------------------------------------

class MetricUnavailableError(RuntimeError):
    pass


def ensure_metrics_available(metrics: Sequence[str],
                             availability: dict[str, bool] | None = None) -> None:
    for name in metrics:
        if availability is not None:
            available = bool(availability.get(name, False))
            reason = "not available"
        elif name == "bleu":
            available = importlib.util.find_spec("sacrebleu") is not None
            reason = "sacrebleu is not installed"
        elif name == "comet":
            available = importlib.util.find_spec("comet") is not None
            reason = "comet is not installed"
        else:
            raise ValueError(f"unknown metric {name!r}")
        if not available:
            raise MetricUnavailableError(f"requested metric {name!r} is unavailable: {reason}")


def _default_bleu_fn(hypotheses, references):
    return _load_pfba(required=True).corpus_bleu(list(hypotheses), list(references))


def _default_comet_fn(hypotheses, sources, references, device="cpu"):
    pfba = _load_pfba(required=True)
    return pfba.comet_score(list(hypotheses), list(sources), list(references),
                            pfba.DEFAULT_COMET, device)


def rollout_metrics(traces: Sequence[RolloutTrace], metrics: Sequence[str] = ("bleu",),
                    bleu_fn: Callable | None = None,
                    comet_fn: Callable | None = None,
                    device: str = "cpu") -> dict[str, Any]:
    """Corpus metrics; fail loudly if a requested metric is unavailable."""
    if not traces:
        raise ValueError("at least one rollout trace is required")
    hypotheses = [trace.hypothesis for trace in traces]
    references = [trace.reference for trace in traces]
    sources = [trace.source for trace in traces]
    result: dict[str, Any] = {"n": len(traces), "metrics": list(metrics)}

    for name in metrics:
        if name == "bleu":
            fn = bleu_fn or _default_bleu_fn
            score = fn(hypotheses, references)
            if score.get("status") != "ok":
                raise MetricUnavailableError(
                    f"BLEU unavailable: {score.get('reason', score)}")
            result["bleu"] = score
        elif name == "comet":
            fn = comet_fn or _default_comet_fn
            score = fn(hypotheses, sources, references, device=device)
            if score.get("status") != "ok":
                raise MetricUnavailableError(
                    f"COMET unavailable: {score.get('reason', score)}")
            result["comet"] = score
        else:
            raise ValueError(f"unknown metric {name!r}")

    laals = [trace.laal for trace in traces if trace.laal is not None]
    completions = sorted(float(trace.completion_s) for trace in traces)
    histogram: Counter[str] = Counter()
    for trace in traces:
        histogram.update(trace.action_histogram)
    result.update({
        "laal": float(np.mean(laals)) if laals else None,
        "completion_s": {
            "mean": float(np.mean(completions)),
            "p95": float(completions[(95 * len(completions) - 1) // 100]),
        },
        # Name used by the task: virtual replay completion.
        "virtual_replay_completion_s": {
            "mean": float(np.mean(completions)),
            "p95": float(completions[(95 * len(completions) - 1) // 100]),
        },
        "inference_s": {
            "mean": float(np.mean([trace.inference_s for trace in traces])),
            "total": float(np.sum([trace.inference_s for trace in traces])),
        },
        "feature_s": float(np.sum([trace.feature_s for trace in traces])),
        "predict_s": float(np.sum([trace.predict_s for trace in traces])),
        "translate_s": float(np.sum([trace.translate_s for trace in traces])),
        "action_histogram": dict(sorted(histogram.items())),
    })
    return result


# ---------------------------------------------------------------------------
# Checkpoints and persistence
# ---------------------------------------------------------------------------

def save_checkpoint(model: GateModel, path: str | Path) -> Path:
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_checkpoint(path: str | Path) -> GateModel:
    import joblib
    return joblib.load(Path(path))


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = asdict(row) if hasattr(row, "__dataclass_fields__") else row
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Feasibility pipeline (injectable adapters make it CPU-testable)
# ---------------------------------------------------------------------------

def _default_behaviors() -> tuple[Any, ...]:
    return (
        FixedBehavior(Action(2, 2)),
        FixedBehavior(FIXED_BASELINE),
        PeriodicBehavior(FIXED_BASELINE, every=3),
    )


def make_qwen_entropy_fn(predictor):
    """One cheap forward pass on the observed prefix; returns (entropy, max_prob)."""
    def entropy_fn(observed: str) -> tuple[float, float]:
        import torch
        inputs = predictor.tok(
            observed, return_tensors="pt", add_special_tokens=False).to(predictor.device)
        with torch.inference_mode():
            logits = predictor.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
            max_prob = float(probs.max())
        return entropy, max_prob
    return entropy_fn


def run_feasibility(dev_sentences: Sequence[Sentence],
                    test_sentences: Sequence[Sentence], predictor, translator,
                    output_dir: str | Path, cfg: GateConfig,
                    behaviors: Sequence[Any] | None = None,
                    metrics: Sequence[str] = ("bleu",),
                    bleu_fn: Callable | None = None,
                    comet_fn: Callable | None = None,
                    extractor: FeatureExtractor | None = None,
                    timer: Any | None = None,
                    actions: Sequence[Action] = ACTION_GRID,
                    tune_fraction: float | None = None,
                    max_states_per_sentence: int | None = None,
                    labels_dir: Path | None = None) -> dict[str, Any]:
    """Small end-to-end feasibility run: label -> fit -> tune -> rollout -> score."""
    timer = timer or NullTimer()
    output_dir = Path(output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "rollouts").mkdir(parents=True, exist_ok=True)
    behaviors = tuple(behaviors or _default_behaviors())
    if extractor is None:
        extractor = FeatureExtractor(
            make_qwen_entropy_fn(predictor) if cfg.entropy_features else None)
    tune_fraction = cfg.tune_fraction if tune_fraction is None else float(tune_fraction)
    per_sentence = (cfg.max_states_per_sentence if max_states_per_sentence is None
                    else int(max_states_per_sentence))

    split_map = split_dev_sentences(dev_sentences, tune_fraction, cfg.seed)
    by_group: dict[str, list[Sentence]] = {"train": [], "tune": []}
    for sentence in dev_sentences:
        by_group[split_map[sentence.sentence_id]].append(sentence)
    sentences_by_id = {s.sentence_id: s for s in [*dev_sentences, *test_sentences]}

    def group_labels(split, sentences):
        path = labels_dir / f"labels_{split}.jsonl" if labels_dir else None
        if path is None or not path.is_file():
            states = collect_states(sentences, split, predictor, translator, behaviors,
                                    extractor, cfg, timer, per_sentence)
            return label_states(states, sentences_by_id, predictor, translator, actions,
                                cfg.seed, cfg.ralcp_threshold, timer, cfg.incorrect_penalty)
        expected = {s.sentence_id: s.source.split() for s in sentences}
        expected_features = set(extractor.feature_names())
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            words = expected.get(row["sentence_id"])
            if (row["split"] != split or words is None or not 0 < row["read_pos"] < len(words)
                    or row["observed"] != " ".join(words[:row["read_pos"]])
                    or set(row["features"]) != expected_features):
                raise ValueError(f"saved labels do not match {split} data or feature configuration")
            rows.append(TrainingRow(**dict(row, committed=tuple(row["committed"]),
                                           new_words=tuple(row["new_words"]))))
        if not rows:
            raise ValueError(f"saved {split} labels are empty")
        return rows

    rows_train = group_labels("train", by_group["train"])
    rows_tune = group_labels("tune", by_group["tune"])
    rows_test = group_labels("test", test_sentences)

    validate_sentence_disjoint(rows_train, rows_tune, rows_test)
    _write_jsonl(output_dir / "labels_train.jsonl", rows_train)
    _write_jsonl(output_dir / "labels_tune.jsonl", rows_tune)
    _write_jsonl(output_dir / "labels_test.jsonl", rows_test)

    model_specs = {
        "joint": tuple(actions),
        "b_only": B_ONLY_ACTIONS,
        "h_only": H_ONLY_ACTIONS,
    }
    if cfg.neural:
        model_specs["joint_neural"] = tuple(actions)
    models: dict[str, GateModel] = {}
    thresholds: dict[str, float] = {}
    offline: dict[str, Any] = {}
    for name, subset in model_specs.items():
        model = fit_gate_model(
            rows_train, rows_tune, subset, cfg.seed, cfg.cost_weight,
            cfg.incorrect_penalty, neural=name == "joint_neural")
        models[name] = model
        thresholds[name] = float(model.threshold)
        save_checkpoint(model, output_dir / "checkpoints" / f"{name}.joblib")
        offline[name] = evaluate_offline(model, rows_test)

    policies: dict[str, Any] = {
        "fixed_4_6": FixedActionPolicy(FIXED_BASELINE),
        "fixed_2_6": FixedActionPolicy(FIXED_DEV_REFERENCE),
        **{name: LearnedGatePolicy(model, name) for name, model in models.items()},
    }
    rollouts: dict[str, Any] = {}
    for name, policy in policies.items():
        traces = [run_online_rollout(
            sentence, predictor, translator, policy, extractor, cfg,
            split="test", timer=timer) for sentence in test_sentences]
        _write_jsonl(output_dir / "rollouts" / f"{name}.jsonl", traces)
        cell = rollout_metrics(
            traces, metrics, bleu_fn=bleu_fn, comet_fn=comet_fn,
            device=cfg.device)
        cell["policy"] = name
        if name in thresholds:
            cell["threshold"] = thresholds[name]
        rollouts[name] = cell

    results = {
        "config": asdict(cfg),
        "action_grid": [action.label for action in actions],
        "baselines": {
            "fixed_4_6": FIXED_BASELINE.label,
            "fixed_2_6": FIXED_DEV_REFERENCE.label,
            "b_only": [action.label for action in B_ONLY_ACTIONS],
            "h_only": [action.label for action in H_ONLY_ACTIONS],
        },
        "splits": {
            "train_sentences": [s.sentence_id for s in by_group["train"]],
            "tune_sentences": [s.sentence_id for s in by_group["tune"]],
            "test_sentences": [s.sentence_id for s in test_sentences],
        },
        "label_counts": {
            "train": len(rows_train), "tune": len(rows_tune),
            "test": len(rows_test),
            "states_train": len(_state_groups(rows_train)), "states_tune": len(_state_groups(rows_tune)),
            "states_test": len(_state_groups(rows_test)),
        },
        "thresholds": thresholds,
        "offline": offline,
        "rollouts": rollouts,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_summary_markdown(results), encoding="utf-8")
    return results


def _summary_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Joint READ / (B,h) gate feasibility", "",
        f"- seed: {results['config']['seed']}; actions: {', '.join(results['action_grid'])}",
        "- Offline labels compare each action's committed words with the full-true-source "
        "Marian continuation; oracle labels are never online features.",
        "", "| policy | BLEU | COMET | LAAL | virtual completion mean (s) | action histogram |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, cell in results["rollouts"].items():
        bleu = cell.get("bleu", {}).get("score")
        comet = cell.get("comet", {}).get("score")
        completion = cell["completion_s"]["mean"]
        histogram = ", ".join(f"{k}:{v}" for k, v in cell["action_histogram"].items())
        lines.append(
            f"| {name} | {bleu if bleu is not None else 'n/a'} "
            f"| {comet if comet is not None else 'n/a'} "
            f"| {cell['laal']} | {completion:.3f} | {histogram} |")
    lines.extend(["", "## Offline held-out decisions", "",
                  "| model | threshold | compute rate | mean reward | mean matched "
                  "| mean incorrect |", "|---|---:|---:|---:|---:|---:|"])
    for name, report in results["offline"].items():
        lines.append(
            f"| {name} | {report['threshold']:.6g} | {report['compute_rate']:.3f} "
            f"| {report['mean_reward']:.3f} | {report['mean_matched']:.3f} "
            f"| {report['mean_incorrect']:.3f} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Official data and real adapters
# ---------------------------------------------------------------------------

def load_official_sentences(split: str, limit: int = 0) -> list[Sentence]:
    pfba = _load_pfba(required=True)
    return [Sentence(s.sentence_id, s.source, s.reference)
            for s in pfba.load_split(split, limit)]


def build_small_adapters(args, cfg: GateConfig):
    """Load small Qwen + small Marian through run_pfba's adapters."""
    pfba = _load_pfba(required=True)
    import torch

    predictor_device, translator_device = pfba.resolve_devices(
        torch, args, "small", "small")
    predictor = pfba.QwenPredictor(
        pfba.PREDICTORS["small"], predictor_device, "auto",
        bool(getattr(args, "local_files_only", False)),
        bool(getattr(args, "trust_remote_code", False)))
    predictor.load()
    stream_cfg = pfba.StreamConfig(
        min_start=cfg.min_start, ralcp_threshold=cfg.ralcp_threshold,
        seed=cfg.seed, arrival_gap=cfg.arrival_gap,
        local_files_only=bool(getattr(args, "local_files_only", False)),
        trust_remote_code=bool(getattr(args, "trust_remote_code", False)))
    translator = pfba.load_translator(
        "small", pfba.TRANSLATORS["small"], translator_device, stream_cfg)

    def sync():
        pfba._sync(predictor_device)
        if translator_device != predictor_device:
            pfba._sync(translator_device)

    return predictor, translator, pfba.StageTimer(sync)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cheap joint READ/(B,h) gate: small Qwen + Marian feasibility run.")
    parser.add_argument("--dev-limit", type=int, default=10)
    parser.add_argument("--test-limit", type=int, default=10)
    parser.add_argument("--test-offset", type=int, default=0)
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--tune-fraction", type=float, default=0.2)
    parser.add_argument("--min-start", type=int, default=3)
    parser.add_argument("--arrival-gap", type=float, default=0.3)
    parser.add_argument("--ralcp-threshold", type=float, default=0.6)
    parser.add_argument("--incorrect-penalty", type=float, default=1.0)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--max-states-per-sentence", type=int, default=4)
    parser.add_argument("--entropy-features", action="store_true")
    parser.add_argument("--neural", action="store_true", help="evaluate a small neural confidence model")
    parser.add_argument("--metrics", nargs="+", choices=("bleu", "comet"),
                        default=["bleu"])
    parser.add_argument("--skip-comet", action="store_true")
    parser.add_argument("--output-dir", default="outputs/joint_gate")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--predictor-device", default=None)
    parser.add_argument("--translator-device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_config(args) -> GateConfig:
    if args.dev_limit < 2 or args.test_limit < 1 or args.test_offset < 0:
        raise ValueError("--dev-limit must be >= 2, --test-limit >= 1, and --test-offset >= 0")
    if not 0.0 <= float(args.ralcp_threshold) <= 1.0:
        raise ValueError("--ralcp-threshold must be in [0, 1]")
    if int(args.min_start) < 1:
        raise ValueError("--min-start must be >= 1")
    if float(args.arrival_gap) < 0:
        raise ValueError("--arrival-gap must be non-negative")
    if float(args.tune_fraction) <= 0 or float(args.tune_fraction) >= 1:
        raise ValueError("--tune-fraction must be in (0, 1)")
    if int(args.max_states_per_sentence) < 1:
        raise ValueError("--max-states-per-sentence must be positive")
    return GateConfig(
        seed=int(args.seed), min_start=int(args.min_start),
        arrival_gap=float(args.arrival_gap),
        ralcp_threshold=float(args.ralcp_threshold),
        incorrect_penalty=float(args.incorrect_penalty),
        cost_weight=float(args.cost_weight),
        tune_fraction=float(args.tune_fraction),
        max_states_per_sentence=int(args.max_states_per_sentence),
        entropy_features=bool(args.entropy_features), neural=bool(args.neural), device=args.device)


def resolve_metrics(args) -> list[str]:
    metrics = list(args.metrics)
    if args.skip_comet:
        metrics = [name for name in metrics if name != "comet"]
    return metrics


def run(args) -> Path:
    cfg = resolve_config(args)
    metrics = resolve_metrics(args)
    if not metrics:
        raise ValueError("at least one metric must remain after --skip-comet")
    ensure_metrics_available(metrics)

    dev = load_official_sentences("dev", int(args.dev_limit))
    test = load_official_sentences("test", args.test_offset + args.test_limit)[args.test_offset:]
    if not dev or not test:
        raise RuntimeError("official dev/test data could not be loaded")

    predictor, translator, timer = build_small_adapters(args, cfg)
    prefix = " ".join(dev[0].source.split()[:cfg.min_start])
    predictor.sample(prefix, FIXED_BASELINE.branches, FIXED_BASELINE.horizon, cfg.seed)
    translator.translate_batch([prefix] * FIXED_BASELINE.branches, "")
    output_dir = Path(args.output_dir)
    if args.run_name:
        output_dir = output_dir / args.run_name
    else:
        output_dir = output_dir / time.strftime("run_%Y%m%d_%H%M%S")
    pfba = _load_pfba(required=True)
    comet_fn = None
    if "comet" in metrics:
        comet_fn = lambda hypotheses, sources, references, device="cpu": (
            pfba.comet_score(list(hypotheses), list(sources), list(references),
                             pfba.DEFAULT_COMET, device))
    run_feasibility(
        dev, test, predictor, translator, output_dir, cfg,
        metrics=metrics, bleu_fn=pfba.corpus_bleu, comet_fn=comet_fn,
        timer=timer, labels_dir=args.labels_dir)
    print(f"[joint-gate] wrote {output_dir / 'results.json'}")
    return output_dir


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        cfg = resolve_config(args)
        metrics = resolve_metrics(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    if args.dry_run:
        print(json.dumps({
            "config": asdict(cfg),
            "action_grid": [a.label for a in ACTION_GRID],
            "baselines": {
                "fixed_4_6": FIXED_BASELINE.label,
                "fixed_2_6": FIXED_DEV_REFERENCE.label,
                "b_only": [a.label for a in B_ONLY_ACTIONS],
                "h_only": [a.label for a in H_ONLY_ACTIONS],
            },
            "metrics": metrics,
            "dev_limit": args.dev_limit,
            "test_limit": args.test_limit,
            "predictor": "Qwen/Qwen2.5-0.5B",
            "translator": "Helsinki-NLP/opus-mt-de-en",
        }, ensure_ascii=False, indent=2))
        return 0
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
