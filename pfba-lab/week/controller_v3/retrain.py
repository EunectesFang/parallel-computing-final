"""Offline retraining from persisted one-step labels.

The expensive part of a CAVC run is sampling and translating counterfactual
futures. `labels.jsonl` stores every action's outcome at every visited state,
so a new reward configuration, cost topology, or loss weighting can be
retrained without calling the predictor or translator again. This module does
exactly that:

* replay the online History counters with the current semantics;
* rewrite the per-action marginal cost for the measured topology;
* rebuild Bellman targets from the raw ``matched``/``incorrect``/``emitted``
  fields under the new reward configuration;
* train a fresh value network and save an independent checkpoint;
* emit offline diagnostics (action distribution, Q/target error, risk
  calibration) so the expensive closed-loop run is only paid for once.

Closed-loop BLEU/LAAL/completion still require the real predictor and
translator and are deliberately out of scope here.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

from .action import READ, Action
from .checkpoint import _deserialize_cost_records, save_checkpoint
from .config import ControllerConfig
from .cost_model import MeasuredCostTable
from .evaluator import risk_calibration
from .features import FeatureBuilder, NUMERIC_FEATURES
from .network import CAVCNetwork
from .rollout import select_guarded_action
from .trainer import CAVCTrainer
from . import kaggle

__all__ = [
    "apply_serial_costs",
    "replay_histories",
    "retrain_from_labels",
]


def replay_histories(rows: Sequence[dict]) -> Sequence[dict]:
    """Rewrite the pre-action History lag counters in each label row.

    The stored snapshots were written before ``no_emit_streak`` was decoupled
    from ``words_since_commit``. Replaying the behavior action sequence gives
    both counters the current semantics: every step advances the source clock,
    only a failed WRITE advances the failed-write streak, and a commit resets
    both. Mutates and returns `rows`.
    """
    for row in rows:
        words_since_commit = 0
        no_emit_streak = 0
        for state in row.get("states") or []:
            history = state.setdefault("history", {})
            history["words_since_commit"] = words_since_commit
            history["no_emit_streak"] = no_emit_streak
            chosen = str(state.get("chosen", ""))
            outcome = next(
                (
                    item
                    for item in state.get("outcomes") or []
                    if str(item.get("action")) == chosen
                ),
                None,
            )
            emitted = int((outcome or {}).get("emitted", 0))
            if emitted > 0:
                words_since_commit = 0
                no_emit_streak = 0
            else:
                words_since_commit += 1
                if chosen != READ.label:
                    no_emit_streak += 1
    return rows


def apply_serial_costs(
    rows: Sequence[dict],
    *,
    overhead_s: float = 0.0,
) -> Sequence[dict]:
    """Replace the scheduled marginal with the single-device stage sum.

    Labels priced on one GPU carry a ``marginal_wall_s`` from the two-device
    pipeline simulation, which can credit overlap that cannot happen. The
    serial topology must rebuild the successor clock from ``pred_s + mt_s``
    (plus an optional measured per-action overhead). Mutates and returns
    `rows`.
    """
    for row in rows:
        for state in row.get("states") or []:
            for outcome in state.get("outcomes") or []:
                if str(outcome.get("action")) == READ.label:
                    outcome["marginal_wall_s"] = 0.0
                else:
                    outcome["marginal_wall_s"] = (
                        float(outcome.get("pred_s", 0.0))
                        + float(outcome.get("mt_s", 0.0))
                        + float(overhead_s)
                    )
    return rows


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Rank correlation without a scipy dependency."""
    if len(left) < 2 or len(left) != len(right):
        return None

    def ranks(values):
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        for rank, index in enumerate(order):
            result[index] = float(rank)
        return result

    a = ranks(left)
    b = ranks(right)
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denom_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    denom_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    if denom_a == 0.0 or denom_b == 0.0:
        return None
    return numerator / (denom_a * denom_b)


def _sigmoid(value: float) -> float:
    exponent = math.exp(-float(value))
    return 1.0 / (1.0 + exponent)


def _mean(values: Sequence[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    values = sorted(float(value) for value in values)
    if not values:
        return None
    index = min(
        len(values) - 1,
        max(0, int(round(float(fraction) * (len(values) - 1)))),
    )
    return values[index]


def _auc(scores: Sequence[float], labels: Sequence[float]) -> float | None:
    """ROC AUC with tie-aware ranks; ``None`` when one class is absent."""
    pairs = sorted(zip((float(v) for v in scores), (float(v) for v in labels)))
    positives = sum(1 for _, label in pairs if label > 0.0)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = [0.0] * len(pairs)
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[position] = average_rank
        index = end + 1
    positive_rank_sum = sum(
        rank for rank, (_, label) in zip(ranks, pairs) if label > 0.0
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _action_counts(actions: Sequence[Action]) -> dict[str, int]:
    return dict(Counter(action.label for action in actions))


def behavior_read_run_stats(rows: Sequence[dict]) -> dict:
    """Consecutive READ runs in the behavior trajectory (diagnostic only).

    The new policy's READ runs can only be measured in a real rollout; these
    numbers describe the fixed behavior policy that produced the labels.
    """
    runs: list[int] = []
    current = 0
    total_states = 0
    for row in rows:
        for state in row.get("states") or []:
            total_states += 1
            if str(state.get("chosen", "")) == READ.label:
                current += 1
            else:
                if current:
                    runs.append(current)
                current = 0
        if current:
            runs.append(current)
            current = 0
    return {
        "n_states": total_states,
        "n_read_runs": len(runs),
        "mean_run": _mean(runs),
        "max_run": max(runs) if runs else 0,
        "read_fraction": (
            sum(runs) / total_states if total_states else None
        ),
    }


def _diagnostics(
    network: CAVCNetwork,
    trainer: CAVCTrainer,
    cfg: ControllerConfig,
    targets: Sequence,
    calibration_targets: Sequence,
    *,
    guard_cfg: ControllerConfig,
) -> dict:
    """Offline action-selection and value-quality diagnostics."""
    import torch

    states_by_key: dict[str, object] = {}
    for target in targets:
        states_by_key.setdefault(str(target.state_key), target.state)
    keys = list(states_by_key)
    states = [states_by_key[key] for key in keys]
    with torch.no_grad():
        encoded = trainer._encode_states(network, states)
        writes = [action for action in cfg.actions if not action.is_read]
        write_outputs = network.write_values(encoded, writes)
        if cfg.tri_value:
            read_q, read_l, read_w = network.read_components(encoded)
            read_values = (
                read_q
                - float(cfg.beta_lag) * read_l
                - float(cfg.beta_wall) * read_w
            ).detach().tolist()
            write_values = (
                write_outputs.quality
                - float(cfg.beta_lag) * write_outputs.delay
                - float(cfg.beta_wall) * write_outputs.wall
            ).detach().tolist()
        else:
            read_values = network.read_value(encoded).detach().tolist()
            write_values = write_outputs.value.detach().tolist()
        no_emit_logits = write_outputs.no_emit.detach().tolist()

    q_by_action_samples: dict[Action, list[float]] = {
        action: [] for action in writes
    }
    read_advantages: list[float] = []
    delta_q_expensive: list[float] = []
    delta_c_expensive: list[float] = []
    expected_cost_v3: list[float] = []
    expected_cost_guarded: list[float] = []
    no_emit_pairs: list[tuple[float, float]] = []

    baseline = (
        Action(int(cfg.advantage_baseline[0]), int(cfg.advantage_baseline[1]))
        if cfg.advantage_baseline
        else None
    )
    v3_actions: list[Action] = []
    guarded_actions: list[Action] = []
    target_by_state: dict[str, list] = {}
    for target in targets:
        target_by_state.setdefault(str(target.state_key), []).append(target)

    q_by_target: list[float] = []
    target_values: list[float] = []
    top1_matches = 0
    top1_states = 0
    for index, key in enumerate(keys):
        state = states[index]
        q_read = float(read_values[index])
        q_by_action = {
            action: float(write_values[index][column])
            for column, action in enumerate(writes)
        }
        no_emit_risk = {
            action: _sigmoid(float(no_emit_logits[index][column]))
            for column, action in enumerate(writes)
        }
        costs = {
            action: float(trainer._cost(state, action)) for action in writes
        }
        if baseline is not None and not baseline.is_read and baseline not in costs:
            costs[baseline] = float(trainer._cost(state, baseline))
        cost_base = costs.get(baseline, 0.0) if baseline is not None else 0.0
        q_base = (
            q_read
            if baseline is not None and baseline.is_read
            else q_by_action.get(baseline, 0.0)
        )

        def score(action: Action) -> float:
            q_value = q_read if action.is_read else q_by_action[action]
            if cfg.tri_value:
                return q_value if baseline is None else q_value - q_base
            cost = 0.0 if action.is_read else costs[action]
            if baseline is None:
                return q_value - cfg.lambda_t * cost
            return (q_value - q_base) - cfg.lambda_t * (cost - cost_base)

        scores = {READ: score(READ), **{action: score(action) for action in writes}}
        best = READ
        for action in writes:
            if scores[action] > scores[best]:
                best = action
        v3_actions.append(best)
        expected_cost_v3.append(0.0 if best.is_read else costs[best])
        if writes:
            read_advantages.append(
                float(scores[READ])
                - max(float(scores[action]) for action in writes)
            )
        if Action(6, 10) in costs and Action(4, 6) in costs:
            delta_q_expensive.append(
                q_by_action[Action(6, 10)] - q_by_action[Action(4, 6)]
            )
            delta_c_expensive.append(
                costs[Action(6, 10)] - costs[Action(4, 6)]
            )
        for action in writes:
            q_by_action_samples[action].append(q_by_action[action])

        numeric = state.numeric
        lag = float(numeric[NUMERIC_FEATURES.index("normalized_lag")])
        no_commit = int(round(float(numeric[NUMERIC_FEATURES.index("words_since_commit")])))
        chosen, _threshold, _admitted = select_guarded_action(
            scores=scores,
            costs={"read": 0.0, **costs},
            candidate_writes=writes,
            baseline=baseline,
            cfg=guard_cfg,
            normalized_lag=lag,
            consecutive_reads=0,
            no_commit_streak=no_commit,
            no_emit_risk=no_emit_risk,
        )
        guarded_actions.append(chosen)
        expected_cost_guarded.append(
            0.0 if chosen.is_read else costs[chosen]
        )

        rows = target_by_state.get(key, [])
        qs = []
        ys = []
        for target in rows:
            q = q_read if target.action.is_read else q_by_action[target.action]
            qs.append(float(q))
            target_utility = (
                float(target.quality_target)
                - float(cfg.beta_lag) * float(target.delay_target or 0.0)
                - float(cfg.beta_wall) * float(target.wall_target or 0.0)
                if cfg.tri_value and target.quality_target is not None
                else float(target.value_target)
            )
            ys.append(target_utility)
            q_by_target.append(float(q))
            target_values.append(target_utility)
            if not target.action.is_read:
                no_emit_pairs.append(
                    (
                        float(no_emit_risk[target.action]),
                        float(target.no_emit_target),
                    )
                )
        if qs:
            q_argmax = rows[max(range(len(qs)), key=lambda i: qs[i])].action
            y_argmax = rows[max(range(len(ys)), key=lambda i: ys[i])].action
            top1_states += 1
            top1_matches += int(q_argmax == y_argmax)

    errors = [abs(q - y) for q, y in zip(q_by_target, target_values)]
    risk = (
        risk_calibration(network, list(calibration_targets))
        if calibration_targets
        else {
            "status": "unavailable",
            "reason": "no calibration targets",
            "n": 0,
            "error_risk": {},
            "no_emit": {},
        }
    )
    return {
        "n_states": len(keys),
        "v3_action_counts": _action_counts(v3_actions),
        "guarded_action_counts": _action_counts(guarded_actions),
        "v3_read_fraction": (
            sum(1 for action in v3_actions if action.is_read) / max(1, len(v3_actions))
        ),
        "v3_b6h10": sum(
            1 for action in v3_actions if action == Action(6, 10)
        ),
        "v3_h10_fraction": (
            sum(
                1
                for action in v3_actions
                if not action.is_read and action.horizon == 10
            )
            / max(1, sum(1 for action in v3_actions if not action.is_read))
        ),
        "q_target_mae": sum(errors) / max(1, len(errors)),
        "q_target_signed_mean": _mean(
            [q - y for q, y in zip(q_by_target, target_values)]
        ),
        "q_target_spearman": _spearman(q_by_target, target_values),
        "q_top1_match_fraction": top1_matches / max(1, top1_states),
        "action_q_stats": {
            action.label: {
                "mean": _mean(values),
                "std": (
                    statistics.pstdev(values) if len(values) > 1 else 0.0
                ),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
            for action, values in q_by_action_samples.items()
        },
        "read_advantage": {
            "mean": _mean(read_advantages),
            "p10": _quantile(read_advantages, 0.10),
            "p50": _quantile(read_advantages, 0.50),
            "p90": _quantile(read_advantages, 0.90),
            "fraction_positive": (
                sum(1 for value in read_advantages if value > 0.0)
                / len(read_advantages)
                if read_advantages
                else None
            ),
        },
        "delta_u_expensive": {
            "n": len(delta_q_expensive),
            "mean_delta_q": _mean(delta_q_expensive),
            "median_delta_q": _quantile(delta_q_expensive, 0.50),
            "mean_delta_c": _mean(delta_c_expensive),
            "fraction_positive_q": (
                sum(1 for value in delta_q_expensive if value > 0.0)
                / len(delta_q_expensive)
                if delta_q_expensive
                else None
            ),
        },
        "expected_cost": {
            "v3": _mean(expected_cost_v3),
            "guarded": _mean(expected_cost_guarded),
        },
        "no_emit_auc": _auc(
            [predicted for predicted, _ in no_emit_pairs],
            [label for _, label in no_emit_pairs],
        ),
        "risk_calibration": risk,
    }


def retrain_from_labels(
    labels_path,
    checkpoint_path,
    output_dir,
    *,
    weight_e: float | None = None,
    lambda_l: float | None = None,
    lambda_t: float | None = None,
    gamma: float | None = None,
    alpha_rank: float | None = None,
    cost_topology: str = "serial",
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    seed: int | None = None,
    tri_value: bool | None = None,
    double_dqn: bool | None = None,
    target_tau: float | None = None,
    beta_lag: float | None = None,
    beta_wall: float | None = None,
    lag_anchor: float | None = None,
    convex_kappa: float | None = None,
    pair_weight: float | None = None,
    tri_rank_epsilon: float | None = None,
    alpha_error: float | None = None,
    alpha_noemit: float | None = None,
    alpha_emit: float | None = None,
    gamma_delay: float | None = None,
    gamma_wall: float | None = None,
    terminal_eos_wall_s: float | None = None,
    feature_normalize: bool | None = None,
    risk_every: int = 5,
    guard_cfg: ControllerConfig | None = None,
) -> dict:
    """Retrain a value network from labels; never calls an adapter.

    The checkpoint supplies the frozen vocabulary, length ratio, cost records,
    action grid and model-pair identity. Only the reward/training knobs are
    overridden. Returns the summary written to ``output_dir/summary.json``.
    """
    import torch

    labels_path = Path(labels_path)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    payload = kaggle._checkpoint_payload(checkpoint_path)
    base_cfg = kaggle._config_from_payload(payload, checkpoint_path)
    overrides = {
        key: value
        for key, value in {
            "weight_e": weight_e,
            "lambda_l": lambda_l,
            "lambda_t": lambda_t,
            "gamma": gamma,
            "alpha_rank": alpha_rank,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "tri_value": tri_value,
            "double_dqn": double_dqn,
            "target_tau": target_tau,
            "beta_lag": beta_lag,
            "beta_wall": beta_wall,
            "lag_anchor": lag_anchor,
            "convex_kappa": convex_kappa,
            "pair_weight": pair_weight,
            "tri_rank_epsilon": tri_rank_epsilon,
            "alpha_error": alpha_error,
            "alpha_noemit": alpha_noemit,
            "alpha_emit": alpha_emit,
            "gamma_delay": gamma_delay,
            "gamma_wall": gamma_wall,
            "terminal_eos_wall_s": terminal_eos_wall_s,
            "feature_normalize": feature_normalize,
        }.items()
        if value is not None
    }
    # The base checkpoint stores the old 34-dim schema; the current feature
    # contract is authoritative for a fresh retrain.
    overrides["numeric_dim"] = len(NUMERIC_FEATURES)
    cfg = dataclasses.replace(base_cfg, **overrides)
    vocab, ratio = kaggle._frozen_vocab(payload)
    if vocab is None:
        raise ValueError(
            f"checkpoint {checkpoint_path} has no stored vocabulary; cannot "
            "rebuild the frozen text ids for retraining"
        )
    builder = FeatureBuilder(
        length_ratio=float(ratio if ratio is not None else 1.0),
        src_vocab=vocab,
        tgt_vocab=vocab,
    )
    records = _deserialize_cost_records(payload.get("cost_records", []))
    base_cost = MeasuredCostTable(records)
    cost_model = kaggle._cost_model_for_topology(base_cost, cost_topology)

    rows = list(kaggle._read_jsonl(labels_path))
    if not rows:
        raise ValueError(f"{labels_path} contains no label rows")
    replay_histories(rows)
    if cost_topology == "serial":
        apply_serial_costs(rows)

    model_pair = dict((payload.get("extra") or {}).get("model_pair") or {})
    predictor_id = int(model_pair.get("predictor_id", 0))
    translator_id = int(model_pair.get("translator_id", 0))
    targets, states = kaggle._targets_from_rows(
        rows, builder, cfg, predictor_id, translator_id
    )
    if not targets:
        raise ValueError(f"{labels_path} produced no training targets")
    train_targets, calibration_targets = kaggle._split_calibration_targets(
        targets, every=risk_every
    )

    mean, std = kaggle._feature_stats(states)
    torch.manual_seed(int(cfg.seed))
    random.seed(int(cfg.seed))
    network = CAVCNetwork(
        cfg, src_vocab_size=len(vocab), tgt_vocab_size=len(vocab)
    )
    network.set_feature_stats(mean, std)
    trainer = CAVCTrainer(network, cfg, cost_model)
    started = time.perf_counter()
    training = trainer.fit(train_targets)
    train_seconds = time.perf_counter() - started
    network.eval()
    trained = (
        int(training.get("epochs", 0)) > 0
        and int(training.get("n_samples", 0)) > 0
    )

    guard = guard_cfg or dataclasses.replace(
        cfg,
        read_threshold=0.10,
        read_threshold_lag=0.15,
        read_threshold_reads=0.15,
        read_threshold_no_commit=0.10,
        expensive_margin=0.25,
        no_emit_veto=1.0,
        no_emit_penalty=0.20,
    )
    eval_targets = list(calibration_targets) if calibration_targets else list(targets)
    diagnostics = _diagnostics(
        network,
        trainer,
        cfg,
        eval_targets,
        calibration_targets,
        guard_cfg=guard,
    )
    diagnostics["evaluation_split"] = (
        "sentence_holdout" if calibration_targets else "all_targets"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_stats = kaggle.feature_diagnostics(states)
    dead_features = list(feature_stats.get("dead_features") or [])
    if dead_features:
        raise ValueError(
            "training states contain constant (dead) numeric features: "
            f"{dead_features}; remove or wire them before retraining"
        )
    extra = {
        "mode": "retrain",
        "split": "dev",
        "training": training,
        "n_targets": len(train_targets),
        "trained": trained,
        "vocab": list(vocab.id_to_token),
        "length_ratio": float(builder.rho),
        "model_pair": model_pair,
        "feature_stats": feature_stats,
        "cost_topology": cost_topology,
        "candidate_mode": (payload.get("extra") or {}).get(
            "candidate_mode", "per_candidate"
        ),
        "post_controller": "disabled",
        "reward_config": {
            "weight_m": float(cfg.weight_m),
            "weight_e": float(cfg.weight_e),
            "lambda_l": float(cfg.lambda_l),
            "lambda_t": float(cfg.lambda_t),
            "gamma": float(cfg.gamma),
            "alpha_rank": float(cfg.alpha_rank),
        },
        "source_labels": str(labels_path),
        "base_checkpoint": str(checkpoint_path),
        "cost_measurement_source": (
            "pred_s/mt_s persisted by the source Kaggle run (cuda:0); local "
            "retraining does not re-measure inference cost"
        ),
        "training_device": "cpu",
    }
    checkpoint_file = output_dir / "checkpoint.pt"
    metadata = save_checkpoint(
        checkpoint_file, network, cfg, mean, std, base_cost, extra=extra
    )
    summary = {
        "mode": "retrain",
        "trained": trained,
        "labels": str(labels_path),
        "base_checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "n_label_rows": len(rows),
        "n_states": len(states),
        "n_targets": len(targets),
        "n_train_targets": len(train_targets),
        "n_risk_calibration_targets": len(calibration_targets),
        "training": training,
        "train_seconds": float(train_seconds),
        "training_device": "cpu",
        "cost_topology": cost_topology,
        "cost_measurement_source": (
            "pred_s/mt_s persisted by the source Kaggle run (cuda:0); local "
            "retraining does not re-measure inference cost"
        ),
        "behavior_diagnostics": behavior_read_run_stats(rows),
        "config": {
            "weight_m": float(cfg.weight_m),
            "weight_e": float(cfg.weight_e),
            "lambda_l": float(cfg.lambda_l),
            "lambda_t": float(cfg.lambda_t),
            "gamma": float(cfg.gamma),
            "alpha_rank": float(cfg.alpha_rank),
            "epochs": int(cfg.epochs),
            "batch_size": int(cfg.batch_size),
            "lr": float(cfg.lr),
            "seed": int(cfg.seed),
            "tri_value": bool(cfg.tri_value),
            "double_dqn": bool(cfg.double_dqn),
            "target_tau": float(cfg.target_tau),
            "beta_lag": float(cfg.beta_lag),
            "beta_wall": float(cfg.beta_wall),
            "lag_anchor": float(cfg.lag_anchor),
            "convex_kappa": float(cfg.convex_kappa),
            "pair_weight": float(cfg.pair_weight),
            "tri_rank_epsilon": float(cfg.tri_rank_epsilon),
            "gamma_delay": float(cfg.gamma_delay),
            "gamma_wall": float(cfg.gamma_wall),
            "terminal_eos_wall_s": float(cfg.terminal_eos_wall_s),
            "feature_normalize": bool(cfg.feature_normalize),
        },
        "diagnostics": diagnostics,
        "checkpoint": {
            "path": str(checkpoint_file),
            "format_version": metadata["format_version"],
            "seed": metadata["seed"],
        },
    }
    kaggle._write_json(output_dir / "summary.json", summary)
    kaggle._write_json(output_dir / "diagnostics.json", diagnostics)
    kaggle._write_json(
        output_dir / "config.json",
        kaggle._config_payload(cfg, model_pair),
    )
    return summary
