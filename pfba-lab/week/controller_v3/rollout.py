"""One-sentence greedy rollout of the CAVC controller.

At every source read position the rollout builds the online feature vector,
scores READ and the configured write actions with the value network, and takes
the argmax of

    score(a) = (Q(a) - Q(baseline)) - lambda_t * (C(a) - C(baseline))

when `cfg.advantage_baseline` names a baseline action, or the plain

    score(a) = Q(a) - lambda_t * C(a)

when it is disabled (`cfg.advantage_baseline` is falsy, e.g. ``None`` or ``()``).
A baseline of ``(0, 0)`` is the READ baseline, not a disabled one. The baseline
Q and cost are computed even when the baseline action is not itself a selectable
candidate, which is what makes the advantage comparable to the trainer's
Bellman backup.

READ performs no prediction and no translation: it only advances `read_pos` and
updates `History`. WRITE calls the injected runtime once and commits only the
final RALCP result, after the optional post-candidate controller has had a
chance to add branches or wait. Every decision, READ or WRITE, consumes one
source position, mirroring `week_run.rollout` and the fixed source cadence: the
next word arrives whether the controller computed something or not.

Costs are **virtual**: the clock advances to the source arrival time of the
observed prefix, then by the measured prediction/translation seconds. The
runtime's timer is therefore the only source of measured time, and provisional
work that the post controller discards is still billed once.

At source EOS the true full source is translated once with the committed prefix
forced, and that final output is appended. LAAL uses the shared upstream
`run_pfba.laal_score` so the rollout is directly comparable to `run_pfba`.
"""

from __future__ import annotations

import inspect
import math
import warnings
import weakref
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

from .action import READ, Action
from .config import ControllerConfig
from .features import (
    NUMERIC_FEATURES,
    FeatureBuilder,
    HistoryTracker,
    advance_virtual_clock,
)
from .post_controller import ADD_2_BRANCHES, PostAction
from .runtime import ControlledRuntime, StepResult, run_pfba

__all__ = [
    "RolloutTrace",
    "read_threshold",
    "run_rollout",
    "run_fixed_rollout",
    "select_guarded_action",
]

# The fixed baseline always writes this budget; it never reads.
FIXED_ACTION = Action(4, 6)


@dataclass(frozen=True)
class RolloutTrace:
    """Everything one sentence's rollout produced.

    `steps` holds one dict per decision plus a final `"eos"` step. Each dict
    carries at least the fields documented in the task contract:
    ``read_pos, action, observed_len, committed_before, committed_after,
    new_words, pred_s, mt_s, spent_s, marginal_wall_s, vote_top1, vote_margin,
    vote_entropy, no_emit, q_read, q_write, cost, post_action``.
    `committed_before` / `committed_after` are word counts; `new_words` is the
    tuple of words this step added. For a write step, `pred_s` / `mt_s` are the
    full decision's measured stage compute (initial execute plus any ADD_2
    extension), `spent_s` is the cumulative serial compute, and
    `marginal_wall_s` is the incremental pipeline wall time that advanced the
    virtual clock. Discarded WAIT work appears in both but not in the committed
    words.
    """

    sentence_id: str
    source: str
    reference: str
    hypothesis: str
    committed: tuple[str, ...]
    delays: tuple[int, ...]
    laal: float | None
    completion_s: float
    read_fraction: float
    action_histogram: dict[str, int]
    steps: tuple[dict, ...]


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _field(sentence, name: str, default):
    """Read a field from a dataclass-like sentence or a plain mapping."""
    if isinstance(sentence, Mapping):
        return sentence.get(name, default)
    return getattr(sentence, name, default)


def _baseline_action(cfg: ControllerConfig) -> Action | None:
    """Return the advantage baseline, or ``None`` when it is disabled."""
    raw = getattr(cfg, "advantage_baseline", None)
    if not raw:
        return None
    branches, horizon = raw
    branches, horizon = int(branches), int(horizon)
    if branches == 0 and horizon == 0:
        return READ
    return Action(branches, horizon)


def _sigmoid(value: float) -> float:
    """Numerically safe logistic used to turn a no-emit logit into a risk."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _score_state(network, vector, query_actions: Sequence[Action]):
    """Encode one state and return `(z, read_q, q_by_action, no_emit_by_action)`.

    The network is fed one-row tensors, exactly like the trainer's batch path,
    under `torch.no_grad()`. Fakes that ignore their tensor inputs and return
    sequences work too because every value is read through `float(...)`. The
    caller is expected to have called `network.eval()`; this function does not
    flip the training flag of a shared module.
    """
    import torch

    numeric = torch.tensor([vector.numeric], dtype=torch.float32)
    predictor_id = torch.tensor([vector.predictor_id], dtype=torch.long)
    translator_id = torch.tensor([vector.translator_id], dtype=torch.long)
    src_ids = (
        torch.tensor([list(vector.src_ids)], dtype=torch.long)
        if vector.src_ids
        else torch.zeros((1, 0), dtype=torch.long)
    )
    tgt_ids = (
        torch.tensor([list(vector.tgt_ids)], dtype=torch.long)
        if vector.tgt_ids
        else torch.zeros((1, 0), dtype=torch.long)
    )

    with torch.no_grad():
        z = network.encode(numeric, predictor_id, translator_id, src_ids, tgt_ids)
        read_q = float(network.read_value(z)[0])
        q_by_action: dict[Action, float] = {}
        no_emit_by_action: dict[Action, float] = {}
        if query_actions:
            outputs = network.write_values(z, list(query_actions))
            value_row = outputs.value[0]
            logits = getattr(outputs, "no_emit", None)
            for index, action in enumerate(query_actions):
                q_by_action[action] = float(value_row[index])
                if logits is not None:
                    no_emit_by_action[action] = _sigmoid(float(logits[0][index]))
                else:
                    no_emit_by_action[action] = 0.0
    return z, read_q, q_by_action, no_emit_by_action


def _score_state_v4(network, vector, query_actions: Sequence[Action]):
    """Three-component READ/WRITE scores for one state (CAVC-Q V4)."""
    import torch

    numeric = torch.tensor([vector.numeric], dtype=torch.float32)
    predictor_id = torch.tensor([vector.predictor_id], dtype=torch.long)
    translator_id = torch.tensor([vector.translator_id], dtype=torch.long)
    src_ids = (
        torch.tensor([list(vector.src_ids)], dtype=torch.long)
        if vector.src_ids
        else torch.zeros((1, 0), dtype=torch.long)
    )
    tgt_ids = (
        torch.tensor([list(vector.tgt_ids)], dtype=torch.long)
        if vector.tgt_ids
        else torch.zeros((1, 0), dtype=torch.long)
    )

    with torch.no_grad():
        z = network.encode(numeric, predictor_id, translator_id, src_ids, tgt_ids)
        read_q, read_l, read_w = network.read_components(z)
        read = (float(read_q[0]), float(read_l[0]), float(read_w[0]))
        write: dict[Action, tuple[float, float, float]] = {}
        no_emit_by_action: dict[Action, float] = {}
        if query_actions:
            outputs = network.write_values(z, list(query_actions))
            for index, action in enumerate(query_actions):
                write[action] = (
                    float(outputs.quality[0][index]),
                    float(outputs.delay[0][index]),
                    float(outputs.wall[0][index]),
                )
                no_emit_by_action[action] = _sigmoid(
                    float(outputs.no_emit[0][index])
                )
    return z, read, write, no_emit_by_action


# Plain cost models warned about at most once each. `WeakSet` keeps this from
# retaining caller objects; an unhashable/non-weakref-able model is warned per
# call because identity cannot be tracked safely.
_WARNED_PLAIN_COST: "weakref.WeakSet" = weakref.WeakSet()


def _warn_plain_cost_once(cost_model) -> None:
    try:
        if cost_model in _WARNED_PLAIN_COST:
            return
        _WARNED_PLAIN_COST.add(cost_model)
    except TypeError:
        pass
    warnings.warn(
        "rollout pricing used a plain cost model without `marginal_wall`; "
        "falling back to the serial pred_s + mt_s diagnostic instead of "
        "pipeline wall time. Wrap the model in PipelineCostModel for the "
        "schedule-aware price.",
        RuntimeWarning,
        stacklevel=3,
    )


def _action_cost(
    cost_model,
    vector,
    action: Action,
    src_words: int,
    tgt_words: int,
    t_pred_free: float = 0.0,
    t_mt_free: float = 0.0,
    release_s: float = 0.0,
    dependency_s: float = 0.0,
) -> float:
    """Pipeline cost estimate of one write action; READ is always free.

    A `PipelineCostModel` is priced through ``marginal_wall`` against the
    runtime's device free times, so the score never charges
    ``pred_s + mt_s`` as wall time. A plain cost model has no marginal
    schedule; its ``wall_s`` is then used only as a documented serial
    diagnostic fallback.
    """
    if action.is_read or cost_model is None:
        return 0.0
    marginal = getattr(cost_model, "marginal_wall", None)
    if callable(marginal):
        return float(
            marginal(
                max(float(t_pred_free), float(release_s)),
                max(float(t_mt_free), float(release_s)),
                float(release_s),
                float(dependency_s),
                vector.predictor_id,
                vector.translator_id,
                action,
                src_words,
                tgt_words,
            )
        )
    _warn_plain_cost_once(cost_model)
    estimate = cost_model.predict(
        vector.predictor_id,
        vector.translator_id,
        action,
        src_words,
        tgt_words,
    )
    return float(getattr(estimate, "wall_s", estimate))


def _arrived_words(elapsed_s: float, gap_s: float, source_len: int) -> int:
    """Fallback copy of `run_pfba.arrived_words` for a stripped checkout."""
    if source_len <= 0:
        return 0
    if gap_s <= 0:
        return source_len
    return min(source_len, int(elapsed_s / gap_s + 1e-9) + 1)


def _delay_at(completion_s: float, gap_s: float, source_len: int) -> int:
    if run_pfba is not None and hasattr(run_pfba, "arrived_words"):
        return int(run_pfba.arrived_words(completion_s, gap_s, source_len))
    return _arrived_words(completion_s, gap_s, source_len)


def _laal(
    delays: Sequence[int],
    source_len: int,
    reference_len: int,
    prediction_len: int,
) -> float | None:
    """Shared upstream LAAL, with a local fallback for a stripped checkout."""
    if run_pfba is not None and hasattr(run_pfba, "laal_score"):
        return run_pfba.laal_score(
            list(delays), source_len, reference_len, prediction_len
        )
    used: list[float] = []
    for delay in delays:
        used.append(float(delay))
        if delay >= source_len:
            break
    target_len = max(int(reference_len), int(prediction_len))
    if not used or target_len <= 0:
        return None
    slope = float(source_len) / target_len
    return sum(d - index * slope for index, d in enumerate(used)) / len(used)


def _step_seed(cfg: ControllerConfig, sentence_id: str, read_pos: int, split: str) -> int:
    if run_pfba is not None and hasattr(run_pfba, "derive_step_seed"):
        return int(
            run_pfba.derive_step_seed(cfg.seed, split, sentence_id, read_pos)
        )
    return int(cfg.seed) + int(read_pos)


def _write_step(
    *,
    read_pos: int,
    observed: str,
    action_label: str,
    committed_before: int,
    committed_after: int,
    new_words: Sequence[str],
    pred_s: float,
    mt_s: float,
    spent_s: float,
    vote_top1: float,
    vote_margin: float,
    vote_entropy: float,
    no_emit: bool,
    q_read,
    q_write,
    cost,
    post_action,
    post_reason=None,
    marginal_wall_s=None,
) -> dict:
    """One step record with a uniform field set across READ/WRITE/EOS."""
    return {
        "read_pos": int(read_pos),
        "action": str(action_label),
        "observed": observed,
        "observed_len": int(read_pos),
        "committed_before": int(committed_before),
        "committed_after": int(committed_after),
        "new_words": tuple(new_words),
        "pred_s": float(pred_s),
        "mt_s": float(mt_s),
        "spent_s": float(spent_s),
        # Diagnostic pipeline wall time; never derive it as pred_s + mt_s.
        "marginal_wall_s": (
            None if marginal_wall_s is None else float(marginal_wall_s)
        ),
        "vote_top1": float(vote_top1),
        "vote_margin": float(vote_margin),
        "vote_entropy": float(vote_entropy),
        "no_emit": bool(no_emit),
        "q_read": None if q_read is None else float(q_read),
        "q_write": None if q_write is None else float(q_write),
        "cost": None if cost is None else float(cost),
        "post_action": None if post_action is None else str(post_action),
        "post_reason": None if post_reason is None else str(post_reason),
    }


def _accepts_keyword(method, name: str) -> bool:
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    if name in parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _execute_step(
    runtime,
    observed: str,
    committed,
    action: Action,
    seed: int,
    release_s: float,
    dependency_s: float,
) -> StepResult:
    """Call `runtime.execute`, passing the schedule when the runtime accepts it."""
    if _accepts_keyword(runtime.execute, "release_s"):
        return runtime.execute(
            observed,
            committed,
            action,
            seed,
            release_s=release_s,
            dependency_s=dependency_s,
        )
    return runtime.execute(observed, committed, action, seed)


def _extend_step(
    runtime,
    observed: str,
    committed,
    current_candidates,
    extra_branches: int,
    horizon: int,
    seed: int,
    spent_s: float,
    release_s: float,
    dependency_s: float,
):
    """Call `runtime.extend`, passing the schedule when it accepts it."""
    if _accepts_keyword(runtime.extend, "release_s"):
        return runtime.extend(
            observed,
            committed,
            current_candidates,
            extra_branches,
            horizon,
            seed,
            spent_s=spent_s,
            release_s=release_s,
            dependency_s=dependency_s,
        )
    return runtime.extend(
        observed,
        committed,
        current_candidates,
        extra_branches,
        horizon,
        seed,
        spent_s=spent_s,
    )


def _marginal_of(result: StepResult) -> float | None:
    value = getattr(result, "marginal_wall_s", None)
    return None if value is None else float(value)


def _billed_seconds(result: StepResult) -> float:
    """Pipeline seconds to bill: measured marginal when available, else serial."""
    marginal = _marginal_of(result)
    return float(result.spent_s) if marginal is None else marginal


def _release_time(completion_s: float, read_pos: int, arrival_gap: float) -> float:
    """Earliest moment an action at `read_pos` may start."""
    arrived = max(0.0, (int(read_pos) - 1) * float(arrival_gap))
    return max(float(completion_s), arrived)


def read_threshold(
    cfg: ControllerConfig,
    *,
    normalized_lag: float,
    consecutive_reads: int,
    no_commit_streak: int,
) -> float:
    """State-dependent admission margin for READ over the best WRITE.

    The threshold rises with normalized source backlog (after the first
    ``read_threshold_lag_ref`` words, scaled by ``read_threshold_lag_scale``)
    and with consecutive waits. ``consecutive_reads`` and
    ``no_commit_streak`` are disjoint event counts: the caller passes READs
    for the first and failed WRITEs for the second, so one wait is never
    charged twice. Streak terms are capped at three. All multipliers default
    to zero, which makes the threshold zero and reproduces V3 selection.
    """
    scale = float(cfg.read_threshold_lag_scale)
    if scale <= 0.0:
        raise ValueError("read_threshold_lag_scale must be positive")
    lag_term = max(
        0.0,
        (max(0.0, float(normalized_lag)) - float(cfg.read_threshold_lag_ref))
        / scale,
    )
    reads = min(3, max(0, int(consecutive_reads)))
    failures = min(3, max(0, int(no_commit_streak)))
    return float(
        cfg.read_threshold
        + cfg.read_threshold_lag * lag_term
        + cfg.read_threshold_reads * reads
        + cfg.read_threshold_no_commit * failures
    )


def select_guarded_action(
    *,
    scores: Mapping[Action, float],
    costs: Mapping[Action, float],
    candidate_writes: Sequence[Action],
    baseline: Action | None,
    cfg: ControllerConfig,
    normalized_lag: float,
    consecutive_reads: int,
    no_commit_streak: int,
    no_emit_risk: Mapping[Action, float] | None = None,
) -> tuple[Action, float, list[Action]]:
    """Choose READ or a WRITE under the V3.1 cost guards.

    Two layers run before the usual argmax:

    1. **Expensive-action margin.** A WRITE whose cost exceeds the baseline's
       must beat the baseline's score by ``cfg.expensive_margin``. Near-ties
       keep the cheaper action. The baseline is always admitted; cheaper
       actions are never filtered by this margin.
    2. **Dynamic READ threshold.** READ wins only when its score exceeds the
       best admitted WRITE by ``read_threshold(...)``. Ties go to READ at a
       zero threshold, matching V3's deterministic tie-break.

    ``scores`` are already in the rollout's scoring space (advantage vs the
    baseline when one is configured). A calibrated no-emit risk can be used
    as a **soft** penalty (``cfg.no_emit_penalty``) or a hard veto
    (``cfg.no_emit_veto``). The soft penalty is the default in the V3.1 guard:
    it lowers an unreliable WRITE without forcing READ when every WRITE is
    unreliable. If the hard veto rejects the best WRITE, or every WRITE is
    filtered, READ is chosen. The caller still submits only through RALCP;
    this function chooses whether to execute a WRITE, not whether to commit
    its output.

    Returns ``(chosen, threshold, admitted_writes)``.
    """
    if READ not in scores:
        raise ValueError("scores must include READ")

    def net_score(action: Action) -> float:
        value = float(scores[action])
        if (
            no_emit_risk is not None
            and not action.is_read
            and float(cfg.no_emit_penalty) != 0.0
        ):
            value -= float(cfg.no_emit_penalty) * float(
                no_emit_risk.get(action, 0.0)
            )
        return value

    adjusted = {action: net_score(action) for action in scores}
    baseline_score = 0.0
    if baseline is not None and not baseline.is_read:
        baseline_score = float(adjusted.get(baseline, 0.0))

    admitted: list[Action] = []
    for action in candidate_writes:
        if (
            baseline is not None
            and not baseline.is_read
            and float(costs.get(action, 0.0))
            > float(costs.get(baseline, 0.0)) + 1e-12
            and float(adjusted[action]) - baseline_score
            <= float(cfg.expensive_margin) + 1e-12
        ):
            continue
        admitted.append(action)

    threshold = read_threshold(
        cfg,
        normalized_lag=normalized_lag,
        consecutive_reads=consecutive_reads,
        no_commit_streak=no_commit_streak,
    )
    if not admitted:
        return READ, threshold, admitted

    best_write = max(admitted, key=lambda action: float(adjusted[action]))
    if (
        no_emit_risk is not None
        and float(no_emit_risk.get(best_write, 0.0))
        > float(cfg.no_emit_veto) + 1e-12
    ):
        return READ, threshold, admitted
    if float(adjusted[READ]) - float(adjusted[best_write]) >= threshold - 1e-12:
        return READ, threshold, admitted
    return best_write, threshold, admitted


def run_rollout(
    sentence,
    predictor,
    translator,
    network,
    builder,
    cost_model,
    cfg: ControllerConfig,
    timer=None,
    *,
    runtime=None,
    post_controller=None,
    predictor_id: int = 0,
    translator_id: int = 0,
) -> RolloutTrace:
    """Run one greedy sentence rollout and return a `RolloutTrace`.

    `sentence` may be a `run_pfba.Sample` or any object/mapping with
    ``sentence_id``, ``source``, and ``reference`` (plus optional ``split``).
    `predictor` / `translator` are only used when `runtime` is not supplied;
    the runtime must expose ``execute``, ``extend`` (used on ADD_2), and
    ``translate`` (used for the EOS flush). `post_controller` is optional and,
    when supplied, must expose
    ``decide(z_pre, current_branches, horizon, vote_entropy, vote_margin,
    no_emit_prob, spent_s, cfg) -> PostDecision``.

    `predictor_id` / `translator_id` are injected into every online feature
    vector through `builder.build`. They select the network's model embeddings
    and key the measured cost lookup, so a real-model caller must pass the
    ids resolved for its adapter pair; the default 0/0 is the small pair.

    The network is forced into eval mode for scoring (its Dropout would
    otherwise make Q values stochastic and non-reproducible) and the previous
    training flag is restored in a `finally`, even when the rollout raises.
    """
    can_eval = callable(getattr(network, "eval", None))
    was_training = getattr(network, "training", None)
    if can_eval:
        network.eval()
    try:
        return _run_rollout_impl(
            sentence,
            predictor,
            translator,
            network,
            builder,
            cost_model,
            cfg,
            timer,
            runtime=runtime,
            post_controller=post_controller,
            predictor_id=predictor_id,
            translator_id=translator_id,
        )
    finally:
        if (
            can_eval
            and was_training is not None
            and callable(getattr(network, "train", None))
        ):
            network.train(bool(was_training))


def _run_rollout_impl(
    sentence,
    predictor,
    translator,
    network,
    builder,
    cost_model,
    cfg: ControllerConfig,
    timer=None,
    *,
    runtime=None,
    post_controller=None,
    predictor_id: int = 0,
    translator_id: int = 0,
) -> RolloutTrace:
    """Body of `run_rollout`; assumes the network is already in eval mode."""
    cfg = cfg if cfg is not None else ControllerConfig()
    source = str(_field(sentence, "source", ""))
    reference = str(_field(sentence, "reference", ""))
    sentence_id = str(_field(sentence, "sentence_id", ""))
    split = str(_field(sentence, "split", "test"))

    words = source.split()
    source_len = len(words)

    runtime = runtime or ControlledRuntime(
        predictor, translator, cfg, timer=timer, cost_model=cost_model
    )
    builder = builder if builder is not None else FeatureBuilder()

    baseline = _baseline_action(cfg)
    candidate_writes: tuple[Action, ...] = tuple(
        dict.fromkeys(action for action in cfg.actions if not action.is_read)
    )
    # Query the network for the baseline even when it is not selectable.
    query_actions: list[Action] = list(candidate_writes)
    if (
        baseline is not None
        and not baseline.is_read
        and baseline not in query_actions
    ):
        query_actions.append(baseline)

    read_pos = min(max(0, int(cfg.min_start)), source_len)
    committed: tuple[str, ...] = ()
    delays: list[int] = []
    steps: list[dict] = []
    # One state-update implementation shared with the offline label path.
    tracker = HistoryTracker(arrival_gap=float(cfg.arrival_gap))
    completion_s = 0.0
    # Consecutive READ decisions. Failed WRITEs are the other half of the
    # no-commit streak; the two counts are disjoint so the dynamic threshold
    # never charges one wait twice.
    consecutive_reads = 0

    # Every decision advances one source position, so a normal run needs at
    # most `source_len - min_start + 2` iterations. The generous cap is a
    # defensive guard against a future edit that forgets to advance.
    max_steps = max(16, (source_len + 1) * (len(candidate_writes) + 4))
    iterations = 0

    while True:
        iterations += 1
        if iterations > max_steps:
            raise RuntimeError(
                "run_rollout exceeded its step cap; check that every decision "
                "advances read_pos"
            )

        # --------------------------------------------------------------
        # EOS: translate the true full source once with the forced prefix.
        # --------------------------------------------------------------
        if read_pos >= source_len:
            prefix = " ".join(committed)
            outputs, mt_s = runtime.translate([source], prefix, final=True)
            new_words = tuple(outputs[0].split()) if outputs else ()
            completion_s = advance_virtual_clock(
                completion_s, source_len, cfg.arrival_gap, mt_s
            )

            committed_before = len(committed)
            committed = committed + new_words
            delay = _delay_at(completion_s, cfg.arrival_gap, source_len)
            delays.extend([delay] * len(new_words))
            steps.append(
                _write_step(
                    read_pos=source_len,
                    observed=source,
                    action_label="eos",
                    committed_before=committed_before,
                    committed_after=len(committed),
                    new_words=new_words,
                    pred_s=0.0,
                    mt_s=mt_s,
                    spent_s=mt_s,
                    vote_top1=0.0,
                    vote_margin=0.0,
                    vote_entropy=0.0,
                    no_emit=not new_words,
                    q_read=None,
                    q_write=None,
                    cost=None,
                    post_action=None,
                    marginal_wall_s=mt_s,
                )
            )
            break

        # --------------------------------------------------------------
        # Decision point.
        # --------------------------------------------------------------
        observed = " ".join(words[:read_pos])
        vector = tracker.build(
            builder,
            observed,
            committed,
            predictor_id=int(predictor_id),
            translator_id=int(translator_id),
        )
        tri_value = bool(getattr(cfg, "tri_value", False))
        read_components = write_components = None
        if tri_value:
            (
                z_pre,
                read_components,
                write_components,
                no_emit_by_action,
            ) = _score_state_v4(network, vector, query_actions)
            q_read = float(read_components[0])
            q_by_action = {
                action: float(values[0])
                for action, values in write_components.items()
            }
        else:
            z_pre, q_read, q_by_action, no_emit_by_action = _score_state(
                network, vector, query_actions
            )

        src_words = len(observed.split())
        tgt_words = len(committed)
        # The action may start only after the decision clock and the source
        # arrival; the translation may start only after the source arrives.
        release_s = _release_time(completion_s, read_pos, cfg.arrival_gap)
        dependency_s = release_s
        t_pred_free = float(getattr(runtime, "t_pred_free", 0.0))
        t_mt_free = float(getattr(runtime, "t_mt_free", 0.0))

        def price(action: Action) -> float:
            """Pipeline marginal for `action`; never the serial stage sum."""
            if action.is_read or cost_model is None:
                return 0.0
            # A ControlledRuntime with a PipelineCostModel prices the action
            # against its own device scheduled state.
            estimator = getattr(runtime, "estimate_marginal", None)
            if callable(estimator):
                value = estimator(
                    vector.predictor_id,
                    vector.translator_id,
                    action,
                    src_words,
                    tgt_words,
                    release_s=release_s,
                    dependency_s=dependency_s,
                )
                if value is not None:
                    return float(value)
            return _action_cost(
                cost_model,
                vector,
                action,
                src_words,
                tgt_words,
                t_pred_free=t_pred_free,
                t_mt_free=t_mt_free,
                release_s=release_s,
                dependency_s=dependency_s,
            )

        cost_by_action = {
            action: price(action) for action in query_actions
        }
        q_baseline = None
        c_baseline = 0.0
        if baseline is not None:
            q_baseline = q_read if baseline.is_read else q_by_action[baseline]
            if baseline.is_read:
                c_baseline = 0.0
            else:
                c_baseline = cost_by_action.get(baseline)
                if c_baseline is None:
                    # The baseline is queried for the network even when it is
                    # not a selectable action; price it here in that case.
                    c_baseline = price(baseline)

        scores: dict[Action, float] = {}
        if tri_value:
            # V4: U = Qq - beta_lag*Ql - beta_wall*Qw. The wall head already
            # carries time, so lambda_t * cost must not be subtracted again.
            def utility(action: Action) -> float:
                components = (
                    read_components
                    if action.is_read
                    else write_components[action]
                )
                return (
                    float(components[0])
                    - float(cfg.beta_lag) * float(components[1])
                    - float(cfg.beta_wall) * float(components[2])
                )

            baseline_utility = utility(baseline) if baseline is not None else 0.0
            for action in (READ, *candidate_writes):
                scores[action] = utility(action) - baseline_utility
            q_read = utility(READ)
            q_by_action = {
                action: utility(action) for action in candidate_writes
            }
            if baseline is not None and not baseline.is_read:
                q_by_action.setdefault(baseline, utility(baseline))
        else:
            for action in (READ, *candidate_writes):
                q_value = q_read if action.is_read else q_by_action[action]
                cost = 0.0 if action.is_read else cost_by_action[action]
                if baseline is not None:
                    scores[action] = (
                        (q_value - q_baseline)
                        - float(cfg.lambda_t) * (cost - c_baseline)
                    )
                else:
                    scores[action] = q_value - float(cfg.lambda_t) * cost

        # Deterministic tie-break: READ is first, and later actions must beat it
        # strictly. This keeps a zero-initialised network from writing at once.
        # V3.1 adds two guards before the same argmax; all-zero defaults make
        # this exactly the V3 selection.
        normalized_lag = float(
            vector.numeric[NUMERIC_FEATURES.index("normalized_lag")]
        )
        no_commit_streak = max(
            0,
            int(tracker.history.words_since_commit) - consecutive_reads,
        )
        chosen, read_threshold_value, admitted_writes = select_guarded_action(
            scores=scores,
            costs=cost_by_action,
            candidate_writes=candidate_writes,
            baseline=baseline,
            cfg=cfg,
            normalized_lag=normalized_lag,
            consecutive_reads=consecutive_reads,
            no_commit_streak=no_commit_streak,
            no_emit_risk=no_emit_by_action,
        )

        best_write_q = max(
            (q_by_action[action] for action in candidate_writes), default=None
        )

        # --------------------------------------------------------------
        # READ: no sampling, no translation, no commit.
        # --------------------------------------------------------------
        if chosen.is_read:
            consecutive_reads += 1
            completion_s = advance_virtual_clock(
                completion_s, read_pos, cfg.arrival_gap
            )
            steps.append(
                _write_step(
                    read_pos=read_pos,
                    observed=observed,
                    action_label=READ.label,
                    committed_before=len(committed),
                    committed_after=len(committed),
                    new_words=(),
                    pred_s=0.0,
                    mt_s=0.0,
                    spent_s=0.0,
                    vote_top1=0.0,
                    vote_margin=0.0,
                    vote_entropy=0.0,
                    no_emit=True,
                    q_read=q_read,
                    q_write=best_write_q,
                    cost=0.0,
                    post_action=None,
                    marginal_wall_s=0.0,
                )
            )
            tracker.on_read(now_s=completion_s)
            read_pos += 1
            continue

        # --------------------------------------------------------------
        # WRITE: provisional execute, optional post controller, commit.
        # --------------------------------------------------------------
        consecutive_reads = 0
        seed = _step_seed(cfg, sentence_id, read_pos, split)
        provisional: StepResult = _execute_step(
            runtime,
            observed,
            committed,
            chosen,
            seed,
            release_s,
            dependency_s,
        )

        post_action = None
        post_reason = None
        final: StepResult | None = provisional
        extra_branches = 0

        if post_controller is not None:
            decision = post_controller.decide(
                z_pre,
                chosen.branches,
                chosen.horizon,
                provisional.vote_entropy,
                provisional.vote_margin,
                no_emit_by_action.get(chosen, 0.0),
                provisional.spent_s,
                cfg,
            )
            post_action = decision.action.value
            post_reason = decision.reason
            if decision.action is PostAction.ADD_2:
                extra_branches = min(
                    ADD_2_BRANCHES,
                    max(0, int(cfg.max_branches) - int(chosen.branches)),
                )
                if extra_branches > 0:
                    # Cached candidates only: `extend` translates the new
                    # requests and re-runs RALCP over the full pool.
                    final = _extend_step(
                        runtime,
                        observed,
                        committed,
                        provisional.candidates,
                        extra_branches,
                        chosen.horizon,
                        seed,
                        provisional.spent_s,
                        release_s,
                        dependency_s,
                    )
                else:
                    # No branch headroom: the proposal was a no-op, so do not
                    # mislabel the committed provisional result as "add_2".
                    post_action = "add_2_noop"
            elif decision.action is PostAction.WAIT:
                final = None

        committed_before = len(committed)
        used_branches = int(chosen.branches) + int(extra_branches)

        if final is None:
            # WAIT: the provisional work is billed but nothing is committed.
            spent = provisional.spent_s
            new_words: tuple[str, ...] = ()
            vote_source = provisional
            decision_marginal = _marginal_of(provisional)
        else:
            spent = final.spent_s
            previous = committed
            new_words = tuple(final.committed_after[len(previous):])
            committed = tuple(final.committed_after)
            delays.extend([read_pos] * len(new_words))
            vote_source = final
            if final is provisional:
                decision_marginal = _marginal_of(provisional)
            else:
                provisional_marginal = _marginal_of(provisional)
                extension_marginal = _marginal_of(final)
                decision_marginal = (
                    provisional_marginal + extension_marginal
                    if provisional_marginal is not None
                    and extension_marginal is not None
                    else None
                )

        # Bill the incremental pipeline wall when the runtime measured it. The
        # serial `spent_s` is only a documented fallback for fake runtimes.
        billed_s = (
            float(spent) if decision_marginal is None else float(decision_marginal)
        )
        completion_s = advance_virtual_clock(
            completion_s, read_pos, cfg.arrival_gap, billed_s
        )

        # `StepResult.pred_s` / `mt_s` from `extend` cover only the extension,
        # while `spent_s` is already cumulative. Report the whole decision's
        # measured stage compute as a diagnostic; the virtual clock is billed
        # from `decision_marginal` (or `spent_s` only for fake runtimes), never
        # from this serial sum.
        if final is None or final is provisional:
            step_pred_s = provisional.pred_s
            step_mt_s = provisional.mt_s
        else:
            step_pred_s = provisional.pred_s + final.pred_s
            step_mt_s = provisional.mt_s + final.mt_s

        steps.append(
            _write_step(
                read_pos=read_pos,
                observed=observed,
                action_label=chosen.label,
                committed_before=committed_before,
                committed_after=len(committed),
                new_words=new_words,
                pred_s=step_pred_s,
                mt_s=step_mt_s,
                spent_s=spent,
                vote_top1=vote_source.vote_top1,
                vote_margin=vote_source.vote_margin,
                vote_entropy=vote_source.vote_entropy,
                no_emit=not new_words,
                q_read=q_read,
                q_write=q_by_action[chosen],
                cost=cost_by_action[chosen],
                post_action=post_action,
                post_reason=post_reason,
                marginal_wall_s=decision_marginal,
            )
        )

        tracker.on_write(
            SimpleNamespace(
                action=Action(used_branches, chosen.horizon),
                emitted=len(new_words),
                no_emit=not new_words,
                vote_top1=vote_source.vote_top1,
                vote_margin=vote_source.vote_margin,
                vote_entropy=vote_source.vote_entropy,
            ),
            now_s=completion_s,
        )

        # One decision per source position, exactly like `week_run.rollout`:
        # the next source word arrives during this step whether or not the
        # controller spent compute on it.
        read_pos += 1

    hypothesis = " ".join(committed)
    laal = _laal(delays, source_len, len(reference.split()), len(committed))
    histogram = Counter(
        step["action"] for step in steps if step["action"] != "eos"
    )
    decisions = sum(histogram.values())
    read_fraction = histogram.get("read", 0) / decisions if decisions else 0.0

    return RolloutTrace(
        sentence_id=sentence_id,
        source=source,
        reference=reference,
        hypothesis=hypothesis,
        committed=committed,
        delays=tuple(delays),
        laal=laal,
        completion_s=float(completion_s),
        read_fraction=float(read_fraction),
        action_histogram=dict(histogram),
        steps=tuple(steps),
    )


def run_fixed_rollout(
    sentence,
    predictor,
    translator,
    builder,
    cost_model,
    cfg: ControllerConfig,
    predictor_id: int = 0,
    translator_id: int = 0,
    timer=None,
    *,
    runtime=None,
) -> RolloutTrace:
    """Roll out the fixed ``(4, 6)`` baseline on the controller's own path.

    This is the matched baseline for a learned rollout: it uses the same
    `ControlledRuntime`, the same per-candidate sampling seeds, the same
    translation call, RALCP gate, and virtual-clock/timing accounting. The only
    differences are the policy (always :data:`FIXED_ACTION`, never READ) and
    the absence of value scoring.

    `builder` and `cost_model` are accepted so the baseline has the same call
    shape as `run_rollout`. The fixed policy scores nothing, so `cost_model` is
    not consulted; timings come from the runtime's measured stages. `runtime`
    may be injected for tests; otherwise one is constructed.
    """
    cfg = cfg if cfg is not None else ControllerConfig()
    source = str(_field(sentence, "source", ""))
    reference = str(_field(sentence, "reference", ""))
    sentence_id = str(_field(sentence, "sentence_id", ""))
    split = str(_field(sentence, "split", "test"))
    words = source.split()
    source_len = len(words)
    arrival_gap = float(cfg.arrival_gap)

    del builder  # The fixed policy has no state features or network scoring.
    runtime = runtime or ControlledRuntime(predictor, translator, cfg, timer=timer)

    action = FIXED_ACTION
    read_pos = min(max(0, int(cfg.min_start)), source_len)
    committed: tuple[str, ...] = ()
    delays: list[int] = []
    steps: list[dict] = []
    completion_s = 0.0

    while read_pos < source_len:
        observed = " ".join(words[:read_pos])
        seed = _step_seed(cfg, sentence_id, read_pos, split)
        release_s = _release_time(completion_s, read_pos, arrival_gap)
        result = _execute_step(
            runtime,
            observed,
            committed,
            action,
            seed,
            release_s,
            release_s,
        )

        committed_before = len(committed)
        new_words = tuple(result.committed_after[committed_before:])
        committed = tuple(result.committed_after)
        delays.extend([read_pos] * len(new_words))
        marginal = _marginal_of(result)
        completion_s = advance_virtual_clock(
            completion_s,
            read_pos,
            arrival_gap,
            _billed_seconds(result),
        )
        steps.append(
            _write_step(
                read_pos=read_pos,
                observed=observed,
                action_label=action.label,
                committed_before=committed_before,
                committed_after=len(committed),
                new_words=new_words,
                pred_s=result.pred_s,
                mt_s=result.mt_s,
                spent_s=result.spent_s,
                vote_top1=result.vote_top1,
                vote_margin=result.vote_margin,
                vote_entropy=result.vote_entropy,
                no_emit=not new_words,
                q_read=None,
                q_write=None,
                cost=None,
                post_action=None,
                marginal_wall_s=marginal,
            )
        )
        read_pos += 1

    # Source EOS: one final translation of the true source with the committed
    # prefix forced, exactly like `run_rollout`.
    prefix = " ".join(committed)
    outputs, mt_s = runtime.translate([source], prefix, final=True)
    new_words = tuple(outputs[0].split()) if outputs else ()
    completion_s = advance_virtual_clock(
        completion_s, source_len, arrival_gap, mt_s
    )
    committed_before = len(committed)
    committed = committed + new_words
    delay = _delay_at(completion_s, arrival_gap, source_len)
    delays.extend([delay] * len(new_words))
    steps.append(
        _write_step(
            read_pos=source_len,
            observed=source,
            action_label="eos",
            committed_before=committed_before,
            committed_after=len(committed),
            new_words=new_words,
            pred_s=0.0,
            mt_s=mt_s,
            spent_s=mt_s,
            vote_top1=0.0,
            vote_margin=0.0,
            vote_entropy=0.0,
            no_emit=not new_words,
            q_read=None,
            q_write=None,
            cost=None,
            post_action=None,
            marginal_wall_s=mt_s,
        )
    )

    hypothesis = " ".join(committed)
    laal = _laal(delays, source_len, len(reference.split()), len(committed))
    histogram = Counter(
        step["action"] for step in steps if step["action"] != "eos"
    )
    decisions = sum(histogram.values())
    read_fraction = histogram.get("read", 0) / decisions if decisions else 0.0
    return RolloutTrace(
        sentence_id=sentence_id,
        source=source,
        reference=reference,
        hypothesis=hypothesis,
        committed=committed,
        delays=tuple(delays),
        laal=laal,
        completion_s=float(completion_s),
        read_fraction=float(read_fraction),
        action_histogram=dict(histogram),
        steps=tuple(steps),
    )
