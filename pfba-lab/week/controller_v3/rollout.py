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

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

from .action import READ, Action
from .config import ControllerConfig
from .features import FeatureBuilder, HistoryTracker
from .post_controller import ADD_2_BRANCHES, PostAction
from .runtime import ControlledRuntime, StepResult, run_pfba

__all__ = ["RolloutTrace", "run_rollout"]


@dataclass(frozen=True)
class RolloutTrace:
    """Everything one sentence's rollout produced.

    `steps` holds one dict per decision plus a final `"eos"` step. Each dict
    carries at least the fields documented in the task contract:
    ``read_pos, action, observed_len, committed_before, committed_after,
    new_words, pred_s, mt_s, spent_s, vote_top1, vote_margin, vote_entropy,
    no_emit, q_read, q_write, cost, post_action``.
    `committed_before` / `committed_after` are word counts; `new_words` is the
    tuple of words this step added. For a write step, `pred_s` / `mt_s` are the
    full decision's measured compute (initial execute plus any ADD_2
    extension), `spent_s` is the cumulative measured compute billed to the
    virtual clock, and discarded WAIT work appears in `spent_s` but not in the
    committed words.
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


def _action_cost(
    cost_model, vector, action: Action, src_words: int, tgt_words: int
) -> float:
    """Virtual wall cost of one write action; READ is always free."""
    if action.is_read or cost_model is None:
        return 0.0
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

    runtime = runtime or ControlledRuntime(predictor, translator, cfg, timer=timer)
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
    tracker = HistoryTracker()
    completion_s = 0.0

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
            completion_s = max(
                completion_s, (source_len - 1) * float(cfg.arrival_gap)
            )
            outputs, mt_s = runtime.translate([source], prefix, final=True)
            new_words = tuple(outputs[0].split()) if outputs else ()
            completion_s += float(mt_s)

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
        z_pre, q_read, q_by_action, no_emit_by_action = _score_state(
            network, vector, query_actions
        )

        src_words = len(observed.split())
        tgt_words = len(committed)
        cost_by_action = {
            action: _action_cost(cost_model, vector, action, src_words, tgt_words)
            for action in query_actions
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
                    c_baseline = _action_cost(
                        cost_model, vector, baseline, src_words, tgt_words
                    )

        scores: dict[Action, float] = {}
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
        chosen = READ
        best_score = scores[READ]
        for action in candidate_writes:
            if scores[action] > best_score:
                chosen = action
                best_score = scores[action]

        best_write_q = max(
            (q_by_action[action] for action in candidate_writes), default=None
        )

        # --------------------------------------------------------------
        # READ: no sampling, no translation, no commit.
        # --------------------------------------------------------------
        if chosen.is_read:
            completion_s = max(
                completion_s, (read_pos - 1) * float(cfg.arrival_gap)
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
                )
            )
            tracker.on_read(now_s=completion_s)
            read_pos += 1
            continue

        # --------------------------------------------------------------
        # WRITE: provisional execute, optional post controller, commit.
        # --------------------------------------------------------------
        seed = _step_seed(cfg, sentence_id, read_pos, split)
        provisional: StepResult = runtime.execute(observed, committed, chosen, seed)

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
                    final = runtime.extend(
                        observed,
                        committed,
                        provisional.candidates,
                        extra_branches,
                        chosen.horizon,
                        seed,
                        spent_s=provisional.spent_s,
                    )
                else:
                    # No branch headroom: the proposal was a no-op, so do not
                    # mislabel the committed provisional result as "add_2".
                    post_action = "add_2_noop"
            elif decision.action is PostAction.WAIT:
                final = None

        completion_s = max(
            completion_s, (read_pos - 1) * float(cfg.arrival_gap)
        )
        committed_before = len(committed)
        used_branches = int(chosen.branches) + int(extra_branches)

        if final is None:
            # WAIT: the provisional work is billed but nothing is committed.
            spent = provisional.spent_s
            completion_s += float(spent)
            new_words: tuple[str, ...] = ()
            vote_source = provisional
        else:
            spent = final.spent_s
            completion_s += float(spent)
            previous = committed
            new_words = tuple(final.committed_after[len(previous):])
            committed = tuple(final.committed_after)
            delays.extend([read_pos] * len(new_words))
            vote_source = final

        # `StepResult.pred_s` / `mt_s` from `extend` cover only the extension,
        # while `spent_s` is already cumulative. Report the whole decision's
        # measured compute so pred_s + mt_s + discarded WAIT work explain the
        # virtual clock without double-counting `spent_s`.
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
