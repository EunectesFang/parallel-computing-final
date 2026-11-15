# CAVC v3 Fix Wave — Modeling Correctness, Runtime Path, Auditability

**Base** `cavc-controller-v3` at code `4984b63` (docs `4d1acbf`).
**Original plan** `pfba-lab/docs/superpowers/plans/2026-11-15-cavc-controller-v3.md`.
**Why** The first implementation passed unit tests but the final reviews found the
Bellman target is action-independent (shared successor), training `History` is all
zeros, the labeled trajectory never selects READ, delay is state-constant, the
terminal reward is unused, and the cost/risk/post-controller machinery is not in
the production decision path. This wave fixes those before any Kaggle dev run.

**Rules** Only edit under `pfba-lab/week/controller_v3/` unless a task says
otherwise. CPU tests use `D:/MINICONDA/envs/py312/python.exe`. No downloads,
GPU, network, push, or submit. Tests use fakes. English docstrings. Do not
dispatch subagents. Keep `week_run.py`, `joint_gate.py`, and `run_pfba.py`
unchanged.

## Task 8: Modeling core — per-action successors, history, READ trajectory, stride, terminal

Files:
- `pfba-lab/week/controller_v3/features.py`
- `pfba-lab/week/controller_v3/reward.py`
- `pfba-lab/week/controller_v3/targets.py`
- `pfba-lab/week/controller_v3/trainer.py`
- `pfba-lab/week/controller_v3/rollout.py`
- `pfba-lab/week/controller_v3/kaggle.py`
- new `tests/test_modeling_core.py`; update `test_trainer.py`,
  `test_labeler.py`, `test_kaggle_wiring.py`, `test_integration.py`.

Requirements:

1. **`HistoryTracker` in `features.py`** — one state-update implementation used
   by both labeling and rollout.
   - Holds `History`, `RuntimeInfo`, and `last_commit_s`.
   - `on_read()`: `words_since_commit += 1`, `no_emit_streak += 1`,
     `last_empty=True`, `last_commit_count=0`, `last_branches=0`,
     `last_horizon=0`; time fields updated.
   - `on_write(outcome)`: set `last_branches`, `last_horizon`,
     `last_vote_top1/margin/entropy`; `last_commit_count=emitted`;
     `last_empty=no_emit`; if `emitted > 0` reset `words_since_commit` and
     `no_emit_streak`, else increment both.
   - `build(builder, observed, committed, predictor_id, translator_id)` and
     `snapshot()`.
   - `rollout.py` must use it instead of hand-updating `History`.

2. **Per-action successor in `targets.py`.**
   - `build_action_targets(state, outcomes, cfg, *, next_states:
     Sequence[StateInput] | None = None, next_state: StateInput | None = None,
     next_actions=(), src_words_after: int | None = None,
     tgt_words_after_per_outcome: Sequence[int] | None = None,
     terminal: bool = False, terminal_value: float | None = None,
     backlog=0.0, source_advance=1, state_key="", rho=1.0)`.
   - Pair `next_states[i]` with `outcomes[i]`; `next_state` is a legacy ablation
     only.
   - Delay penalty uses post-action backlog:
     `backlog_after = max(0, src_words_after - tgt_words_after / rho)`, with
     `tgt_words_after = current_tgt + outcome.emitted`.
   - Terminal: `terminal=True`, `next_state=None`, `value_target=terminal_value`;
     no bootstrap.
   - `state_key` non-empty; `next_actions=(READ, *cfg.actions)`.

3. **`trainer.compute_targets`** — use each target's own `next_state`; terminal
   or missing next state uses `value_target` only; cache the one-step bonus per
   unique `(next_state, next_actions)` within one call and batch unique next
   states when possible. `fit` returns actual `epochs` and `n_samples`; zero
   epochs/none targets means zero optimization and `epochs=0`.

4. **`kaggle.label_sentence` true one-step trajectories.**
   - Default `stride=1`; raise `ValueError` for `stride>1` unless a real
     multi-step target is implemented with accumulated discounted rewards.
   - At every kept position, build the state through `HistoryTracker`, label
     `cfg.actions`, build the READ outcome, and construct one successor per
     action:
     - READ: `observed_next = words[:read_pos+1]`, committed unchanged,
       `tracker.on_read()`.
     - WRITE: `observed_next = words[:read_pos+1]`,
       `committed = outcome.next_committed`, `tracker.on_write(outcome)`.
   - Last decision position: `terminal=True`, `terminal_value` from the final
     hypothesis/reference proxy (documented; not BLEU/COMET).
   - Behavior policy must include READ: `read_prob` (default 0.25) chooses READ;
     otherwise fixed `(4,6)`. The behavior policy is only for state coverage;
     the Bellman target comes from labeled outcomes, not an immediate-reward
     argmax.
   - Persist per-outcome `next_committed`, `emitted`, vote stats, and the
     successor state inputs; do not store full candidate text.
   - Record `behavior` (`read_prob`, `fixed_action`) in the summary.

5. **Terminal reward** — `reward.terminal_reward` remains a documented proxy
   (word-length difference plus optional delay); used for the last kept state.
   Do not mix BLEU/COMET into local rewards.

6. **Tests:**
   - Different writes produce different successors; READ keeps `committed` and
     advances `observed`.
   - READ can win when its network Q is higher; behavior can produce READ
     transitions.
   - Offline History replay equals online History construction.
   - `compute_targets` uses per-target successors; cache is correct.
   - `stride=1` is a true one-step transition; `stride>1` raises unless the
     multi-step path exists.
   - Terminal has no bootstrap and uses the terminal value.
   - Delay changes when `emitted` changes the post-action backlog.
   - Zero-epoch fit reports `epochs=0`.

7. Run the full suite. Commit:
   `Fix CAVC modeling core: per-action successors, history, READ trajectory, terminal`

## Task 9: Runtime cost, risk/post-controller, fixed baseline, feature diagnostics

Files:
- `cost_model.py`, `runtime.py`, `rollout.py`, `post_controller.py`,
  `features.py`, `checkpoint.py`, `kaggle.py`, `evaluator.py`
- update `test_cost_model.py`, `test_rollout.py`, `test_post_controller.py`,
  `test_kaggle_wiring.py`; optional new `test_runtime_diagnostics.py`.

Requirements:

1. **Runtime cost accounting.** `ControlledRuntime` tracks `t_pred_free`,
   `t_mt_free`, and accepts a `PipelineCostModel`; `StepResult` adds
   `marginal_wall_s`. `execute`/`extend` return `pred_s`, `mt_s`, `spent_s`,
   `marginal_wall_s`. No caller may call `pred_s + mt_s` pipeline wall time.
   `calibrate_cost_records` records all three; rollout uses
   `marginal_wall_s` for scoring/completion when available.

2. **Risk/post-controller.** Construct `PostController()` in `kaggle.py` when
   `--post-controller` is on, passing the selected action's `no_emit`,
   vote stats, and `spent_s`. When off, record `"post_controller": "disabled"`
   and state that risk heads are training-only auxiliaries. Add Brier, ECE-style
   bucket error, and no-emit calibration to dev/confirm summaries. Never use an
   uncalibrated sigmoid as a hard threshold.

3. **Fixed `(4,6)` baseline on the same path.** Add `run_fixed_rollout(...)`
   using `ControlledRuntime`, the same per-candidate sampling/translation/RALCP/
   timing, always `(4,6)`, never READ. `run_dev`/`run_confirm` evaluate both
   learned and fixed; write sentence-paired BLEU/COMET/LAAL/completion intervals
   to `paired.json`.

4. **Feature diagnostics.** `features.feature_diagnostics(vectors)` returns per
   feature mean/std/nonzero/missing and train-serve mean difference.
   `run_preflight`/`run_dev` write `feature_stats.json` and warn when a feature
   is constant in training but varies in serving. Add synthetic tests.

5. **Checkpoint pair/vocab metadata.** Store `model_pair`, `vocab`, and
   `feature_stats` in checkpoint `extra`; `run_confirm` validates them against
   resolved adapters and refuses mismatches. Never rebuild a test-derived vocab.

6. **Candidate-generation parity.** Make `labeler` use the same per-candidate
   `seed + i` path as `ControlledRuntime.predict` (or add an explicit
   `candidate_mode` and validate both). A labeler that uses one batched
   `num_return_sequences` call while the runtime uses per-candidate calls makes
   the labeled action return distribution differ from the executed one. Add a
   parity test: the same observed prefix, action, and seed produce the same
   candidate token rows/strings (or a documented, tested distribution match).
   Record the chosen mode in labels and summaries.

7. **Tests:** marginal wall vs serial sum; post-controller on/off; fixed
   baseline uses the same path; diagnostics detect constant features;
   pair/vocab mismatch raises; paired bootstrap written; candidate-generation
   parity or a tested distribution match.

7. Run the full suite. Commit:
   `Wire runtime cost, risk/post controller, fixed baseline, feature diagnostics`

## Task 10: Strict checks, target performance, artifacts, bundle, docs

Files:
- `kaggle.py`, `trainer.py`, `checkpoint.py`, `README.md`,
  `pfba-lab/prepare_cavc_kernel.py`
- update `test_kaggle_wiring.py`, `test_integration.py`.

Requirements:

1. `trained=True` only when `fit` ran at least one optimization epoch and saw at
   least one sample; otherwise `trained=False` and no checkpoint.
2. `compute_targets` caching/batching enforced; target wall time logged in the
   dev summary; regression test with many targets sharing a successor.
3. `run_dev`/`run_confirm` strict model-pair, vocab, cost-record, and
   normalization agreement; mismatch raises.
4. Deadline-stopped runs must not overwrite a non-empty `summary.json`/
   `traces.jsonl` with empty content; use a suffixed file or skip.
5. Test-derived vocabulary fallback removed; `run_confirm` requires stored vocab.
6. Real-adapters path uses `run_pfba.StageTimer(sync)` for calibration/rollout.
7. `prepare_cavc_kernel.py` verifies committed bundle hashes; add a test that
   the committed `kaggle-runs/pfba-cavc-v3/kaggle_entry.py` and
   `kernel-metadata.json` byte-match generator output.
8. README corrections: `test:0000–0139` consumed, `test:0140+` clean; resume
   uses `labels.jsonl`/`checkpoint.pt`; post-controller on/off; no quality claim
   from smoke.
9. Add paired-bootstrap summaries to dev/confirm; `test:0140+` never tuned.
10. Confirm risk calibration: persist dev holdout calibration targets with the
    checkpoint, or explicitly state in the confirm summary that calibration
    comes from dev. Never calibrate on test.
11. Deadline early-return schema: emit `risk_calibration`, `rollout_fixed`,
    `paired`, and `artifacts` as `null` with a reason, so consumers can
    distinguish "not run" from "absent".
12. Regenerate the committed `kaggle-runs/pfba-cavc-v3/{kaggle_entry.py,
    kernel-metadata.json}` at the final HEAD and verify byte/hash equality.
13. Add the missing tests: `estimate_marginal -> PipelineCostModel` pricing in
    rollout; post-controller actually changes a step (not just the summary
    string); preflight writes `feature_stats.json`; dev calibration values
    match an independently computed holdout; legacy-vs-six-field cost-record
    checkpoint round-trip; untrained risk calibration is `unavailable`;
    test-derived vocab is refused.
14. Cheap hardening: warn (not silently fall back) when a plain cost model is
    used instead of `PipelineCostModel`, and when `per_candidate` downgrades to
    `batched`; validate `risk_calibration` `buckets > 0`; validate feature-stat
    names in list form; compare checkpoint pair identity to resolved adapters,
    not caller-declared names.
15. Run the full suite and prepare script. Commit:
    `Harden CAVC experiment artifacts, timing, and kernel bundle`

## Final acceptance

- `pytest -q pfba-lab/week/controller_v3/tests` passes.
- Tests exist and pass for: per-action successor, READ selection, History
  replay, terminal value, marginal wall, fixed baseline, feature diagnostics,
  pair/vocab mismatch, bundle staleness.
- Re-review the three original structural findings: shared successor, training
  History, READ trajectory.
- No claim of BLEU/COMET/LAAL/value-learning from CPU smoke.
