# CAVC v3: cost-aware value controller

CAVC v3 replaces the `week_run.py` one-hot action classifier with an
action-value controller for source-streaming translation. At every source
position it either **READs** (waits for the next source word) or **WRITEs** a
`(branches, horizon)` budget, and chooses the action whose value minus explicit
runtime cost is largest.

The four design rules are:

1. **Value learning, not classification.** One network scores READ and every
   write action on a shared scale.
2. **Unified READ/WRITE scale.** READ is a first-class action in training and
   in the Bellman backup.
3. **Explicit runtime cost.** Cost comes from a measured cost model, not a
   training-time constant.
4. **Counterfactual labels, no leakage.** Online features never see the full
   source, the reference, or future words; labels do, in a separate path.

**Status: this code has not been trained on Kaggle data.** No BLEU or COMET
result is claimed anywhere in this directory. `sacrebleu` is not installed in
the local CPU environment, so every BLEU result is reported as
`{"status": "unavailable", "reason": ...}` and the score field is absent.
COMET is opt-in only and was not run; loading it downloads model weights.

## Architecture

```text
observed source prefix        committed target prefix        history / runtime /
(src words, <= 16 tokens)     (tgt words, <= 16 tokens)      confidence / costs
          |                            |                              |
   src embedding + GRU          tgt embedding + GRU           34-value numeric
   masked mean pool             masked mean pool              MLP + LayerNorm
          \____________________________|____________________________/
                                       |
                        predictor / translator embeddings (8 x 8)
                                       |
                          fusion MLP -> state z (64)
                          /                              \
                  READ head                       action encoder (6 -> 16)
                  Q_read                          concat(z, action) -> write head
                                                  /       |       |       \
                                             Q_write  error  no_emit  expected
                                                       risk   risk     emit
```

The network encodes a state once. `read_value(z)` and
`write_values(z, actions)` share the same `z`, so their Q values are directly
comparable. Write actions are described by six numbers:
`(branches, horizon, branches*horizon, log branches, log horizon,
ceil(threshold*branches)/branches)`.

## Module map

| Module | Responsibility |
|---|---|
| `action.py` | `Action`, READ, 9-action default grid, 25-action extended grid, 6 action features |
| `config.py` | Frozen `ControllerConfig` shared by every module |
| `features.py` | 34-value online state vector, `StateInput`, history/runtime/confidence records |
| `text_encoder.py` | Word `Vocab` and the two-GRU CPU text encoder |
| `network.py` | `CAVCNetwork`: READ head plus action-conditioned write heads |
| `reward.py` | Immediate quality/latency reward and the terminal length proxy |
| `labeler.py` | Counterfactual `(state, action)` outcomes; training-only oracle path |
| `targets.py` | Converts `ActionOutcome` records into `ActionTarget` rows |
| `trainer.py` | Multi-task losses, one-step Bellman targets, pairwise ranking |
| `cost_model.py` | Measured table, power-law fit, pipeline marginal-wall model |
| `runtime.py` | Candidate sampling, translation, RALCP, per-candidate seeds |
| `post_controller.py` | Optional STOP / ADD_2 / WAIT decision after a provisional write |
| `rollout.py` | Greedy one-sentence rollout and `RolloutTrace` |
| `evaluator.py` | Trace summaries, status-aware BLEU/COMET, paired bootstrap |
| `checkpoint.py` | Save/load state, config, feature stats, cost records, seed, git commit |
| `cli.py` | `--dry-run`, synthetic `--smoke`, and explicit unimplemented real-data paths |
| `kaggle.py` | Real-model adapters, cost calibration, dev labeling with resume, preflight/dev/confirm runs |
| `tests/` | CPU-only unit and integration tests; all model adapters are fakes |

## Numeric features (exactly 34)

The order below is `features.NUMERIC_FEATURES`; the trainable network uses this
order and checkpoints validate the length against `cfg.numeric_dim`.

Surface and history (26):

1. `src_words`
2. `src_chars`
3. `tgt_words`
4. `tgt_chars`
5. `length_ratio`
6. `normalized_lag`
7. `words_since_commit`
8. `no_emit_streak`
9. `last_B`
10. `last_H`
11. `last_vote_top1`
12. `last_vote_margin`
13. `last_vote_entropy`
14. `last_commit_count`
15. `last_empty`
16. `src_end_punct`
17. `tgt_end_punct`
18. `last_capital`
19. `subordinate_marker`
20. `current_clause_len`
21. `pred_cost_ref`
22. `mt_cost_ref`
23. `pred_queue_work`
24. `mt_queue_work`
25. `time_since_commit`
26. `arrival_slack`

Confidence values (4):

27. `mt_next_max_prob`
28. `mt_next_entropy`
29. `mt_prefix_logprob`
30. `pred_next_entropy`

Availability masks (4):

31. `mt_next_max_prob_available`
32. `mt_next_entropy_available`
33. `mt_prefix_logprob_available`
34. `pred_next_entropy_available`

An unavailable confidence value is stored as `0.0` **and** its mask as `0.0`,
so the network can distinguish "zero measurement" from "no measurement".

## Action grid

`READ = Action(0, 0)`.

`DEFAULT_ACTIONS` has **9 write actions**:

```text
B \ H    4     6     10
2      b2h4  b2h6  b2h10
4      b4h4  b4h6  b4h10
6      b6h4  b6h6  b6h10
```

`EXTENDED_ACTIONS` has **25 write actions** with `B, H in {2, 4, 6, 8, 10}`.
There is normally one additional selectable action, READ, for 10 candidates in
the default configuration.

A write action samples `B` source-side candidates, truncates each to `H`
tokens, translates every request with the committed prefix forced, and runs
RALCP over the translated pool. READ performs no prediction and no translation.

## Per-candidate sampling rule

The rollout runtime samples each candidate with its own call:

```python
for i in range(action.branches):
    predictor.sample(observed, branches=1, horizon=action.horizon, seed=seed + i)
```

Candidate `i` therefore always uses `seed + i`, independent of the budget. This
makes candidate sets **nested**: `B=2` is exactly the first two candidates of
`B=4`, and growing the budget adds candidates instead of redrawing the stream.
The labeler expects the same nesting from `sample_token_rows(...)`.

Any matched baseline must use this per-candidate path. A baseline that calls
`model.generate(..., num_return_sequences=B)` in one call draws from a different
sampling stream and cannot be compared directly with a controller using
`seed + i` calls, even at the same nominal seed.

## Loss definitions

For a labelled row `(s, a)` with immediate reward `r` and successor `s'`:

```text
y = r                                           if s' is None or terminal
    r + gamma * max_{a' in (READ, *actions)} [ Q_target(s', a') - lambda_t * C(s', a') ]
                                                otherwise

value_loss    = Huber(Q(s, a), y, delta=huber_delta)
rank_loss     = mean over same-state write pairs (i, j) with |y_i - y_j| > rank_epsilon
                of softplus(-(Q_i - Q_j)), at most 32 pairs per state
error_loss    = BCEWithLogits(error_logit, 1[incorrect > 0])
no_emit_loss  = BCEWithLogits(no_emit_logit, 1[no_emit])
emit_loss     = Huber(softplus(emit_logit), emitted, delta=huber_delta)

total = alpha_q * value_loss
      + alpha_rank * rank_loss
      + alpha_error * error_loss
      + alpha_noemit * no_emit_loss
      + alpha_emit * emit_loss
```

`r` is the immediate reward:

```text
quality = weight_m * matched - weight_e * incorrect
delay   = lambda_l * max(0, backlog) * max(0, source_advance)
r       = quality - delay
```

READ contributes `matched = incorrect = 0`, so its only immediate reward is the
negative delay penalty. The trainer adds the bootstrap term; `build_action_targets`
never does. Rows with an empty `state_key` are skipped by the ranking loss, which
is why `build_action_targets` derives a stable SHA-256 key when the caller leaves
it empty. Production callers should still pass an explicit key such as
`f"{sentence_id}:{read_pos}"`.

The ranking loss only pairs **write** actions from the same state. READ is
excluded because its role is captured by the value target, and pairing READ
against writes would train the two heads against each other on the same row set.

## Leakage boundary

Two paths are deliberately separate:

* **Online features** (`FeatureBuilder.build`) see only the observed source
  prefix, the committed target prefix, history statistics produced earlier in
  the same sentence, runtime/cost information, and optional confidence values.
  They never receive the full source, source length, reference, oracle
  continuation, or any future source word. `tests/test_integration.py` asserts
  that the method signature has no `source`, `reference`, `future`, or
  `full_source` parameter and no `**kwargs` escape hatch.
* **Training labels** (`CounterfactualLabeler.label_state` and
  `oracle_words`) may read the full source and the oracle translation. They run
  offline to produce `ActionOutcome` records. `targets.build_action_targets`
  consumes only those labelled outcomes; it never reads raw text.
* **Rollout** (`run_rollout`) streams the source one position at a time. The
  reference is used only after the hypothesis exists, for LAAL/BLEU reporting.

Only the commit path appends to `committed`. Provisional RALCP results and
post-controller proposals never mutate it.

## `ActionTarget` data format

```python
@dataclass(frozen=True)
class ActionTarget:
    state: StateInput             # 34 numeric values, model ids, text ids
    action: Action                # READ = (0, 0); writes = (branches, horizon)
    value_target: float           # IMMEDIATE reward; trainer adds bootstrap
    error_target: float           # 1.0 if incorrect > 0 else 0.0
    no_emit_target: float         # 1.0 if RALCP emitted nothing else 0.0
    emit_target: float            # number of words emitted
    state_key: str                # non-empty identity used to group actions
    next_state: StateInput | None # successor state for the one-step backup
    next_actions: tuple[Action, ...]  # (READ, *cfg.actions) when next_state is set
    terminal: bool                # True disables the bootstrap term
```

`StateInput` is `(numeric: tuple[float, ...], predictor_id: int,
translator_id: int, src_ids: tuple[int, ...], tgt_ids: tuple[int, ...])`.
Text id tuples may be empty; the text encoder maps empty rows to zeros.

`build_action_targets` emits one row per outcome, all with the same state and
state key. READ rows carry zero auxiliary targets because they never use the
write heads.

## Checkpoints

`save_checkpoint(path, network, cfg, feature_mean, feature_std, cost_model,
extra=None)` writes:

* the network `state_dict`;
* the config as a plain dict (actions as `[branches, horizon]` lists);
* feature mean and standard deviation;
* measured cost records `(predictor_id, translator_id, action, pred_s, mt_s)`;
* the seed and, when available, the current git commit.

`load_checkpoint(path, network)` compares every key and tensor shape against the
target network before copying anything. Missing keys, unexpected keys, shape
mismatches, and feature-stat length mismatches raise a `ValueError` that names
the offending entries, so a wrong architecture never leaves a partially loaded
network.

## Metric honesty

* `evaluator.corpus_bleu` delegates to `run_pfba.corpus_bleu`. Without
  `sacrebleu` it returns `status="unavailable"` and no score.
* The `--smoke` path never calls `corpus_bleu` and always writes
  `{"status": "unavailable", "reason": "synthetic smoke data"}`. This holds
  even when `sacrebleu` is installed: the smoke sentences are synthetic, so no
  score would be meaningful.
* `evaluator.comet_score` delegates to `run_pfba.comet_score`;
  `summarize_traces(include_comet=True)` is the opt-in API because the model
  download needs network access. The smoke always skips COMET.
* `reward.terminal_reward` is only a negative absolute word-count difference.
  It is not COMET and is never reported as a translation-quality score.
* `paired_bootstrap` raises `MetricUnavailableError` when per-sentence BLEU
  cannot be computed instead of substituting a different metric.

## Post-candidate controller

After a provisional write, `PostController` may propose:

* `STOP`: commit the provisional RALCP result;
* `ADD_2`: sample two more branches through the runtime and re-run RALCP over
  the full pool (the extra candidates can change the winner);
* `WAIT`: discard the provisional result and commit nothing.

WAIT is deliberately **non-monotonic**: spending compute can leave the committed
prefix unchanged. ADD_2 can also change the selected prefix, not merely extend
it. The thresholds in `post_controller.py` are hand-picked defaults, not tuned
on data.

## Running the CPU tests

From the repository root:

```bash
"D:/MINICONDA/envs/py312/python.exe" -m pytest -q pfba-lab/week/controller_v3/tests
```

The suite uses fake predictors/translators/tokenizers and a tiny synthetic
dataset. It needs no model weights, GPU, network access, or downloaded data.

Inspect the configuration without loading models:

```bash
cd pfba-lab/week
"D:/MINICONDA/envs/py312/python.exe" -m controller_v3.cli --dry-run
```

Run the synthetic end-to-end smoke (three sentences, three states x three
actions, five epochs, checkpoint round trip, two rollouts):

```bash
cd pfba-lab/week
"D:/MINICONDA/envs/py312/python.exe" -m controller_v3.cli --smoke \
  --output-dir outputs/cavc_smoke
```

This writes `checkpoint.pt`, `summary.json`, and `traces.jsonl` under the
output directory. `summary.json` contains the training losses, the checkpoint
metadata, and the rollout summary; `traces.jsonl` contains one JSON record per
sentence with every decision step. The smoke forces BLEU to `unavailable` and
COMET to `skipped` because its three sentences are synthetic; no
translation-quality score is produced or implied.

`--train-labels PATH` and `--rollout` are parsed but raise
`NotImplementedError`: they need a real predictor/translator adapter, and the
CLI does not download or load model weights.

## Deliberate simplifications

* Branch variants at one horizon share the max-`B` translation time measured for
  that horizon. The measured cost table is keyed by `(branches, horizon)` and
  cannot separate variants produced by a single translation batch.
* The trainer's cost term calls the cost model on the successor state; it does
  not simulate the full two-device pipeline schedule. `PipelineCostModel`
  exposes `marginal_wall` for callers that need the schedule-aware estimate.
* `terminal_reward` is a length-difference proxy, not a learned metric.
* Post-controller thresholds are fixed heuristic constants.
* The smoke path is synthetic and its outputs are never reported as translation
  quality.

## Real-model Kaggle commands

`controller_v3.kaggle` is the only module that loads real weights. It reads
`QwenPredictor`, `load_translator`, `resolve_devices`, and `StreamConfig` from
`pfba-lab/paired/run_pfba.py`. Model-pair ids are `small -> (0, 0)` and
`large -> (1, 1)`; those ids index the measured cost table and are only
meaningful together with calibration records measured for the same adapter
pair.

Run every command from `pfba-lab/week` (or from `/tmp/pfba_cavc/week` inside
the packaged Kaggle kernel):

```bash
cd pfba-lab/week

# Plan only: prints the split policy and expected hours. Loads no model and no
# dataset.
"D:/MINICONDA/envs/py312/python.exe" -m controller_v3.kaggle --mode dry-run

# Preflight (dev sentences only; ~0.5-1 h): calibrate the cost table, label a
# few states, train, checkpoint, roll out a few sentences.
"D:/MINICONDA/envs/py312/python.exe" -m controller_v3.kaggle \
  --mode preflight --limit 8 --predictor small --translator small \
  --label-states 3 --rollout-sentences 2 \
  --output-dir outputs/cavc_preflight

# Development run (~4-6 h): label dev hash buckets 2-9, train, and roll out
# dev holdout bucket 1. `--max-hours` bounds the labeling/rollout loops, and
# labels.jsonl flushes one row per sentence so a later session resumes by
# sentence_id.
"D:/MINICONDA/envs/py312/python.exe" -m controller_v3.kaggle \
  --mode dev --limit 0 --stride 4 --max-hours 10 \
  --predictor small --translator small \
  --output-dir outputs/cavc_dev

# Frozen confirmation (~1-2 h): test split only, never tuned. The clean slice
# starts at test:0140. An earlier --offset raises ValueError unless
# --allow-consumed is passed explicitly for a diagnostic rerun.
"D:/MINICONDA/envs/py312/python.exe" -m controller_v3.kaggle \
  --mode confirm --offset 140 --limit 64 \
  --checkpoint outputs/cavc_dev/checkpoint.pt \
  --output-dir outputs/cavc_confirm
```

Each real run writes `summary.json`, `traces.jsonl`, `config.json`,
`cost_records.json`, and `checkpoint.pt` under `--output-dir`, plus the git
commit recorded in `summary.json` and the checkpoint. BLEU/COMET carry the
status returned by `run_pfba`: without `sacrebleu` BLEU is
`{"status": "unavailable"}` with no score, and COMET is `skipped` because
loading it downloads model weights. No metric is ever invented.

`confirm` refuses an offset below `test:0140` unless `--allow-consumed` is
given. The test slice is never used to choose losses, thresholds, budgets, or
fallback rules.

### Packaging the Kaggle kernel

```bash
"D:/MINICONDA/envs/py312/python.exe" pfba-lab/prepare_cavc_kernel.py
```

This writes `pfba-lab/kaggle-runs/pfba-cavc-v3/kaggle_entry.py` and
`kernel-metadata.json`, embedding every `controller_v3/*.py`,
`paired/run_pfba.py`, and `decoupled/cpu_embedding.py` under `/tmp/pfba_cavc`
with the repository-relative layout (`week/controller_v3/*.py`,
`week/cpu_embedding.py`, `paired/run_pfba.py`) so `parents[2] / "paired"`
resolves. It prints the generated paths and the SHA-256 of every bundled file.
It never pushes or submits to Kaggle.

Expected total: roughly **7-11 h** (preflight 0.5-1 h, dev 4-6 h, confirm
1-2 h), leaving buffer inside a ~23-30 h weekly quota. A single Kaggle session
is capped near 12 h; resume a dev run by rerunning the same command.

### Tests without models

The wiring tests inject fake adapters, so the whole suite stays CPU-only:

```bash
"D:/MINICONDA/envs/py312/python.exe" -m pytest -q pfba-lab/week/controller_v3/tests
```

## Kaggle plan (wiring implemented; not yet run on Kaggle)

Data is WMT German-to-English: **dev = newstest2013** (3,003 sentences) and
**test = newstest2015** (2,169 sentences).

1. **Local CPU tests** - minutes. `pytest` suite plus `--dry-run`/`--smoke`.
2. **Preflight and cost calibration** - 8-16 dev sentences for the small pair
   and the 7B pair - about 0.5-1 h.
3. **Feasibility** - label 200-300 dev sentences (buckets 2-9), train, roll out
   60-100 dev holdout sentences with `(0.5B, Marian)` - about 1.5-2.5 h.
4. **Full development run** - label 2,422 dev sentences, evaluate the 295
   sentence holdout, plus a 64-sentence 7B slice - about 4-6 h.
5. **Frozen confirmation** - untouched `test:0140 onward`, 64-128 sentences -
   about 1-2 h.

Total: roughly **7-11 h** of a ~23-30 h weekly quota, leaving buffer for reruns.
A single Kaggle session is capped near 12 h, so split the run and resume from
`checkpoint.py`.

Split policy:

* Development train: dev hash buckets 2-9 (2,422 sentences).
* Development tune: dev bucket 0 (286 sentences).
* Development holdout: dev bucket 1 (295 sentences).
* Clean confirmation: `test:0140 onward` (test sentences 0012-0139 are already
  consumed by earlier folds and must not be reused for tuning).

Buckets 0 and 1 were used by the earlier fold1/fold2 experiments. V3 is a new
model, but those buckets are historically consumed; report that caveat rather
than presenting them as pristine. Never tune on the frozen test slice.
