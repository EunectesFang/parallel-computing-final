# CAVC Controller V3 Implementation Plan

**Goal** Implement a CPU-testable `controller_v3` package that turns the existing
`week/week_run.py` one-hot action classifier into a cost-aware value controller
following the Astra design: state features, READ/WRITE value heads,
action-conditioned B/H scoring, explicit runtime cost, counterfactual labels,
multi-task losses, and an optional post-candidate budget controller.

**Branch** `cavc-controller-v3` (created from `7aca8ec`). Do not modify
`main`. Do not modify existing `week/week_run.py`, `joint-gate/joint_gate.py`,
or `paired/run_pfba.py`; the new package imports them.

**Spec** The Astra design document delivered in the user message on
2026-11-15. The plan below is the binding implementation contract.

## Global Constraints

- Only create/edit files under `pfba-lab/week/controller_v3/`. Do not touch
  existing modules, `results/`, `kaggle-runs/`, or data files.
- Python is `D:/MINICONDA/envs/py312/python.exe`. Run tests with
  `"D:/MINICONDA/envs/py312/python.exe" -m pytest -q pfba-lab/week/controller_v3/tests`.
- No downloads, no GPU, no Kaggle submission, no network. Tests use fakes.
- `torch==2.6.0`, `numpy==2.4.3`, `scikit-learn==1.6.1`, `pytest==9.1.1` are
  available. `sacrebleu` is NOT installed; BLEU must degrade to
  `{"status": "unavailable"}` and must never invent a score.
- Strict online feature boundary: features may use only the observed source
  prefix, the committed target prefix, history statistics already produced,
  and runtime/cost information. Never use the full source, source length,
  reference, oracle continuation, or future words in online features.
- Training labels may use the next true source word, the full source, and the
  oracle translation. Keep the two paths in separate functions.
- Only the commit path may append to `committed`. Provisional RALCP results
  must never mutate it.
- All randomness is seeded from `seed, split, sentence_id, read_pos, candidate
  index`. Candidate sets must be nested: B=2 must be the first two candidates
  of B=4 so budget comparisons are paired.
- Every checkpoint saves: model state dict, config, feature mean/std, cost
  table, seed, and the git commit.
- Sentence-disjoint splits. Test sentences are never used to choose losses,
  thresholds, budgets, or fallback rules.
- Docstrings and comments in English. Code must be importable on CPU without
  downloading models.
- Import `run_pfba` by adding `Path(__file__).resolve().parents[2] / "paired"`
  to `sys.path` (controller_v3 -> week -> pfba-lab). `parents[1]` is `week`,
  not `pfba-lab`. If a module sits elsewhere, compute the repo-relative path
  explicitly rather than guessing the parent depth.
- Implementers must not dispatch subagents.

## Shared Contract

These are the authoritative interfaces. Later tasks may read the files from
earlier tasks; do not change a signature without a recorded ruling.

`run_pfba` lives at `pfba-lab/paired/run_pfba.py`. From
`controller_v3/<module>.py`, add `Path(__file__).resolve().parents[2] /
"paired"` to `sys.path`; `parents[1]` is `week`, not `pfba-lab`.

```python
# action.py
@dataclass(frozen=True, order=True)
class Action:
    branches: int
    horizon: int
    @property is_read -> bool
    @property label -> str          # "read" or "b2h4"
READ = Action(0, 0)
DEFAULT_ACTIONS: tuple[Action, ...] = tuple(Action(b, h) for b in (2,4,6) for h in (4,6,10))
EXTENDED_ACTIONS: tuple[Action, ...] = tuple(Action(b, h) for b in (2,4,6,8,10) for h in (2,4,6,8,10))
def action_features(action: Action, threshold: float = 0.6) -> tuple[float, ...]  # 6 values
```

```python
# config.py
@dataclass(frozen=True)
class ControllerConfig:
    seed: int = 13
    min_start: int = 3
    arrival_gap: float = 0.3
    ralcp_threshold: float = 0.6
    gamma: float = 0.99
    lambda_l: float = 0.05
    lambda_t: float = 0.10
    weight_m: float = 1.0
    weight_e: float = 2.0
    advantage_baseline: tuple[int, int] = (4, 6)
    actions: tuple[Action, ...] = DEFAULT_ACTIONS
    numeric_dim: int = 34
    text_emb: int = 32
    text_hidden: int = 32
    text_window_src: int = 16
    text_window_tgt: int = 16
    model_emb: int = 8
    value_hidden: int = 64
    dropout: float = 0.10
    lr: float = 3e-4
    batch_size: int = 128
    epochs: int = 20
    huber_delta: float = 1.0
    rank_epsilon: float = 0.05
    alpha_q: float = 1.0
    alpha_rank: float = 0.5
    alpha_error: float = 0.5
    alpha_noemit: float = 0.5
    alpha_emit: float = 0.5
    target_update_every: int = 1
    max_branches: int = 10
    max_horizon: int = 10
```

```python
# features.py
HISTORY_FEATURES = ("last_B", "last_H", "last_vote_top1", "last_vote_margin",
                    "last_vote_entropy", "last_commit_count", "last_empty")
SURFACE_FEATURES = ("src_words", "src_chars", "tgt_words", "tgt_chars",
                    "length_ratio", "normalized_lag", "words_since_commit",
                    "no_emit_streak", "last_B", "last_H", "last_vote_top1",
                    "last_vote_margin", "last_vote_entropy", "last_commit_count",
                    "last_empty", "src_end_punct", "tgt_end_punct",
                    "last_capital", "subordinate_marker", "current_clause_len",
                    "pred_cost_ref", "mt_cost_ref", "pred_queue_work",
                    "mt_queue_work", "time_since_commit", "arrival_slack")
CONFIDENCE_FEATURES = ("mt_next_max_prob", "mt_next_entropy",
                       "mt_prefix_logprob", "pred_next_entropy")
MASK_FEATURES = tuple(f"{name}_available" for name in CONFIDENCE_FEATURES)
NUMERIC_FEATURES = SURFACE_FEATURES + CONFIDENCE_FEATURES + MASK_FEATURES  # 34
@dataclass(frozen=True)
class History:
    last_branches: int = 0
    last_horizon: int = 0
    last_vote_top1: float = 0.0
    last_vote_margin: float = 0.0
    last_vote_entropy: float = 0.0
    last_commit_count: int = 0
    last_empty: bool = False
    words_since_commit: int = 0
    no_emit_streak: int = 0
@dataclass(frozen=True)
class RuntimeInfo:
    pred_cost_ref: float = 0.0
    mt_cost_ref: float = 0.0
    pred_queue_work: float = 0.0
    mt_queue_work: float = 0.0
    time_since_commit: float = 0.0
    arrival_slack: float = 0.0
@dataclass(frozen=True)
class OptionalConfidence:
    values: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    available: tuple[bool, bool, bool, bool] = (False, False, False, False)
@dataclass(frozen=True)
class FeatureVector:
    numeric: tuple[float, ...]       # len 34
    predictor_id: int
    translator_id: int
    src_ids: tuple[int, ...]
    tgt_ids: tuple[int, ...]
@dataclass(frozen=True)
class StateInput:
    numeric: tuple[float, ...]
    predictor_id: int
    translator_id: int
    src_ids: tuple[int, ...] = ()
    tgt_ids: tuple[int, ...] = ()
class FeatureBuilder:
    def __init__(self, length_ratio: float = 1.0, src_vocab=None, tgt_vocab=None)
    def build(self, observed: str, committed: Sequence[str], *,
              history: History = History(), runtime: RuntimeInfo = RuntimeInfo(),
              confidence: OptionalConfidence = OptionalConfidence(),
              predictor_id: int = 0, translator_id: int = 0) -> FeatureVector
def estimate_length_ratio(samples: Sequence[tuple[str, str]]) -> float
```

```python
# cost_model.py
@dataclass(frozen=True)
class CostEstimate:
    pred_s: float
    mt_s: float
    wall_s: float
class CostModel(Protocol):
    def predict(self, predictor_id: int, translator_id: int, action: Action,
                src_words: int, tgt_words: int) -> CostEstimate: ...
class MeasuredCostTable:
    def __init__(self, records: Sequence[tuple[int, int, Action, float, float]] = ())
    def fit(self, records) -> "MeasuredCostTable"
    def predict(self, predictor_id, translator_id, action, src_words, tgt_words) -> CostEstimate
class PowerLawCostModel:
    def fit(self, records) -> "PowerLawCostModel"
    def predict(self, predictor_id, translator_id, action, src_words, tgt_words) -> CostEstimate
class PipelineCostModel:
    def __init__(self, base: CostModel, arrival_gap: float = 0.3)
    def marginal_wall(self, t_pred_free: float, t_mt_free: float,
                      release_s: float, dependency_s: float,
                      predictor_id: int, translator_id: int,
                      action: Action, src_words: int, tgt_words: int) -> float
```

```python
# reward.py
@dataclass(frozen=True)
class RewardConfig:
    weight_m: float = 1.0
    weight_e: float = 2.0
    lambda_l: float = 0.05
    gamma: float = 0.99
def quality_reward(matched: int, incorrect: int, cfg: RewardConfig) -> float
def delay_penalty(backlog: float, source_advance: float, cfg: RewardConfig) -> float
def immediate_reward(matched: int, incorrect: int, backlog: float,
                     source_advance: float, cfg: RewardConfig) -> float
def terminal_reward(hypothesis: str, reference: str, cfg: RewardConfig) -> float
```

```python
# labeler.py
@dataclass(frozen=True)
class ActionOutcome:
    action: Action
    new_words: tuple[str, ...]
    matched: int
    incorrect: int
    emitted: int
    no_emit: bool
    next_committed: tuple[str, ...]
    pred_s: float
    mt_s: float
    vote_top1: float
    vote_margin: float
    vote_entropy: float
    candidates: tuple[str, ...] = ()
class CounterfactualLabeler:
    def __init__(self, predictor, translator, cfg: ControllerConfig, timer=None)
    def sample_candidates(self, observed: str, max_b: int, max_h: int, seed: int) -> list[list[int]]
    def oracle_words(self, source: str, committed: Sequence[str]) -> tuple[str, ...]
    def label_state(self, observed: str, committed: Sequence[str], source: str,
                    actions: Sequence[Action], seed: int) -> list[ActionOutcome]
    def read_outcome(self, committed: Sequence[str]) -> ActionOutcome
```

```python
# network.py
@dataclass
class WriteOutputs:
    value: Tensor          # [B, A]
    error_risk: Tensor     # [B, A] logits
    no_emit: Tensor        # [B, A] logits
    expected_emit: Tensor  # [B, A] softplus
class CAVCNetwork(nn.Module):
    def __init__(self, cfg: ControllerConfig, src_vocab_size: int, tgt_vocab_size: int)
    def encode(self, numeric, predictor_id, translator_id, src_ids, tgt_ids) -> Tensor
    def read_value(self, z: Tensor) -> Tensor
    def write_values(self, z: Tensor, actions: Sequence[Action]) -> WriteOutputs
    def forward(self, numeric, predictor_id, translator_id, src_ids, tgt_ids,
                actions) -> tuple[Tensor, WriteOutputs]
```

```python
# trainer.py
@dataclass(frozen=True)
class ActionTarget:
    state: StateInput
    action: Action
    value_target: float
    error_target: float
    no_emit_target: float
    emit_target: float
    state_key: str
    next_state: StateInput | None = None
    next_actions: tuple[Action, ...] = ()
    terminal: bool = False
@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 3e-4
class CAVCTrainer:
    def __init__(self, network: CAVCNetwork, cfg: ControllerConfig, cost_model=None)
    def fit(self, targets: Sequence[ActionTarget]) -> dict
    def compute_targets(self, targets: Sequence[ActionTarget]) -> list[float]
```

```python
# rollout.py
@dataclass(frozen=True)
class RolloutTrace:
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
def run_rollout(sentence, predictor, translator, network, builder, cost_model,
                cfg, timer=None) -> RolloutTrace
```

```python
# post_controller.py
class PostAction(Enum):
    STOP = "stop"
    ADD_2 = "add_2"
    WAIT = "wait"
@dataclass(frozen=True)
class PostDecision:
    action: PostAction
    reason: str
class PostController:
    def decide(self, z_pre, current_branches: int, horizon: int,
               vote_entropy: float, vote_margin: float,
               no_emit_prob: float, spent_s: float,
               cfg: ControllerConfig) -> PostDecision
```

## Task 1: Core scaffold — config, actions, features, text encoder

Create the package skeleton and the shared contract.

Files:
- `pfba-lab/week/controller_v3/__init__.py`
- `pfba-lab/week/controller_v3/config.py`
- `pfba-lab/week/controller_v3/action.py`
- `pfba-lab/week/controller_v3/features.py`
- `pfba-lab/week/controller_v3/text_encoder.py`
- `pfba-lab/week/controller_v3/tests/__init__.py`
- `pfba-lab/week/controller_v3/tests/test_action.py`
- `pfba-lab/week/controller_v3/tests/test_features.py`
- `pfba-lab/week/controller_v3/tests/test_text_encoder.py`

Requirements:
1. `action.py`: exact `Action`, `READ`, `DEFAULT_ACTIONS`, `EXTENDED_ACTIONS`,
   `action_features`. `action_features` returns
   `(b, h, b*h, log(b), log(h), ceil(threshold*b)/b)`. READ returns six zeros.
   `label` is `"read"` for READ and `f"b{b}h{h}"` otherwise.
2. `config.py`: exact `ControllerConfig` with the fields above. No mutable
   defaults; use `field(default_factory=...)` only if needed. `actions` defaults
   to `DEFAULT_ACTIONS`.
3. `features.py`: exact names and dataclasses above. `FeatureBuilder.build`
   returns exactly 34 numeric values in `NUMERIC_FEATURES` order. Implement:
   - `length_ratio` is the **current** ratio `tgt_words / max(1, src_words)`;
     the builder's constructor argument `length_ratio` is the training-estimated
     `rho`, used only for the lag feature, and must not replace this feature.
   - `normalized_lag = max(0, src_words - tgt_words / rho)` where `rho` is the
     constructor argument (default 1.0), never negative.
   - `words_since_commit` and `no_emit_streak` from `History`.
   - `src_end_punct` / `tgt_end_punct` check the last character in a punctuation
     set; `subordinate_marker` checks a small German **subordinating
     conjunction** set (`dass`, `weil`, `ob`, `wenn`, `obwohl`, `während`,
     `nachdem`, `bevor`, `sobald`, `falls`, `da`, `damit`, `sodass`). Do not
     include articles (`der`, `die`, `das`, `den`, `dem`); they are not clause
     markers. `current_clause_len` counts words since the last clause
     punctuation.
   - Confidence values are copied; unavailable values are replaced with 0.0 and
     the corresponding mask is 0.0. Available masks are 1.0.
   - `pred_cost_ref`, `mt_cost_ref`, queue work, `time_since_commit`,
     `arrival_slack` come from `RuntimeInfo`.
   - Text ids may be empty; `FeatureBuilder` accepts optional `src_vocab` and
     `tgt_vocab` with `.encode(words, window)`; if `None`, ids stay empty.
   - `estimate_length_ratio` returns `mean(tgt_words / src_words)` over
     non-empty pairs, clamped to `[0.5, 2.0]`.
4. `text_encoder.py`: a `Vocab` class with `PAD=0`, `UNK=1`, `from_texts`,
   `encode(words, window)`, `__len__`; a `TextEncoder(nn.Module)` with two
   embeddings (src/tgt), two single-layer GRUs (`batch_first=True`), and an
   output projection to `2 * hidden` (default 64). Empty id sequences must
   produce a zero vector without crashing. Use `torch.nn.utils.rnn.pack_padded_sequence`
   or a simple mask; document the choice.
5. Tests:
   - `test_action.py`: READ label/is_read; default 9 actions; extended 25;
     `action_features` length 6 and exact formula for (4,6); READ zeros.
   - `test_features.py`: exact 34 values; confidence mask behavior; normalized
     lag; punctuation and clause features; no future/full-source argument in
     `build` (assert via `inspect.signature` that names do not include
     `source`, `reference`, `future`, `full_source`).
   - `test_text_encoder.py`: vocab PAD/UNK; encode window; forward shapes;
     empty sequence returns zeros; gradient flows.
6. Run the three test files. All pass.

Commit message: `Add controller_v3 core scaffold: actions, features, text encoder`

## Task 2: CAVC network and multi-task trainer

Files:
- `pfba-lab/week/controller_v3/network.py`
- `pfba-lab/week/controller_v3/trainer.py`
- `pfba-lab/week/controller_v3/tests/test_network.py`
- `pfba-lab/week/controller_v3/tests/test_trainer.py`

Requirements:
1. `network.py` implements the exact `WriteOutputs` and `CAVCNetwork` contract.
   - Numeric encoder: `Linear(34, 64) -> LayerNorm -> SiLU`.
   - Text encoder: use `TextEncoder`; output 64.
   - Model embeddings: `Embedding(8, 8)` for predictor and translator; unknown
     ids are clamped to `[0, 7]`.
   - Fusion: `Linear(64 + 64 + 8 + 8, 128) -> LayerNorm -> SiLU -> Dropout ->
     Linear(128, 64) -> SiLU`.
   - `read_value`: `Linear(64,32) -> SiLU -> Linear(32,1)`, squeeze to `[B]`.
   - `write_values`: action encoder `Linear(6,32) -> SiLU -> Linear(32,16)`;
     head `Linear(64+16,64) -> LayerNorm -> SiLU -> Dropout -> Linear(64,32) ->
     SiLU`; four output linears. `expected_emit` uses `softplus`.
   - `forward` returns `(read_value, WriteOutputs)`.
   - Empty `src_ids`/`tgt_ids` tensors of shape `[B,0]` must work.
2. `trainer.py` implements `ActionTarget`, `TrainConfig`, `CAVCTrainer`.
   - `compute_targets`: if `next_state` is None or `terminal`, target =
     `value_target`. Otherwise one-step Bellman:
     `y = value_target + gamma * max_a' [Q_target(next_state, a') - lambda_t *
     C(next_state, a')]`. Use the frozen target network and the injected cost
     model; if no cost model, use zero cost.
   - `fit`: AdamW, grad clip 1.0, Huber loss for value, BCE-with-logits for
     error/no-emit (write actions only), Huber on `softplus` emit (write only),
     and a pairwise ranking loss. Ranking only uses pairs in the same
     `state_key` with `|y_i - y_j| > rank_epsilon`; cap at 32 pairs per state.
     Loss weights come from `ControllerConfig.alpha_*`.
   - Target network: deep copy, frozen, updated every `target_update_every`
     epochs.
   - Return a dict with `epochs`, `final_loss`, `value_loss`, `rank_loss`,
     `error_loss`, `no_emit_loss`, `emit_loss`, `n_samples`.
   - No NaN when a state has only READ or only one action.
3. Tests:
   - `test_network.py`: output shapes for B=3 and 9 actions; empty text ids;
     `expected_emit >= 0`; model id clamping; gradient exists for all
     parameters.
   - `test_trainer.py`: tiny synthetic dataset (3 states x 3 actions) with a
     fake cost model; `fit` reduces total loss over 30 epochs; `compute_targets`
     equals immediate reward for terminal; ranking ignores pairs within
     epsilon; no NaN; target network changes after an update.
4. Run both test files. All pass.

Commit message: `Add CAVC value network and multi-task trainer`

## Task 3: Cost model and controlled runtime

Files:
- `pfba-lab/week/controller_v3/cost_model.py`
- `pfba-lab/week/controller_v3/runtime.py`
- `pfba-lab/week/controller_v3/tests/test_cost_model.py`

Requirements:
1. `cost_model.py` implements the exact contract.
   - `CostEstimate(pred_s, mt_s, wall_s)` with `wall_s = pred_s + mt_s`.
   - `MeasuredCostTable`: keyed by `(predictor_id, translator_id, action)`;
     stores median `(pred_s, mt_s)`; falls back to nearest measured action with
     the same model pair, then to a global median, then to a 1e-3 floor.
   - `PowerLawCostModel`: fit `pred = a * b^r_b * h^r_h`, `mt = c * b^s_b *
     h^s_h` by log-linear least squares on records; clamp exponents to `[0,1]`;
     fall back to `MeasuredCostTable` when fewer than 4 records.
   - `PipelineCostModel.marginal_wall`: simulate one prediction and one
     translation with two device-free times. Return
     `max(t_mt_end, t_pred_end) - max(t_pred_free, t_mt_free)` for the
     incremental wall time, never negative. Use `release_s` for prediction
     start and `dependency_s` for translation start.
1b. READ handling: `MeasuredCostTable.predict` and `PowerLawCostModel.predict`
   must accept `READ` (`branches=0`, `horizon=0`) and return
   `CostEstimate(0.0, 0.0, 0.0)`. Add a test for this: the trainer's Bellman
   backup now queries the cost model with READ for every non-terminal target.
   `ControlledRuntime.execute` is never called with READ by the rollout, but it
   should raise a clear `ValueError` if it is, rather than predict with B=0.
2. `runtime.py` implements `ControlledRuntime`:
   - `__init__(predictor, translator, cfg, timer=None)`.
   - `predict(observed, action, seed)` -> `(continuations, seconds)`. Sample
     each candidate separately with `seed + index` and `branches=1`, so
     candidate index `i` is stable when the budget grows from B=2 to B=4.
     Document the trade-off: this is B small calls, not one batched
     `num_return_sequences=B` call, and it can be slower. The matched baseline
     for any controller experiment must use the same per-candidate path;
     otherwise budget comparisons mix sampling-stream differences with the
     controller's decisions.
   - `translate(requests, prefix, final=False)` -> `(outputs, seconds)`.
   - `ralcp(candidates)` -> selected string.
   - `execute(observed, committed, action, seed)` -> a `StepResult` dataclass
     with `candidates`, `selected`, `committed_after`, `pred_s`, `mt_s`,
     `spent_s`, `vote_top1`, `vote_margin`, `vote_entropy`, `no_emit`.
     `spent_s = pred_s + mt_s` for the action.
   - `extend(observed, committed, current_candidates, extra_branches, horizon,
     seed, spent_s=0.0)` -> a new `StepResult` that appends only the extra
     candidates (using candidate-index seeds `seed + current_branches + index`
     so the first candidates stay identical), re-runs RALCP over the full pool,
     and never mutates the shared committed list. This is the execution path
     for the post-candidate controller's `ADD_2`. The returned `spent_s` is
     `spent_s + pred_s + mt_s` for the extension only.
   - Use `run_pfba.call_translate_batch` and `run_pfba.upstream_ralcp` when
     available; otherwise call the adapter directly / a local RALCP fallback.
   - `execute` and `extend` must NOT mutate a shared committed list; return the
     new tuple.
   - Vote statistics are computed at the **decisive depth**: the first word
     depth where the RALCP threshold fails, or the last depth shared by all
     candidates when no depth fails. Support is the count of identical word
     prefixes at that depth divided by the original candidate count; top1 =
     max support, margin = top1 - second support, entropy over the support
     distribution.
3. Tests with a fake predictor/translator:
   - measured table exact lookup and fallback; power law fits monotone exponents;
     pipeline overlap reduces marginal wall time versus serial sum.
   - runtime returns candidates and no-emit correctly; provisional RALCP does
     not mutate the input list.
4. Run tests. All pass.

Commit message: `Add measured/pipeline cost model and controlled runtime`

## Task 4: Reward and counterfactual labeler

Files:
- `pfba-lab/week/controller_v3/reward.py`
- `pfba-lab/week/controller_v3/labeler.py`
- `pfba-lab/week/controller_v3/tests/test_labeler.py`

Requirements:
1. `reward.py`: exact contract. `quality_reward = weight_m * matched -
   weight_e * incorrect`. `delay_penalty = lambda_l * max(0, backlog) *
   max(0, source_advance)`. `immediate_reward` subtracts the delay penalty.
   `terminal_reward` uses a simple negative absolute word-length difference
   between hypothesis and reference (documented proxy, not COMET).
2. `labeler.py`: exact contract.
   - `sample_candidates`: prefer `predictor.sample_token_rows(observed, max_b,
     max_h, seed)`; else if `predictor.model` and `predictor.tok` exist, use
     `model.generate` with temperature 1.0, top_p 0.9, top_k 0,
     `num_return_sequences=max_b`, `max_new_tokens=max_h`, seeded exactly like
     `week/week_run.py:251-266`; else call `predictor.sample` and tokenize via
     `predictor.tok` if available. If no tokenizer is available, raise a clear
     `RuntimeError` — never silently return an empty candidate pool.
   - `label_state`: generate one nested pool of `max(actions.branches)` x
     `max(actions.horizon)` candidates with one seed; for each action take the
     first b rows, truncate to h tokens, decode, build the request
     `observed + " " + continuation`, translate all b in one batch with the
     committed prefix, run RALCP, and compute `matched`/`incorrect` against
     `oracle_words(source, committed)`. Record measured `pred_s`/`mt_s` and
     vote statistics. Reuse the pool; do not re-sample per action.
   - `oracle_words`: translate `[source]` with the committed prefix using
     `run_pfba.call_translate_batch`; strip.
   - `read_outcome`: returns READ with no compute and the same committed tuple.
   - Strict boundary: `label_state` may read `source`; no online feature
     function may. Add a module docstring stating this.
3. Tests with a fake predictor/translator:
   - nested candidates: B=2 output equals the first two of B=4.
   - `matched`/`incorrect` computed correctly against a fixed oracle.
   - READ outcome has zero compute and no words.
   - one translate call per action horizon, not one per (b,h), when b varies.
   - no mutation of the input committed list.
4. Run tests. All pass.

Commit message: `Add counterfactual action labeler and reward functions`

## Task 5: Rollout, post-candidate controller, evaluator

Files:
- `pfba-lab/week/controller_v3/rollout.py`
- `pfba-lab/week/controller_v3/post_controller.py`
- `pfba-lab/week/controller_v3/evaluator.py`
- `pfba-lab/week/controller_v3/tests/test_rollout.py`
- `pfba-lab/week/controller_v3/tests/test_post_controller.py`

Requirements:
1. `rollout.py`: exact `RolloutTrace` and `run_rollout`.
   - At each read position, build features, call the network, score READ and all
     actions as
     `score = (Q(a) - Q(baseline)) - lambda_t * (C(a) - C(baseline))` when
     `advantage_baseline` is enabled; otherwise `Q(a) - lambda_t * C(a)`.
   - Choose the argmax over READ + actions. READ advances `read_pos`, increments
     `History.words_since_commit` and `no_emit_streak`, sets `last_empty=True`
     and `last_commit_count=0`, and updates `RuntimeInfo.time_since_commit`.
     WRITE calls the runtime, commits only the final RALCP result, and updates
     history: on commit reset `words_since_commit` and `no_emit_streak`, set
     `last_commit_count`; on no emit increment both and set `last_empty=True`.
   - If a `post_controller` is supplied, run it after the initial `execute`,
     before committing. `ADD_2` calls `runtime.extend` (cached candidates only),
     then re-runs RALCP over the full candidate pool; `WAIT` discards the
     provisional result without committing; `STOP` commits the initial RALCP
     result. Provisional results never mutate `committed`.
   - Virtual completion time: `max(completion_s, (read_pos - 1) * arrival_gap)`
     plus measured pred/mt time; never double-count.
   - At EOS, translate the true full source once with the committed prefix and
     append; use `run_pfba.laal_score`.
   - The network, cost model, runtime, and post controller are injectable; tests
     use a fake network with deterministic values and a fake runtime.
2. `post_controller.py`: exact `PostAction`, `PostDecision`,
   `PostController.decide`.
   - Default rule: if `vote_margin` is high and `no_emit_prob` is low, STOP;
     if `vote_entropy` is high and `current_branches < max_branches`, ADD_2;
     if `no_emit_prob` is high and `spent_s` exceeds a budget, WAIT.
   - The post controller only proposes; it must not commit. Document that the
     caller must re-run RALCP over all candidates after ADD_2.
3. `evaluator.py`:
   - `corpus_bleu(hypotheses, references)` delegates to
     `run_pfba.corpus_bleu`; if unavailable, return
     `{"status": "unavailable", "reason": ...}`.
   - `comet_score` delegates to `run_pfba.comet_score` and returns the same
     status shape.
   - `summarize_traces(traces)` returns n, status-aware BLEU/COMET, mean LAAL,
     mean/P95 completion, read fraction, action histogram, per-step records.
   - `paired_bootstrap(traces_a, traces_b, metric, samples=2000, seed=13)`
     returns point estimate and 95% interval for sentence-paired differences;
     skip `None` LAAL values.
4. Tests:
   - rollout with a fake network that always prefers a specific action; verify
     histograms, commit path, EOS flush, LAAL not None.
   - READ path increments read_pos and does not call the runtime.
   - post controller STOP/ADD_2/WAIT cases.
   - evaluator returns unavailable BLEU without sacrebleu; bootstrap interval
     contains 0 for identical traces.
5. Run tests. All pass.

Commit message: `Add rollout, post-candidate controller, and evaluator`

## Task 6: CLI, checkpointing, integration smoke, README

Files:
- `pfba-lab/week/controller_v3/cli.py`
- `pfba-lab/week/controller_v3/checkpoint.py`
- `pfba-lab/week/controller_v3/targets.py`
- `pfba-lab/week/controller_v3/README.md`
- `pfba-lab/week/controller_v3/tests/test_integration.py`
- update `pfba-lab/week/controller_v3/__init__.py` with public exports.

Requirements:
1. `targets.py`:
   - `build_action_targets(state: StateInput, outcomes: Sequence[ActionOutcome],
     cfg: ControllerConfig, *, next_state: StateInput | None = None,
     next_actions: Sequence[Action] = (), backlog: float = 0.0,
     source_advance: int = 1, state_key: str = "") -> list[ActionTarget]`.
   - **`state_key` must be non-empty** in real use (for example
     `f"{sentence_id}:{read_pos}:{hash(committed)}"`). If the caller leaves it
     `""`, derive a stable key from `state.numeric`, `state.predictor_id`,
     `state.translator_id`, and the text ids. Empty keys silently disable the
     ranking loss, so do not let production data use them. Tests may still use
     empty keys to exercise the skip.
   - Convert each `ActionOutcome` to an `ActionTarget` using
     `reward.immediate_reward`; READ gets the delay penalty and the same next
     state (the caller supplies the post-READ state). Write actions get
     `error_target = 1.0 if incorrect > 0 else 0.0`,
     `no_emit_target = 1.0 if no_emit else 0.0`, `emit_target = emitted`.
   - When building the bootstrap, pass `next_actions=(READ, *cfg.actions)` so
     the one-step backup maximizes over READ as well as writes.
   - `value_target` is the **immediate** reward; the trainer adds the bootstrap
     term. This function must not read the reference or future words.
2. `checkpoint.py`:
   - `save_checkpoint(path, network, cfg, feature_mean, feature_std, cost_model,
     extra=None)`; save `state_dict`, config dict, mean/std, cost records,
     seed, git commit if available.
   - `load_checkpoint(path, network)` restores and returns metadata; strict
     shape checks with a clear error.
3. `cli.py`:
   - `--dry-run` prints config, actions, and feature count without loading
     models.
   - `--smoke` runs an end-to-end CPU test with a tiny fake predictor/translator
     and a 3-sentence synthetic dataset; writes `summary.json` and
     `traces.jsonl`.
   - `--train-labels PATH --rollout` are parsed but may raise a clear
     `NotImplementedError` if a real model adapter is not supplied; do not
     fake scores.
4. `README.md`:
   - Architecture diagram (text), module map, exact feature list, action grid,
     loss definitions, leakage rules, data format for `ActionTarget`, and how
     to run CPU tests.
   - Explicitly state: not trained on Kaggle yet; no BLEU/COMET claimed;
     `sacrebleu` unavailable locally.
   - State the per-candidate sampling rule: candidate `i` uses `seed + i`, so
     B=2 is the first two of B=4. Any matched baseline must use the same
     per-candidate path; a batched `num_return_sequences` baseline is a
     different sampling stream and cannot be compared directly.
   - Kaggle plan stub: which datasets/splits to use first, estimated hours.
5. `tests/test_integration.py`:
   - Build `ControllerConfig`, `FeatureBuilder`, `TextEncoder`, `CAVCNetwork`,
     `MeasuredCostTable`, `CounterfactualLabeler`, `CAVCTrainer`,
     `run_rollout` with fakes.
   - Train on 3 states x 3 actions for 5 epochs, save/load checkpoint, roll out
     2 sentences, assert finite metrics and correct trace keys.
   - Verify the feature builder signature cannot see future/source/reference.
6. Run the full test suite:
   `"D:/MINICONDA/envs/py312/python.exe" -m pytest -q pfba-lab/week/controller_v3/tests`
   All pass.
7. Run the smoke CLI:
   `"D:/MINICONDA/envs/py312/python.exe" -m pfba_lab...` — because the package
   is not installed, run from `pfba-lab/week`:
   `cd pfba-lab/week && "D:/MINICONDA/envs/py312/python.exe" -m controller_v3.cli --smoke --output-dir /tmp/cavc_smoke`
   Confirm `summary.json` is written.

Commit message: `Add controller_v3 CLI, checkpoints, integration smoke, and docs`

## Final acceptance

- `pytest -q pfba-lab/week/controller_v3/tests` passes.
- `controller_v3.cli --dry-run` and `--smoke` run on CPU.
- No changes outside `pfba-lab/week/controller_v3/`.
- A final reviewer checks:
  1. Code correctness and test coverage.
  2. Modeling consistency with the Astra spec: value-based not one-hot, unified
     READ/WRITE scale, explicit cost, counterfactual labels, no leakage, nested
     candidates, post-controller non-monotonicity, calibration/metric honesty.
- Produce a short report mapping each Astra design element to its file/function,
  and list any deliberate simplifications.

## Task 7: Real-model Kaggle wiring, cost calibration, and kernel packaging

**Task 7 scope exception:** this task may create files under
`pfba-lab/kaggle-runs/pfba-cavc-v3/` and `pfba-lab/prepare_cavc_kernel.py` in
addition to `controller_v3/`. It must not modify existing files outside those
paths. It must not submit or push to Kaggle.

Files:
- `pfba-lab/week/controller_v3/kaggle.py`
- `pfba-lab/week/controller_v3/tests/test_kaggle_wiring.py`
- `pfba-lab/prepare_cavc_kernel.py`
- `pfba-lab/kaggle-runs/pfba-cavc-v3/kaggle_entry.py` (generated)
- `pfba-lab/kaggle-runs/pfba-cavc-v3/kernel-metadata.json` (generated)
- update `pfba-lab/week/controller_v3/README.md` with the Kaggle commands.

Requirements:
1. `kaggle.py`:
   - `resolve_adapters(predictor_name="small", translator_name="small",
     local_files_only=False, trust_remote_code=False, device=None)` returns
     `(predictor, translator, stream_cfg, predictor_device, translator_device)`.
     Use `run_pfba.QwenPredictor`, `run_pfba.load_translator`,
     `run_pfba.resolve_devices`, and `run_pfba.StreamConfig`.
   - Model-pair ids: small predictor/translator = 0/0, large = 1/1. Document
     that these ids are only meaningful with matching calibration records.
   - `calibrate_cost_records(predictor, translator, sentences, actions, cfg,
     limit=8, states_per_sentence=2)` measures `pred_s` and `mt_s` for every
     action with the same per-candidate path the runtime uses
     (`predictor.sample(observed, 1, h, seed+i)`), records
     `(predictor_id, translator_id, action, pred_s, mt_s)`, and returns the
     list plus a summary. This is the measured table Task 3 consumes.
   - `run_preflight(limit=8, output_dir=..., label_states=3,
     rollout_sentences=2, seed=13)`: load `dev` sentences only; build the
     feature builder, network, cost model, labeler, targets, trainer;
     label a few states, train, save a checkpoint, roll out a few sentences,
     and write `summary.json`, `traces.jsonl`, `config.json`,
     `cost_records.json`, `checkpoint.pt`, and the git commit. BLEU/COMET must
     report their real status (`unavailable`/`skipped`), never invented.
   - `run_dev(limit=0, stride=4, output_dir=..., max_hours=...)`: label dev
     train buckets 2–9, train, roll out dev holdout bucket 1, save the same
     artifacts. Respect a wall-clock deadline and write per-sentence
     checkpoints so a Kaggle session can resume.
   - `label_sentence(labeler, sentence, stride, cfg)`: walk the sentence from
     `min_start` to the penultimate word. At kept positions (every `stride`-th
     decision position), call `labeler.label_state` for `cfg.actions`, compute
     `oracle_words`, and convert outcomes to immediate rewards. Advance
     `committed` by the write action with the highest immediate reward; on
     ties prefer smaller `b*h`; READ leaves `committed` unchanged. Write one
     JSONL row per sentence with `sentence_id`, `read_pos`, outcomes
     (omitting full candidate text to keep the file small), and the committed
     prefix. Unkept positions still advance `read_pos` but do not relabel.
     This is the V3 analogue of `week_run.label_split` but with the 9-action
     grid and the value-style immediate reward.
   - `run_dev` must resume from a partially written labels file by
     `sentence_id`, never relabel a completed sentence, and never include
     holdout bucket 1 or test sentences in labeling.
   - `run_confirm(offset=140, limit=64, output_dir=...)`: test split only,
     frozen config. Raise `ValueError` for `offset < 140` unless
     `allow_consumed=True` is passed explicitly; never tune on this slice.
   - `main(argv)` with `--mode {preflight,dev,confirm,dry-run}`,
     `--predictor {small,large}`, `--translator {small,large}`,
     `--limit`, `--label-states`, `--rollout-sentences`, `--offset`,
     `--output-dir`, `--local-files-only`, `--allow-consumed`.
2. `prepare_cavc_kernel.py`:
   - Read every `controller_v3/*.py`, `paired/run_pfba.py`, and
     `decoupled/cpu_embedding.py`.
   - Generate `kaggle-runs/pfba-cavc-v3/kaggle_entry.py` that writes those files
     under `/tmp/pfba_cavc` **preserving the repo-relative layout**:
     `/tmp/pfba_cavc/week/controller_v3/*.py` and
     `/tmp/pfba_cavc/paired/run_pfba.py`. This matters because
     `controller_v3` resolves `run_pfba` via `parents[2]/paired`; a flat
     `/tmp/pfba_cavc/controller_v3` + `/tmp/pfba_cavc/run_pfba.py` layout
     would break the import. Also write `cpu_embedding.py` under
     `/tmp/pfba_cavc/week/`. Run from `/tmp/pfba_cavc/week`:
     `python -m controller_v3.kaggle --mode preflight --limit 8
     --output-dir /kaggle/working/cavc`.
   - Generate `kernel-metadata.json` with id
     `eunectesqaq/pfba-cavc-v3-preflight`, title `pfba cavc v3 preflight`,
     `code_file: kaggle_entry.py`, `language: python`,
     `kernel_type: script`, `is_private: true`, `enable_gpu: true`,
     `enable_internet: true`, empty sources.
   - Print the generated paths and the SHA-256 of each bundled file. Do not
     push.
3. `tests/test_kaggle_wiring.py`:
   - Fake adapters; `run_preflight` writes all required artifacts and reports
     honest metric statuses.
   - `calibrate_cost_records` produces one record per action with non-negative
     times and uses per-candidate seeds.
   - `run_confirm` rejects `offset < 140` and accepts `allow_consumed=True`.
   - `prepare_cavc_kernel.py` can be imported and its bundle function returns
     the expected file list; the metadata JSON parses.
4. README update: exact preflight/dev/confirm commands, expected hours, and the
   rule that the test slice starts at `test:0140`.
5. Run the full controller test suite and the prepare script; verify the
   generated `kaggle_entry.py` and `kernel-metadata.json` exist. Do not run a
   real model, do not push, do not submit.

Commit message: `Add real-model Kaggle wiring, cost calibration, and kernel packaging`
