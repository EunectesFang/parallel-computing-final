"""Cost-aware value controller (CAVC) v3.

A CPU-testable redesign of the `week_run.py` one-hot action classifier around
four principles:

1. action-value learning instead of one-hot classification;
2. one unified READ/WRITE value scale;
3. explicit measured runtime cost inside the Bellman target;
4. counterfactual labels with a strict online-feature leakage boundary.

Importing the package exposes the public API below. Heavy dependencies (torch)
are imported here only because `CAVCNetwork`, the trainer, and checkpointing
are part of the public surface; every module remains individually importable.
"""

from .action import (
    DEFAULT_ACTIONS,
    EXTENDED_ACTIONS,
    READ,
    Action,
    action_features,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .config import ControllerConfig
from .cost_model import (
    CostEstimate,
    CostModel,
    MeasuredCostTable,
    PipelineCostModel,
    PowerLawCostModel,
)
from .evaluator import (
    MetricUnavailableError,
    comet_score,
    corpus_bleu,
    paired_bootstrap,
    summarize_traces,
)
from .features import (
    CONFIDENCE_FEATURES,
    HISTORY_FEATURES,
    MASK_FEATURES,
    NUMERIC_FEATURES,
    SURFACE_FEATURES,
    FeatureBuilder,
    FeatureVector,
    History,
    OptionalConfidence,
    RuntimeInfo,
    StateInput,
    estimate_length_ratio,
)
from .labeler import ActionOutcome, CounterfactualLabeler
from .network import CAVCNetwork, WriteOutputs
from .post_controller import PostAction, PostController, PostDecision
from .reward import (
    RewardConfig,
    delay_penalty,
    immediate_reward,
    quality_reward,
    terminal_reward,
)
from .rollout import RolloutTrace, run_rollout
from .runtime import ControlledRuntime, StepResult, vote_statistics
from .targets import build_action_targets, derive_state_key
from .text_encoder import TextEncoder, Vocab
from .trainer import ActionTarget, CAVCTrainer, TrainConfig

__all__ = [
    "Action",
    "READ",
    "DEFAULT_ACTIONS",
    "EXTENDED_ACTIONS",
    "action_features",
    "ControllerConfig",
    "NUMERIC_FEATURES",
    "SURFACE_FEATURES",
    "CONFIDENCE_FEATURES",
    "MASK_FEATURES",
    "HISTORY_FEATURES",
    "FeatureVector",
    "StateInput",
    "History",
    "RuntimeInfo",
    "OptionalConfidence",
    "FeatureBuilder",
    "estimate_length_ratio",
    "Vocab",
    "TextEncoder",
    "CAVCNetwork",
    "WriteOutputs",
    "ActionTarget",
    "TrainConfig",
    "CAVCTrainer",
    "CostModel",
    "CostEstimate",
    "MeasuredCostTable",
    "PowerLawCostModel",
    "PipelineCostModel",
    "RewardConfig",
    "quality_reward",
    "delay_penalty",
    "immediate_reward",
    "terminal_reward",
    "ActionOutcome",
    "CounterfactualLabeler",
    "StepResult",
    "ControlledRuntime",
    "vote_statistics",
    "RolloutTrace",
    "run_rollout",
    "PostAction",
    "PostDecision",
    "PostController",
    "MetricUnavailableError",
    "corpus_bleu",
    "comet_score",
    "summarize_traces",
    "paired_bootstrap",
    "build_action_targets",
    "derive_state_key",
    "save_checkpoint",
    "load_checkpoint",
]
