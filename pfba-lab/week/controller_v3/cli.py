"""Command-line entry point for the CAVC v3 controller.

Three modes exist:

* ``--dry-run`` prints the configuration, the action grid, and the exact
  numeric feature list. It imports no model weights and opens no files.
* ``--smoke`` runs a fully synthetic CPU end-to-end path: three German/English
  toy sentences, a fake predictor/translator/tokenizer, three states x three
  actions, five training epochs, a checkpoint round trip, and two rollouts.
  It writes ``summary.json`` and ``traces.jsonl`` under ``--output-dir``.
* ``--train-labels PATH --rollout`` are parsed but raise `NotImplementedError`:
  they need a real predictor/translator adapter, and this repository does not
  download or load model weights from the CLI. No metric is ever fabricated.

The smoke summary never reports a translation-quality score. Its sentences are
synthetic, so BLEU is forced to ``{"status": "unavailable", "reason":
"synthetic smoke data"}`` without calling `corpus_bleu`, even when `sacrebleu`
is installed. COMET is always skipped in the smoke because loading it downloads
model weights; `evaluator.summarize_traces(include_comet=True)` remains the
opt-in API for real-data callers.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .action import DEFAULT_ACTIONS, EXTENDED_ACTIONS, READ, Action
from .config import ControllerConfig
from .features import NUMERIC_FEATURES

__all__ = ["build_parser", "main"]


# Three synthetically aligned sentences. The fake translator glosses German
# source words to English so RALCP has something to vote on; none of this is
# Kaggle data and no score from it is meaningful.
SMOKE_SENTENCES = (
    ("smoke-0", "hallo welt .", "hello world ."),
    ("smoke-1", "der hund schläft .", "the dog sleeps ."),
    ("smoke-2", "wir lernen heute .", "we learn today ."),
)

SMOKE_GLOSS = {
    "hallo": "hello",
    "welt": "world",
    "der": "the",
    "hund": "dog",
    "schläft": "sleeps",
    "wir": "we",
    "lernen": "learn",
    "heute": "today",
    ".": ".",
}

SMOKE_POOL = (
    "heute",
    "der",
    "lernen",
    ".",
    "hund",
    "wir",
    "welt",
    "schläft",
)

SMOKE_ACTIONS = (READ, Action(2, 4), Action(4, 6))

SMOKE_EPOCHS = 5
SMOKE_STAGE_SECONDS = 0.05

# The smoke dataset is synthetic, so no BLEU/COMET score is meaningful.
SMOKE_BLEU_STATUS = {
    "status": "unavailable",
    "reason": "synthetic smoke data",
}
SMOKE_COMET_STATUS = {
    "status": "skipped",
    "reason": "synthetic smoke data; COMET would download weights and score "
    "meaningless text",
}


# ----------------------------------------------------------------------
# Small fakes: deterministic, CPU-only, no downloads
# ----------------------------------------------------------------------
class _SmokeTokenizer:
    """Word-level tokenizer over a `Vocab`; never stops on EOS."""

    eos_token_id = -1

    def __init__(self, vocab) -> None:
        self.vocab = vocab

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        words: list[str] = []
        for token in token_ids:
            token = int(token)
            if token == self.eos_token_id:
                break
            if 0 <= token < len(self.vocab.id_to_token):
                word = self.vocab.id_to_token[token]
                if word is not None:
                    words.append(word)
        return " ".join(words)

    def __call__(self, text: str, add_special_tokens: bool = False, **kwargs):
        del add_special_tokens, kwargs
        return {
            "input_ids": [
                self.vocab.token_to_id.get(word, self.vocab.UNK)
                for word in str(text).split()
            ]
        }


class _SmokePredictor:
    """Deterministic source-side candidate generator.

    `sample_token_rows` feeds the labeler and `sample` feeds the rollout
    runtime. Both draw from the same toy word pool and honour the seed, so the
    two paths are reproducible.
    """

    def __init__(self, vocab, tokenizer, pool=SMOKE_POOL) -> None:
        self.vocab = vocab
        self.tok = tokenizer
        self.pool = tuple(pool)

    def _words(self, seed: int, index: int, length: int) -> list[str]:
        return [
            self.pool[(int(seed) + int(index) + depth) % len(self.pool)]
            for depth in range(max(1, int(length)))
        ]

    def sample_token_rows(
        self, observed: str, max_b: int, max_h: int, seed: int
    ) -> list[list[int]]:
        del observed
        rows: list[list[int]] = []
        for index in range(int(max_b)):
            words = self._words(seed, index, int(max_h))
            rows.append(
                [self.vocab.token_to_id.get(word, self.vocab.UNK) for word in words]
            )
        return rows

    def sample(self, observed: str, branches: int, horizon: int, seed: int) -> list[str]:
        del observed
        return [
            " ".join(self._words(seed, index, horizon))
            for index in range(max(1, int(branches)))
        ]


class _SmokeTranslator:
    """Translates each whitespace token through a fixed German/English gloss."""

    def __init__(self, gloss=SMOKE_GLOSS) -> None:
        self.gloss = dict(gloss)

    def translate_batch(self, sources, prefix, final: bool = False, **kwargs):
        del prefix, final, kwargs
        return [
            " ".join(self.gloss.get(word, word) for word in str(source).split())
            for source in sources
        ]


class _FixedTimer:
    """Every timed stage costs the same fixed number of seconds."""

    def __init__(self, seconds: float = SMOKE_STAGE_SECONDS) -> None:
        self.seconds = float(seconds)

    def run(self, name, function):
        del name
        value = function()
        return value, self.seconds


@dataclass(frozen=True)
class _SmokeSentence:
    """Minimal sentence shape accepted by `run_rollout`."""

    sentence_id: str
    source: str
    reference: str
    split: str = "smoke"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _feature_stats(states) -> tuple[list[float], list[float]]:
    """Elementwise mean and population std; std defaults to 1.0 when zero."""
    states = list(states)
    if not states:
        return [], []
    width = len(states[0].numeric)
    count = float(len(states))
    mean = [
        sum(float(state.numeric[index]) for state in states) / count
        for index in range(width)
    ]
    std: list[float] = []
    for index in range(width):
        variance = sum(
            (float(state.numeric[index]) - mean[index]) ** 2 for state in states
        ) / count
        std.append(math.sqrt(variance) if variance > 0.0 else 1.0)
    return mean, std


def _trace_record(trace) -> dict:
    """Convert one `RolloutTrace` to a JSON-serializable dict."""
    return {
        "sentence_id": trace.sentence_id,
        "source": trace.source,
        "reference": trace.reference,
        "hypothesis": trace.hypothesis,
        "committed": list(trace.committed),
        "delays": list(trace.delays),
        "laal": trace.laal,
        "completion_s": trace.completion_s,
        "read_fraction": trace.read_fraction,
        "action_histogram": dict(trace.action_histogram),
        "steps": [dict(step) for step in trace.steps],
    }


def _config_summary(cfg: ControllerConfig) -> dict:
    """A compact, JSON-serializable config view for the smoke report."""
    return {
        "seed": int(cfg.seed),
        "min_start": int(cfg.min_start),
        "arrival_gap": float(cfg.arrival_gap),
        "ralcp_threshold": float(cfg.ralcp_threshold),
        "gamma": float(cfg.gamma),
        "lambda_l": float(cfg.lambda_l),
        "lambda_t": float(cfg.lambda_t),
        "numeric_dim": int(cfg.numeric_dim),
        "epochs": int(cfg.epochs),
        "batch_size": int(cfg.batch_size),
        "lr": float(cfg.lr),
        "actions": [action.label for action in cfg.actions],
    }


# ----------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------
def _run_dry_run(args) -> int:
    """Print the config, action grid, and feature list without loading models."""
    cfg = ControllerConfig(seed=int(args.seed))
    print("controller_v3 dry run - no model weights are loaded")
    print("")
    print("config:")
    for field in dataclasses.fields(cfg):
        value = getattr(cfg, field.name)
        if field.name == "actions":
            value = [action.label for action in cfg.actions]
        elif field.name == "advantage_baseline":
            value = tuple(cfg.advantage_baseline)
        print(f"  {field.name}: {value}")

    print("")
    print(f"actions: READ + {len(cfg.actions)} write actions")
    print(f"  {READ.label}")
    for action in cfg.actions:
        print(
            f"  {action.label} (branches={action.branches}, "
            f"horizon={action.horizon})"
        )
    print(f"DEFAULT_ACTIONS: {len(DEFAULT_ACTIONS)} write actions")
    print(f"EXTENDED_ACTIONS: {len(EXTENDED_ACTIONS)} write actions")

    print("")
    print(f"numeric features: {len(NUMERIC_FEATURES)}")
    for index, name in enumerate(NUMERIC_FEATURES, start=1):
        print(f"  {index:02d}. {name}")
    return 0


def _run_smoke(args) -> int:
    """Run the synthetic end-to-end CPU path and write report artifacts."""
    # Imported lazily so `--dry-run` constructs no model and loads no weights.
    # The package import itself still imports torch through the public exports.
    from dataclasses import replace

    import torch

    from .checkpoint import load_checkpoint, save_checkpoint
    from .cost_model import MeasuredCostTable
    from .evaluator import summarize_traces
    from .features import FeatureBuilder
    from .labeler import CounterfactualLabeler
    from .network import CAVCNetwork
    from .rollout import run_rollout
    from .targets import build_action_targets
    from .text_encoder import Vocab
    from .trainer import CAVCTrainer

    torch.manual_seed(int(args.seed))
    cfg = replace(
        ControllerConfig(),
        seed=int(args.seed),
        epochs=int(args.epochs),
        actions=(Action(2, 4), Action(4, 6)),
        batch_size=8,
    )

    source_texts = [source for _, source, _ in SMOKE_SENTENCES]
    target_texts = [reference for _, _, reference in SMOKE_SENTENCES]
    vocab = Vocab.from_texts(
        [*source_texts, *target_texts, " ".join(SMOKE_POOL)]
    )
    tokenizer = _SmokeTokenizer(vocab)
    predictor = _SmokePredictor(vocab, tokenizer)
    translator = _SmokeTranslator()
    timer = _FixedTimer()
    builder = FeatureBuilder(length_ratio=1.0, src_vocab=vocab, tgt_vocab=vocab)
    labeler = CounterfactualLabeler(predictor, translator, cfg, timer=timer)
    network = CAVCNetwork(cfg, src_vocab_size=len(vocab), tgt_vocab_size=len(vocab))

    # ------------------------------------------------------------------
    # Labels: three states x three actions (READ + two writes)
    # ------------------------------------------------------------------
    targets = []
    states = []
    cost_records: list[tuple[int, int, Action, float, float]] = []
    for index, (sentence_id, source, _reference) in enumerate(SMOKE_SENTENCES):
        words = source.split()
        observed = " ".join(words[:2])
        successor_observed = " ".join(words[:3])
        committed: tuple[str, ...] = ()

        state = builder.build(observed, committed, predictor_id=0, translator_id=0)
        outcomes = labeler.label_state(
            observed, committed, source, SMOKE_ACTIONS, seed=int(args.seed) + index
        )
        # One successor per action: READ keeps `committed`, each write appends
        # its own `next_committed`. This exercises the corrected target, not the
        # legacy shared-successor ablation.
        next_states = [
            builder.build(
                successor_observed,
                outcome.next_committed,
                predictor_id=0,
                translator_id=0,
            )
            for outcome in outcomes
        ]
        targets.extend(
            build_action_targets(
                state,
                outcomes,
                cfg,
                next_states=next_states,
                src_words_after=len(successor_observed.split()),
                source_advance=1,
                state_key=f"{sentence_id}:2",
            )
        )
        states.append(state)
        for outcome in outcomes:
            if not outcome.action.is_read:
                cost_records.append(
                    (
                        0,
                        0,
                        outcome.action,
                        float(outcome.pred_s),
                        float(outcome.mt_s),
                    )
                )

    # ------------------------------------------------------------------
    # Train, checkpoint, reload
    # ------------------------------------------------------------------
    cost_model = MeasuredCostTable(cost_records)
    trainer = CAVCTrainer(network, cfg, cost_model)
    training = trainer.fit(targets)
    feature_mean, feature_std = _feature_stats(states)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        network,
        cfg,
        feature_mean,
        feature_std,
        cost_model,
        extra={"mode": "smoke", "training": training},
    )
    metadata = load_checkpoint(checkpoint_path, network)

    # ------------------------------------------------------------------
    # Roll out two sentences
    # ------------------------------------------------------------------
    traces = []
    for sentence_id, source, reference in SMOKE_SENTENCES[1:3]:
        traces.append(
            run_rollout(
                _SmokeSentence(sentence_id, source, reference),
                predictor,
                translator,
                network,
                builder,
                cost_model,
                cfg,
                timer=timer,
            )
        )
    # Synthetic sentences: skip BLEU entirely and force the no-score status.
    # This is deliberate even when `sacrebleu` is installed, so the smoke can
    # never emit a real-looking translation-quality number.
    rollout_summary = summarize_traces(
        traces, include_comet=False, include_bleu=False
    )
    rollout_summary["bleu"] = dict(SMOKE_BLEU_STATUS)
    rollout_summary["comet"] = dict(SMOKE_COMET_STATUS)

    report = {
        "mode": "smoke",
        "synthetic": True,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "n_features": len(NUMERIC_FEATURES),
        "n_states": len(SMOKE_SENTENCES),
        "n_actions": len(SMOKE_ACTIONS),
        "n_targets": len(targets),
        "actions": [action.label for action in SMOKE_ACTIONS],
        "feature_names": list(NUMERIC_FEATURES),
        "config": _config_summary(cfg),
        "training": training,
        "checkpoint": {
            "path": str(checkpoint_path),
            "format_version": metadata["format_version"],
            "seed": metadata["seed"],
            "git_commit": metadata["git_commit"],
            "feature_mean_len": len(metadata["feature_mean"]),
            "feature_std_len": len(metadata["feature_std"]),
            "cost_records": len(metadata["cost_records"]),
        },
        "rollout": rollout_summary,
        "notes": [
            "Synthetic three-sentence smoke data; not Kaggle data.",
            "BLEU is forced to unavailable for the smoke and corpus_bleu is "
            "never called, even when sacrebleu is installed.",
            "COMET is always skipped for the smoke because its text is "
            "synthetic; loading it would also download model weights.",
        ],
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    traces_path = output_dir / "traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(_trace_record(trace), sort_keys=True) + "\n")

    bleu = rollout_summary.get("bleu", {})
    print("smoke complete")
    print(f"  targets:    {len(targets)}")
    print(f"  rollouts:   {len(traces)}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  summary:    {summary_path}")
    print(f"  traces:     {traces_path}")
    print(f"  BLEU:       {bleu.get('status', 'unknown')}")
    print(f"  COMET:      {rollout_summary.get('comet', {}).get('status', 'unknown')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser; every mode is explicit and CPU-only."""
    parser = argparse.ArgumentParser(
        prog="python -m controller_v3.cli",
        description=(
            "Cost-aware value controller v3. Use --dry-run to inspect the "
            "configuration or --smoke for a synthetic CPU end-to-end run."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print config, actions, and the 34 numeric features without loading models",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the synthetic CPU end-to-end smoke test and write report files",
    )
    parser.add_argument(
        "--train-labels",
        metavar="PATH",
        default=None,
        help="path to labelled training data (requires a real model adapter)",
    )
    parser.add_argument(
        "--rollout",
        action="store_true",
        help="roll out a trained controller (requires a real model adapter)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/cavc",
        help="directory for summary.json and traces.jsonl",
    )
    parser.add_argument(
        "--epochs", type=int, default=SMOKE_EPOCHS, help="smoke training epochs"
    )
    parser.add_argument(
        "--seed", type=int, default=13, help="random seed for the smoke path"
    )
    return parser


def main(argv=None) -> int:
    """Parse arguments and dispatch one mode."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.dry_run:
        return _run_dry_run(args)
    if args.smoke:
        return _run_smoke(args)
    if args.train_labels is not None or args.rollout:
        raise NotImplementedError(
            "--train-labels/--rollout need a real predictor and translator "
            "adapter, and controller_v3 does not download or load model "
            "weights from the CLI. Use --dry-run or --smoke for the CPU path."
        )

    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
