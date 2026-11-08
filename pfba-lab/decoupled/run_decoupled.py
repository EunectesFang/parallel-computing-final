"""Compare batch TAF against a separately scheduled prediction/translation pipeline."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter

from checkpoint import Journal, atomic_json
from pipeline import Config, DEFAULT_MODES, MODES, Span, Step, Trace, run_sentence
import run_pfba as adapters


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor", choices=adapters.PREDICTORS, default="large")
    parser.add_argument("--translator", choices=adapters.TRANSLATORS, default="large")
    parser.add_argument("--predictor-device", default="cuda:0")
    parser.add_argument("--prediction-batch-size", type=int, default=0)
    parser.add_argument("--cpu-predictor-embedding", action="store_true")
    parser.add_argument("--translator-device", default="cuda:1")
    parser.add_argument("--allow-shared-device", action="store_true")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--min-start", type=int, default=3)
    parser.add_argument("--arrival-gap", type=float, default=0.3)
    parser.add_argument("--ralcp-threshold", type=float, default=0.6)
    parser.add_argument("--anchor-threshold", type=float)
    parser.add_argument("--anchor-policy", choices=("vote", "veto"), default="vote")
    parser.add_argument("--ready-limit", type=int, default=1)
    parser.add_argument("--bounded-threshold", type=float)
    parser.add_argument("--bounded-initial-steps", type=int, default=2)
    parser.add_argument("--bounded-initial-words", type=int, default=1)
    parser.add_argument("--bounded-skip-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--modes", choices=MODES, nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-comet", action="store_true")
    parser.add_argument("--metric-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def summarize(traces, samples, mode, skip_comet, replay_s, metric_device="cpu"):
    hypotheses = [trace.hypothesis for trace in traces]
    references = [sample.reference for sample in samples]
    sources = [sample.source for sample in samples]
    bleu = adapters.corpus_bleu(hypotheses, references)
    comet = ({"status": "skipped"} if skip_comet else adapters.comet_score(
        hypotheses, sources, references, adapters.DEFAULT_COMET, metric_device))
    if bleu["status"] != "ok" or (not skip_comet and comet["status"] != "ok"):
        raise RuntimeError(f"scoring failed: BLEU {bleu}, COMET {comet}")
    completions = sorted(trace.completion_s for trace in traces)
    first_output = [trace.first_output_s for trace in traces if trace.first_output_s is not None]
    skipped = [right.read_pos - left.read_pos - 1 for trace in traces
               for left, right in zip(trace.steps, trace.steps[1:])]
    return {
        "mode": mode, "n": len(traces), "bleu": bleu, "comet": comet,
        "laal": adapters.corpus_laal(traces),
        "mean_first_output_s": mean(first_output) if first_output else None,
        "mean_completion_s": mean(completions),
        "p95_completion_s": completions[(95 * len(completions) - 1) // 100],
        "mean_drain_s": mean(trace.drain_s for trace in traces),
        "actual_replay_s": replay_s,
        "prediction_service_s": sum(span.end_s - span.start_s for trace in traces
                                    for span in trace.predictions),
        "translation_service_s": sum(step.translation.end_s - step.translation.start_s
                                     for trace in traces for step in trace.steps),
        "overlap_s": sum(trace.overlap_s for trace in traces),
        "prediction_calls": sum(len(trace.predictions) for trace in traces),
        "bounded_skip_events": sum(span.skipped_before > 0 for trace in traces
                                   for span in trace.predictions),
        "translation_calls": sum(len(trace.steps) for trace in traces),
        "translation_requests": sum(step.translation_requests for trace in traces for step in trace.steps),
        "peak_ready": max(trace.peak_ready for trace in traces),
        "discarded_predictions": sum(trace.discarded_predictions for trace in traces),
        "skipped_source_positions": sum(skipped),
        "max_skipped_between_translations": max(skipped, default=0),
    }


def main():
    args = parse_args()
    cfg = Config(
        branches=args.branches, horizon=args.horizon, min_start=args.min_start,
        arrival_gap=args.arrival_gap, threshold=args.ralcp_threshold, seed=args.seed,
        ready_limit=args.ready_limit,
        anchor_policy=args.anchor_policy, anchor_threshold=args.anchor_threshold,
        bounded_threshold=args.bounded_threshold, bounded_initial_steps=args.bounded_initial_steps,
        bounded_initial_words=args.bounded_initial_words,
        bounded_skip_every=args.bounded_skip_every,
    )
    if args.limit < 0 or args.offset < 0 or args.prediction_batch_size < 0:
        raise ValueError("limit and offset must be nonnegative")
    if len(args.modes) != len(set(args.modes)):
        raise ValueError("modes must be unique")
    pdev, tdev = args.predictor_device, args.translator_device
    if pdev == tdev and not args.allow_shared_device:
        raise ValueError("prediction and translation require separate devices")
    metadata = {
        "config": asdict(cfg), "split": args.split, "offset": args.offset,
        "limit": args.limit, "predictor": adapters.PREDICTORS[args.predictor],
        "translator": adapters.TRANSLATORS[args.translator],
        "predictor_device": pdev, "translator_device": tdev, "modes": list(args.modes),
        "upstream_commit": adapters.UPSTREAM_COMMIT,
        "clock": "monotonic wall-clock, paced text replay",
        "prediction_batch_size": args.prediction_batch_size,
        "cpu_predictor_embedding": args.cpu_predictor_embedding,
    }
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return
    samples = adapters.load_split(args.split, args.offset + args.limit if args.limit else 0)
    samples = samples[args.offset:]
    if not samples:
        raise ValueError("no sentences selected")

    torch = adapters._torch()
    predictor = adapters.QwenPredictor(
        metadata["predictor"], pdev, "auto", args.local_files_only, False)
    predictor.prediction_batch_size = args.prediction_batch_size
    predictor.load()
    if args.cpu_predictor_embedding:
        from cpu_embedding import offload_input_embedding

        metadata["predictor_embedding_offload"] = offload_input_embedding(predictor.model)
    translator_cfg = adapters.StreamConfig(local_files_only=args.local_files_only)
    translator = adapters.load_translator(
        args.translator, metadata["translator"], tdev, translator_cfg)
    metadata.update({
        "predictor_dtype": str(predictor.model.dtype), "translator_dtype": str(translator.model.dtype),
        "predictor_revision": getattr(predictor.model.config, "_commit_hash", None),
        "translator_revision": getattr(translator.model.config, "_commit_hash", None),
        "torch": torch.__version__, "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    })
    metadata["code_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(__file__).with_name("pipeline.py"),
                     Path(__file__).with_name("checkpoint.py"), Path(adapters.__file__))
    }
    if args.cpu_predictor_embedding:
        helper = Path(__file__).with_name("cpu_embedding.py")
        metadata["code_sha256"][helper.name] = hashlib.sha256(helper.read_bytes()).hexdigest()
    journal = Journal(args.output_dir, metadata, samples, args.modes, args.resume)

    def sync_predict():
        adapters._sync(pdev)

    def sync_translate():
        adapters._sync(tdev)

    warm = " ".join(samples[0].source.split()[:cfg.min_start])
    predictor.sample(warm, cfg.branches, cfg.horizon, cfg.seed)
    sync_predict()
    translator.translate_batch([warm] * cfg.branches, "")
    sync_translate()

    for i, sample in enumerate(samples):
        offset = i % len(args.modes)
        for mode in (*args.modes[offset:], *args.modes[:offset]):
            if i < len(journal.rows[mode]):
                continue
            start = perf_counter()
            trace = run_sentence(
                sample.source, sample.reference, predictor, translator, cfg, mode,
                sentence_id=sample.sentence_id, split=args.split,
                sync_predict=sync_predict, sync_translate=sync_translate,
            )
            journal.append(mode, {
                "sentence_id": sample.sentence_id, "source": sample.source,
                "reference": sample.reference, **asdict(trace),
                "replay_s": perf_counter() - start,
            })
        print(f"[pipeline] {i + 1}/{len(samples)} sentences", flush=True)

    del predictor, translator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    journal.require_complete()
    traces = {mode: [restore_trace(record) for record in rows] for mode, rows in journal.rows.items()}
    replay_s = {mode: sum(record["replay_s"] for record in rows)
                for mode, rows in journal.rows.items()}

    cells = [summarize(traces[mode], samples, mode, args.skip_comet, replay_s[mode], args.metric_device)
             for mode in args.modes]
    if "taf" in traces:
        for cell in cells:
            pairs = list(zip(traces["taf"], traces[cell["mode"]], strict=True))
            cell["same_hypothesis_and_content_delays"] = sum(
                a.hypothesis == b.hypothesis and a.delays == b.delays for a, b in pairs)
            compared = matched = 0
            for baseline, alternative in pairs:
                baseline_hashes = {step.read_pos: step.candidate_sha256 for step in baseline.steps}
                for step in alternative.steps[:-1]:
                    if step.read_pos in baseline_hashes:
                        compared += 1
                        matched += step.candidate_sha256 == baseline_hashes[step.read_pos]
            cell["paired_candidate_steps"] = compared
            cell["matching_candidate_steps"] = matched
    summary = {"metadata": metadata, "cells": cells}
    path = args.output_dir / "summary.json"
    atomic_json(path, summary)
    print(f"[pipeline] wrote {path}")


def restore_trace(record):
    values = {name: record[name] for name in Trace.__dataclass_fields__}
    values["steps"] = tuple(Step(
        **{**step, "new_words": tuple(step["new_words"]), "translation": Span(**step["translation"])}
    ) for step in record["steps"])
    values["predictions"] = tuple(Span(**span) for span in record["predictions"])
    values["delays"] = tuple(record["delays"])
    return Trace(**values)


if __name__ == "__main__":
    main()
