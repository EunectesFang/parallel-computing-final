"""Tune a neural controller on new development sentences at matched LAAL."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

import joint_gate as gate


class BoundedReadPolicy(gate.LearnedGatePolicy):
    def __init__(self, model, name, limit):
        super().__init__(model, name)
        self.limit = limit
        self.streak = 0

    def decide(self, features):
        scores = self.model.score_actions(features)
        best = max(scores, key=scores.get)
        if scores[best] < self.model.threshold and self.streak < self.limit:
            self.streak += 1
            return gate.READ
        self.streak = 0
        return best


def read_labels(root, name):
    rows = (root / f"labels_{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [gate.TrainingRow(**dict(row, committed=tuple(row["committed"]), new_words=tuple(row["new_words"])))
            for row in map(json.loads, rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=24)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--cost-weight", type=float, default=0.6)
    parser.set_defaults(device="cuda", predictor_device=None, translator_device=None)
    args = parser.parse_args()
    train, tune = (read_labels(args.labels_dir, name) for name in ("train", "tune"))
    model = gate.fit_gate_model(train, tune, cost_weight=args.cost_weight, neural=True)
    sentences = gate.load_official_sentences("dev", args.offset + args.limit)[args.offset:]
    seen = {row.sentence_id for row in train + tune}
    if len(sentences) != args.limit or seen.intersection(s.sentence_id for s in sentences):
        raise ValueError("validation sentences must be disjoint from the training and calibration data")

    scores = [max(model.score_actions(rows[0].features).values())
              for rows in gate._state_groups(tune).values()]
    thresholds = np.quantile(scores, (0.0, 0.2, 0.4)).tolist()
    policies = {
        "fixed_4_6": lambda: gate.FixedActionPolicy(gate.FIXED_BASELINE),
        "fixed_2_6": lambda: gate.FixedActionPolicy(gate.FIXED_DEV_REFERENCE),
    }
    settings = {}
    for i, threshold in enumerate(thresholds):
        name = f"neural_q{i}"
        candidate = replace(model, threshold=float(threshold))
        policies[name] = lambda candidate=candidate, name=name: gate.LearnedGatePolicy(candidate, name)
        settings[name] = {"threshold": float(threshold)}
    policies["neural_bounded"] = lambda: BoundedReadPolicy(model, "neural_bounded", 1)
    settings["neural_bounded"] = {"threshold": model.threshold, "max_consecutive_reads": 1}

    cfg = gate.GateConfig(entropy_features=True, neural=True, device="cuda", cost_weight=args.cost_weight)
    predictor, translator, timer = gate.build_small_adapters(args, cfg)
    warm = " ".join(sentences[0].source.split()[:cfg.min_start])
    predictor.sample(warm, 4, 6, cfg.seed)
    translator.translate_batch([warm] * 4, "")
    extractor = gate.FeatureExtractor(gate.make_qwen_entropy_fn(predictor))
    traces = {name: [] for name in policies}
    names = list(policies)
    for i, sentence in enumerate(sentences):
        offset = i % len(names)
        for name in names[offset:] + names[:offset]:
            traces[name].append(gate.run_online_rollout(
                sentence, predictor, translator, policies[name](), extractor, cfg, split="dev", timer=timer))
        print(f"[gate-sweep] {i + 1}/{len(sentences)}", flush=True)

    output = args.output_dir
    rollouts = {}
    for name, rows in traces.items():
        gate._write_jsonl(output / "rollouts" / f"{name}.jsonl", rows)
        rollouts[name] = gate.rollout_metrics(rows, ("bleu", "comet"), device="cuda")
    summary = {"config": asdict(cfg), "validation_ids": [s.sentence_id for s in sentences],
               "settings": settings, "rollouts": rollouts}
    (output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    gate.save_checkpoint(model, output / "confidence.joblib")


if __name__ == "__main__":
    main()
