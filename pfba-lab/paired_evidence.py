"""Compare two matched translation rollouts with sentence-paired intervals."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import fmean


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compare(baseline, candidate, baseline_comet, candidate_comet, repeats, seed, laal_margin):
    from sacrebleu.metrics import BLEU

    pairs = list(zip(baseline, candidate, baseline_comet, candidate_comet, strict=True))
    if not pairs or repeats < 100 or laal_margin < 0:
        raise ValueError("nonempty matched traces, at least 100 draws, and a nonnegative margin are required")
    ids = set()
    for left, right, left_comet, right_comet in pairs:
        keys = ("sentence_id", "source", "reference")
        if any(left[key] != right[key] for key in keys) or left["sentence_id"] in ids:
            raise ValueError("the traces must contain the same unique, ordered sentences and references")
        ids.add(left["sentence_id"])
        values = (left_comet, right_comet, left["laal"], right["laal"],
                  left["completion_s"], right["completion_s"])
        if any(value is None or not math.isfinite(value) for value in values):
            raise ValueError("each pair requires finite COMET, LAAL, and completion time")
    metric = BLEU()

    def deltas(sample):
        refs = [[left["reference"] for left, _, _, _ in sample]]
        left_bleu = metric.corpus_score([left["hypothesis"] for left, _, _, _ in sample], refs).score
        right_bleu = metric.corpus_score([right["hypothesis"] for _, right, _, _ in sample], refs).score
        return {
            "bleu": right_bleu - left_bleu,
            "comet": fmean(right - left for _, _, left, right in sample),
            "laal": fmean(right["laal"] - left["laal"] for left, right, _, _ in sample),
            "completion_s": fmean(right["completion_s"] - left["completion_s"] for left, right, _, _ in sample),
        }

    observed = deltas(pairs)
    draws = {key: [] for key in observed}
    rng = random.Random(seed)
    for _ in range(repeats):
        for key, value in deltas(rng.choices(pairs, k=len(pairs))).items():
            draws[key].append(value)
    intervals = {}
    for key, values in draws.items():
        values.sort()
        intervals[key] = [values[math.floor(0.025 * (repeats - 1))],
                          values[math.ceil(0.975 * (repeats - 1))]]
    return {
        "n": len(pairs), "repeats": repeats, "seed": seed, "delta_candidate_minus_baseline": observed,
        "paired_95_percent_intervals": intervals, "laal_margin": laal_margin,
        "passes": {
            "laal_equivalent": intervals["laal"][0] >= -laal_margin and intervals["laal"][1] <= laal_margin,
            "bleu_increase": intervals["bleu"][0] > 0,
            "comet_increase": intervals["comet"][0] > 0,
            "completion_time_reduction": intervals["completion_s"][1] < 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="gate or pipeline summary with sentence COMET scores")
    parser.add_argument("--baseline", default="fixed_4_6")
    parser.add_argument("--candidate", default="joint_neural")
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--laal-margin", type=float, default=0.5)
    args = parser.parse_args()
    document = json.loads(args.result.read_text(encoding="utf-8"))
    if "rollouts" in document:
        rollouts = document["rollouts"]
        trace_root = args.result.parent / "rollouts"
    else:
        rollouts = {cell["mode"]: cell for cell in document["cells"]}
        trace_root = args.result.parent
    print(json.dumps(compare(
        read_rows(trace_root / f"{args.baseline}.jsonl"),
        read_rows(trace_root / f"{args.candidate}.jsonl"),
        rollouts[args.baseline]["comet"]["sentence_scores"],
        rollouts[args.candidate]["comet"]["sentence_scores"],
        args.repeats, args.seed, args.laal_margin,
    ), indent=2))


if __name__ == "__main__":
    main()
