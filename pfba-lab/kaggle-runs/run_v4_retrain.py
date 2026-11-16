"""Train CAVC-Q V4 variants offline from the persisted 200-sentence labels.

Usage:
    python run_v4_retrain.py linear 13
    python run_v4_retrain.py convex 13
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "week"))

from controller_v3.retrain import retrain_from_labels  # noqa: E402

LABELS = ROOT / "kaggle-runs" / "results" / "cavc-dev-200" / "labels.jsonl"
BASE = ROOT / "kaggle-runs" / "results" / "cavc-dev-200" / "checkpoint.pt"
OUT = ROOT / "kaggle-runs" / "results" / "cavc-v4"

VARIANTS = {
    "linear": 0.0,
    "convex": 0.5,
}


def main(argv: list[str]) -> int:
    name = argv[0] if argv else "convex"
    seed = int(argv[1]) if len(argv) > 1 else 13
    if name not in VARIANTS:
        raise SystemExit(f"variant must be one of {sorted(VARIANTS)}")
    output_dir = OUT / name / f"seed{seed}"
    summary = retrain_from_labels(
        LABELS,
        BASE,
        output_dir,
        tri_value=True,
        double_dqn=True,
        target_tau=0.01,
        beta_lag=0.25,
        beta_wall=0.30,
        lag_anchor=3.0,
        convex_kappa=VARIANTS[name],
        pair_weight=0.25,
        tri_rank_epsilon=0.10,
        alpha_rank=0.10,
        alpha_error=0.20,
        alpha_noemit=0.20,
        alpha_emit=0.0,
        weight_e=3.0,
        epochs=30,
        lr=3e-4,
        batch_size=128,
        seed=seed,
        cost_topology="serial",
    )
    print(
        name,
        seed,
        "epochs",
        summary["training"]["epochs"],
        "loss",
        round(summary["training"]["final_loss"], 4),
        "train_seconds",
        round(summary["train_seconds"], 1),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
