"""Train CAVC-Q V5 (dead-feature-free, non-negative cost heads, EOS wall)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

THREADS = int(os.environ.get("CAVC_THREADS", "14"))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "week"))

import torch  # noqa: E402

torch.set_num_threads(THREADS)

from controller_v3.retrain import retrain_from_labels  # noqa: E402

LABELS = ROOT / "kaggle-runs" / "results" / "cavc-dev-200" / "labels.jsonl"
BASE = ROOT / "kaggle-runs" / "results" / "cavc-dev-200" / "checkpoint.pt"
OUT = ROOT / "kaggle-runs" / "results" / "cavc-v5"


def main(argv: list[str]) -> int:
    name = argv[0] if argv else "v5"
    seed = int(argv[1]) if len(argv) > 1 else 13
    kappa = float(argv[2]) if len(argv) > 2 else 0.5
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
        convex_kappa=kappa,
        pair_weight=0.25,
        tri_rank_epsilon=0.10,
        alpha_rank=0.10,
        alpha_error=0.20,
        alpha_noemit=0.20,
        alpha_emit=0.0,
        weight_e=3.0,
        gamma_delay=1.0,
        gamma_wall=1.0,
        terminal_eos_wall_s=1.0,
        feature_normalize=True,
        epochs=30,
        lr=3e-4,
        batch_size=128,
        seed=seed,
        cost_topology="pipeline",
    )
    print(
        name,
        seed,
        "epochs",
        summary["training"]["epochs"],
        "loss",
        round(summary["training"]["final_loss"], 4),
        "seconds",
        round(summary["train_seconds"], 1),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
