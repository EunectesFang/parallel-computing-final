"""Save and load CAVC checkpoints with architecture and metadata checks.

A checkpoint carries everything needed to reproduce a controller run:

* the network `state_dict`;
* the frozen `ControllerConfig` as a plain dict;
* the feature mean and standard deviation used for reporting / normalisation
  experiments;
* the measured cost records (`(predictor_id, translator_id, action,
  pred_s, mt_s)` tuples);
* the seed and, when the checkout is a git repository, the current commit.

Loading is strict: every key, tensor shape, and feature-stat length is checked
against the target network and the stored config before any parameter is
copied. A mismatch raises `ValueError` naming every offending tensor instead of
letting `load_state_dict` fail with a partially loaded module.
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .action import Action
from .config import ControllerConfig

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "save_checkpoint",
    "load_checkpoint",
]

CHECKPOINT_FORMAT_VERSION = 1


# ----------------------------------------------------------------------
# Serialisation helpers
# ----------------------------------------------------------------------
def _config_to_dict(cfg: ControllerConfig) -> dict[str, Any]:
    """Return a JSON-friendly dict for `cfg` (actions become `[b, h]` lists)."""
    data = dataclasses.asdict(cfg)
    data["actions"] = [
        [int(action.branches), int(action.horizon)] for action in cfg.actions
    ]
    baseline = cfg.advantage_baseline
    data["advantage_baseline"] = (
        [int(baseline[0]), int(baseline[1])] if baseline else None
    )
    return data


def _to_float_list(values) -> list[float]:
    """Normalise a sequence, tensor, or scalar to a plain list of floats."""
    if values is None:
        return []
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    if isinstance(values, (int, float)):
        return [float(values)]
    return [float(value) for value in values]


def _to_cpu(value):
    """Move a tensor to CPU without copying non-tensor values."""
    return value.detach().cpu() if hasattr(value, "detach") else value


def _cost_model_records(cost_model) -> list[tuple[int, int, Action, float, float]]:
    """Extract measured records from a cost model, tolerating wrappers.

    `MeasuredCostTable` exposes `.records`. `PowerLawCostModel` keeps its
    measured fallback table under `.table`, so the same lookup is used for
    both. A model that exposes neither stores no records.
    """
    if cost_model is None:
        return []
    records = getattr(cost_model, "records", None)
    if records is None:
        table = getattr(cost_model, "table", None)
        records = getattr(table, "records", None)
    if records is None:
        return []
    return [
        (
            int(predictor_id),
            int(translator_id),
            action,
            float(pred_s),
            float(mt_s),
        )
        for predictor_id, translator_id, action, pred_s, mt_s in records
        if isinstance(action, Action)
    ]


def _serialize_cost_records(
    records: Sequence[tuple[int, int, Action, float, float]],
) -> list[list]:
    """Turn records into primitive lists so any torch loader can read them."""
    return [
        [
            int(predictor_id),
            int(translator_id),
            [int(action.branches), int(action.horizon)],
            float(pred_s),
            float(mt_s),
        ]
        for predictor_id, translator_id, action, pred_s, mt_s in records
    ]


def _deserialize_cost_records(raw: Any) -> list[tuple[int, int, Action, float, float]]:
    """Rebuild `Action` objects from the stored primitive records."""
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("checkpoint cost_records must be a list")
    records: list[tuple[int, int, Action, float, float]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 5:
            raise ValueError(
                f"checkpoint cost_records[{index}] must be "
                "(predictor_id, translator_id, [branches, horizon], pred_s, mt_s)"
            )
        predictor_id, translator_id, action_pair, pred_s, mt_s = item
        if (
            not isinstance(action_pair, (list, tuple))
            or len(action_pair) != 2
        ):
            raise ValueError(
                f"checkpoint cost_records[{index}] action must be [branches, horizon]"
            )
        records.append(
            (
                int(predictor_id),
                int(translator_id),
                Action(int(action_pair[0]), int(action_pair[1])),
                float(pred_s),
                float(mt_s),
            )
        )
    return records


def _git_commit() -> str | None:
    """Return the current HEAD commit, or ``None`` outside a git checkout."""
    repo_root = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


def _metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build the public metadata view of a checkpoint payload."""
    config = payload.get("config") or {}
    if not isinstance(config, dict):
        raise ValueError("checkpoint config must be a dict")
    return {
        "format_version": int(payload.get("format_version", 0)),
        "config": config,
        "feature_mean": [float(value) for value in payload.get("feature_mean", [])],
        "feature_std": [float(value) for value in payload.get("feature_std", [])],
        "cost_records": _deserialize_cost_records(payload.get("cost_records", [])),
        "seed": int(payload.get("seed", config.get("seed", 0))),
        "git_commit": payload.get("git_commit"),
        "extra": dict(payload.get("extra") or {}),
    }


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def save_checkpoint(
    path,
    network,
    cfg: ControllerConfig,
    feature_mean,
    feature_std,
    cost_model,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one checkpoint and return its metadata view.

    `feature_mean` / `feature_std` may be sequences or 1-D tensors. They are
    stored as plain float lists. `cost_model` may be ``None``; otherwise its
    measured records are stored as `(predictor_id, translator_id, action,
    pred_s, mt_s)` tuples (an empty list when the model carries none).

    `extra` is an optional mapping of additional primitive/tensor metadata.
    """
    if not hasattr(network, "state_dict"):
        raise TypeError("network must be a torch nn.Module with state_dict()")
    if not isinstance(cfg, ControllerConfig):
        raise TypeError("cfg must be a ControllerConfig")
    if extra is not None and not isinstance(extra, Mapping):
        raise TypeError("extra must be a mapping or None")

    mean = _to_float_list(feature_mean)
    std = _to_float_list(feature_std)
    if len(mean) != len(std):
        raise ValueError(
            f"feature_mean has {len(mean)} values but feature_std has {len(std)}"
        )
    if mean and len(mean) != int(cfg.numeric_dim):
        raise ValueError(
            f"feature statistics have {len(mean)} values but cfg.numeric_dim "
            f"is {cfg.numeric_dim}"
        )

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "state_dict": {
            key: _to_cpu(value) for key, value in network.state_dict().items()
        },
        "config": _config_to_dict(cfg),
        "feature_mean": mean,
        "feature_std": std,
        "cost_records": _serialize_cost_records(_cost_model_records(cost_model)),
        "seed": int(cfg.seed),
        "git_commit": _git_commit(),
        "extra": dict(extra or {}),
    }

    destination = Path(path)
    if destination.parent and not destination.parent.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return _metadata(payload)


def load_checkpoint(path, network) -> dict[str, Any]:
    """Restore `network` from `path` and return the checkpoint metadata.

    The saved and target state dicts are compared key by key first. Missing
    keys, unexpected keys, and shape mismatches all raise a single
    `ValueError` that lists the offending tensors, so a mismatched
    architecture never leaves a half-loaded network behind.
    """
    if not hasattr(network, "state_dict"):
        raise TypeError("network must be a torch nn.Module with state_dict()")

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {source}")

    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(
            f"{source} is not a controller_v3 checkpoint: missing 'state_dict'"
        )

    incoming = payload["state_dict"]
    if not isinstance(incoming, Mapping):
        raise ValueError("checkpoint state_dict must be a mapping")

    current = network.state_dict()
    problems: list[str] = []
    for key, reference in current.items():
        if key not in incoming:
            problems.append(f"missing key {key!r}")
            continue
        reference_shape = tuple(getattr(reference, "shape", ()))
        incoming_shape = tuple(getattr(incoming[key], "shape", ()))
        if reference_shape != incoming_shape:
            problems.append(
                f"shape mismatch for {key!r}: checkpoint {incoming_shape} vs "
                f"network {reference_shape}"
            )
    for key in incoming:
        if key not in current:
            problems.append(f"unexpected key {key!r}")
    if problems:
        raise ValueError(
            "checkpoint does not match the network architecture: "
            + "; ".join(problems)
        )

    metadata = _metadata(payload)
    mean, std = metadata["feature_mean"], metadata["feature_std"]
    if mean and std and len(mean) != len(std):
        raise ValueError(
            f"checkpoint feature_mean has {len(mean)} values but feature_std "
            f"has {len(std)}"
        )
    expected_dim = int(metadata["config"].get("numeric_dim", 0) or 0)
    if expected_dim and mean and len(mean) != expected_dim:
        raise ValueError(
            f"checkpoint feature statistics have {len(mean)} values but "
            f"config.numeric_dim is {expected_dim}"
        )

    network.load_state_dict(incoming, strict=True)
    return metadata
