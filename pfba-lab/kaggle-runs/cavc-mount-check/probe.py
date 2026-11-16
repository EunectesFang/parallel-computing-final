"""CPU-only probe: print exactly what Kaggle mounts under /kaggle/input."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def main() -> int:
    root = Path("/kaggle/input")
    print(f"input_root={root} exists={root.is_dir()}")
    for entry in sorted(root.iterdir()) if root.is_dir() else []:
        print(f"entry={entry} is_dir={entry.is_dir()}")

    files = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        files += 1
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        print(f"file={path} size={path.stat().st_size} sha256_16={digest}")
    print(f"total_files={files}")

    expected = Path("/kaggle/input/cavc-retrain-checkpoints/v3_baseline.pt")
    print(f"expected_checkpoint={expected} exists={expected.is_file()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
