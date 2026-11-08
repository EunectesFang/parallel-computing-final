"""Package the local paired experiment as one private Kaggle script."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOOTSTRAP = """import subprocess
import sys
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6",
     "sacrebleu==2.5.1", "sentencepiece", "accelerate", "sacremoses",
     "entmax", "jsonargparse==3.13.1", "torchmetrics==0.10.3", "pytorch-lightning"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "unbabel-comet==2.2.7"],
    check=True,
)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--predictor", choices=("small", "large"), required=True)
    parser.add_argument("--translator", choices=("small", "large"), required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    ablation = parser.add_mutually_exclusive_group()
    ablation.add_argument("--kv", action="store_true", help="include the predictor KV ablation")
    ablation.add_argument("--latest", action="store_true", help="test latest-arrived prefix scheduling")
    options = parser.parse_args()

    slug = f"pfba-{options.predictor}-{options.translator}-{options.split}"
    if options.kv:
        slug += "-kv"
    if options.latest:
        slug += "-latest"
    stage = ROOT / "kaggle-runs" / slug
    stage.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "paired" / "run_pfba.py").read_text(encoding="utf-8")
    compile(source, "run_pfba.py", "exec")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    command_args = [
        "--predictor", options.predictor, "--translator", options.translator,
        "--limit", str(options.limit), "--branches", str(options.branches),
        "--horizon", str(options.horizon), "--split", options.split,
        "--device", "cuda", "--output-dir", "/kaggle/working/pfba",
    ]
    cache_setup = ""
    if options.kv:
        kv_source = (ROOT / "kv-experiment" / "kv_prefix.py").read_text(encoding="utf-8")
        compile(kv_source, "kv_prefix.py", "exec")
        cache_setup = f'Path("/tmp/kv_prefix.py").write_text({kv_source!r}, encoding="utf-8")'
        command_args.extend(("--modes", "taf", "pfba", "pfba-kv"))
    if options.latest:
        command_args.extend(("--modes", "taf", "pfba-latest"))
    script = BOOTSTRAP + f"""
entry = Path("/tmp/run_pfba.py")
entry.write_text({source!r}, encoding="utf-8")
{cache_setup}
print("run_pfba.py SHA-256:", {source_hash!r}, flush=True)
subprocess.run([sys.executable, "-u", str(entry), *{command_args!r}], check=True)
"""
    (stage / "kaggle_entry.py").write_text(script, encoding="utf-8")
    metadata = {
        "id": f"{options.owner}/{slug}",
        "title": slug.replace("-", " ").upper(),
        "code_file": "kaggle_entry.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (stage / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(stage)


if __name__ == "__main__":
    main()
