"""Package paced two-device PFBA comparisons for private Kaggle execution."""
from __future__ import annotations

import argparse
import hashlib
import json

from prepare_kernel import BOOTSTRAP, ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--name", default="pfba-decoupled-dev")
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    stage = ROOT / "kaggle-runs" / options.name
    stage.mkdir(parents=True, exist_ok=True)
    sources = {
        "run_pfba.py": ROOT / "paired" / "run_pfba.py",
        "pipeline.py": ROOT / "decoupled" / "pipeline.py",
        "checkpoint.py": ROOT / "decoupled" / "checkpoint.py",
        "cpu_embedding.py": ROOT / "decoupled" / "cpu_embedding.py",
        "run_decoupled.py": ROOT / "decoupled" / "run_decoupled.py",
        "paired_evidence.py": ROOT / "paired_evidence.py",
    }
    script = BOOTSTRAP + '\nroot = Path("/tmp/pfba_pipeline")\nroot.mkdir(exist_ok=True)\n'
    for name, path in sources.items():
        code = path.read_text(encoding="utf-8")
        compile(code, name, "exec")
        digest = hashlib.sha256(code.encode()).hexdigest()
        script += f'(root / {name!r}).write_text({code!r}, encoding="utf-8")\n'
        script += f'print("{name} SHA-256: {digest}", flush=True)\n'
    args = list(options.runner_args)
    if args and args[0] == "--":
        args.pop(0)
    args.extend(("--output-dir", "/kaggle/working/decoupled"))
    script += f"subprocess.run([sys.executable, '-u', 'run_decoupled.py', *{args!r}], cwd=root, check=True)\n"
    compile(script, "kaggle_entry.py", "exec")
    (stage / "kaggle_entry.py").write_text(script, encoding="utf-8")

    metadata = {
        "id": f"{options.owner}/{options.name}", "title": options.name.replace("-", " "),
        "code_file": "kaggle_entry.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }
    (stage / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(stage)


if __name__ == "__main__":
    main()
