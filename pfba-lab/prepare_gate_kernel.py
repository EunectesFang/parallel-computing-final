"""Package the joint confidence experiment as a private Kaggle job."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import zlib
from pathlib import Path

from prepare_kernel import BOOTSTRAP, ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--dev-limit", type=int, default=20)
    parser.add_argument("--test-limit", type=int, default=12)
    parser.add_argument("--test-offset", type=int, default=0)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--entropy-features", action="store_true")
    parser.add_argument("--neural", action="store_true")
    parser.add_argument("--labels-from", type=Path, help="reuse a prior experiment's public-corpus labels")
    parser.add_argument("--sweep", action="store_true", help="tune the neural policy on disjoint dev sentences")
    args = parser.parse_args()

    if args.sweep and not args.labels_from:
        parser.error("--sweep requires --labels-from")
    slug = "pfba-gate-matched-dev" if args.sweep else "pfba-joint-gate"
    stage = ROOT / "kaggle-runs" / slug
    stage.mkdir(parents=True, exist_ok=True)
    sources = {
        "run_pfba.py": ROOT / "paired" / "run_pfba.py",
        "joint_gate.py": ROOT / "joint-gate" / "joint_gate.py",
    }
    if args.sweep:
        sources["sweep_gate.py"] = ROOT / "joint-gate" / "sweep_gate.py"
    writes = []
    for name, path in sources.items():
        code = path.read_text(encoding="utf-8")
        compile(code, name, "exec")
        digest = hashlib.sha256(code.encode()).hexdigest()
        writes.append(f'(root / {name!r}).write_text({code!r}, encoding="utf-8")')
        writes.append(f'print("{name} SHA-256: {digest}", flush=True)')

    argv = [
        "--dev-limit", str(args.dev_limit), "--test-limit", str(args.test_limit),
        "--test-offset", str(args.test_offset),
        "--cost-weight", str(args.cost_weight), "--device", "cuda", "--metrics", "bleu", "comet",
        "--output-dir", "/kaggle/working/joint_gate",
    ]
    if args.entropy_features:
        argv.append("--entropy-features")
    if args.neural:
        argv.append("--neural")
    if args.labels_from:
        argv.extend(("--labels-dir", "/tmp/pfba_gate/labels"))
        writes.extend(("import base64, zlib", '(root / "labels").mkdir(exist_ok=True)'))
        groups = ("train", "tune") if args.test_offset or args.sweep else ("train", "tune", "test")
        for split in groups:
            name = f"labels_{split}.jsonl"
            payload = base64.b64encode(zlib.compress((args.labels_from / name).read_bytes()))
            writes.append(f'(root / "labels" / {name!r}).write_bytes(zlib.decompress(base64.b64decode({payload!r})))')
    script = BOOTSTRAP + '\nroot = Path("/tmp/pfba_gate")\nroot.mkdir(exist_ok=True)\n'
    script += "\n".join(writes) + "\n"
    entrypoint = "from joint_gate import main; raise SystemExit(main())"
    if args.sweep:
        entrypoint = "from sweep_gate import main; main()"
        argv = ["--labels-dir", "/tmp/pfba_gate/labels", "--cost-weight", str(args.cost_weight),
                "--output-dir", "/kaggle/working/gate_sweep"]
    script += f"subprocess.run([sys.executable, '-u', '-c', {entrypoint!r}, *{argv!r}], cwd=root, check=True)\n"
    compile(script, "kaggle_entry.py", "exec")
    (stage / "kaggle_entry.py").write_text(script, encoding="utf-8")

    metadata = {
        "id": f"{args.owner}/{slug}", "title": slug.replace("-", " "), "code_file": "kaggle_entry.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }
    (stage / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(stage)


if __name__ == "__main__":
    main()
