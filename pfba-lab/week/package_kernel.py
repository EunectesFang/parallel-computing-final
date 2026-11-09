"""Write the private Kaggle script for this week's one-split run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = """import subprocess
import sys
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6",
     "sacrebleu==2.5.1", "sentencepiece", "accelerate", "sacremoses",
     "entmax", "jsonargparse==3.13.1", "torchmetrics==0.10.3", "pytorch-lightning"],
    check=True,
)
"""


def main() -> None:
    stage = ROOT / "kaggle-runs" / "pfba-week-confidence"
    stage.mkdir(parents=True, exist_ok=True)
    files = {
        "run_pfba.py": ROOT / "paired" / "run_pfba.py",
        "week_run.py": ROOT / "week" / "week_run.py",
        "cpu_embedding.py": ROOT / "decoupled" / "cpu_embedding.py",
    }
    writes = []
    for name, path in files.items():
        code = path.read_text(encoding="utf-8")
        compile(code, name, "exec")
        digest = hashlib.sha256(code.encode()).hexdigest()
        writes.append(f"(root / {name!r}).write_text({code!r}, encoding='utf-8')")
        writes.append(f"print({name!r} + ' SHA-256: {digest}', flush=True)")
    script = BOOTSTRAP + "\nroot = Path('/tmp/pfba_week')\nroot.mkdir(exist_ok=True)\n"
    script += "\n".join(writes) + "\n"
    script += (
        "subprocess.run([sys.executable, '-u', str(root / 'week_run.py'), "
        "'--output-dir', '/kaggle/working/week', '--stride', '4', "
        "'--max-hours', '8', '--large-limit', '64'], cwd=root, check=True)\n"
    )
    compile(script, "kaggle_entry.py", "exec")
    (stage / "kaggle_entry.py").write_text(script, encoding="utf-8")
    metadata = {
        "id": "eunectesqaq/pfba-week-confidence",
        "title": "pfba week confidence",
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
