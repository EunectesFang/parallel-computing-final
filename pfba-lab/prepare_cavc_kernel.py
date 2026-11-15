"""Package the CAVC v3 controller as one private Kaggle script.

The generated ``kaggle_entry.py`` embeds every ``week/controller_v3/*.py``
module, ``paired/run_pfba.py``, and ``decoupled/cpu_embedding.py`` as string
literals, then writes them under ``/tmp/pfba_cavc`` **preserving the
repository-relative layout**::

    /tmp/pfba_cavc/week/controller_v3/*.py
    /tmp/pfba_cavc/week/cpu_embedding.py
    /tmp/pfba_cavc/paired/run_pfba.py

The layout matters: ``controller_v3/runtime.py`` resolves ``run_pfba`` through
``Path(__file__).resolve().parents[2] / "paired"``. A flat
``/tmp/pfba_cavc/controller_v3`` plus ``/tmp/pfba_cavc/run_pfba.py`` would make
that import fail. The script then runs::

    python -m controller_v3.kaggle --mode preflight --limit 8 \
        --output-dir /kaggle/working/cavc

from ``/tmp/pfba_cavc/week``.

This module never pushes or submits to Kaggle. Run it with::

    python pfba-lab/prepare_cavc_kernel.py

It prints the generated paths and the SHA-256 of every bundled source file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTROLLER_DIR = ROOT / "week" / "controller_v3"
STAGE_DIR = ROOT / "kaggle-runs" / "pfba-cavc-v3"

KERNEL_ID = "eunectesqaq/pfba-cavc-v3-preflight"
KERNEL_TITLE = "pfba cavc v3 preflight"

# Kaggle's base image ships torch; these packages cover transformers, BLEU,
# tokenizers, and the translator adapters. COMET is not installed because
# preflight never requests it.
BOOTSTRAP = """import subprocess
import sys
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6",
     "sacrebleu==2.5.1", "sentencepiece", "accelerate", "sacremoses",
     "entmax", "jsonargparse==3.13.1", "torchmetrics==0.10.3",
     "pytorch-lightning"],
    check=True,
)
"""

# Default command executed inside the bundled tree.
DEFAULT_COMMAND = (
    "--mode",
    "preflight",
    "--limit",
    "8",
    "--output-dir",
    "/kaggle/working/cavc",
)

# Repo-relative destination -> repository source.
FIXED_SOURCES = (
    ("week/cpu_embedding.py", ROOT / "decoupled" / "cpu_embedding.py"),
    ("paired/run_pfba.py", ROOT / "paired" / "run_pfba.py"),
)


def _read_source(path: Path) -> str:
    """Read one UTF-8 source file byte-for-byte and validate that it compiles.

    Reading raw bytes (rather than `read_text`) avoids universal-newline
    translation, so the printed SHA-256 matches the file on disk and the
    generated kernel writes the same bytes that were hashed.
    """
    source = Path(path).read_bytes().decode("utf-8")
    compile(source, str(path), "exec")
    return source


def bundle_paths() -> list[str]:
    """Return the sorted repo-relative paths that the kernel will bundle."""
    controller = sorted(
        f"week/controller_v3/{path.name}" for path in CONTROLLER_DIR.glob("*.py")
    )
    return sorted([*controller, *(destination for destination, _ in FIXED_SOURCES)])


def bundle_sources() -> dict[str, str]:
    """Return ``{repo_relative_path: source}`` for every bundled file.

    Controller modules are sorted by file name so repeated runs produce the
    same script. Tests (``controller_v3/tests``) and ``__pycache__`` are not
    bundled; the runtime needs only the package modules.
    """
    sources: dict[str, str] = {}
    for path in sorted(CONTROLLER_DIR.glob("*.py")):
        sources[f"week/controller_v3/{path.name}"] = _read_source(path)
    for destination, path in FIXED_SOURCES:
        sources[destination] = _read_source(path)
    return sources


def source_hashes(sources: dict[str, str]) -> dict[str, str]:
    """SHA-256 of each bundled source, keyed by repo-relative path."""
    return {
        relative: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for relative, text in sources.items()
    }


def build_entry_script(
    sources: dict[str, str] | None = None,
    command: tuple[str, ...] | list[str] = DEFAULT_COMMAND,
) -> str:
    """Build the text of the generated ``kaggle_entry.py``.

    Every source is embedded with ``repr`` so newlines and quotes survive
    exactly. ``compile`` runs on the result before it is returned; the caller
    can therefore trust the generated file to be syntactically valid.
    """
    sources = bundle_sources() if sources is None else dict(sources)
    hashes = source_hashes(sources)
    script = (
        BOOTSTRAP
        + f"""
import hashlib

FILES = {sources!r}
EXPECTED = {hashes!r}

root = Path("/tmp/pfba_cavc")
for relative, text in FILES.items():
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"{{relative}} SHA-256: {{actual}}", flush=True)
    assert actual == EXPECTED[relative], f"hash mismatch for {{relative}}"

command = {tuple(command)!r}
subprocess.run(
    [sys.executable, "-u", "-m", "controller_v3.kaggle", *command],
    cwd=root / "week",
    check=True,
)
"""
    )
    compile(script, "kaggle_entry.py", "exec")
    return script


def build_metadata() -> dict:
    """Return the private GPU/internet kernel metadata payload."""
    return {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
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


def write_kernel(
    output_dir: Path | str = STAGE_DIR,
    sources: dict[str, str] | None = None,
    command: tuple[str, ...] | list[str] = DEFAULT_COMMAND,
) -> dict:
    """Write ``kaggle_entry.py`` and ``kernel-metadata.json`` under `output_dir`.

    Returns the stage path, both generated file paths, and every source hash.
    No Kaggle API call is made.
    """
    stage = Path(output_dir)
    stage.mkdir(parents=True, exist_ok=True)
    sources = bundle_sources() if sources is None else dict(sources)
    script = build_entry_script(sources, command)
    entry_path = stage / "kaggle_entry.py"
    metadata_path = stage / "kernel-metadata.json"
    # Pin LF so the committed bytes are identical on every platform and the
    # `--check` hash comparison is meaningful.
    entry_path.write_bytes(script.encode("utf-8"))
    metadata_path.write_bytes(
        (json.dumps(build_metadata(), indent=2) + "\n").encode("utf-8")
    )
    return {
        "stage": stage,
        "entry": entry_path,
        "metadata": metadata_path,
        "hashes": source_hashes(sources),
    }


def expected_outputs(
    sources: dict[str, str] | None = None,
    command: tuple[str, ...] | list[str] = DEFAULT_COMMAND,
) -> tuple[str, str]:
    """Return the exact ``(entry_text, metadata_text)`` a regeneration writes."""
    sources = bundle_sources() if sources is None else dict(sources)
    entry = build_entry_script(sources, command)
    metadata = json.dumps(build_metadata(), indent=2) + "\n"
    return entry, metadata


def verify_kernel(
    output_dir: Path | str = STAGE_DIR,
    sources: dict[str, str] | None = None,
    command: tuple[str, ...] | list[str] = DEFAULT_COMMAND,
) -> tuple[bool, list[str]]:
    """Compare the committed bundle byte-for-byte with generator output.

    Returns ``(ok, problems)``. ``problems`` names every stale or missing file
    so a caller can fail a check or a test can report exactly what changed. No
    file is written.
    """
    stage = Path(output_dir)
    expected_entry, expected_metadata = expected_outputs(sources, command)
    problems: list[str] = []
    for name, expected in (
        ("kaggle_entry.py", expected_entry),
        ("kernel-metadata.json", expected_metadata),
    ):
        path = stage / name
        if not path.is_file():
            problems.append(f"{path} is missing")
            continue
        actual = path.read_bytes().decode("utf-8")
        if actual != expected:
            problems.append(
                f"{path} is stale: regenerate with "
                "`python pfba-lab/prepare_cavc_kernel.py`"
            )
    return (not problems), problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(STAGE_DIR),
        help="directory for the generated kaggle_entry.py and metadata",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed bundle instead of regenerating it; exits 1 "
        "when kaggle_entry.py or kernel-metadata.json is stale",
    )
    return parser


def main(argv=None) -> int:
    """Generate the kernel and print every bundled file's SHA-256."""
    args = build_parser().parse_args(argv)
    if args.check:
        ok, problems = verify_kernel(args.output_dir)
        if ok:
            print(f"bundle up to date: {Path(args.output_dir)}")
            return 0
        for problem in problems:
            print(f"stale bundle: {problem}")
        return 1
    result = write_kernel(args.output_dir)
    print(f"generated: {result['entry']}")
    print(f"generated: {result['metadata']}")
    for relative, digest in result["hashes"].items():
        print(f"{relative} SHA-256: {digest}")
    print("not pushed; submit from the Kaggle UI or the CLI when ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
