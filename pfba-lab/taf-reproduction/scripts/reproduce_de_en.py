#!/usr/bin/env python
"""Run a De-En streaming policy and emit auditable artifacts.

Examples
--------
CPU smoke on the authors' development pair (no downloads):
    python scripts/reproduce_de_en.py \
        --data ../taf-upstream/data/de_en/dev.de \
        --ref-data ../taf-upstream/data/de_en/dev.en \
        --llm-backend mock --mt-backend mock --policy ralcp_taf \
        --n 4 --beam-width 4 --max-sentences 5 \
        --output-dir results/author_dev_mock

GPU paper-style substitution:
    python scripts/reproduce_de_en.py \
        --data ../taf-upstream/data/de_en/test.de \
        --ref-data ../taf-upstream/data/de_en/test.en \
        --llm-backend hf --llm-model meta-llama/Llama-2-7b-hf \
        --mt-backend hf --mt-model Helsinki-NLP/opus-mt-de-en \
        --policy ralcp_taf --n 10 --horizon 10 --top-k 10 \
        --beam-width 40 --agree-thres 0.6 --min-start 3 \
        --comet-model Unbabel/XCOMET-XXL \
        --output-dir results/wmt15_ralcp_taf_opus
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taf.config import StreamingConfig  # noqa: E402
from taf.mocks import MockLLM, MockMT, make_vocab  # noqa: E402
from taf.runner import build_manifest, run_dataset, write_run_artifacts  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True,
                        help="TSV (src<TAB>ref), JSONL, or plain text (source only)")
    parser.add_argument("--ref-data", default=None,
                        help="parallel plain-text reference file (e.g. the authors' "
                             "data/de_en/dev.en paired with dev.de)")
    parser.add_argument("--src-column", type=int, default=0)
    parser.add_argument("--ref-column", type=int, default=1)
    parser.add_argument("--src-field", default="src")
    parser.add_argument("--ref-field", default="ref")
    parser.add_argument("--max-sentences", type=int, default=0)

    parser.add_argument("--policy",
                        choices=("ralcp_taf", "ralcp+taf", "taf", "ralcp", "waitk"),
                        default="ralcp_taf",
                        help="ralcp_taf is the authors' agents/ralcp_taf.py protocol")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10, help="LLM continuation length l")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--beam-width", type=int, default=40)
    parser.add_argument("--ralcp-fraction", "--agree-thres", dest="ralcp_fraction",
                        type=float, default=0.5,
                        help="authors' --agree-thres for word-prefix voting")
    parser.add_argument("--ralcp-content", action="store_true",
                        help="retained for compatibility; ignored by ralcp_taf")
    parser.add_argument("--min-start", type=int, default=1,
                        help="authors' --min-start (source units before first write)")
    parser.add_argument("--wait-k", type=int, default=3)
    parser.add_argument("--wait-stride", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--initial-read", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--src-unit", choices=("word", "char"), default="word")
    parser.add_argument("--tgt-unit", choices=("word", "char"), default="word")

    parser.add_argument("--llm-backend", choices=("hf", "mock"), default="mock")
    parser.add_argument("--llm-model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--llm-prompt-template", default="{source}")
    parser.add_argument("--llm-load-in-4bit", action="store_true")
    parser.add_argument("--llm-max-context", type=int, default=2048)

    parser.add_argument("--mt-backend", choices=("hf", "mock"), default="mock")
    parser.add_argument("--mt-model", default="Helsinki-NLP/opus-mt-de-en")
    parser.add_argument("--mt-arch", choices=("seq2seq", "causal"), default="seq2seq")
    parser.add_argument("--mt-prompt-template",
                        default="Translate German to English:\n{source}\nEnglish:")
    parser.add_argument("--mt-max-context", type=int, default=1024)

    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dtype", default="auto",
                        choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--comet-model", default=None,
                        help="e.g. Unbabel/XCOMET-XXL (paper footnote). "
                             "Omit to record COMET as disabled.")
    parser.add_argument("--no-cuda-sync", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def load_examples(path: str, args) -> tuple[list[str], list[str] | None]:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"data file not found: {file}")
    suffix = file.suffix.lower()
    sources: list[str] = []
    references: list[str] = []

    if args.ref_data:
        ref_file = Path(args.ref_data)
        if not ref_file.exists():
            raise FileNotFoundError(f"reference file not found: {ref_file}")
        sources = file.read_text(encoding="utf-8").splitlines()
        references = ref_file.read_text(encoding="utf-8").splitlines()
        if len(sources) != len(references):
            raise ValueError(
                f"--data and --ref-data have different line counts: "
                f"{len(sources)} != {len(references)}"
            )
    elif suffix == ".jsonl":
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sources.append(str(row[args.src_field]))
                if args.ref_field in row and row[args.ref_field] is not None:
                    references.append(str(row[args.ref_field]))
                else:
                    references.append("")
    elif suffix in (".tsv", ".csv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with file.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            rows = list(reader)
        if rows and any(cell.lower() in ("src", "source") for cell in rows[0]):
            header = [cell.lower() for cell in rows[0]]
            src_idx = header.index(args.src_field) if args.src_field in header else args.src_column
            ref_idx = header.index(args.ref_field) if args.ref_field in header else args.ref_column
        else:
            src_idx, ref_idx = args.src_column, args.ref_column
        for row in rows:
            if not row:
                continue
            if len(row) <= src_idx:
                continue
            sources.append(row[src_idx])
            references.append(row[ref_idx] if len(row) > ref_idx else "")
    else:
        sources = [line.rstrip("\n") for line in file.open(encoding="utf-8") if line.strip()]
        references = None

    if args.max_sentences and args.max_sentences > 0:
        sources = sources[: args.max_sentences]
        if references is not None:
            references = references[: args.max_sentences]
    if references is not None and all(ref == "" for ref in references):
        references = None
    return sources, references


def build_mock_llm(sources: list[str], args) -> MockLLM:
    """Oracle-ish table: n-1 samples predict the actual next word, one is empty."""
    table: dict[str, list[str]] = {}
    for source in sources:
        words = source.split()
        for pos in range(1, len(words) + 1):
            partial = " ".join(words[:pos])
            next_word = words[pos] if pos < len(words) else ""
            if next_word:
                table[partial] = [next_word] * (args.n - 1) + [""]
            else:
                table[partial] = [""] * args.n
    return MockLLM(table=table)


def build_backends(sources: list[str], args):
    if args.llm_backend == "mock":
        llm = build_mock_llm(sources, args)
        llm_name = "mock-table"
    else:
        from taf.hf_backends import HFLLM

        llm = HFLLM(
            args.llm_model,
            device=args.device,
            dtype=args.dtype,
            load_in_4bit=args.llm_load_in_4bit,
            prompt_template=args.llm_prompt_template,
            max_context=args.llm_max_context,
        )
        llm_name = args.llm_model

    if args.mt_backend == "mock":
        words: set[str] = set()
        for source in sources:
            words.update(source.split())
        mt = MockMT(mapping={}, vocab=make_vocab(words or {"empty"}))
        mt_name = "mock-identity"
    else:
        from taf.hf_backends import HFMT

        mt = HFMT(
            args.mt_model,
            device=args.device,
            dtype=args.dtype,
            mt_arch=args.mt_arch,
            max_context=args.mt_max_context,
            prompt_template=args.mt_prompt_template,
        )
        mt_name = args.mt_model
    return llm, mt, llm_name, mt_name


def main(argv=None) -> int:
    args = parse_args(argv)
    sources, references = load_examples(args.data, args)
    if not sources:
        print("no examples loaded", file=sys.stderr)
        return 2

    cfg = StreamingConfig(
        policy=args.policy,
        n=args.n,
        horizon=args.horizon,
        top_k=args.top_k,
        tau=args.tau,
        beam_width=args.beam_width,
        ralcp_fraction=args.ralcp_fraction,
        ralcp_content=args.ralcp_content,
        min_start=args.min_start,
        wait_k=args.wait_k,
        wait_stride=args.wait_stride,
        max_new_tokens=args.max_new_tokens,
        initial_read=args.initial_read,
        seed=args.seed,
        src_unit=args.src_unit,
        tgt_unit=args.tgt_unit,
        cuda_sync=not args.no_cuda_sync,
    )

    llm, mt, llm_name, mt_name = build_backends(sources, args)

    comet = None
    if args.comet_model:
        from taf.comet import CometScorer

        comet = CometScorer.load(args.comet_model, device=args.device)
        if comet.status != "loaded" and not args.quiet:
            print(f"[comet] unavailable: {comet.reason}", file=sys.stderr)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "results" / f"{stamp}_{args.policy}_{args.llm_backend}_{args.mt_backend}"
    )

    def progress(idx: int, total: int, hypothesis: str) -> None:
        if not args.quiet and (idx < 3 or (idx + 1) % 25 == 0 or idx + 1 == total):
            print(f"[{idx + 1}/{total}] {hypothesis[:120]}")

    run = run_dataset(sources, references, llm, mt, cfg, comet=comet, progress=progress)
    manifest = build_manifest(
        cfg,
        llm_name=llm_name,
        mt_name=mt_name,
        comet_name=args.comet_model,
        data_path=args.data,
        argv=sys.argv if argv is None else ["reproduce_de_en.py", *argv],
    )
    manifest["run"] = {
        "timestamp_utc": stamp,
        "output_dir": str(out_dir),
        "implementation_version": "0.1.0",
    }
    write_run_artifacts(run, manifest, out_dir)

    summary = run.summary
    print(json.dumps({
        "output_dir": str(out_dir),
        "n_sentences": summary["n_sentences"],
        "bleu": summary["bleu"].get("score"),
        "comet": (summary["comet"] or {}).get("score"),
        "comet_status": (summary["comet"] or {}).get("status"),
        "laal": summary["laal"],
        "avg_step_ms": summary["timing"]["avg_total_ms"],
        "classification": manifest["classification"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
