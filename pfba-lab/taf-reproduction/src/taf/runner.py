"""Dataset-level streaming runner, word-unit metrics, and artifact writing."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .bleu import compute_bleu, corpus_bleu
from .config import StreamingConfig
from .laal import average_lagging, laal
from .policy import PolicyResult, StepTrace, build_policy


@dataclass
class SentenceResult:
    index: int
    source: str
    reference: str | None
    hypothesis: str
    token_ids: list[int]
    token_strings: list[str]
    word_delays: list[int]
    token_delays: list[int]
    traces: list[StepTrace]
    timing: dict[str, Any]
    metrics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "source": self.source,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
            "token_ids": self.token_ids,
            "token_strings": self.token_strings,
            "word_delays": self.word_delays,
            "token_delays": self.token_delays,
            "timing": self.timing,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "traces": [t.to_dict() for t in self.traces],
        }


@dataclass
class RunResult:
    sentences: list[SentenceResult]
    summary: dict[str, Any]
    config: dict[str, Any]
    wall_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "config": self.config,
            "wall_ms": self.wall_ms,
            "sentences": [s.to_dict() for s in self.sentences],
        }


def split_units(text: str, unit: str) -> list[str]:
    if unit == "word":
        return text.split()
    if unit == "char":
        return list(text)
    raise ValueError(f"unknown source unit {unit!r}")


def _is_word_start(token: str) -> bool:
    return token.startswith("▁") or token.startswith("Ġ")


def word_level_delays(
    token_strings: Sequence[str],
    token_delays: Sequence[int],
    word_level: bool,
) -> list[int]:
    """Group subword token delays into word-level delays.

    A word's delay is the delay of its first subword.  SentencePiece (``▁``)
    and GPT-2 BPE (``Ġ``) boundary markers are recognized.  Backends whose
    tokens are already words set ``word_level=True``.
    """
    if len(token_strings) != len(token_delays):
        raise ValueError("token_strings and token_delays must have equal length")
    if word_level:
        return list(token_delays)

    delays: list[int] = []
    for token, delay in zip(token_strings, token_delays):
        if not delays or _is_word_start(token):
            delays.append(delay)
        # continuation subwords inherit the word's existing (earliest) delay
    return delays


def _word_prefix_length(
    token_strings: Sequence[str], token_count: int, word_level: bool
) -> int:
    """Number of complete target words covered by the first ``token_count`` tokens."""
    token_count = max(0, min(token_count, len(token_strings)))
    if word_level:
        return token_count
    words = 0
    for index, token in enumerate(token_strings[:token_count]):
        if index == 0 or _is_word_start(token):
            words += 1
    return words


def source_completion_target_cutoff(
    traces: Sequence[StepTrace],
    source_length: int,
    token_strings: Sequence[str],
    word_level: bool,
    total_words: int,
) -> int:
    """SimulEval cutoff in target-word units.

    The first write that occurs once the source is fully read is the cutoff
    emission; everything after it is excluded.  If no write occurs at full
    source, every emission is kept.
    """
    for trace in traces:
        if trace.read_pos >= source_length and trace.committed_after > trace.committed_before:
            return _word_prefix_length(
                token_strings, trace.committed_after, word_level
            )
    return total_words


def _mean(values: Sequence[float | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _aggregate_timing(sentences: Sequence[SentenceResult], wall_ms: float) -> dict:
    stages = ("llm_sample", "mt_decode", "aggregate", "total")
    sums = {key: 0.0 for key in stages}
    n_steps = 0
    flush_ms = 0.0
    for sent in sentences:
        timing = sent.timing
        n_steps += int(timing.get("n_steps", 0))
        flush_ms += float(timing.get("flush_ms", 0.0))
        for key in stages:
            sums[key] += float(timing.get(f"sum_{key}_ms", 0.0))
    out = {"n_steps": n_steps, "wall_ms": round(wall_ms, 6),
           "sum_flush_ms": round(flush_ms, 6)}
    for key in stages:
        out[f"sum_{key}_ms"] = round(sums[key], 6)
        out[f"avg_{key}_ms"] = round(sums[key] / n_steps, 6) if n_steps else 0.0
    return out


def run_dataset(
    sources: Sequence[str],
    references: Sequence[str | None] | None,
    llm,
    mt,
    cfg: StreamingConfig,
    comet=None,
    progress=None,
) -> RunResult:
    if references is not None and len(references) != len(sources):
        raise ValueError("sources and references must have equal length")

    started = time.perf_counter()
    sentences: list[SentenceResult] = []
    for idx, source in enumerate(sources):
        units = split_units(source, cfg.src_unit)
        policy = build_policy(llm, mt, cfg)
        result: PolicyResult = policy.run(units)
        token_strings = mt.token_strings(result.committed_ids)
        word_delays = word_level_delays(
            token_strings,
            result.delays,
            getattr(mt, "word_level", False),
        )
        reference = references[idx] if references is not None else None
        reference_units = (
            split_units(reference, cfg.tgt_unit) if reference is not None else None
        )
        reference_length = len(reference_units) if reference_units is not None else 0
        cutoff = source_completion_target_cutoff(
            result.traces,
            len(units),
            token_strings,
            getattr(mt, "word_level", False),
            len(word_delays),
        )
        laal_value = laal(
            len(units), reference_length, word_delays, cutoff=cutoff
        )
        laal_no_cutoff = laal(len(units), reference_length, word_delays)
        al_value = average_lagging(len(units), len(word_delays), word_delays)
        sentence_bleu = (
            corpus_bleu([result.text], [[reference]])["score"] if reference is not None else None
        )
        metrics = {
            "laal": laal_value,
            "laal_no_cutoff": laal_no_cutoff,
            "laal_cutoff": cutoff,
            "laal_reference_length": reference_length,
            "laal_target_length": max(reference_length, cutoff),
            "average_lagging": al_value,
            "bleu": sentence_bleu,
            "n_source_units": len(units),
            "n_target_tokens": len(result.committed_ids),
            "n_target_words": len(word_delays),
        }
        sentences.append(
            SentenceResult(
                index=idx,
                source=source,
                reference=reference,
                hypothesis=result.text,
                token_ids=result.committed_ids,
                token_strings=token_strings,
                word_delays=word_delays,
                token_delays=result.delays,
                traces=result.traces,
                timing=result.timing,
                metrics=metrics,
                metadata=result.metadata,
            )
        )
        if progress is not None:
            progress(idx, len(sources), result.text)

    wall_ms = (time.perf_counter() - started) * 1000.0
    hypotheses = [s.hypothesis for s in sentences]
    has_refs = references is not None and all(r is not None for r in references)
    refs = list(references) if has_refs else None

    bleu = compute_bleu(hypotheses, refs) if has_refs else {
        "score": None,
        "status": "unavailable",
        "reason": "no references",
    }
    if comet is None:
        comet_result = {
            "score": None,
            "status": "disabled",
            "reason": "no COMET scorer configured",
            "model": None,
        }
    elif has_refs:
        comet_result = comet.score(
            [s.source for s in sentences], hypotheses, [str(r) for r in refs]
        )
    else:
        comet_result = {
            "score": None,
            "status": "unavailable",
            "reason": "no references",
            "model": getattr(comet, "model_name", None),
        }

    total_words = sum(len(s.word_delays) for s in sentences)
    summary = {
        "n_sentences": len(sentences),
        "n_source_units": sum(len(split_units(s.source, cfg.src_unit)) for s in sentences),
        "n_target_tokens": sum(len(s.token_ids) for s in sentences),
        "n_target_words": total_words,
        "bleu": bleu,
        "comet": comet_result,
        "laal": _mean([s.metrics["laal"] for s in sentences]),
        "laal_no_cutoff": _mean(
            [s.metrics["laal_no_cutoff"] for s in sentences]
        ),
        "average_lagging": _mean([s.metrics["average_lagging"] for s in sentences]),
        "timing": _aggregate_timing(sentences, wall_ms),
        "tokens_per_second": (
            round(total_words / (wall_ms / 1000.0), 3) if wall_ms > 0 and total_words else 0.0
        ),
        "policy": sentences[0].metadata if sentences else None,
    }
    return RunResult(sentences=sentences, summary=summary, config=cfg.to_dict(), wall_ms=wall_ms)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def build_manifest(
    cfg: StreamingConfig,
    llm_name: str | None,
    mt_name: str | None,
    comet_name: str | None,
    data_path: str | None,
    argv: Sequence[str] | None = None,
) -> dict:
    """Machine-readable exact/substitution record.  No score claims."""
    def classify(llm: str | None, mt: str | None, comet: str | None) -> dict:
        llm_kind = (
            "paper-exact (LLaMA-2-7B base)"
            if llm and "llama-2-7b" in llm.lower()
            else ("substitution" if llm else "none")
        )
        mt_kind = (
            "paper-exact (author Transformer-Big checkpoint)"
            if mt and "transformer-big" in mt.lower()
            else ("substitution" if mt else "none")
        )
        if comet is None:
            comet_kind = "none"
        elif "xcomet" in comet.lower():
            comet_kind = "paper-cited (XCOMET-XXL)"
        else:
            comet_kind = "substitution"
        return {"llm": llm_kind, "mt": mt_kind, "comet": comet_kind}

    return {
        "paper": {
            "title": "Anticipating Future with Large Language Model for Simultaneous Machine Translation",
            "venue": "NAACL 2025 long",
            "anthology": "2025.naacl-long.286",
            "pages": "5547-5557",
            "arxiv": "2410.22499v2",
            "official_code": (
                "https://github.com/owaski/TAF "
                "(local checkout ../taf-upstream @ 99e373a)"
            ),
            "author_checkout": {
                "path": "../taf-upstream",
                "commit": "99e373a887e0556c4d50e9ea08497889037876f8",
                "files_read": [
                    "agents/ralcp_taf.py",
                    "agents/ralcp.py",
                    "agents/base.py",
                    "agents/utils.py",
                ],
                "data": [
                    "../taf-upstream/data/de_en/dev.de",
                    "../taf-upstream/data/de_en/dev.en",
                ],
            },
        },
        "implementation": "independent reimplementation; author checkout inspected",
        "config": cfg.to_dict(),
        "models": {"llm": llm_name, "mt": mt_name, "comet": comet_name},
        "classification": classify(llm_name, mt_name, comet_name),
        "data": data_path,
        "argv": list(argv) if argv is not None else sys.argv,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in (
                    "torch",
                    "transformers",
                    "sentencepiece",
                    "datasets",
                    "sacrebleu",
                    "unbabel-comet",
                )
            },
        },
        "notes": [
            "Scores are valid only for the recorded backends; they are not paper reproduction numbers unless all exact artifacts are used.",
            "COMET is null when unavailable; it is never fabricated.",
            "Metric implementations: sacrebleu when available, otherwise internal 13a/exp BLEU; LAAL follows SimulEval (max(reference, prediction) length, source-completion cutoff, no per-token clamp).",
        ],
    }


def write_run_artifacts(
    run: RunResult,
    manifest: dict,
    out_dir: str | Path,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def write_lines(name: str, values: Sequence[str]) -> None:
        (out / name).write_text(
            "\n".join(values) + ("\n" if values else ""), encoding="utf-8"
        )

    write_lines("sources.txt", [s.source for s in run.sentences])
    write_lines("references.txt", [s.reference or "" for s in run.sentences])
    write_lines("hypotheses.txt", [s.hypothesis for s in run.sentences])

    with (out / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for sentence in run.sentences:
            for trace in sentence.traces:
                record = {"sentence": sentence.index, **trace.to_dict()}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.write(
                json.dumps(
                    {
                        "sentence": sentence.index,
                        "event": "flush",
                        "committed_tokens": len(sentence.token_ids),
                        "flush_added_tokens": sentence.metadata.get("flush_added_tokens", 0),
                        "timing": sentence.timing,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with (out / "per_sentence.jsonl").open("w", encoding="utf-8") as handle:
        for sentence in run.sentences:
            handle.write(json.dumps(sentence.to_dict(), ensure_ascii=False) + "\n")

    (out / "summary.json").write_text(
        json.dumps(run.summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out
