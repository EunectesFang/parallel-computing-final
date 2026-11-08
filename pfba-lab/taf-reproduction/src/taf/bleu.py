"""BLEU reimplementation compatible with sacrebleu's 13a/exp corpus BLEU.

The paper evaluates with SimulEval, which delegates BLEU to sacrebleu.  When
``sacrebleu`` is importable, ``compute_bleu`` uses it directly and records the
implementation.  The internal path exists so CPU invariant tests do not need
network access and so every number is auditable.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

_RE_PUNCT = re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])")
_RE_PERIOD_COMMA = re.compile(r"([^0-9])([\.,])")
_RE_PERIOD_COMMA_2 = re.compile(r"([\.,])([^0-9])")
_RE_DASH = re.compile(r"([0-9])(-)")


def tokenize_13a(text: str) -> list[str]:
    """mteval-v13a tokenizer (same regex sequence as sacrebleu)."""
    line = text.replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
    line = f" {line} "
    line = _RE_PUNCT.sub(r" \1 ", line)
    line = _RE_PERIOD_COMMA.sub(r"\1 \2 ", line)
    line = _RE_PERIOD_COMMA_2.sub(r" \1 \2", line)
    line = _RE_DASH.sub(r"\1 \2 ", line)
    return line.split()


def _ngrams(tokens: Sequence[str], order: int) -> Counter:
    return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))


def _corpus_bleu_internal(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
    max_order: int = 4,
) -> dict:
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")

    correct = [0] * (max_order + 1)
    total = [0] * (max_order + 1)
    hyp_len_total = 0
    ref_len_total = 0

    for hyp, refs in zip(hypotheses, references):
        hyp_tokens = tokenize_13a(hyp)
        ref_tokens = [tokenize_13a(r) for r in refs]
        hyp_len_total += len(hyp_tokens)

        # Closest reference length (sacrebleu uses the closest, ties shorter).
        if ref_tokens:
            ref_lens = [len(r) for r in ref_tokens]
            closest = min(ref_lens, key=lambda n: (abs(n - len(hyp_tokens)), n))
            ref_len_total += closest

        for order in range(1, max_order + 1):
            hyp_ngrams = _ngrams(hyp_tokens, order)
            total[order] += max(0, len(hyp_tokens) - order + 1)
            max_ref_counts: Counter = Counter()
            for ref in ref_tokens:
                ref_counts = _ngrams(ref, order)
                for ngram, count in ref_counts.items():
                    if count > max_ref_counts[ngram]:
                        max_ref_counts[ngram] = count
            correct[order] += sum(min(count, max_ref_counts[ngram]) for ngram, count in hyp_ngrams.items())

    precisions = [0.0] * (max_order + 1)
    for order in range(1, max_order + 1):
        if total[order] == 0:
            precisions[order] = 0.0
        elif correct[order] == 0:
            # sacrebleu exp smoothing: add 1 / 2**order correct n-grams.
            precisions[order] = (1.0 / (2**order)) / total[order]
        else:
            precisions[order] = correct[order] / total[order]

    if hyp_len_total == 0:
        bp = 0.0
    elif hyp_len_total < ref_len_total:
        bp = math.exp(1.0 - ref_len_total / hyp_len_total)
    else:
        bp = 1.0

    # Standard corpus BLEU: no unigram match means BLEU 0.  Higher orders use
    # exp smoothing only when the unigram precision is positive (sacrebleu
    # behaviour for corpus BLEU).
    if correct[1] == 0 or any(p <= 0.0 for p in precisions[1:]):
        score = 0.0
    else:
        score = bp * math.exp(sum(math.log(p) for p in precisions[1:]) / max_order) * 100.0

    return {
        "score": score,
        "precisions": precisions[1:],
        "brevity_penalty": bp,
        "hyp_len": hyp_len_total,
        "ref_len": ref_len_total,
        "correct": correct[1:],
        "total": total[1:],
        "implementation": "internal-13a-exp",
    }


def compute_bleu(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
) -> dict:
    """Corpus BLEU with sacrebleu when available, else the internal fallback."""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")
    refs = [[r] if isinstance(r, str) else list(r) for r in references]
    try:
        import sacrebleu  # type: ignore

        reference_streams = [list(stream) for stream in zip(*refs, strict=True)]
        result = sacrebleu.corpus_bleu(list(hypotheses), reference_streams)
        return {
            "score": float(result.score),
            "precisions": [float(p) for p in result.precisions],
            "brevity_penalty": float(result.bp),
            "hyp_len": int(result.sys_len),
            "ref_len": int(result.ref_len),
            "implementation": f"sacrebleu-{getattr(sacrebleu, '__version__', 'unknown')}",
        }
    except Exception:
        return _corpus_bleu_internal(hypotheses, refs)


def corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[str] | Sequence[Sequence[str]],
) -> dict:
    """Internal implementation, exposed for deterministic tests."""
    refs = [[r] if isinstance(r, str) else list(r) for r in references]
    return _corpus_bleu_internal(hypotheses, refs)
