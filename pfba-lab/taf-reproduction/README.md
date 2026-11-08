# TAF De-En Reproduction (NAACL 2025)

Independent reimplementation of **Translation by Anticipating Future (TAF)** for German→English, with the authors' RALCP word-prefix protocol, forced committed prefixes, fresh full-source EOS, BLEU/COMET/SimulEval-LAAL, and wall-clock instrumentation. Built for the parent PFBA project.

**Read first:** [`docs/REPRODUCTION_AUDIT.md`](docs/REPRODUCTION_AUDIT.md) — artifact availability, exact-vs-substitution classification, paper settings, and what is *not* verified.

> This folder does **not** contain the authors' code or checkpoints and does **not** claim reproduced paper scores. The authors' release is checked out locally at `../taf-upstream` (commit `99e373a`) and was inspected; the streaming protocol is ported from `agents/ralcp_taf.py` and `agents/utils.py:ralcp`. Every run writes a manifest recording which artifacts were exact and which were substitutions.

## What is implemented

| Paper element | Location |
|---|---|
| **Author RALCP_TAF (default):** source-prefix LLM continuations, `n × beam` MT candidates, word-prefix voting with `agree_thres`, no exact-majority gate | `src/taf/policy.py` (`AuthorRALCPTAFPolicy`), `src/taf/ralcp.py` (`author_ralcp`) |
| Forced committed target prefix on every MT decode (no rewrite/delete) | `src/taf/policy.py`, `src/taf/hf_backends.py` (`decoder_input_ids`) |
| Fresh full-source EOS decode | `src/taf/policy.py` |
| Eq. 4–7 exact-majority interpretation (`--policy taf`) | `src/taf/policy.py`, `src/taf/majority.py` |
| RALCP baseline with author word-prefix voting | `src/taf/policy.py`, `src/taf/ralcp.py` |
| RALCP+TAF with `n × per-candidate beam = 40` budget | `src/taf/config.py`, `src/taf/policy.py` |
| Wait-K-Stride-N baseline | `src/taf/policy.py` |
| SimulEval LAAL (reference length, source-completion cutoff, no clamp) | `src/taf/laal.py` |
| BLEU (sacrebleu when available; internal 13a/exp fallback) | `src/taf/bleu.py` |
| COMET (XCOMET-XXL optional; `null` when unavailable, never invented) | `src/taf/comet.py` |
| Per-step clock: LLM / MT / aggregate / total, CUDA sync | `src/taf/instrumentation.py` |
| Dataset runner, traces, manifest, artifacts | `src/taf/runner.py`, `scripts/reproduce_de_en.py` |

## Quickstart (CPU, no downloads)

Local Python is restricted to `D:/MINICONDA/envs/py312/python.exe`.

```bash
D:/MINICONDA/envs/py312/python.exe -m pytest -q

# Mock smoke on the authors' development pair (no downloads).
D:/MINICONDA/envs/py312/python.exe scripts/reproduce_de_en.py \
  --data ../taf-upstream/data/de_en/dev.de \
  --ref-data ../taf-upstream/data/de_en/dev.en \
  --llm-backend mock --mt-backend mock \
  --policy ralcp_taf --n 4 --beam-width 4 --max-sentences 5 \
  --output-dir results/author_dev_mock
```

Outputs: `sources.txt`, `references.txt`, `hypotheses.txt`, `traces.jsonl`, `per_sentence.jsonl`, `summary.json`, `manifest.json`.

## GPU / Kaggle run

See [`kaggle/README_KAGGLE.md`](kaggle/README_KAGGLE.md) and `kaggle/run_de_en.sh`.

```bash
python scripts/reproduce_de_en.py \
  --data ../taf-upstream/data/de_en/test.de \
  --ref-data ../taf-upstream/data/de_en/test.en \
  --llm-backend hf --llm-model meta-llama/Llama-2-7b-hf \
  --mt-backend hf --mt-model Helsinki-NLP/opus-mt-de-en \
  --policy ralcp_taf --n 10 --horizon 10 --top-k 10 \
  --beam-width 40 --agree-thres 0.6 --min-start 3 \
  --comet-model Unbabel/XCOMET-XXL \
  --output-dir results/wmt15_ralcp_taf_opus
```

`Helsinki-NLP/opus-mt-de-en` is a **substitution** for the unavailable author Transformer-Big. Scores from it are not paper-comparable; the manifest says so.

## Layout

```
docs/REPRODUCTION_AUDIT.md   audit, exact/substitution table, formulas
src/taf/                     policy, RALCP, majority, metrics, HF/mock backends
tests/                       CPU invariant tests (written first)
scripts/reproduce_de_en.py   main CLI
kaggle/                      Kaggle setup/run scripts + instructions
data/sample_de_en.tsv        toy smoke data
results/                     run artifacts (generated)
```

## Safety / scope

- No uploads, no external publication, no access to secrets.
- Only this folder is modified.
- All local Python execution uses `D:/MINICONDA/envs/py312/python.exe`.
- COMET/BLEU/LAAL values are reported only when actually computed; unavailable artifacts produce `null` plus a reason.
