# TAF De-En Reproduction Audit

**Paper:** Siqi Ouyang, Oleksii Hrinchuk, Zhehuai Chen, Vitaly Lavrukhin, Jagadeesh Balam, Lei Li, Boris Ginsburg. *Anticipating Future with Large Language Model for Simultaneous Machine Translation.* NAACL 2025 (long), **pp. 5547–5557**. ACL Anthology `2025.naacl-long.286`. Local source: `../../2410.22499v2.pdf` (arXiv v2, 31 May 2025).

**Scope of this subproject:** De-En only. The parent PFBA project consumes the streaming traces produced here.

**Audit date:** 2026-11-07.

**Sources actually read:**
- the local arXiv v2 PDF (`../../2410.22499v2.pdf`) including Appendix A/B;
- the **authors' checked-out release** at `../taf-upstream`, commit `99e373a887e0556c4d50e9ea08497889037876f8`: `agents/ralcp_taf.py`, `agents/ralcp.py`, `agents/base.py`, `agents/utils.py`, `README.md`;
- the authors' De-En data at `../taf-upstream/data/de_en/` (`dev.de`, `dev.en`, 3003 lines each; `test.de`, `test.en`, 2169 lines each).

The official SimulEval source was not fetched in this sandbox (WebFetch permission was denied). The LAAL implementation follows the SimulEval scorer semantics recorded in `../INTEGRATION.md`: normalization by `max(reference length, prediction length)`, cutoff at the first full-source emission batch (trace-based: a later EOS write at the same delay is excluded), and no per-token clamping.

**Bottom line:** this folder is an independent reimplementation whose streaming protocol now follows the authors' `RALCP_TAF` agent file rather than a paper-only interpretation. It is **not** a release of the authors' code or checkpoints, and it does **not** claim reproduced paper scores. The authors' code was inspected locally and is not vendored. Every run writes a manifest recording which artifacts were exact and which were substitutions.

---

## 1. Artifact availability (exact vs. substitution)

| Artifact | Paper specification | Local status | Classification |
|---|---|---|---|
| Author code | `github.com/owaski/TAF` (paper footnote 1) | **Present at `../taf-upstream` (commit 99e373a).** Files above were read; source-prefix continuation, word-prefix RALCP, and EOS behavior are ported. Not copied into this folder. | **Author code inspected.** This implementation is independent but protocol-faithful. |
| Offline MT (De-En) | Transformer-Big; Appendix A says "12 × 6 layers", `d_model=1024`, `d_inner=4096`, 16 heads, joint BPE 16384, Adam + inverse-sqrt, 7.5K warmup, LR 1e-3, ≤100K steps, dropout 0.1, label smoothing 0.2. Trained on WMT15 De-En 4.5M pairs. | Authors' checkpoint not present in the checkout. | **Substitution required.** Public De-En MT passed via `--mt-model` (e.g. `Helsinki-NLP/opus-mt-de-en`) is a different architecture/data recipe; BLEU/COMET are not paper-comparable. |
| Future predictor LLM | LLaMA-2-7B **base** (paper §4.2) | Public, gated on Hugging Face. Not downloaded here. | **Exact if access is granted:** `meta-llama/Llama-2-7b-hf`. Any other causal LM is a substitution. |
| Test set | WMT15 De-En `newstest2015` | Authors' `data/de_en/test.de` / `test.en` (2169 lines) present in the checkout. | **Exact if these are the paper files.** |
| Validation set | `newstest2013` | Authors' `data/de_en/dev.de` / `dev.en` (3003 lines) present in the checkout. | **Exact if these are the paper files.** |
| COMET | Paper §4.3, footnote 2: `Unbabel/XCOMET-XXL`. | Package/model not installed locally. | **Optional exact.** `Unbabel/wmt22-comet-da` is a substitution. COMET is emitted only if the model actually loads; otherwise `null` plus a reason. |
| BLEU | SimulEval BLEU (sacrebleu-style, 13a) | `sacrebleu` is used automatically when importable; otherwise a self-contained 13a/exp corpus BLEU. | **Reimplementation / sacrebleu when available.** |
| LAAL | SimulEval length-adaptive average lagging, word unit for De/En | Implemented in `src/taf/laal.py` with reference length and source-completion cutoff. | **Reimplementation of the official semantics**; SimulEval is not vendored. |
| Timing | Average wall-clock per step (Table 1); A6000 + EPYC 9354 | Instrumentation implemented; no benchmark run performed here. | **Method reproduced, numbers not claimed.** |

## 2. Paper settings extracted (De-En focus)

| Parameter | Value | Source |
|---|---|---|
| LLM | LLaMA-2-7B base | §4.2 |
| Number of continuations `n` | 10 (default); authors' `--prediction-num-continuations 10` | §4.2; author README |
| Continuation length `l` | 10 tokens (default; sweeps 2,4,6,8,10) | §4.2, §5.4 |
| Top-k sampling | `k=10` (author README example: `-1` with `top_p=0.9`) | §4.2; author README |
| RALCP agreement threshold | authors' `--agree-thres`, code default 0.5; author README example 0.6 | `agents/ralcp.py`; author README |
| `--min-start` | code default 1; author README example 3 | `agents/ralcp.py`; author README |
| RALCP beam width | 40 for De-En | §4.2 |
| TAF+RALCP constraint | `n × per-candidate beam width = RALCP beam width` (40); authors' base `--beam-size 4`, `n=10` | §4.2; `agents/base.py` |
| Wait-K-Stride-N | K ∈ {3,6,9,12,15}, N=1 for De-En | §4.2 |
| MT training | 100K steps max, Adam, inverse-sqrt, 7.5K warmup, LR 1e-3, dropout 0.1, label smoothing 0.2 | Appendix A |
| Evaluation units | word for En/De; character for Zh/Ja | §4.3 |
| Reported latency metric | LAAL | §4.3 |
| Reported quality metrics | BLEU, COMET | §4.3 |

## 3. Implemented policy semantics

### 3.1 Author `RALCP_TAF` (default: `--policy ralcp_taf`)

Ported from `agents/ralcp_taf.py`:

1. At each read position (including the fully read source but before EOS), sample `n` **source-prefix continuations** from the LLM, and append each to the partial source.
2. Translate every predicted source with the **committed target prefix forced** and a per-call beam width. The candidate pool is `n × beam` (40 with the defaults).
3. Vote on **decoded word prefixes** exactly as `agents/utils.py:ralcp`:
   - at each target-word depth count identical candidate prefixes;
   - the denominator is always the number of candidates, including candidates that ended early (they keep voting for their full sequence at deeper depths);
   - accept the most common prefix while `count >= len(candidates) * agree_thres`;
   - ties go to the prefix first seen (Python `Counter` insertion order, matching `max(cnt, key=cnt.get)`);
   - the winner at a depth need not extend the winner at the previous depth.
4. There is **no exact-sequence majority or `τ` prerequisite**; the emitted content is the relaxed word prefix. Write it if non-empty, otherwise read the next source unit.
5. On EOS (`states.source_finished`), perform a **fresh full-source translation** with the committed prefix (single returned sequence) and finish. Cached candidates are not reused.
6. Committed target tokens are never rewritten or deleted. `--min-start` suppresses early writes.

The `ralcp+taf` policy name is a backward-compatible alias. The separate `taf` policy remains the Eq. 4–7 exact-majority interpretation and is not the author protocol.

### 3.2 RALCP baseline (no LLM)

Same word-prefix voting and fresh full-source EOS as above, but candidates come from a single beam decode of the partial source with `--beam-width` beams (`agents/ralcp.py`).

### 3.3 Wait-K-Stride-N baseline

Read until `K` source units, then alternate writing one target token (greedy, prefix-forced) and reading `N` source units. Final flush at source end. Word units.

## 4. Evaluation definitions

- **BLEU:** corpus BLEU, 13a tokenization, exp smoothing, up to 4-grams. `sacrebleu` is used if importable; the internal fallback is labelled in the manifest. A 3-word identical corpus necessarily scores 0 because there are no 4-grams; tests asserting 100 use at least four identical words.
- **COMET:** optional; XCOMET-XXL when available. Otherwise `score = null` with an explicit status. No score is synthesized.
- **LAAL:** let `S` be source length, `R` reference length, `P` the number of target tokens up to the first full-source emission (all emissions if the source never completes), and `T = max(R, P)`:
  
  `LAAL = (1/T) Σ_{i=0}^{P-1} (g_i − i·(S−1)/(T−1))`
  
  where `g_i` is the number of source units read when target token `i` was emitted. Per-token terms are **not** clamped. The cutoff is the first write made once the source is fully read; that whole write is kept and all later writes (including the fresh EOS decode) are dropped. If no write occurs at full source, every emission is kept. Delays are word-level for De/En.
- **Timing:** each streaming step records `llm_sample`, `mt_decode`, `aggregate`, and `total` in milliseconds, with optional CUDA synchronization.

## 5. What is verified vs. not verified

Verified locally by CPU invariant tests (no network, no model weights):
- author RALCP word-prefix voting, including a winner that changes between depths, first-seen tie handling, ended-candidate denominators, and voting without an exact majority;
- source-prefix continuations, forced committed prefix, `n × beam = 40` candidate budget, and fresh full-source EOS;
- `--min-start` gating;
- RALCP baseline and Wait-K behavior;
- SimulEval LAAL arithmetic: reference length, cutoff, and no clamp, including a case where a full-source anticipation write is kept but a later EOS write at the same delay is excluded;
- BLEU, including the 3-word four-gram zero case;
- end-to-end mock pipeline and artifact/manifest writing;
- the authors' `dev.de` / `dev.en` pair loading through `--ref-data` and a mock smoke run;
- COMET unavailability reported as `null`.

Not verified here:
- numerical equivalence to the authors' checkpoints (not present);
- paper BLEU/COMET/LAAL values (no exact MT checkpoint, no GPU run);
- LLaMA-2 behavior (not downloaded);
- wall-clock overhead on A6000/EPYC (hardware not present);
- the exact SimulEval source file (not fetched; implemented from the recorded scorer semantics).

## 6. Reproduction commands

Local invariants (CPU only, no downloads):

```bash
D:/MINICONDA/envs/py312/python.exe -m pytest -q
```

Mock smoke on the authors' development pair:

```bash
D:/MINICONDA/envs/py312/python.exe scripts/reproduce_de_en.py \
  --data ../taf-upstream/data/de_en/dev.de \
  --ref-data ../taf-upstream/data/de_en/dev.en \
  --llm-backend mock --mt-backend mock --policy ralcp_taf \
  --n 4 --beam-width 4 --max-sentences 5 \
  --output-dir results/author_dev_mock
```

Kaggle / GPU run: see `kaggle/README_KAGGLE.md`. No upload or external publication is performed by any script in this folder.

## 7. Reproducibility manifest

Every run writes `results/<run>/manifest.json` containing: paper reference and final page range, the local author checkout path/commit/files, policy config, model IDs, seed, command line, package availability, and an explicit `classification` block. This is the machine-readable version of the table above.

## 8. Local verification status (2026-11-07)

- The suite has **61 tests**; the latest commanded run was:
  `D:/MINICONDA/envs/py312/python.exe -m pytest -q` -> **61 passed**.
- The end-to-end mock BLEU fixture uses the identical 5-word sentence `a b c d e` / `A B C D E`. The 3-word four-gram limitation is now covered explicitly by `test_identical_three_word_corpus_is_zero_4gram_bleu`.
- Tests that touch author data read `../taf-upstream/data/de_en/dev.de` and `dev.en`; they skip if the checkout is absent.
- No GPU, model download, or network scoring run was performed, so no BLEU/COMET/LAAL *paper* numbers are reported or implied.
