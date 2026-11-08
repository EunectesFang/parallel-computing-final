# Kaggle Runbook — TAF De-En

This runbook produces auditable artifacts on Kaggle. It does **not** upload or publish anything, and it never reads or prints secrets. Scripts are in this folder; outputs stay under `/kaggle/working/results`.

## 1. Hardware and internet

- Recommended accelerator: **GPU T4 ×2** or **P100** (16 GB each). The paper used an A6000; wall-clock numbers here are not comparable to Table 1.
- Enable **Internet** for pip, Hugging Face, and sacrebleu downloads, or pre-mount datasets/models as Kaggle inputs.

## 2. Data: WMT15 De-En (exact paper test set)

The paper's De-En test set is `newstest2015` from WMT15.

```bash
bash kaggle/setup_kaggle.sh
```

This installs `sacrebleu` and writes `data/wmt15_newstest2015.tsv` (`source<TAB>reference`). If Internet is off, upload `newstest2015.de`/`.en` as a Kaggle Dataset and build the TSV with a small Python cell:

```python
import csv
src = open('/kaggle/input/wmt15/newstest2015.de', encoding='utf-8').read().splitlines()
ref = open('/kaggle/input/wmt15/newstest2015.en', encoding='utf-8').read().splitlines()
with open('/kaggle/working/wmt15_newstest2015.tsv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(src, ref))
```

## 3. Models

### 3.1 LLM (paper-exact if available)

Paper: **LLaMA-2-7B base** → `meta-llama/Llama-2-7b-hf`.

- This model is **gated**. The Kaggle account owner must accept the license on Hugging Face.
- Set a Hugging Face token through **Kaggle Secrets** / Add-ons → Secrets as `HF_TOKEN`. Do **not** hard-code or print it. The scripts here never read it directly; `transformers`/`huggingface_hub` pick it up from the environment.
- If the 7B model does not fit on one GPU, either:
  - run with `--llm-load-in-4bit` (requires `bitsandbytes`), or
  - leave `--device auto` so `device_map="auto"` shards across both T4s.
- 4-bit is a **substitution** for the paper's fp16 7B run; record it.

Any other causal LM (e.g. Mistral-7B, MicroLlama-300M) is a **substitution**; the manifest labels it.

### 3.2 MT (paper-exact checkpoint is unavailable)

Paper: authors' Transformer-Big (12 encoder / 6 decoder layers, `d_model=1024`, 16384 BPE) trained on WMT15. It was not found or released in this reproduction workspace.

Default substitution:

```bash
--mt-model Helsinki-NLP/opus-mt-de-en --mt-arch seq2seq
```

If you obtain the author checkpoint, mount it and pass:

```bash
--mt-model /kaggle/input/taf-checkpoints/transformer-big-de-en --mt-arch seq2seq
```

The manifest classifies MT as paper-exact only when the model string contains `transformer-big`; otherwise it is a substitution. Do not compare substitution BLEU/COMET to the paper table.

## 4. COMET

Paper footnote 2 points to `Unbabel/XCOMET-XXL`.

```bash
pip install -q unbabel-comet
--comet-model Unbabel/XCOMET-XXL
```

If the model/package cannot load, the run records `"score": null` and the reason. `Unbabel/wmt22-comet-da` is a substitution.

## 5. Smoke test, then scale

Always start small:

```bash
MAX_SENTENCES=20 bash kaggle/run_de_en.sh
```

Then 200, then the full test set. TAF makes `n=10` MT calls per read step; full-set runs are expensive. The paper's main settings are `n=10`, `l=10`, `top-k=10`, RALCP beam 40; the authors' README example uses `agree_thres=0.6` and `min_start=3` (code defaults 0.5 and 1).

## 6. Paper policy grid

```bash
# Wait-K-Stride-N: K in {3,6,9,12,15}, N=1
# RALCP: fraction in {0.1,0.2,...,1.0}, beam 40
# TAF (Eq. 7 majority): tau in {0.5,0.6,...,1.0}
# Author RALCP_TAF: agree threshold sweep, min-start sweep, n*beam=40
```

`kaggle/run_de_en.sh` runs one representative of each policy. Loop over values by setting `TAU`, `FRACTION`, `AGREE`, `MIN_START`, or `WAIT_K` and re-running; each invocation writes a separate `results/<timestamp>_...` directory with `manifest.json`.

## 7. Artifacts

Each run writes:

```
hypotheses.txt          detokenized MT output
sources.txt
references.txt
traces.jsonl            one record per read step + flush
per_sentence.jsonl      tokens, word delays, LAAL, timing
summary.json            corpus BLEU, COMET (or null), LAAL, per-step clock
manifest.json           exact/substitution classification, packages, argv
```

`summary.json` is the only place scores appear. It never contains a fabricated COMET value.

## 8. Known limitations

- Author code is checked out at `../taf-upstream` and inspected, but the author MT checkpoint is unavailable; no bit-level equivalence is claimed.
- MT substitution dominates quality; use the same MT for all policies when making paired comparisons.
- T4 wall-clock is not comparable to the paper's A6000/EPYC numbers.
- BLEU is `sacrebleu` when installed; LAAL follows SimulEval semantics but SimulEval itself was not vendored.
- LLaMA-2 7B base, not chat; the paper uses the base model.
