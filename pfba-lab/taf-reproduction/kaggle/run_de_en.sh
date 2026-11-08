#!/usr/bin/env bash
# Kaggle runner for one representative of each paper policy.
# Configure via environment variables; no secrets are read or printed.
#
#   MAX_SENTENCES=20 bash kaggle/run_de_en.sh
#   TAU=0.6 AGREE=0.6 MIN_START=3 WAIT_K=3 MAX_SENTENCES=200 bash kaggle/run_de_en.sh
#
# Set EXACT_MT=/path/to/transformer-big-de-en if the author checkpoint is mounted.
# Set LLM_MODEL to another causal LM only as a documented substitution.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA="${DATA:-data/wmt15_newstest2015.tsv}"
OUT_ROOT="${OUT_ROOT:-/kaggle/working/results}"
MAX_SENTENCES="${MAX_SENTENCES:-20}"
LLM_MODEL="${LLM_MODEL:-meta-llama/Llama-2-7b-hf}"
MT_MODEL="${EXACT_MT:-Helsinki-NLP/opus-mt-de-en}"
COMET_MODEL="${COMET_MODEL:-}"
TAU="${TAU:-0.6}"
FRACTION="${FRACTION:-0.5}"
AGREE="${AGREE:-0.6}"
MIN_START="${MIN_START:-3}"
WAIT_K="${WAIT_K:-3}"
N="${N:-10}"
HORIZON="${HORIZON:-10}"
TOP_K="${TOP_K:-10}"
BEAM="${BEAM:-40}"

COMMON=(--data "$DATA" --max-sentences "$MAX_SENTENCES" --quiet)
LLM_ARGS=(--llm-backend hf --llm-model "$LLM_MODEL")
if [[ "${LLM_4BIT:-0}" == "1" ]]; then
  LLM_ARGS+=(--llm-load-in-4bit)
fi
MT_ARGS=(--mt-backend hf --mt-model "$MT_MODEL" --mt-arch seq2seq)
COMET_ARGS=()
if [[ -n "$COMET_MODEL" ]]; then
  COMET_ARGS=(--comet-model "$COMET_MODEL")
fi

mkdir -p "$OUT_ROOT"

run() {
  local name="$1"; shift
  echo "============================================================"
  echo "[run] $name"
  echo "============================================================"
  python scripts/reproduce_de_en.py \
    "${COMMON[@]}" "$@" "${COMET_ARGS[@]}" \
    --output-dir "$OUT_ROOT/$name"
}

# Wait-K-Stride-N: no LLM needed, so use the mock predictor backend.
run "waitk_K${WAIT_K}" \
  --policy waitk --wait-k "$WAIT_K" --wait-stride 1 \
  --llm-backend mock "${MT_ARGS[@]}"

# RALCP: no LLM; beam 40, relaxed fraction.
run "ralcp_f${FRACTION}" \
  --policy ralcp --beam-width "$BEAM" --ralcp-fraction "$FRACTION" \
  --llm-backend mock "${MT_ARGS[@]}"

# TAF (Eq. 7).
run "taf_tau${TAU}" \
  --policy taf --n "$N" --horizon "$HORIZON" --top-k "$TOP_K" --tau "$TAU" \
  "${LLM_ARGS[@]}" "${MT_ARGS[@]}"

# Author RALCP_TAF: n * per-candidate beam = 40; word-prefix voting.
run "ralcp_taf_a${AGREE}_minstart${MIN_START}" \
  --policy ralcp_taf --n "$N" --horizon "$HORIZON" --top-k "$TOP_K" \
  --beam-width "$BEAM" --agree-thres "$AGREE" --min-start "$MIN_START" \
  "${LLM_ARGS[@]}" "${MT_ARGS[@]}"

echo "[run] artifacts under: $OUT_ROOT"
echo "[run] inspect each manifest.json classification before comparing scores"
