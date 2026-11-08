#!/usr/bin/env bash
# Kaggle setup: dependencies + WMT15 De-En newstest2015 TSV.
# Does not read or print secrets. Does not upload anything.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[setup] installing Python deps"
pip install -q sacrebleu
if [[ "${INSTALL_COMET:-0}" == "1" ]]; then
  pip install -q unbabel-comet
fi
if [[ "${INSTALL_BNB:-0}" == "1" ]]; then
  pip install -q bitsandbytes
fi

echo "[setup] downloading WMT15 De-En (newstest2015)"
mkdir -p data
sacrebleu -t wmt15 -l de-en --echo src > data/wmt15.de
sacrebleu -t wmt15 -l de-en --echo ref > data/wmt15.en

python - <<'PY'
import csv
from pathlib import Path
root = Path("data")
src = (root / "wmt15.de").read_text(encoding="utf-8").splitlines()
ref = (root / "wmt15.en").read_text(encoding="utf-8").splitlines()
assert len(src) == len(ref), (len(src), len(ref))
with (root / "wmt15_newstest2015.tsv").open("w", encoding="utf-8", newline="") as f:
    csv.writer(f, delimiter="\t").writerows(zip(src, ref))
print(f"[setup] wrote {len(src)} pairs to data/wmt15_newstest2015.tsv")
PY

sacrebleu -t wmt15 -l de-en --echo ref > data/wmt15_ref.txt
echo "[setup] done"
