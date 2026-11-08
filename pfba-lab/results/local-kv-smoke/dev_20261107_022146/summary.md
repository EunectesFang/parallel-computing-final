# Source-streaming TAF/PFBA

- split: dev; B=4; h=6; RALCP=0.6; min_start=3; seed=13
- LAAL: SimulEval length-adaptive, first full-source-emission cutoff, unclamped.

| predictor | translator | mode | n | BLEU | COMET | LAAL | predict s | translate s | reuse hits |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| small | small | taf | 3 | 13.80358212874601 | skipped | 2.910712520084044 | 23.636 | 7.717 | 0 |
| small | small | pfba | 3 | 13.80358212874601 | skipped | 2.910712520084044 | 24.268 | 7.856 | 1 |
| small | small | pfba-kv | 3 | 13.800069448265413 | skipped | 3.0180555555555557 | 25.534 | 6.783 | 2 |
