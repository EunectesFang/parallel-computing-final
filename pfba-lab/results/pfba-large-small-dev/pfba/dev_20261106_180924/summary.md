# Source-streaming TAF/PFBA

- split: dev; B=4; h=6; RALCP=0.6; min_start=3; seed=13
- LAAL: SimulEval length-adaptive, first full-source-emission cutoff, unclamped.

| predictor | translator | mode | n | BLEU | COMET | LAAL | predict s | translate s | reuse hits |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| large | small | taf | 12 | 18.352058669648727 | 0.755995104710261 | 2.379815369241459 | 336.337 | 16.222 | 0 |
| large | small | pfba | 12 | 18.352058669648727 | 0.755995104710261 | 2.379815369241459 | 337.837 | 16.237 | 9 |
