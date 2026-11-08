# Source-streaming TAF/PFBA

- split: dev; B=4; h=6; RALCP=0.6; min_start=3; seed=13
- LAAL: SimulEval length-adaptive, first full-source-emission cutoff, unclamped.

| predictor | translator | mode | n | BLEU | COMET | LAAL | predict s | translate s | reuse hits |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| small | large | taf | 12 | 26.41925708538648 | 0.7902980893850327 | 2.1962565199491433 | 28.514 | 151.475 | 0 |
| small | large | pfba | 12 | 26.41925708538648 | 0.7902980893850327 | 2.1962565199491433 | 28.314 | 153.390 | 2 |
