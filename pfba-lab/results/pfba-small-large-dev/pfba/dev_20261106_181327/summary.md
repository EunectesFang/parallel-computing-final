# Source-streaming TAF/PFBA

- split: dev; B=4; h=6; RALCP=0.6; min_start=3; seed=13
- LAAL: SimulEval length-adaptive, first full-source-emission cutoff, unclamped.

| predictor | translator | mode | n | BLEU | COMET | LAAL | predict s | translate s | reuse hits |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| small | large | taf | 12 | 24.840326118396987 | 0.8104457159837087 | 2.4068051510659867 | 46.694 | 819.665 | 0 |
| small | large | pfba | 12 | 24.840326118396987 | 0.8104457159837087 | 2.4068051510659867 | 46.585 | 823.001 | 3 |
