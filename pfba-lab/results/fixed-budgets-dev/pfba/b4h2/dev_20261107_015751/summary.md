# Source-streaming TAF/PFBA

- split: dev; B=4; h=2; RALCP=0.6; min_start=3; seed=13
- LAAL: SimulEval length-adaptive, first full-source-emission cutoff, unclamped.

| predictor | translator | mode | n | BLEU | COMET | LAAL | predict s | translate s | reuse hits |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| small | small | taf | 24 | 18.419611665638538 | 0.7345990786949793 | 2.1092017491448707 | 22.134 | 14.917 | 0 |
