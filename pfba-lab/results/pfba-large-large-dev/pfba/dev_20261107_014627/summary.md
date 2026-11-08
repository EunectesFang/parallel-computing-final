# Source-streaming TAF/PFBA

- split: dev; B=4; h=6; RALCP=0.6; min_start=3; seed=13
- LAAL: SimulEval length-adaptive, first full-source-emission cutoff, unclamped.

| predictor | translator | mode | n | BLEU | COMET | LAAL | predict s | translate s | reuse hits |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| large | large | taf | 12 | 23.21826723467505 | 0.807565708955129 | 2.596716445390981 | 63.505 | 161.756 | 0 |
| large | large | pfba | 12 | 23.21826723467505 | 0.807565708955129 | 2.596716445390981 | 63.307 | 162.790 | 9 |
