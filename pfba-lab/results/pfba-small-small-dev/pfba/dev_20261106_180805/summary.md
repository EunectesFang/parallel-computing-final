# Source-streaming TAF/PFBA

- split: dev; B=4; h=6; RALCP=0.6; min_start=3; seed=13
- LAAL: SimulEval length-adaptive, first full-source-emission cutoff, unclamped.

| predictor | translator | mode | n | BLEU | COMET | LAAL | predict s | translate s | reuse hits |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| small | small | taf | 12 | 17.894554448071837 | 0.7576559285322825 | 2.5652780618211075 | 29.288 | 13.502 | 0 |
| small | small | pfba | 12 | 17.894554448071837 | 0.7576559285322825 | 2.5652780618211075 | 29.645 | 13.600 | 3 |
