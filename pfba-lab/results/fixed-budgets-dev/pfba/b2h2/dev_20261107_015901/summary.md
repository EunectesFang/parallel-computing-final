# Source-streaming TAF/PFBA

- split: dev; B=2; h=2; RALCP=0.6; min_start=3; seed=13
- LAAL: SimulEval length-adaptive, first full-source-emission cutoff, unclamped.

| predictor | translator | mode | n | BLEU | COMET | LAAL | predict s | translate s | reuse hits |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| small | small | taf | 24 | 18.502866881205662 | 0.7469112041095892 | 2.351597804370748 | 21.737 | 14.074 | 0 |
