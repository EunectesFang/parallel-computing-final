# Source-streaming TAF/PFBA

- split: dev; B=4; h=6; RALCP=0.6; min_start=3; seed=13
- LAAL: SimulEval length-adaptive, first full-source-emission cutoff, unclamped.

| predictor | translator | mode | n | BLEU | COMET | LAAL | predict s | translate s | reuse hits |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| small | small | taf | 24 | 19.06630445838203 | 0.7631521212557951 | 2.4765735324752733 | 63.103 | 24.844 | 0 |
