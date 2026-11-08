# Source-streaming TAF/PFBA

- split: dev; B=4; h=6; RALCP=0.6; min_start=3; seed=13
- LAAL: SimulEval length-adaptive, first full-source-emission cutoff, unclamped.

| predictor | translator | mode | n | BLEU | COMET | LAAL | predict s | translate s | reuse hits |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| large | small | taf | 12 | 18.65639926969641 | 0.7573413004477819 | 2.483334937621709 | 62.321 | 11.048 | 0 |
| large | small | pfba | 12 | 18.65639926969641 | 0.7573413004477819 | 2.483334937621709 | 64.436 | 11.016 | 9 |
