# Source-streaming TAF/PFBA

- split: dev; B=4; h=6; RALCP=0.6; min_start=3; seed=13
- LAAL: SimulEval length-adaptive, first full-source-emission cutoff, unclamped.

| predictor | translator | mode | n | BLEU | COMET | LAAL | predict s | translate s | reuse hits |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| small | small | taf | 12 | 19.080095656924325 | 0.7570668210585912 | 2.8473999359651656 | 25.809 | 9.880 | 0 |
| small | small | pfba | 12 | 19.080095656924325 | 0.7570668210585912 | 2.8473999359651656 | 25.895 | 9.965 | 2 |
| small | small | pfba-kv | 12 | 18.828451527762237 | 0.7517697612444559 | 2.756305164723335 | 30.716 | 9.875 | 2 |
