# Joint READ / (B,h) gate feasibility

- seed: 13; actions: b2h2, b2h6, b4h2, b4h6
- Offline labels compare each action's committed words with the full-true-source Marian continuation; oracle labels are never online features.

| policy | BLEU | COMET | LAAL | virtual completion mean (s) | action histogram |
|---|---:|---:|---:|---:|---|
| fixed_4_6 | 22.647417687342855 | 0.822944367925326 | 2.200812054421321 | 5.523 | b4h6:195, eos:12 |
| fixed_2_6 | 16.48444458220903 | 0.8288588474194208 | 2.202151115591778 | 5.507 | b2h6:195, eos:12 |
| joint | 17.285625053458816 | 0.7917509526014328 | 1.9747422024087555 | 5.533 | b2h2:32, b2h6:70, b4h2:24, b4h6:65, eos:12, read:4 |
| b_only | 19.271993775063294 | 0.8262973874807358 | 2.0888760457791746 | 5.531 | b2h6:89, b4h6:104, eos:12, read:2 |
| h_only | 23.103136331348487 | 0.8223961194356283 | 1.8314098527425733 | 5.525 | b4h2:55, b4h6:115, eos:12, read:25 |
| joint_neural | 17.086841615476114 | 0.791302889585495 | 1.9209740046561183 | 5.524 | b2h2:38, b2h6:72, b4h2:11, b4h6:70, eos:12, read:4 |

## Offline held-out decisions

| model | threshold | compute rate | mean reward | mean matched | mean incorrect |
|---|---:|---:|---:|---:|---:|
| joint | 0.216873 | 0.958 | 0.527 | 1.375 | 0.729 |
| b_only | 0.154023 | 0.958 | 0.565 | 1.375 | 0.667 |
| h_only | 0.0236222 | 0.917 | 0.469 | 1.083 | 0.500 |
| joint_neural | 0.162892 | 0.958 | 0.588 | 1.333 | 0.625 |
