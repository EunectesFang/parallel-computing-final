# Joint READ / (B,h) gate feasibility

- seed: 13; actions: b2h2, b2h6, b4h2, b4h6
- Offline labels compare each action's committed words with the full-true-source Marian continuation; oracle labels are never online features.

| policy | BLEU | COMET | LAAL | virtual completion mean (s) | action histogram |
|---|---:|---:|---:|---:|---|
| fixed_4_6 | 22.647417687342855 | 0.822944367925326 | 2.200812054421321 | 5.553 | b4h6:195, eos:12 |
| fixed_2_6 | 16.48444458220903 | 0.8288588474194208 | 2.202151115591778 | 5.519 | b2h6:195, eos:12 |
| joint | 20.97478370151935 | 0.7951550036668777 | 1.9985431443674069 | 5.539 | b2h2:59, b2h6:54, b4h2:14, b4h6:62, eos:12, read:6 |
| b_only | 22.8816935193043 | 0.8437170932690302 | 3.176602447768047 | 5.622 | b2h6:61, b4h6:88, eos:12, read:46 |
| h_only | 18.121187413792562 | 0.8082335342963537 | 1.7159535844049747 | 5.517 | b4h2:88, b4h6:100, eos:12, read:7 |
| joint_neural | 22.54301680083444 | 0.8427513142426809 | 3.7437287619166493 | 5.551 | b2h2:17, b2h6:34, b4h2:1, b4h6:8, eos:12, read:135 |

## Offline held-out decisions

| model | threshold | compute rate | mean reward | mean matched | mean incorrect |
|---|---:|---:|---:|---:|---:|
| joint | -0.806603 | 0.979 | 0.825 | 1.458 | 0.521 |
| b_only | -0.267174 | 0.792 | 0.626 | 0.938 | 0.188 |
| h_only | -1.10751 | 0.938 | 0.562 | 1.104 | 0.438 |
| joint_neural | 0.258588 | 0.479 | 0.426 | 0.604 | 0.125 |
