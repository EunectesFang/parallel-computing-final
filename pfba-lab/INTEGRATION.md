# Integration notes for Claude Code workers

The parent checked out the author release at ../taf-upstream (commit 99e373a887e0556c4d50e9ea08497889037876f8). Read that local copy. Its agents/ralcp_taf.py sends all branches to a batched translator; serial execution is only a diagnostic. Main comparison must be matched batched TAF versus PFBA reuse and scheduling. SGLang is used by the author predictor and the Tower greedy translator. A bare Hugging Face no-prefix-cache path must be identified as such. Source files are in ../taf-upstream/data/de_en. The published paper is pp. 5547–5557. For LAAL verify the official SimulEval scorer, including max(reference length, prediction length) and the cutoff at source completion.

RALCP gate: at each target depth count identical candidate prefixes and divide by the original number of candidates (including candidates that ended). A matched token count divided by a single candidate length is a different statistic. Add a test where the winning prefix changes between depths and preserve author tie handling when running the fidelity baseline.

LAAL reference: https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py . Use the first full-source target emission as the cutoff, or all emissions if it never occurs. Do not clamp per-token terms to zero.
