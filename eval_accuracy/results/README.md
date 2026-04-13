# Accuracy results (simple-evals)

Store per-run logs (stdout, simple-evals JSON if emitted) and a small summary file per model + method.

## Summary table (template)

| Model | Method | Benchmark | Score | Notes |
|-------|--------|-----------|-------|-------|
| — | BF16 | MMLU | — | simple-evals task id |
| — | INT4 | MMLU | — | |
| — | BDR (K-only) | MMLU | — | HADAMARD_ORDER=… |
| — | K-means + INT4 | MMLU | — | N_CLUSTERS, centroid dir |
| — | K-means + BDR | MMLU | — | |

Add rows for each benchmark reported in the paper (GPQA, HumanEval, MATH-500, etc.).
