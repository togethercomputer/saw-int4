# Ablation accuracy results (k-means)

Store per-run logs and summaries for **k-means + INT4** and **k-means + BDR** only (server: **sglang-kmeans**).

## Summary table (template)

| Model | Method | Benchmark | Score | Notes |
|-------|--------|-----------|-------|-------|
| — | K-means + INT4 | MMLU | — | `N_CLUSTERS`, centroid dir |
| — | K-means + BDR | MMLU | — | `HADAMARD_ORDER`, centroids |

Add rows for each benchmark reported in the paper for ablations.
