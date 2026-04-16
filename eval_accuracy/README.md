# Ablation study — accuracy (k-means, k-means + rotation)

This folder holds **ablation** accuracy logs for **k-means + INT4** and **k-means + BDR** on **`third_party/sglang-kmeans`**.

For **primary** BF16 / INT4 / BDR accuracy (fast-rotation only), use [../eval_primary/README.md](../eval_primary/README.md) and [../scripts/run_primary_eval_matrix.sh](../scripts/run_primary_eval_matrix.sh).

**Canonical instructions:** [../README.md](../README.md#ablation-study-k-means-k-means--rotation)  
**KV dump and centroids:** [../README.md#kv-calibration-ablation-only](../README.md#kv-calibration-ablation-only)  
**Helper script:** [../scripts/run_eval_matrix.sh](../scripts/run_eval_matrix.sh) (`kmeans`, `kmeans_bdr` only; requires `CENTROIDS=`)  
**Fit centroids:** [../tools/fit_kv_centroids.py](../tools/fit_kv_centroids.py)

## Server

Build **[third_party/sglang-kmeans](../third_party/sglang-kmeans)**. Use **MHA** + **`--prefill-attention-backend fa3`** + **`--decode-attention-backend triton`** unless your stack requires a different Flash Attention option; see [../README.md#server-requirements](../README.md#server-requirements).

## Client

**simple-evals** is available at **`third_party/simple-evals`** — run `git submodule update --init --checkout third_party/simple-evals` from the repo root if not yet initialized, then install per the main README ([Prepare](../README.md#prepare)); run GPQA against the k-means server the same way as primary accuracy ([Accuracy (primary)](../README.md#accuracy-primary)). Set `MODEL_PATH` to match the server you calibrate and evaluate.

From the **repository root**:

```bash
CENTROIDS=/path/to/centroids_out ./scripts/run_eval_matrix.sh kmeans
CENTROIDS=/path/to/centroids_out ./scripts/run_eval_matrix.sh kmeans_bdr
```

## Results

Store logs under **[results/](results/)**; mirror summary rows in the main [README.md](../README.md) ablation table.
