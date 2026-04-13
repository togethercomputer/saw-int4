# Ablation study — accuracy (k-means, k-means + rotation)

This folder holds **ablation** accuracy logs for **k-means + INT4** and **k-means + BDR** on **`third_party/sglang-kmeans`**.

For **primary** BF16 / INT4 / BDR accuracy (fast-rotation only), use [../eval_primary/README.md](../eval_primary/README.md) and [../scripts/run_primary_eval_matrix.sh](../scripts/run_primary_eval_matrix.sh).

**Canonical doc:** [../docs/03-evaluation-matrix.md](../docs/03-evaluation-matrix.md#ablation-study-track)  
**KV dump and centroids:** [../docs/04-kv-calibration.md](../docs/04-kv-calibration.md)  
**Helper script:** [../scripts/run_eval_matrix.sh](../scripts/run_eval_matrix.sh) (`kmeans`, `kmeans_bdr` only; requires `CENTROIDS=`)  
**Fit centroids:** [../tools/fit_kv_centroids.py](../tools/fit_kv_centroids.py)

## Server

Build **[third_party/sglang-kmeans](../third_party/sglang-kmeans)**. Use **MHA** + **`--prefill-attention-backend fa3`** + **`--decode-attention-backend triton`** unless your stack requires a different Flash Attention option; see [../docs/01-preparation.md](../docs/01-preparation.md#attention-backends-and-model-support-bdr-and-k-means).

## Client

Install **[simple-evals](https://github.com/openai/simple-evals)** and point `OPENAI_BASE_URL` at the k-means server `/v1` endpoint (same as primary track).

From the **repository root**:

```bash
CENTROIDS=/path/to/centroids_out ./scripts/run_eval_matrix.sh kmeans
CENTROIDS=/path/to/centroids_out ./scripts/run_eval_matrix.sh kmeans_bdr
```

## Results

Store logs under **[results/](results/)**; mirror summary rows in the main [README.md](../README.md) ablation table.
