# KV calibration (dump → k-means centroids)

**Ablation track only** (k-means server). Primary BF16 / INT4 / BDR does not require this step.

The **sglang-kmeans** fork can dump floating-point KV activations once enough tokens are present in the cache, then you **fit k-means centroids offline** and reload them for **centroid-subtract INT4** inference.

## 1. Dump KV activations

Run SGLang from **sglang-kmeans** with a **BF16 (or fp16) KV cache** so dumps are in calibration space, not already quantized.

```bash
cd third_party/sglang-kmeans/python

export DUMP_KVCACHE=true
export DUMP_KVCACHE_TOKENS=512      # minimum tokens before a layer dumps once
export DUMP_KVCACHE_DIR=/path/to/kv_dumps

python -m sglang.launch_server \
  --prefill-attention-backend fa3 \
  --decode-attention-backend triton \
  --model-path "Qwen/Qwen3-8B" \
  --port 30000 \
  --kv-cache-dtype auto
```

Use **MHA** models; see [01-preparation.md](01-preparation.md#attention-backends-and-model-support-bdr-and-k-means).

Drive enough prefill/decode traffic so each layer hits the threshold at least once. Files appear as:

`kv_calibration_layer_<layer_id>.pt`

Each file is a dict with `k`, `v`, and `indices` tensors (CPU); see `triton_backend.py` in the submodule for the exact selection logic.

## 2. Fit centroids offline

From the **paper repo root**:

```bash
python tools/fit_kv_centroids.py \
  --dump-dir /path/to/kv_dumps \
  --out-dir /path/to/centroids_out \
  --n-clusters 16 \
  --seed 0
```

This writes, for each global layer index `L`:

- `k_layer_L_clusters_<N>_centers.pt`  
- `v_layer_L_clusters_<N>_centers.pt`  

with tensors shaped `(N, num_kv_heads_global * head_dim)`, matching `MHATokenToKVPool._load_kv_centroids` in the submodule.

## 3. Run INT4 + k-means inference

```bash
export N_CLUSTERS=16   # must match filename suffix
export SGLANG_KV_CENTROIDS_PATH=/path/to/centroids_out

python -m sglang.launch_server \
  --prefill-attention-backend fa3 \
  --decode-attention-backend triton \
  --model-path "Qwen/Qwen3-8B" \
  --port 30000 \
  --kv-cache-dtype int4
```

## BDR + k-means

Set the same `SGLANG_KV_CENTROIDS_PATH` and additionally `HADAMARD=1` (and optionally `ROTATE_V=1`) with `HADAMARD_ORDER` consistent with your head dimension, as in [03-evaluation-matrix.md](03-evaluation-matrix.md).
