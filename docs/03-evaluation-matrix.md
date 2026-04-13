# Accuracy evaluation (simple-evals, two tracks)

**Policy:** all accuracy uses the open-source **[simple-evals](https://github.com/openai/simple-evals)** repository. We do **not** use tore-eval in the documented workflow.

| Track | Submodule | Methods | Results hub | Script |
|-------|-----------|---------|-------------|--------|
| **Primary** | [third_party/sglang-fast-rotation](../third_party/sglang-fast-rotation) | BF16, INT4, BDR | [../eval_primary/README.md](../eval_primary/README.md) | [../scripts/run_primary_eval_matrix.sh](../scripts/run_primary_eval_matrix.sh) |
| **Ablation** | [third_party/sglang-kmeans](../third_party/sglang-kmeans) | k-means + INT4, k-means + BDR | [../eval_accuracy/README.md](../eval_accuracy/README.md) | [../scripts/run_eval_matrix.sh](../scripts/run_eval_matrix.sh) |

Reproduce with the **same simple-evals tasks** and **same sampling hyperparameters** per track; vary only the **SGLang server** (submodule + env).

**Client (both tracks):** clone [simple-evals](https://github.com/openai/simple-evals), `pip install -e .`, then:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:30000/v1"
export OPENAI_API_KEY="dummy"
python -m simple-evals.simple_evals --list-models
python -m simple-evals.simple_evals --model <model_id> --examples 200
```

See upstream [running the evals](https://github.com/openai/simple-evals/blob/main/README.md#running-the-evals).

## Primary evaluation track

**Server:** `cd third_party/sglang-fast-rotation/python`, then `launch_server` with **`--prefill-attention-backend fa3`** (or `fa4` if needed) and **`--decode-attention-backend triton`**. Use **MHA** models; see [01-preparation.md](01-preparation.md#attention-backends-and-model-support-bdr-and-k-means).

Example — BF16 KV:

```bash
cd third_party/sglang-fast-rotation/python
export OPENAI_API_KEY=dummy
export HADAMARD=0
export ROTATE_V=0
python -m sglang.launch_server \
  --prefill-attention-backend fa3 \
  --decode-attention-backend triton \
  --model-path "Qwen/Qwen3-8B" --port 30000 --kv-cache-dtype auto
```

Example — INT4 KV (no rotation):

```bash
export HADAMARD=0
export ROTATE_V=0
python -m sglang.launch_server \
  --prefill-attention-backend fa3 \
  --decode-attention-backend triton \
  --model-path "Qwen/Qwen3-8B" --port 30000 --kv-cache-dtype int4
```

Example — BDR + INT4 (K only):

```bash
export HADAMARD=1
export ROTATE_V=0
export HADAMARD_ORDER=16
python -m sglang.launch_server \
  --prefill-attention-backend fa3 \
  --decode-attention-backend triton \
  --model-path "Qwen/Qwen3-8B" --port 30000 --kv-cache-dtype int4
```

### Primary method matrix

| Method | `HADAMARD` | `ROTATE_V` | `HADAMARD_ORDER` | `--kv-cache-dtype` | `SGLANG_KV_CENTROIDS_PATH` |
|--------|------------|------------|------------------|---------------------|----------------------------|
| BF16 | `0` | `0` | n/a | `auto` | unset |
| INT4 | `0` | `0` | n/a | `int4` | unset |
| BDR (K only) | `1` | `0` | e.g. `16` | `int4` | unset |
| BDR (K+V) | `1` | `1` | e.g. `16` | `int4` | unset |

**Automation:** [../scripts/run_primary_eval_matrix.sh](../scripts/run_primary_eval_matrix.sh) (`bf16`, `int4`, `bdr`, `bdr_kv`).

## Ablation study track

**Server:** `cd third_party/sglang-kmeans/python`. Same **MHA + fa3 + triton** flags. For k-means, set `SGLANG_KV_CENTROIDS_PATH` to centroids from [04-kv-calibration.md](04-kv-calibration.md) and [../tools/fit_kv_centroids.py](../tools/fit_kv_centroids.py).

Example — k-means + INT4 (after centroids exist):

```bash
cd third_party/sglang-kmeans/python
export OPENAI_API_KEY=dummy
export N_CLUSTERS=16
export SGLANG_KV_CENTROIDS_PATH=/path/to/centroids_out
export HADAMARD=0
export ROTATE_V=0
python -m sglang.launch_server \
  --prefill-attention-backend fa3 \
  --decode-attention-backend triton \
  --model-path "Qwen/Qwen3-8B" --port 30000 --kv-cache-dtype int4
```

Example — k-means + BDR:

```bash
export HADAMARD=1
export ROTATE_V=0
export HADAMARD_ORDER=16
export N_CLUSTERS=16
export SGLANG_KV_CENTROIDS_PATH=/path/to/centroids_out
python -m sglang.launch_server \
  --prefill-attention-backend fa3 \
  --decode-attention-backend triton \
  --model-path "Qwen/Qwen3-8B" --port 30000 --kv-cache-dtype int4
```

### Ablation method matrix

| Method | `HADAMARD` | `ROTATE_V` | `HADAMARD_ORDER` | `--kv-cache-dtype` | `SGLANG_KV_CENTROIDS_PATH` | `N_CLUSTERS` |
|--------|------------|------------|------------------|---------------------|----------------------------|--------------|
| K-means + INT4 | `0` | `0` | n/a | `int4` | required | match files |
| K-means + BDR | `1` | `0` or `1` | set | `int4` | required | match files |

**Automation:** `CENTROIDS=/path/to/centroids ./scripts/run_eval_matrix.sh kmeans` or `kmeans_bdr`.

## Full reproduction bundle

Raw JSON logs and frozen environments may be published outside this repo. The main [README.md](../README.md) links to that bundle when available.
