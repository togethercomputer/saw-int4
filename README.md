# System-Aware 4-Bit KV Cache Quantization

Official companion code for the paper **System-Aware 4-Bit KV Cache Quantization** (Together). *Venue / arXiv / DOI: add at publication time.*

## Contents

- [Introduction](#introduction)
- [How to run BDR](#how-to-run-bdr)
  - [Get the code](#get-the-code)
  - [Server requirements](#server-requirements)
  - [Install BDR (sglang-fast-rotation)](#install-bdr-sglang-fast-rotation)
  - [BDR environment variables](#bdr-environment-variables)
- [Primary accuracy and throughput](#primary-accuracy-and-throughput)
  - [Accuracy (primary)](#accuracy-primary)
    - [Prepare](#prepare)
  - [Throughput and latency (primary)](#throughput-and-latency-primary)
    - [Prepare (genai-bench)](#prepare-genai-bench)
- [Ablation study (k-means, k-means + rotation)](#ablation-study-k-means-k-means--rotation)
  - [Install sglang-kmeans](#install-sglang-kmeans)
- [Repository layout](#repository-layout)
- [Full reproduction](#full-reproduction)
- [License](#license)

## Introduction

This work studies **4-bit KV cache quantization** with a **system-aware** recipe. Our primary method, **BDR (block-diagonal rotation)**, is **block Hadamard rotation on keys** (optional rotation on values) **before INT4 KV write**, implemented inside a **fork of [SGLang](https://github.com/sgl-project/sglang)**.

We ship two submodule branches on the same fork remote:

- **[third_party/sglang-fast-rotation](third_party/sglang-fast-rotation)** — **Our proposed BDR:** fused INT4 + Rotation. Use this fork for **both accuracy and throughput** on **BF16**, **INT4**, and **BDR** (the main paper numbers).
- **[third_party/sglang-kmeans](third_party/sglang-kmeans)** — **Ablation study for kmeans, kmeans+rotation:** KV dump, k-means centroids, and k-means + rotation variants. Not required to reproduce the core BDR vs BF16 vs INT4 story.

Pinned commits: [SUBMODULE_VERSIONS.md](SUBMODULE_VERSIONS.md).

## How to run BDR

This section is only about **BDR on `third_party/sglang-fast-rotation`**: clone the repo if needed, server flags (MHA / fa3 / triton), install that fork + **`fast_hadamard_transform`**, and the **`HADAMARD`** / **`ROTATE_V`** / **`HADAMARD_ORDER`** / **`--kv-cache-dtype`** knobs. **K-means ablations** use a different fork; install that under [Ablation study](#ablation-study-k-means-k-means--rotation).

### Get the code

```bash
git clone --recurse-submodules https://github.com/togethercomputer/System-Aware-4-Bit-KV-Cache-Quantization.git
cd System-Aware-4-Bit-KV-Cache-Quantization
```

If you cloned without submodules: `git submodule update --init --recursive`. If that fails on nested deps, use [scripts/clone_submodules.sh](scripts/clone_submodules.sh) or `git submodule update --init third_party/sglang-fast-rotation third_party/sglang-kmeans`. Pinned SHAs: [SUBMODULE_VERSIONS.md](SUBMODULE_VERSIONS.md).

### Server requirements

The paths in this README assume:

- **MHA models only** — **MLA** and other non-MHA layouts are **not supported** for these KV / BDR settings.
- **Prefill:** **`--prefill-attention-backend fa3`** (use **`fa4`** only if your GPU and SGLang build require it; see [SGLang attention backend](https://docs.sglang.ai/advanced_features/attention_backend.html)).
- **Decode:** **`--decode-attention-backend triton`**.

### Install BDR (sglang-fast-rotation)

**BDR** (block Hadamard rotation before INT4 KV) is implemented in **`third_party/sglang-fast-rotation`**. Install that fork, then the Hadamard extra used on the BDR path:

```bash
cd third_party/sglang-fast-rotation/python
pip install -e ".[all]"
pip install fast_hadamard_transform
```

Check that the CLI is available:

```bash
python -m sglang.launch_server --help | head
```

Use a **fresh virtualenv** and match **CUDA / PyTorch** to this submodule’s `python/pyproject.toml` (and upstream SGLang docs).

### BDR environment variables

Set these in the shell **before** `python -m sglang.launch_server` on **sglang-fast-rotation** (read in `memory_pool.py`). Combine with **`--kv-cache-dtype`**.

| Variable | Role |
|----------|------|
| **`HADAMARD`** | `0` = no rotation; `1` = block Hadamard on **K** before INT4 KV (with matching **Q** at decode). |
| **`ROTATE_V`** | `0` = K only (default BDR style); `1` = also rotate **V** and counter-rotate the attention output. |
| **`HADAMARD_ORDER`** | Block size (e.g. `16`); **must divide head dim**; ignored when `HADAMARD=0`. |
| **`--kv-cache-dtype`** | `auto` = BF16 KV baseline; `int4` = INT4 KV (with or without `HADAMARD=1` for BDR). |

Shell helpers live under [scripts/](scripts/) (see [Repository layout](#repository-layout)).

## Primary accuracy and throughput

**Accuracy** (simple-evals / GPQA) and **throughput** ([genai-bench](https://github.com/sgl-project/genai-bench)) both use **`third_party/sglang-fast-rotation`**; server setup is in [How to run BDR](#how-to-run-bdr). **Accuracy model:** **`Qwen/Qwen3-4B-Thinking-2507`**. **Throughput model:** **`Qwen/Qwen3-8B`** (override `MODEL_PATH` in scripts if you align checkpoints).

### Accuracy (primary)

#### Prepare

**Prerequisite (GPQA client):** install a local checkout of **[openai/simple-evals](https://github.com/openai/simple-evals)** (not tore-eval):

```bash
git clone https://github.com/openai/simple-evals.git
cd simple-evals
pip install -e .
pip install openai tqdm numpy
```

How to run evals (models, **`--eval gpqa`**, **`OPENAI_BASE_URL`**, registering a sampler for your SGLang `--model-path`, etc.) follows upstream [simple-evals README](https://github.com/openai/simple-evals/blob/main/README.md#running-the-evals).

With **simple-evals** installed as above and the SGLang server up, point the client at **`http://127.0.0.1:<port>/v1`** and run **GPQA** as in the upstream doc. **[scripts/run_primary_eval_matrix.sh](scripts/run_primary_eval_matrix.sh)** can print a `cd` into your checkout if you set **`SIMPLE_EVALS_DIR`**.

Pick **one** server row per run (same **`--model-path`** and port; vary only env + **`--kv-cache-dtype`**). **BDR + INT4 (K+V)** is the same pattern with **`ROTATE_V=1`** and **`HADAMARD_ORDER`** set. Install **`fast_hadamard_transform`** and env semantics: [How to run BDR](#how-to-run-bdr).

| Mode | Purpose | `HADAMARD` | `ROTATE_V` | `HADAMARD_ORDER` | `--kv-cache-dtype` |
|------|---------|------------|------------|------------------|---------------------|
| **BF16 KV** | Baseline, KV in bf16 | `0` | `0` | unset | `auto` |
| **INT4 KV** | 4-bit KV, no rotation | `0` | `0` | unset | `int4` |
| **INT4 + BDR (K only)** | 4-bit KV after block Hadamard on **K** (and matching **Q** at decode) | `1` | `0` | set (e.g. `16`; must divide head dim) | `int4` |

Start the server from `third_party/sglang-fast-rotation/python` to match a row above, or use **[scripts/run_primary_eval_matrix.sh](scripts/run_primary_eval_matrix.sh)** (`bf16`, `int4`, `bdr`, `bdr_kv`; default **`MODEL_PATH`** is **`Qwen/Qwen3-4B-Thinking-2507`**) and run the printed `launch_server` command.

**Hub for logs / summary tables:** [eval_primary/](eval_primary/)

#### Accuracy results (primary)

| Model | Method | Benchmark | Score |
|-------|--------|-----------|-------|
| Qwen/Qwen3-4B-Thinking-2507 | BF16 | GPQA | |
| Qwen/Qwen3-4B-Thinking-2507 | INT4 | GPQA | |
| Qwen/Qwen3-4B-Thinking-2507 | BDR (K-only) | GPQA | |

Fill from the paper or from [eval_primary/results/](eval_primary/results/).

### Throughput and latency (primary)

Speed results use **sglang-fast-rotation** (fused INT4 + BDR kernels) with **`Qwen/Qwen3-8B`**, driven by **[genai-bench](https://github.com/sgl-project/genai-bench)** against the server’s OpenAI-compatible HTTP API. Helper: [scripts/run_genai_bench_example.sh](scripts/run_genai_bench_example.sh) (default `MODEL_PATH`). Full CLI, traffic scenarios, Excel/plots: [GenAI Bench docs](https://docs.sglang.ai/genai-bench/getting-started/) and [Run benchmark](https://docs.sglang.ai/genai-bench/user-guide/run-benchmark/).

#### Prepare (genai-bench)

**Prerequisite (throughput client):** install genai-bench (separate from the SGLang venv if you prefer):

```bash
pip install genai-bench
```

Optional (quieter HF logs during tokenizer load): `export TRANSFORMERS_VERBOSITY=error`. For Docker / dev installs, see the upstream [installation guide](https://docs.sglang.ai/genai-bench/getting-started/installation/).

**Terminal 1 — server** (example BF16 KV):

```bash
cd third_party/sglang-fast-rotation/python
python -m sglang.launch_server \
  --prefill-attention-backend fa3 \
  --decode-attention-backend triton \
  --model-path "Qwen/Qwen3-8B" \
  --port 30000 \
  --kv-cache-dtype auto
```

**Terminal 2 — client** (after `pip install genai-bench`; matches ~256 input / 32 output tokens and concurrency 16 — see [traffic scenarios](https://docs.sglang.ai/genai-bench/user-guide/scenario-definition/)):

```bash
genai-bench benchmark --api-backend sglang \
  --api-base "http://127.0.0.1:30000" \
  --api-key "dummy" \
  --api-model-name "Qwen/Qwen3-8B" \
  --model-tokenizer "Qwen/Qwen3-8B" \
  --task text-to-text \
  --traffic-scenario "D(256,32)" \
  --num-concurrency 16 \
  --max-time-per-run 5 \
  --max-requests-per-run 200 \
  --server-engine "SGLang" \
  --server-gpu-type "local" \
  --server-version "custom" \
  --server-gpu-count 1
```

Tune `--max-time-per-run`, `--max-requests-per-run`, `--num-concurrency`, and `--traffic-scenario` using `genai-bench benchmark --help` and the docs above. Label runs with accurate `--server-gpu-type` / `--server-version` when publishing numbers.

**Sweep BF16 vs INT4 vs BDR:** restart the server with the right env and `--kv-cache-dtype`, then rerun **genai-bench** with **identical** client flags.

| Config | Env | `--kv-cache-dtype` |
|--------|-----|-------------------|
| BF16 KV | `HADAMARD=0` or unset | `auto` |
| INT4 KV | `HADAMARD=0` | `int4` |
| BDR + INT4 | `HADAMARD=1` `ROTATE_V=0` `HADAMARD_ORDER=16` | `int4` |

SGLang’s built-in `bench_serving` ([bench_serving](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/bench_serving.md)) is optional; this repo standardizes on **genai-bench** for comparable sweeps and reporting.

**Hub:** [eval_speed/](eval_speed/)  
**Helper:** [scripts/run_genai_bench_example.sh](scripts/run_genai_bench_example.sh)

#### Speed results (primary)

| Model | KV config | Output tok/s | TTFT (ms) | TPOT (ms) | Workload |
|-------|-----------|--------------|-----------|-----------|----------|
| Qwen/Qwen3-8B | BF16 / auto | — | — | — | — |
| Qwen/Qwen3-8B | INT4 | — | — | — | — |
| Qwen/Qwen3-8B | INT4 + BDR (K-only) | — | — | — | — |

Fill from [eval_speed/results/](eval_speed/results/).

## Ablation study (k-means, k-means + rotation)

Use **`third_party/sglang-kmeans`**: KV dump for calibration, [tools/fit_kv_centroids.py](tools/fit_kv_centroids.py), then `SGLANG_KV_CENTROIDS_PATH` for **k-means + INT4** and **k-means + BDR** (optional `HADAMARD` / `ROTATE_V`). Accuracy still uses **simple-evals** ([Prepare](#prepare); run GPQA per upstream docs).

### Install sglang-kmeans

Not needed for primary BF16 / INT4 / BDR ([How to run BDR](#how-to-run-bdr)). For this fork only:

```bash
cd third_party/sglang-kmeans/python
pip install -e ".[all]"
pip install "flash-kmeans @ git+https://github.com/jindajia/flash-kmeans.git"
```

### KV calibration (ablation only)

Primary BF16 / INT4 / BDR does **not** need this step.

**1. Dump KV activations** — run from **sglang-kmeans** with a **BF16 KV cache** (`auto`) so dumps are in calibration space:

```bash
cd third_party/sglang-kmeans/python

export DUMP_KVCACHE=true
export DUMP_KVCACHE_TOKENS=512
export DUMP_KVCACHE_DIR=/path/to/kv_dumps

python -m sglang.launch_server \
  --prefill-attention-backend fa3 \
  --decode-attention-backend triton \
  --model-path "Qwen/Qwen3-8B" \
  --port 30000 \
  --kv-cache-dtype auto
```

Drive enough traffic so each layer hits the threshold at least once. Files appear as `kv_calibration_layer_<layer_id>.pt` (dict with `k`, `v`, `indices` on CPU; see `triton_backend.py` in the submodule for selection logic).

**2. Fit centroids offline** — from the **repository root**:

```bash
python tools/fit_kv_centroids.py \
  --dump-dir /path/to/kv_dumps \
  --out-dir /path/to/centroids_out \
  --n-clusters 16 \
  --seed 0
```

This writes `k_layer_L_clusters_<N>_centers.pt` and `v_layer_L_clusters_<N>_centers.pt` per global layer `L`, shaped `(N, num_kv_heads_global * head_dim)`, for loading in the submodule.

**3. Run INT4 + k-means inference**

```bash
export N_CLUSTERS=16
export SGLANG_KV_CENTROIDS_PATH=/path/to/centroids_out

python -m sglang.launch_server \
  --prefill-attention-backend fa3 \
  --decode-attention-backend triton \
  --model-path "Qwen/Qwen3-8B" \
  --port 30000 \
  --kv-cache-dtype int4
```

**K-means + BDR:** keep `SGLANG_KV_CENTROIDS_PATH`, set `HADAMARD=1`, optional `ROTATE_V`, and `HADAMARD_ORDER` consistent with head dimension (same as primary BDR).

### Ablation method matrix

| Method | `HADAMARD` | `ROTATE_V` | `HADAMARD_ORDER` | `--kv-cache-dtype` | `SGLANG_KV_CENTROIDS_PATH` | `N_CLUSTERS` |
|--------|------------|------------|------------------|---------------------|----------------------------|--------------|
| K-means + INT4 | `0` | `0` | n/a | `int4` | required | match files |
| K-means + BDR | `1` | `0` or `1` | set | `int4` | required | match files |

**K-means + INT4 example:**

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

**K-means + BDR example:**

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

**Hub:** [eval_accuracy/](eval_accuracy/)  
**Helper:** `CENTROIDS=/path/to/centroids_out ./scripts/run_eval_matrix.sh kmeans` or `kmeans_bdr`.

#### Accuracy results (ablation)

| Model | Method | Benchmark | Score |
|-------|--------|-----------|-------|
| — | K-means + INT4 | — | — |
| — | K-means + BDR | — | — |

Fill from [eval_accuracy/results/](eval_accuracy/results/).

## Repository layout

| Path | Role |
|------|------|
| [third_party/sglang-fast-rotation/](third_party/sglang-fast-rotation/) | **Primary** BF16 / INT4 / BDR — accuracy + speed |
| [third_party/sglang-kmeans/](third_party/sglang-kmeans/) | **Ablation** k-means KV + dump / centroids |
| [scripts/](scripts/) | `run_primary_eval_matrix.sh`, `run_eval_matrix.sh`, `run_genai_bench_example.sh`, `clone_submodules.sh` |
| [tools/](tools/) | `fit_kv_centroids.py` (ablation calibration) |
| [eval_primary/](eval_primary/) | Primary **accuracy** logs / tables |
| [eval_speed/](eval_speed/) | Primary **throughput** logs / tables |
| [eval_accuracy/](eval_accuracy/) | Ablation **accuracy** logs / tables |

## Full reproduction

Large raw bundles may live outside this repo.

- **Full reproduction bundle:** *TBD — add URL*

Submodule SHAs: [SUBMODULE_VERSIONS.md](SUBMODULE_VERSIONS.md).

## License

See [LICENSE](LICENSE).
