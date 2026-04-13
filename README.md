# System-Aware 4-Bit KV Cache Quantization

Official companion code for the paper **System-Aware 4-Bit KV Cache Quantization** (Together). *Venue / arXiv / DOI: add at publication time.*

## Contents

- [Introduction](#introduction)
- [Preparation](#preparation)
- [Primary evaluation (BF16, INT4, BDR)](#primary-evaluation-bf16-int4-bdr)
- [Ablation study (k-means, k-means + rotation)](#ablation-study-k-means-k-means--rotation)
- [Repository layout](#repository-layout)
- [Full reproduction](#full-reproduction)
- [License](#license)

## Introduction

This work studies **4-bit KV cache quantization** with a **system-aware** recipe. Our primary method, **BDR (block-diagonal rotation)**, is **block Hadamard rotation on keys** (optional rotation on values) **before INT4 KV write**, implemented inside a **fork of [SGLang](https://github.com/sgl-project/sglang)**.

We ship two submodule branches on the same fork remote:

- **[third_party/sglang-fast-rotation](third_party/sglang-fast-rotation)** — **Primary evaluation:** fused INT4 + BDR. Use this fork for **both accuracy and throughput** on **BF16**, **INT4**, and **BDR** (the main paper numbers).
- **[third_party/sglang-kmeans](third_party/sglang-kmeans)** — **Ablation study:** KV dump, k-means centroids, and k-means + rotation variants. Not required to reproduce the core BDR vs BF16 vs INT4 story.

Pinned commits: [SUBMODULE_VERSIONS.md](SUBMODULE_VERSIONS.md).

## Preparation

```bash
git clone --recurse-submodules https://github.com/togethercomputer/System-Aware-4-Bit-KV-Cache-Quantization.git
cd System-Aware-4-Bit-KV-Cache-Quantization
```

**Accuracy** everywhere uses the open-source **[simple-evals](https://github.com/openai/simple-evals)** client against SGLang’s OpenAI-compatible API (not tore-eval). If `git submodule update --init --recursive` fails on optional nested deps inside a fork, use [scripts/clone_submodules.sh](scripts/clone_submodules.sh) or init only the two top-level submodules; see [SUBMODULE_VERSIONS.md](SUBMODULE_VERSIONS.md).

### Install SGLang

**Primary track only:** install **fast-rotation** first:

```bash
cd third_party/sglang-fast-rotation/python && pip install -e ".[all]"
```

**Ablations:** add the k-means fork:

```bash
cd ../../sglang-kmeans/python && pip install -e ".[all]"
```

Full steps: [docs/01-preparation.md](docs/01-preparation.md#install-sglang-from-submodules).

### Prerequisites (MHA, Flash Attention prefill, Triton decode)

| Requirement | Why |
|-------------|-----|
| **MHA models only** | These KV paths target **multi-head attention**. **MLA** and other non-MHA layouts are **not supported** here. |
| **Prefill** | e.g. **`--prefill-attention-backend fa3`** (or **`fa4`** per GPU / SGLang). |
| **Decode** | **`--decode-attention-backend triton`**. |

Details: [docs/01-preparation.md](docs/01-preparation.md#attention-backends-and-model-support-bdr-and-k-means). For BDR also: `pip install fast_hadamard_transform`. For k-means ablations: `flash-kmeans` per [docs/01-preparation.md](docs/01-preparation.md).

## Primary evaluation (BF16, INT4, BDR)

Use **`third_party/sglang-fast-rotation`** for **all** primary numbers: **accuracy** (simple-evals) and **speed** (`bench_serving`). Same env knobs (`HADAMARD`, `ROTATE_V`, `HADAMARD_ORDER`, `--kv-cache-dtype`) and the same **MHA + fa3 + triton** attention stack.

### Running the server (example: BDR, K-only)

From [third_party/sglang-fast-rotation/python](third_party/sglang-fast-rotation/python):

```bash
export HADAMARD=1
export ROTATE_V=0
export HADAMARD_ORDER=16

python -m sglang.launch_server \
  --prefill-attention-backend fa3 \
  --decode-attention-backend triton \
  --model-path "Qwen/Qwen3-8B" \
  --port 30000 \
  --kv-cache-dtype int4
```

| Setting | `HADAMARD` | `ROTATE_V` | `--kv-cache-dtype` |
|---------|------------|------------|--------------------|
| BF16 KV | `0` | `0` | `auto` |
| INT4 KV (no rotation) | `0` | `0` | `int4` |
| BDR + INT4 (K only) | `1` | `0` | `int4` |
| BDR + INT4 (K+V) | `1` | `1` | `int4` |

More detail: [docs/02-bdr-inference.md](docs/02-bdr-inference.md), [third_party/sglang-fast-rotation/EVAL_NOTES.md](third_party/sglang-fast-rotation/EVAL_NOTES.md).

### Accuracy (primary)

Point simple-evals at `http://127.0.0.1:<port>/v1` with the server above. Method matrix and copy-paste flow: [docs/03-evaluation-matrix.md](docs/03-evaluation-matrix.md#primary-evaluation-track).

**Hub for logs / summary tables:** [eval_primary/](eval_primary/)  
**Helper:** [scripts/run_primary_eval_matrix.sh](scripts/run_primary_eval_matrix.sh) (`bf16`, `int4`, `bdr`, `bdr_kv`).

#### Accuracy results (primary)

| Model | Method | Benchmark | Score |
|-------|--------|-----------|-------|
| — | BF16 | — | — |
| — | INT4 | — | — |
| — | BDR (K-only) | — | — |

Fill from the paper or from [eval_primary/results/](eval_primary/results/).

### Speed (primary)

Throughput / latency on the **same** fast-rotation build: [docs/05-throughput-benchmarking.md](docs/05-throughput-benchmarking.md), [eval_speed/](eval_speed/), [scripts/run_bench_serving_example.sh](scripts/run_bench_serving_example.sh).

#### Speed results (primary)

| Model | KV config | Output tok/s | TTFT (ms) | TPOT (ms) | Workload |
|-------|-----------|--------------|-----------|-----------|----------|
| — | BF16 / auto | — | — | — | — |
| — | INT4 | — | — | — | — |
| — | INT4 + BDR (K-only) | — | — | — | — |

Fill from [eval_speed/results/](eval_speed/results/).

## Ablation study (k-means, k-means + rotation)

Use **`third_party/sglang-kmeans`**: KV dump for calibration, [tools/fit_kv_centroids.py](tools/fit_kv_centroids.py), then `SGLANG_KV_CENTROIDS_PATH` for **k-means + INT4** and **k-means + BDR** (optional `HADAMARD` / `ROTATE_V`). Accuracy still via **simple-evals**.

**Docs:** [docs/03-evaluation-matrix.md](docs/03-evaluation-matrix.md#ablation-study-track), [docs/04-kv-calibration.md](docs/04-kv-calibration.md).

**Hub:** [eval_accuracy/](eval_accuracy/)  
**Helper:** [scripts/run_eval_matrix.sh](scripts/run_eval_matrix.sh) (`kmeans`, `kmeans_bdr`; requires `CENTROIDS=`).

#### Accuracy results (ablation)

| Model | Method | Benchmark | Score |
|-------|--------|-----------|-------|
| — | K-means + INT4 | — | — |
| — | K-means + BDR | — | — |

Fill from [eval_accuracy/results/](eval_accuracy/results/).

## Repository layout

| Path | Role |
|------|------|
| [docs/](docs/) | Setup, BDR, evaluation (primary + ablation), calibration, throughput |
| [third_party/sglang-fast-rotation/](third_party/sglang-fast-rotation/) | **Primary** BF16 / INT4 / BDR — accuracy + speed |
| [third_party/sglang-kmeans/](third_party/sglang-kmeans/) | **Ablation** k-means KV + dump / centroids |
| [scripts/](scripts/) | `run_primary_eval_matrix.sh`, `run_eval_matrix.sh` (ablation), `run_bench_serving_example.sh`, `clone_submodules.sh` |
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
