# System-Aware 4-Bit KV Cache Quantization

Official companion code for the paper **System-Aware 4-Bit KV Cache Quantization** (Together). *Venue / arXiv / DOI: add at publication time.*

## Contents

- [Introduction](#introduction)
- [Preparation](#preparation)
- [Run BDR](#run-bdr)
- [Speed evaluation](#speed-evaluation)
- [Accuracy evaluation](#accuracy-evaluation)
- [Repository layout](#repository-layout)
- [Full reproduction](#full-reproduction)
- [License](#license)

## Introduction

This work studies **4-bit KV cache quantization** with a **system-aware** recipe. Our primary on-path method, **BDR (block-diagonal rotation)**, is implemented as **block Hadamard rotation on keys** (optional rotation on values) **before INT4 KV write**, inside a **fork of [SGLang](https://github.com/sgl-project/sglang)**—not a standalone runtime.

We ship two submodule branches on the same fork remote:

- **[third_party/sglang-fast-rotation](third_party/sglang-fast-rotation)** — fused INT4 + BDR kernels; use for **throughput / latency** (`bench_serving`).
- **[third_party/sglang-kmeans](third_party/sglang-kmeans)** — full **accuracy** matrix (BF16, INT4, BDR, k-means, k-means + rotation), KV dump, and centroid loading.

Pinned commits: [SUBMODULE_VERSIONS.md](SUBMODULE_VERSIONS.md).

## Preparation

```bash
git clone --recurse-submodules https://github.com/togethercomputer/System-Aware-4-Bit-KV-Cache-Quantization.git
cd System-Aware-4-Bit-KV-Cache-Quantization
```

**Accuracy** in this repository is reproduced only with the open-source **[simple-evals](https://github.com/openai/simple-evals)** client against SGLang’s OpenAI-compatible API (not tore-eval). If `git submodule update --init --recursive` fails on an optional nested dependency inside a fork, initialize the two top-level submodules only; see [SUBMODULE_VERSIONS.md](SUBMODULE_VERSIONS.md) and [scripts/clone_submodules.sh](scripts/clone_submodules.sh).

Full install (CUDA, PyTorch, `fast_hadamard_transform`, `flash-kmeans`, simple-evals): [docs/01-preparation.md](docs/01-preparation.md). Overview of docs: [docs/00-overview.md](docs/00-overview.md).

## Run BDR

Default setting: **rotate K only** (`ROTATE_V=0`). From [third_party/sglang-fast-rotation/python](third_party/sglang-fast-rotation/python) after `pip install -e ".[all]"` and `pip install fast_hadamard_transform`:

```bash
export HADAMARD=1
export ROTATE_V=0
export HADAMARD_ORDER=16

python -m sglang.launch_server \
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

## Speed evaluation

Throughput and latency use **sglang-fast-rotation** and SGLang’s **`bench_serving`** (see [docs/05-throughput-benchmarking.md](docs/05-throughput-benchmarking.md) and upstream [Benchmark and Profiling](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md)).

**Commands, sweep checklist, and raw log layout:** [eval_speed/](eval_speed/)  
**Helper:** [scripts/run_bench_serving_example.sh](scripts/run_bench_serving_example.sh)

### Speed results (summary)

| Model | KV config | Output tok/s | TTFT (ms) | TPOT (ms) | Workload |
|-------|-----------|--------------|-----------|-----------|----------|
| — | BF16 / auto | — | — | — | — |
| — | INT4 | — | — | — | — |
| — | INT4 + BDR (K-only) | — | — | — | — |

Fill from paper or from logs under [eval_speed/results/](eval_speed/results/).

## Accuracy evaluation

Accuracy uses **[OpenAI simple-evals](https://github.com/openai/simple-evals)** against the OpenAI-compatible HTTP API from **sglang-kmeans**. Method matrix (env + `--kv-cache-dtype`): [docs/03-evaluation-matrix.md](docs/03-evaluation-matrix.md). KV dump → centroids: [docs/04-kv-calibration.md](docs/04-kv-calibration.md).

**Commands, method checklist, and result logs:** [eval_accuracy/](eval_accuracy/)  
**Helper:** [scripts/run_eval_matrix.sh](scripts/run_eval_matrix.sh)

### Accuracy results (summary)

| Model | Method | Benchmark | Score |
|-------|--------|-----------|-------|
| — | BF16 | — | — |
| — | INT4 | — | — |
| — | BDR (K-only) | — | — |
| — | K-means + INT4 | — | — |
| — | K-means + BDR | — | — |

Fill from paper or from logs under [eval_accuracy/results/](eval_accuracy/results/).

## Repository layout

| Path | Role |
|------|------|
| [docs/](docs/) | Detailed setup, BDR, accuracy matrix, calibration, throughput |
| [third_party/sglang-fast-rotation/](third_party/sglang-fast-rotation/) | Speed / production-style BDR + INT4 |
| [third_party/sglang-kmeans/](third_party/sglang-kmeans/) | Accuracy matrix, KV dump, k-means |
| [scripts/](scripts/) | Submodule init, eval env printer, bench template |
| [tools/](tools/) | `fit_kv_centroids.py` for calibration |
| [eval_speed/](eval_speed/) | Throughput experiment hub + [eval_speed/results/](eval_speed/results/) |
| [eval_accuracy/](eval_accuracy/) | Accuracy experiment hub + [eval_accuracy/results/](eval_accuracy/results/) |

## Full reproduction

Large raw bundles (logs, conda lockfiles, exact simple-evals invocations) may live outside this repo.

- **Full reproduction bundle:** *TBD — add URL*

Submodule SHAs for paper alignment: [SUBMODULE_VERSIONS.md](SUBMODULE_VERSIONS.md).

## License

See [LICENSE](LICENSE).
