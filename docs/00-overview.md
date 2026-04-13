# Overview

This repository is the official companion for **System-Aware 4-Bit KV Cache Quantization**. It does not reimplement inference from scratch: methods are delivered as **forks of [SGLang](https://github.com/sgl-project/sglang)** with KV-cache quantization and optional **block-diagonal Hadamard rotation (BDR)** before INT4 storage. Documented experiments assume **MHA** models with **Flash Attention prefill** (`fa3` / `fa4` as appropriate) and **Triton decode**; see [01-preparation.md](01-preparation.md#attention-backends-and-model-support-bdr-and-k-means).

## Two SGLang submodules

| Path | Role |
|------|------|
| [third_party/sglang-fast-rotation](../third_party/sglang-fast-rotation) | **Primary** path: fused INT4 KV + BDR. Use for **BF16 / INT4 / BDR accuracy and throughput** (simple-evals + `bench_serving`). |
| [third_party/sglang-kmeans](../third_party/sglang-kmeans) | **Ablation** path: KV dump, k-means centroids, k-means + rotation. |

## Documentation map

1. [01-preparation.md](01-preparation.md) — environment and builds  
2. [02-bdr-inference.md](02-bdr-inference.md) — running BDR (K-only by default) on `sglang-fast-rotation`  
3. [03-evaluation-matrix.md](03-evaluation-matrix.md) — **Primary** vs **ablation** accuracy with [simple-evals](https://github.com/openai/simple-evals) (no tore-eval)  
4. [04-kv-calibration.md](04-kv-calibration.md) — dump KV → fit centroids → `SGLANG_KV_CENTROIDS_PATH` (ablation)  
5. [05-throughput-benchmarking.md](05-throughput-benchmarking.md) — `bench_serving` on `sglang-fast-rotation` (primary throughput)
