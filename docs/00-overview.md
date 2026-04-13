# Overview

This repository is the official companion for **System-Aware 4-Bit KV Cache Quantization**. It does not reimplement inference from scratch: methods are delivered as **forks of [SGLang](https://github.com/sgl-project/sglang)** with KV-cache quantization and optional **block-diagonal Hadamard rotation (BDR)** before INT4 storage.

## Two SGLang submodules

| Path | Role |
|------|------|
| [third_party/sglang-fast-rotation](../third_party/sglang-fast-rotation) | **Production-oriented** path: fused INT4 KV + BDR kernels. Use this fork for **throughput / latency** studies with SGLang’s official benchmark tools (`bench_serving`, etc.). |
| [third_party/sglang-kmeans](../third_party/sglang-kmeans) | **Research / ablation** path: INT4/INT8 KV, BDR env flags, **KV cache dump** for calibration, and **k-means centroid** residual quantization. Use this fork for the full **accuracy matrix** (BF16, INT4, BDR, k-means, k-means + rotation). |

## Documentation map

1. [01-preparation.md](01-preparation.md) — environment and builds  
2. [02-bdr-inference.md](02-bdr-inference.md) — running BDR (K-only by default) on `sglang-fast-rotation`  
3. [03-evaluation-matrix.md](03-evaluation-matrix.md) — accuracy matrix on `sglang-kmeans` + open-source [simple-evals](https://github.com/openai/simple-evals) only (no tore-eval)  
4. [04-kv-calibration.md](04-kv-calibration.md) — dump KV → fit centroids → `SGLANG_KV_CENTROIDS_PATH`  
5. [05-throughput-benchmarking.md](05-throughput-benchmarking.md) — SGLang `bench_serving` on `sglang-fast-rotation`
