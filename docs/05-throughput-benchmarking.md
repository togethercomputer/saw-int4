# Throughput and latency (sglang-fast-rotation)

**Experiment hub:** [../eval_speed/README.md](../eval_speed/README.md) (commands, results folder).

Speed results should use the **fast-rotation** submodule, which contains the fused INT4 + BDR kernels intended for serving.

Official SGLang documentation (upstream):

- [Benchmark and Profiling](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md)  
- [bench_serving](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/bench_serving.md) (linked from the guide above)

## Recommended tool: `bench_serving`

Start the server from **sglang-fast-rotation**, then in another shell (same Python env):

```bash
cd third_party/sglang-fast-rotation/python

# BF16 KV example
python -m sglang.launch_server \
  --model-path "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --port 30000 \
  --kv-cache-dtype auto
```

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --num-prompts 80 \
  --max-concurrency 16 \
  --random-input-len 256 \
  --random-output-len 32 \
  --dataset-name random
```

Upstream recommends `num-prompts >= 5 * max-concurrency` for steady-state throughput.

## Sweep BF16 vs INT4 vs BDR (K-only)

For each configuration, **restart** the server with the right env vars and `--kv-cache-dtype`, then rerun `bench_serving` with identical client flags.

| Config | Env | `--kv-cache-dtype` |
|--------|-----|-------------------|
| BF16 KV | `HADAMARD=0` or unset | `auto` |
| INT4 KV | `HADAMARD=0` | `int4` |
| BDR + INT4 | `HADAMARD=1` `ROTATE_V=0` `HADAMARD_ORDER=16` | `int4` |

Example BDR server:

```bash
export HADAMARD=1
export ROTATE_V=0
export HADAMARD_ORDER=16
python -m sglang.launch_server \
  --model-path "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --port 30000 \
  --kv-cache-dtype int4
```

## Other tools

The upstream guide compares:

- `bench_one_batch_server` — single HTTP batch (not steady state)  
- `bench_offline_throughput` — in-process `Engine`, no HTTP  
- `bench_one_batch` — kernel-level static batch  

Use these only when you understand the bias they introduce.

## Profiling (optional)

To collect PyTorch profiler traces while benchmarking, follow the **Profile with PyTorch Profiler** section in the same upstream doc (`SGLANG_TORCH_PROFILER_DIR`, `--profile` on `bench_serving`, etc.).

## Helper script

See [../scripts/run_bench_serving_example.sh](../scripts/run_bench_serving_example.sh) for a minimal two-step template (server env + client command).
