# Speed evaluation (throughput / latency)

This folder is the **hub for throughput experiments**: commands, conventions, and where to store raw `bench_serving` outputs.

**Canonical doc:** [../docs/05-throughput-benchmarking.md](../docs/05-throughput-benchmarking.md)  
**Helper script:** [../scripts/run_bench_serving_example.sh](../scripts/run_bench_serving_example.sh)

## Server build

Use **[third_party/sglang-fast-rotation](../third_party/sglang-fast-rotation)** (fused INT4 KV + BDR). Install from `third_party/sglang-fast-rotation/python` per [docs/01-preparation.md](../docs/01-preparation.md).

## Recommended workload (template)

Match upstream guidance: [Benchmark and Profiling](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md). Prefer `num-prompts >= 5 * max-concurrency` for steady state.

**Terminal 1 — server** (example: BDR, K-only):

```bash
cd third_party/sglang-fast-rotation/python
export HADAMARD=1
export ROTATE_V=0
export HADAMARD_ORDER=16
python -m sglang.launch_server \
  --model-path "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --port 30000 \
  --kv-cache-dtype int4
```

**Terminal 2 — client:**

```bash
cd third_party/sglang-fast-rotation/python
python -m sglang.bench_serving \
  --backend sglang \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --num-prompts 80 \
  --max-concurrency 16 \
  --random-input-len 256 \
  --random-output-len 32 \
  --dataset-name random
```

Sweep **BF16** / **INT4** / **BDR** by changing server env and `--kv-cache-dtype` only; keep client flags fixed for comparability.

## Results

Store machine-readable logs and summarized tables under **[results/](results/)**. Use the table template there; copy a one-row summary into the main [README.md](../README.md) when you publish numbers.
