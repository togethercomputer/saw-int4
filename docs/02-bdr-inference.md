# BDR inference (SGLang fork)

## What “BDR” is in code

In our SGLang forks, **block-diagonal rotation before INT4 KV quantization** is implemented as a **block Hadamard** transform on the head dimension (see `HADAMARD_ORDER`), applied to **K** before writing INT4 KV, with a matching transform on **Q** at decode so attention scores are unchanged. Optionally, **V** can be rotated and the attention output counter-rotated (`ROTATE_V=1`).

Details and evaluation notes live in the submodule:

- [third_party/sglang-fast-rotation/EVAL_NOTES.md](../third_party/sglang-fast-rotation/EVAL_NOTES.md)

Implementation entry points include:

- `python/sglang/srt/mem_cache/memory_pool.py` — env toggles and write path  
- `python/sglang/srt/layers/attention/triton_backend.py` — decode-side Q transform  

## Default recipe: rotate **K** only (recommended)

This matches the common **QuIP# / QuaRot-style** setting: rotate K (and Q at decode), leave V unrotated unless you explicitly ablate `ROTATE_V`.

```bash
cd third_party/sglang-fast-rotation/python

export HADAMARD=1
export ROTATE_V=0
export HADAMARD_ORDER=16   # must divide head_dim; try 16 / 64 / 128 per model

python -m sglang.launch_server \
  --model-path "Qwen/Qwen3-8B" \
  --port 30000 \
  --kv-cache-dtype int4
```

- **BF16 KV baseline:** omit rotation and use `--kv-cache-dtype auto` (or your SGLang version’s bf16 KV setting).  
- **INT4 KV without rotation:** `HADAMARD=0` `ROTATE_V=0` and `--kv-cache-dtype int4`.

## Optional: rotate K and V

```bash
export HADAMARD=1
export ROTATE_V=1
export HADAMARD_ORDER=16
python -m sglang.launch_server --model-path "Qwen/Qwen3-8B" --port 30000 --kv-cache-dtype int4
```

## Dependency

BDR requires `fast_hadamard_transform` (see [01-preparation.md](01-preparation.md)).

## Accuracy evaluation (paper workflow)

For **all accuracy numbers**, use the open-source **[simple-evals](https://github.com/openai/simple-evals)** client against a server built from **sglang-kmeans**, following [03-evaluation-matrix.md](03-evaluation-matrix.md). This repository does **not** use tore-eval for benchmarking.

The `sglang-fast-rotation` submodule may still contain legacy scripts (for example `eval_kv_rotation.sh`) used internally by the fork; they are **not** part of the official reproduction path documented here.
