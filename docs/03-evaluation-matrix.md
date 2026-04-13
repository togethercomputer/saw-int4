# Accuracy evaluation matrix (sglang-kmeans + simple-evals)

**Experiment hub:** [../eval_accuracy/README.md](../eval_accuracy/README.md) (commands, results folder).

All **accuracy** rows in the paper should be reproduced with the **same OpenAI simple-evals tasks** and the **same sampling hyperparameters**, varying only the **SGLang server** configuration.

- **Server:** build and run [third_party/sglang-kmeans](../third_party/sglang-kmeans) (not `sglang-fast-rotation`), so k-means and dump-based methods are available.  
- **Client:** [openai/simple-evals](https://github.com/openai/simple-evals) using the **OpenAI** sampler against SGLang’s **OpenAI-compatible** HTTP API.

## Start SGLang (OpenAI-compatible)

Example (adjust model and port):

```bash
cd third_party/sglang-kmeans/python
export OPENAI_API_KEY=dummy   # not used by local SGLang but some clients require a value

# Example: BF16 KV
python -m sglang.launch_server --model-path "Qwen/Qwen3-8B" --port 30000 --kv-cache-dtype auto
```

Point simple-evals at the server (OpenAI Python SDK v1 style):

```bash
export OPENAI_BASE_URL="http://127.0.0.1:30000/v1"
export OPENAI_API_KEY="dummy"
```

Use a **model name** that SGLang accepts on `/v1/chat/completions` (often the same string as `--model-path` or a short name; match what your SGLang build documents).

## Run simple-evals

From a checkout of simple-evals:

```bash
python -m simple-evals.simple_evals --list-models
python -m simple-evals.simple_evals --model <your_registered_or_custom_model> --examples 200
```

If your model name is not built in, add a small adapter in simple-evals (OpenAI-compatible base URL + model id) per upstream instructions, or use `--model` values that already map to `OPENAI_BASE_URL`.

Supported benchmarks in simple-evals include **MMLU**, **GPQA**, **MATH**, **HumanEval**, **DROP**, **MGSM**, **SimpleQA**, and others; match the subset reported in the paper.

## Method matrix (environment + `--kv-cache-dtype`)

Set variables **before** launching `sglang.launch_server`. For k-means, also set `SGLANG_KV_CENTROIDS_PATH` to a directory produced by [04-kv-calibration.md](04-kv-calibration.md) and [../tools/fit_kv_centroids.py](../tools/fit_kv_centroids.py).

| Method | `HADAMARD` | `ROTATE_V` | `HADAMARD_ORDER` | `--kv-cache-dtype` | `SGLANG_KV_CENTROIDS_PATH` | `N_CLUSTERS` |
|--------|------------|------------|------------------|---------------------|----------------------------|--------------|
| BF16 | unset / `0` | unset / `0` | n/a | `auto` | unset | n/a |
| INT4 | `0` | `0` | n/a | `int4` | unset | n/a |
| BDR (K only) | `1` | `0` | e.g. `16` | `int4` | unset | n/a |
| BDR (K+V) | `1` | `1` | e.g. `16` | `int4` | unset | n/a |
| K-means + INT4 | `0` | `0` | n/a | `int4` | path to `*_centers.pt` | must match files (default `16`) |
| K-means + BDR | `1` | `0` or `1` | set | `int4` | path to `*_centers.pt` | same |

## Automation

See [../scripts/run_eval_matrix.sh](../scripts/run_eval_matrix.sh) for a shell helper that prints exact `export` lines and a template `launch_server` command per method.

## Full reproduction bundle

Raw JSON logs, exact simple-evals revisions, and machine configuration may be published **outside** this repo (for example Hugging Face, Zenodo, or an internal artifact store). The main [README.md](../README.md) links to that bundle once it is available.
