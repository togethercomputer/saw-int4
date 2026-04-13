# Accuracy evaluation (simple-evals + SGLang)

This folder is the **hub for accuracy experiments**: method matrix, calibration, and where to store simple-evals outputs.

**Canonical doc:** [../docs/03-evaluation-matrix.md](../docs/03-evaluation-matrix.md)  
**KV dump and centroids:** [../docs/04-kv-calibration.md](../docs/04-kv-calibration.md)  
**Helper script (server env per method):** [../scripts/run_eval_matrix.sh](../scripts/run_eval_matrix.sh)  
**Fit centroids from dumps:** [../tools/fit_kv_centroids.py](../tools/fit_kv_centroids.py)

## Server

Build and run **[third_party/sglang-kmeans](../third_party/sglang-kmeans)** so you can evaluate **BF16**, **INT4**, **BDR**, **k-means**, and **k-means + rotation** with the same OpenAI-compatible API.

## Client (open-source simple-evals only)

Accuracy is **not** run through tore-eval. Install **[simple-evals](https://github.com/openai/simple-evals)** from GitHub:

```bash
git clone https://github.com/openai/simple-evals.git
cd simple-evals
pip install -e .
pip install openai tqdm numpy
```

Then point it at SGLang and run tasks from the [simple-evals README](https://github.com/openai/simple-evals/blob/main/README.md):

```bash
export OPENAI_BASE_URL="http://127.0.0.1:30000/v1"
export OPENAI_API_KEY="dummy"
python -m simple-evals.simple_evals --list-models
python -m simple-evals.simple_evals --model <model_id> --examples 200
```

Use the benchmarks you report in the paper (MMLU, GPQA, HumanEval, etc.).

From the **repository root**, print server settings for each method:

```bash
./scripts/run_eval_matrix.sh bf16
./scripts/run_eval_matrix.sh int4
./scripts/run_eval_matrix.sh bdr
CENTROIDS=/path/to/centroids ./scripts/run_eval_matrix.sh kmeans
CENTROIDS=/path/to/centroids ./scripts/run_eval_matrix.sh kmeans_bdr
```

## Results

Store run logs and parsed scores under **[results/](results/)**. Use the table template there; mirror headline numbers in the main [README.md](../README.md).
