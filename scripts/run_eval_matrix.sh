#!/usr/bin/env bash
# Ablation study only: k-means / k-means+BDR — server from third_party/sglang-kmeans.
# For BF16 / INT4 / BDR primary accuracy use: ./scripts/run_primary_eval_matrix.sh
#
# Client: open-source simple-evals https://github.com/openai/simple-evals
#
# Usage:
#   CENTROIDS=/path/to/centroids ./scripts/run_eval_matrix.sh <METHOD>
# Methods: kmeans | kmeans_bdr
#
# Optional env:
#   MODEL_PATH, PORT, N_CLUSTERS, CENTROIDS (required), SIMPLE_EVALS_DIR,
#   PREFILL_ATTENTION_BACKEND (default fa3), DECODE_ATTENTION_BACKEND (default triton),
#   ROTATE_V (for kmeans_bdr, default 0)

set -euo pipefail
METHOD="${1:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KM="$ROOT/third_party/sglang-kmeans/python"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
PORT="${PORT:-30000}"
N_CLUSTERS="${N_CLUSTERS:-16}"
PREFILL_ATTENTION_BACKEND="${PREFILL_ATTENTION_BACKEND:-fa3}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-triton}"

if [[ -z "$METHOD" ]]; then
  echo "Usage: CENTROIDS=/path/to/centroids $0 <kmeans|kmeans_bdr>" >&2
  echo "Primary eval (bf16|int4|bdr|bdr_kv): ./scripts/run_primary_eval_matrix.sh" >&2
  exit 1
fi

if [[ "$METHOD" == "bf16" || "$METHOD" == "int4" || "$METHOD" == "bdr" || "$METHOD" == "bdr_kv" ]]; then
  echo "Primary methods belong on sglang-fast-rotation. Use:" >&2
  echo "  ./scripts/run_primary_eval_matrix.sh $METHOD" >&2
  exit 1
fi

echo "Repo root: $ROOT"
echo "SGLang (k-means fork, ablation) python: $KM"
echo "Model: $MODEL_PATH  Port: $PORT"
echo ""

case "$METHOD" in
  kmeans|kmeans_bdr)
    C="${CENTROIDS:-}"
    if [[ -z "$C" || ! -d "$C" ]]; then
      echo "For $METHOD set CENTROIDS=/path/to/centroid_dir (from tools/fit_kv_centroids.py)" >&2
      exit 1
    fi
    if [[ "$METHOD" == "kmeans" ]]; then
      cat <<EOF
export HADAMARD=0
export ROTATE_V=0
export N_CLUSTERS=$N_CLUSTERS
export SGLANG_KV_CENTROIDS_PATH="$C"
cd "$KM"
python -m sglang.launch_server \
  --prefill-attention-backend "$PREFILL_ATTENTION_BACKEND" \
  --decode-attention-backend "$DECODE_ATTENTION_BACKEND" \
  --model-path "$MODEL_PATH" --port "$PORT" --kv-cache-dtype int4
EOF
    else
      cat <<EOF
export HADAMARD=1
export ROTATE_V="${ROTATE_V:-0}"
export HADAMARD_ORDER="${HADAMARD_ORDER:-16}"
export N_CLUSTERS=$N_CLUSTERS
export SGLANG_KV_CENTROIDS_PATH="$C"
cd "$KM"
python -m sglang.launch_server \
  --prefill-attention-backend "$PREFILL_ATTENTION_BACKEND" \
  --decode-attention-backend "$DECODE_ATTENTION_BACKEND" \
  --model-path "$MODEL_PATH" --port "$PORT" --kv-cache-dtype int4
EOF
    fi
    ;;
  *)
    echo "Unknown method: $METHOD (expected kmeans|kmeans_bdr)" >&2
    exit 1
    ;;
esac

echo ""
SE="${SIMPLE_EVALS_DIR:-}"
if [[ -n "$SE" ]]; then
  SE_CMD="cd $(printf '%q' "$SE")"
else
  SE_CMD='cd /path/to/simple-evals   # install deps: pip install openai pandas requests jinja2 tqdm numpy'
fi
cat <<EOF
--- Client (open-source simple-evals, separate terminal) ---
# Same as primary: README.md#prepare  README.md#accuracy-primary
# https://github.com/openai/simple-evals/blob/main/README.md#running-the-evals
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY="dummy"
${SE_CMD}
EOF
if [[ -z "$SE" ]]; then
  echo "# Tip: export SIMPLE_EVALS_DIR=/abs/path/to/simple-evals before $0 to emit a concrete cd." >&2
fi
