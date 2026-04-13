#!/usr/bin/env bash
# Print server environment and launch hints for each accuracy method.
# Accuracy: third_party/sglang-kmeans server + open-source simple-evals client
# https://github.com/openai/simple-evals (not tore-eval).
#
# Usage:
#   ./scripts/run_eval_matrix.sh <METHOD>
# Methods: bf16 | int4 | bdr | bdr_kv | kmeans | kmeans_bdr
#
# Optional env:
#   MODEL_PATH         (default Qwen/Qwen3-8B)
#   PORT               (default 30000)
#   CENTROIDS          directory for SGLANG_KV_CENTROIDS_PATH (kmeans*)
#   N_CLUSTERS         (default 16)
#   SIMPLE_EVALS_DIR   path to your simple-evals clone (printed in client block)

set -euo pipefail
METHOD="${1:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KM="$ROOT/third_party/sglang-kmeans/python"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
PORT="${PORT:-30000}"
N_CLUSTERS="${N_CLUSTERS:-16}"

if [[ -z "$METHOD" ]]; then
  echo "Usage: $0 <bf16|int4|bdr|bdr_kv|kmeans|kmeans_bdr>" >&2
  exit 1
fi

echo "Repo root: $ROOT"
echo "SGLang (k-means fork) python: $KM"
echo "Model: $MODEL_PATH  Port: $PORT"
echo ""

case "$METHOD" in
  bf16)
    cat <<EOF
unset HADAMARD ROTATE_V HADAMARD_ORDER SGLANG_KV_CENTROIDS_PATH N_CLUSTERS || true
export HADAMARD=0
export ROTATE_V=0
cd "$KM"
python -m sglang.launch_server --model-path "$MODEL_PATH" --port "$PORT" --kv-cache-dtype auto
EOF
    ;;
  int4)
    cat <<EOF
export HADAMARD=0
export ROTATE_V=0
unset SGLANG_KV_CENTROIDS_PATH || true
cd "$KM"
python -m sglang.launch_server --model-path "$MODEL_PATH" --port "$PORT" --kv-cache-dtype int4
EOF
    ;;
  bdr)
    cat <<EOF
export HADAMARD=1
export ROTATE_V=0
export HADAMARD_ORDER="${HADAMARD_ORDER:-16}"
unset SGLANG_KV_CENTROIDS_PATH || true
cd "$KM"
python -m sglang.launch_server --model-path "$MODEL_PATH" --port "$PORT" --kv-cache-dtype int4
EOF
    ;;
  bdr_kv)
    cat <<EOF
export HADAMARD=1
export ROTATE_V=1
export HADAMARD_ORDER="${HADAMARD_ORDER:-16}"
unset SGLANG_KV_CENTROIDS_PATH || true
cd "$KM"
python -m sglang.launch_server --model-path "$MODEL_PATH" --port "$PORT" --kv-cache-dtype int4
EOF
    ;;
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
python -m sglang.launch_server --model-path "$MODEL_PATH" --port "$PORT" --kv-cache-dtype int4
EOF
    else
      cat <<EOF
export HADAMARD=1
export ROTATE_V="${ROTATE_V:-0}"
export HADAMARD_ORDER="${HADAMARD_ORDER:-16}"
export N_CLUSTERS=$N_CLUSTERS
export SGLANG_KV_CENTROIDS_PATH="$C"
cd "$KM"
python -m sglang.launch_server --model-path "$MODEL_PATH" --port "$PORT" --kv-cache-dtype int4
EOF
    fi
    ;;
  *)
    echo "Unknown method: $METHOD" >&2
    exit 1
    ;;
esac

echo ""
SE="${SIMPLE_EVALS_DIR:-}"
if [[ -n "$SE" ]]; then
  SE_CMD="cd $(printf '%q' "$SE")"
else
  SE_CMD='cd /path/to/simple-evals   # git clone https://github.com/openai/simple-evals.git && pip install -e .'
fi
cat <<EOF
--- Client (open-source simple-evals, separate terminal) ---
# https://github.com/openai/simple-evals
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY="dummy"
${SE_CMD}
python -m simple-evals.simple_evals --list-models
python -m simple-evals.simple_evals --model <model_id> --examples 200
EOF
if [[ -z "$SE" ]]; then
  echo "# Tip: export SIMPLE_EVALS_DIR=/abs/path/to/simple-evals before $0 to emit a concrete cd." >&2
fi
