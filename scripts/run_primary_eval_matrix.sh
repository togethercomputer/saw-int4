#!/usr/bin/env bash
# Primary evaluation: BF16 / INT4 / BDR — server from sglang-fast-rotation only.
# Client: open-source simple-evals https://github.com/openai/simple-evals
#
# Usage:
#   ./scripts/run_primary_eval_matrix.sh <METHOD>
# Methods: bf16 | int4 | bdr | bdr_kv
#
# Optional env: MODEL_PATH, PORT, SIMPLE_EVALS_DIR,
#   PREFILL_ATTENTION_BACKEND (default fa3), DECODE_ATTENTION_BACKEND (default triton)

set -euo pipefail
METHOD="${1:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FR="$ROOT/third_party/sglang-fast-rotation/python"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
PORT="${PORT:-30000}"
PREFILL_ATTENTION_BACKEND="${PREFILL_ATTENTION_BACKEND:-fa3}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-triton}"

if [[ -z "$METHOD" ]]; then
  echo "Usage: $0 <bf16|int4|bdr|bdr_kv>" >&2
  echo "For k-means ablations use: ./scripts/run_eval_matrix.sh kmeans|kmeans_bdr" >&2
  exit 1
fi

echo "Repo root: $ROOT"
echo "SGLang (fast-rotation fork) python: $FR"
echo "Model: $MODEL_PATH  Port: $PORT"
echo ""

case "$METHOD" in
  bf16)
    cat <<EOF
unset HADAMARD ROTATE_V HADAMARD_ORDER SGLANG_KV_CENTROIDS_PATH N_CLUSTERS || true
export HADAMARD=0
export ROTATE_V=0
cd "$FR"
python -m sglang.launch_server \
  --prefill-attention-backend "$PREFILL_ATTENTION_BACKEND" \
  --decode-attention-backend "$DECODE_ATTENTION_BACKEND" \
  --model-path "$MODEL_PATH" --port "$PORT" --kv-cache-dtype auto
EOF
    ;;
  int4)
    cat <<EOF
export HADAMARD=0
export ROTATE_V=0
unset SGLANG_KV_CENTROIDS_PATH || true
cd "$FR"
python -m sglang.launch_server \
  --prefill-attention-backend "$PREFILL_ATTENTION_BACKEND" \
  --decode-attention-backend "$DECODE_ATTENTION_BACKEND" \
  --model-path "$MODEL_PATH" --port "$PORT" --kv-cache-dtype int4
EOF
    ;;
  bdr)
    cat <<EOF
export HADAMARD=1
export ROTATE_V=0
export HADAMARD_ORDER="${HADAMARD_ORDER:-16}"
unset SGLANG_KV_CENTROIDS_PATH || true
cd "$FR"
python -m sglang.launch_server \
  --prefill-attention-backend "$PREFILL_ATTENTION_BACKEND" \
  --decode-attention-backend "$DECODE_ATTENTION_BACKEND" \
  --model-path "$MODEL_PATH" --port "$PORT" --kv-cache-dtype int4
EOF
    ;;
  bdr_kv)
    cat <<EOF
export HADAMARD=1
export ROTATE_V=1
export HADAMARD_ORDER="${HADAMARD_ORDER:-16}"
unset SGLANG_KV_CENTROIDS_PATH || true
cd "$FR"
python -m sglang.launch_server \
  --prefill-attention-backend "$PREFILL_ATTENTION_BACKEND" \
  --decode-attention-backend "$DECODE_ATTENTION_BACKEND" \
  --model-path "$MODEL_PATH" --port "$PORT" --kv-cache-dtype int4
EOF
    ;;
  *)
    echo "Unknown method: $METHOD (expected bf16|int4|bdr|bdr_kv)" >&2
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
