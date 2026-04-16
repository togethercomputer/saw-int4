#!/usr/bin/env bash
# Run GPQA with openai/simple-evals against an already-running SGLang server.
#
# Usage:
#   SIMPLE_EVALS_MODEL=<registered_simple_evals_model> ./scripts/run_primary_eval_matrix.sh
#
# Optional env:
#   SIMPLE_EVALS_DIR  (default: <repo>/third_party/simple-evals)
#   SIMPLE_EVALS_MODEL  required; model key registered in simple-evals
#   OPENAI_BASE_URL  (default: http://127.0.0.1:30000/v1)
#   OPENAI_API_KEY   (default: dummy)
#   GPQA_EXAMPLES    optional --examples value for debugging
#   GPQA_N_REPEATS   optional --n-repeats override
#   GPQA_DEBUG       set to 1 to pass --debug
#   PYTHON_BIN       (default: python)
#
# Any extra CLI args are forwarded to simple-evals.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SE="${SIMPLE_EVALS_DIR:-$ROOT/third_party/simple-evals}"
MODEL="${SIMPLE_EVALS_MODEL:-}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:30000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "$MODEL" ]]; then
  cat >&2 <<EOF
Set SIMPLE_EVALS_MODEL to the model key you registered in simple-evals.

Example:
  SIMPLE_EVALS_MODEL=qwen3_4b_sglang ./scripts/run_primary_eval_matrix.sh
EOF
  exit 1
fi

if [[ ! -d "$SE" ]]; then
  cat >&2 <<EOF
simple-evals directory not found: $SE

Initialize it first:
  git submodule update --init --checkout third_party/simple-evals
  cd third_party/simple-evals
  pip install openai pandas requests jinja2 tqdm numpy
EOF
  exit 1
fi

echo "Repo root: $ROOT"
echo "simple-evals dir: $SE"
echo "simple-evals model: $MODEL"
echo "OpenAI-compatible endpoint: $OPENAI_BASE_URL"
echo ""

export OPENAI_BASE_URL
export OPENAI_API_KEY

CMD=(
  "$PYTHON_BIN" simple_evals.py
  --model "$MODEL"
  --eval gpqa
)

if [[ -n "${GPQA_EXAMPLES:-}" ]]; then
  CMD+=(--examples "$GPQA_EXAMPLES")
fi

if [[ -n "${GPQA_N_REPEATS:-}" ]]; then
  CMD+=(--n-repeats "$GPQA_N_REPEATS")
fi

if [[ "${GPQA_DEBUG:-0}" == "1" ]]; then
  CMD+=(--debug)
fi

CMD+=("$@")

cd "$SE"
echo "Running: ${CMD[*]}"
"${CMD[@]}"
