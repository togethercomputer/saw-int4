#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git submodule update --init --recursive || {
  echo "Warning: recursive submodule init failed (optional nested deps inside a fork)." >&2
  echo "Initializing paper submodules only (accuracy uses open-source simple-evals, not nested eval repos):" >&2
  git submodule update --init third_party/sglang-fast-rotation third_party/sglang-kmeans
}
