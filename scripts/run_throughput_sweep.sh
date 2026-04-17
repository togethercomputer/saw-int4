#!/usr/bin/env bash
# scripts/run_throughput_sweep.sh — SGLang throughput benchmarking with genai-bench
#
# Sweeps BF16 / INT4 / BDR configs using sglang-fast-rotation servers and
# genai-bench as the load client. Multiple configs on non-overlapping GPUs
# launch in parallel.
#
# Usage:
#   bash scripts/run_throughput_sweep.sh
#
# Results: eval_speed/results/<timestamp>/<model>/<config_label>/
# Logs:    eval_speed/logs/<timestamp>/<model>/<config_label>.log
#
# Env overrides (set before running):
#   HF_HOME            (default: /data/jinda/huggingface)
#   TRITON_CACHE_DIR   (default: /tmp/triton-cache)
#   GPU_FREE_MEM_MB    (default: 500 — threshold to consider a GPU free)
#   GPU_POLL_INTERVAL  (default: 60s)

set -eo pipefail

export HF_HOME="${HF_HOME:-/data/jinda/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache}"

cleanup() {
    trap '' INT TERM
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Interrupted — killing all child processes..."
    kill -9 -- -$$ 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

# =============================================================================
# Benchmark parameters
# =============================================================================

# genai-bench traffic scenarios: D(input_tokens, output_tokens)
IFS=' ' read -r -a TRAFFIC_SCENARIOS <<< "${TRAFFIC_SCENARIOS_OVERRIDE:-D(8192,1024)}"

# Concurrency levels to sweep (one genai-bench run per concurrency)
IFS=' ' read -r -a CONCURRENCIES <<< "${CONCURRENCIES_OVERRIDE:-1 8 16 32 256}"

# Per-concurrency max requests (must align 1:1 with CONCURRENCIES above)
IFS=' ' read -r -a MAX_REQUESTS_PER_CONC <<< "${MAX_REQUESTS_PER_CONC_OVERRIDE:-16 16 16 32 256}"

# genai-bench run limits
MAX_TIME_PER_RUN="${MAX_TIME_PER_RUN:-20}"      # minutes

# GPU to consider "free" (MB used below this threshold)
GPU_FREE_MEM_MB="${GPU_FREE_MEM_MB:-500}"
GPU_POLL_INTERVAL="${GPU_POLL_INTERVAL:-60}"

# =============================================================================
# Model configs
# Format: "mode|hadamard|rotate_v|hadamard_order|kv_dtype|model_name|eval_gpus|tp_size"
#
#   mode          : BASE (BF16 or INT4, no rotation) | BDR (rotation + INT4)
#   hadamard      : 0 or 1
#   rotate_v      : 0 or 1
#   hadamard_order: e.g. 64, 128 (ignored for BASE)
#   kv_dtype      : BF16 | INT4
#   model_name    : HuggingFace model ID
#   eval_gpus     : comma-separated CUDA device IDs for this server
#   tp_size       : tensor-parallel size
# =============================================================================
MODEL_CONFIGS=(
    "BASE|0|0|0  |BF16|Qwen/Qwen3-8B|6,7|2"
    "BASE|0|0|0  |INT4|Qwen/Qwen3-8B|2,3|2"
    "BDR |1|0|128|INT4|Qwen/Qwen3-8B|4,5|2"
)

# =============================================================================
# Paths
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SGLANG_DIR="$REPO_ROOT/third_party/sglang-fast-rotation/python"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_BASE="$REPO_ROOT/eval_speed/results/$TIMESTAMP"
LOGS_BASE="$REPO_ROOT/eval_speed/logs/$TIMESTAMP"

BASE_PORT=30200   # configs get port BASE_PORT+i, genai-bench master BASE_PORT+100+i

CONDA_ENV_NAME="colm"
PYTHON="$(conda run -n "$CONDA_ENV_NAME" which python 2>/dev/null || echo python)"

# Resolve genai-bench binary in the same env
GENAI_BENCH="$(conda run -n "$CONDA_ENV_NAME" which genai-bench 2>/dev/null || which genai-bench)"

mkdir -p "$RESULTS_BASE" "$LOGS_BASE"

# =============================================================================
# Helpers
# =============================================================================
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

config_label() {
    local mode="$1" hadamard="$2" rotate_v="$3" hadamard_order="$4" kv_dtype="$5"
    local kv="${kv_dtype,,}"
    case "$mode" in
        BASE) echo "baseline_${kv}" ;;
        BDR)  echo "bdr_${kv}_h${hadamard}_rv${rotate_v}_ord${hadamard_order}" ;;
        *)    echo "unknown_${kv}" ;;
    esac
}

wait_for_server() {
    local port="$1" pid="$2" label="$3"
    local max_wait=1800 elapsed=0
    log "Waiting for $label on port $port..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            log "✓ $label ready (${elapsed}s)"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            log "✗ $label process died"
            return 1
        fi
        if [ $((elapsed % 60)) -eq 0 ] && [ $elapsed -gt 0 ]; then log "  Still waiting... ${elapsed}s"; fi
        sleep 5 & wait $!
        elapsed=$((elapsed + 5))
    done
    log "✗ $label timed out after ${max_wait}s"
    return 1
}

stop_server() {
    local pid="$1" label="$2"
    log "Stopping $label (PID $pid)..."
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    log "✓ $label stopped"
}

gpus_are_free() {
    local gpu_list="$1"
    IFS=',' read -ra IDS <<< "$gpu_list"
    for id in "${IDS[@]}"; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$id" 2>/dev/null | tr -d '[:space:]')
        [ -z "$used" ] && return 1
        [ "$used" -ge "$GPU_FREE_MEM_MB" ] && return 1
    done
    return 0
}

wait_for_gpus_free() {
    local gpu_list="$1" label="$2" waited=0
    while ! gpus_are_free "$gpu_list"; do
        if [ $((waited % 300)) -eq 0 ]; then log "Waiting for GPU(s) [$gpu_list] ($((waited/60))min) [$label]"; fi
        sleep "$GPU_POLL_INTERVAL" & wait $!
        waited=$((waited + GPU_POLL_INTERVAL))
    done
    if [ "$waited" -gt 0 ]; then log "GPU(s) [$gpu_list] free after $((waited/60))min"; fi
}

# =============================================================================
# benchmark_single_config
# =============================================================================
benchmark_single_config() {
    local mode="$1" hadamard="$2" rotate_v="$3" hadamard_order="$4" \
          kv_dtype="$5" model_name="$6" gpu_devices="$7" tp_size="$8" \
          server_port="$9" client_master_port="${10}"

    local label
    label="$(config_label "$mode" "$hadamard" "$rotate_v" "$hadamard_order" "$kv_dtype")"
    local model_short
    model_short="$(basename "$model_name")"

    local result_dir="$RESULTS_BASE/$model_short/$label"
    local log_dir="$LOGS_BASE/$model_short"
    mkdir -p "$result_dir" "$log_dir"
    local batch_log="$log_dir/${label}.log"

    # kv-cache-dtype flag
    local kv_cache_dtype
    case "$kv_dtype" in
        BF16) kv_cache_dtype="auto" ;;
        INT4) kv_cache_dtype="int4" ;;
        *)    kv_cache_dtype="auto" ;;
    esac

    # BASE: force rotation off
    if [[ "$mode" == "BASE" ]]; then hadamard=0; rotate_v=0; fi

    {
        log "=========================================="
        log "Label:     $label"
        log "Model:     $model_name"
        log "GPUs:      $gpu_devices  TP=$tp_size"
        log "KV dtype:  $kv_dtype (--kv-cache-dtype $kv_cache_dtype)"
        log "HADAMARD=$hadamard  ROTATE_V=$rotate_v  HADAMARD_ORDER=$hadamard_order"
        log "Port:      $server_port"
        log "Scenarios: ${TRAFFIC_SCENARIOS[*]}"
        log "Concurrency: ${CONCURRENCIES[*]}"
        log "Results:   $result_dir"
        log "=========================================="
    } | tee -a "$batch_log"

    # ------------------------------------------------------------------
    # Start SGLang server
    # ------------------------------------------------------------------
    local server_log="$log_dir/${label}_server.log"
    {
        log "Starting server..."
        log "Server command:"
        log "  HADAMARD=$hadamard ROTATE_V=$rotate_v HADAMARD_ORDER=$hadamard_order CUDA_VISIBLE_DEVICES=$gpu_devices \\"
        log "  $PYTHON -m sglang.launch_server \\"
        log "    --model-path \"$model_name\" \\"
        log "    --kv-cache-dtype $kv_cache_dtype \\"
        log "    --prefill-attention-backend fa3 \\"
        log "    --decode-attention-backend triton \\"
        log "    --tensor-parallel-size $tp_size \\"
        log "    --mem-fraction-static 0.8 \\"
        log "    --host 0.0.0.0 \\"
        log "    --port $server_port \\"
        log "    --trust-remote-code"
    } | tee -a "$batch_log"

    HADAMARD=$hadamard \
    ROTATE_V=$rotate_v \
    HADAMARD_ORDER=$hadamard_order \
    CUDA_VISIBLE_DEVICES=$gpu_devices \
    "$PYTHON" -m sglang.launch_server \
            --model-path "$model_name" \
            --kv-cache-dtype "$kv_cache_dtype" \
            --prefill-attention-backend fa3 \
            --decode-attention-backend triton \
            --tensor-parallel-size "$tp_size" \
            --mem-fraction-static 0.8 \
            --host 0.0.0.0 \
            --port "$server_port" \
            --trust-remote-code \
        > "$server_log" 2>&1 &
    local server_pid=$!
    log "Server PID: $server_pid" | tee -a "$batch_log"

    if ! wait_for_server "$server_port" "$server_pid" "$label" 2>&1 | tee -a "$batch_log"; then
        log "Server failed to start — last 30 lines of log:" | tee -a "$batch_log"
        tail -30 "$server_log" | tee -a "$batch_log"
        stop_server "$server_pid" "$label" 2>&1 | tee -a "$batch_log"
        return 1
    fi

    # ------------------------------------------------------------------
    # Warmup — small genai-bench run, results discarded
    # ------------------------------------------------------------------
    log "Warmup..." | tee -a "$batch_log"
    "$GENAI_BENCH" benchmark \
            --api-backend sglang \
            --api-base "http://127.0.0.1:${server_port}" \
            --api-key dummy \
            --api-model-name "$model_name" \
            --model-tokenizer "$model_name" \
            --task text-to-text \
            --traffic-scenario "D(128,64)" \
            --num-concurrency 4 \
            --max-time-per-run 2 \
            --max-requests-per-run 16 \
            --server-engine SGLang \
            --server-gpu-type H100 \
            --server-version custom \
            --server-gpu-count "$tp_size" \
            --experiment-base-dir /tmp \
            --experiment-folder-name warmup_discard \
            --master-port "$client_master_port" \
        >> "$batch_log" 2>&1 || true
    log "✓ Warmup done" | tee -a "$batch_log"

    # ------------------------------------------------------------------
    # Benchmark sweep — one genai-bench run per concurrency level so each
    # can have its own --max-requests-per-run limit
    # ------------------------------------------------------------------
    local scenario_args=()
    for sc in "${TRAFFIC_SCENARIOS[@]}"; do
        scenario_args+=(--traffic-scenario "$sc")
    done

    log "Running benchmark (${#CONCURRENCIES[@]} concurrency levels)..." | tee -a "$batch_log"
    local exit_code=0
    for ci in "${!CONCURRENCIES[@]}"; do
        local conc="${CONCURRENCIES[$ci]}"
        local max_req="${MAX_REQUESTS_PER_CONC[$ci]}"
        log "  conc=$conc  max_requests=$max_req" | tee -a "$batch_log"
        "$GENAI_BENCH" benchmark \
                --api-backend sglang \
                --api-base "http://127.0.0.1:${server_port}" \
                --api-key dummy \
                --api-model-name "$model_name" \
                --model-tokenizer "$model_name" \
                --task text-to-text \
                "${scenario_args[@]}" \
                --num-concurrency "$conc" \
                --max-time-per-run "$MAX_TIME_PER_RUN" \
                --max-requests-per-run "$max_req" \
                --server-engine SGLang \
                --server-gpu-type H100 \
                --server-version custom \
                --server-gpu-count "$tp_size" \
                --experiment-base-dir "$result_dir" \
                --experiment-folder-name "${model_short}_${label}_conc${conc}" \
                --master-port "$client_master_port" \
            2>&1 | tee -a "$batch_log" || { exit_code=$?; log "✗ conc=$conc failed (exit $exit_code)" | tee -a "$batch_log"; }
    done

    if [ $exit_code -ne 0 ]; then
        log "✗ Benchmark had failures (last exit $exit_code)" | tee -a "$batch_log"
    else
        log "✓ Benchmark complete — results in $result_dir" | tee -a "$batch_log"
    fi

    stop_server "$server_pid" "$label" 2>&1 | tee -a "$batch_log"
    return $exit_code
}

# =============================================================================
# Preflight checks
# =============================================================================
if [ ! -d "$SGLANG_DIR" ]; then
    echo "ERROR: $SGLANG_DIR not found — run: git submodule update --init third_party/sglang-fast-rotation"
    exit 1
fi

if ! "$PYTHON" -c "import sglang" 2>/dev/null; then
    echo "ERROR: sglang not importable via $PYTHON"
    exit 1
fi

if [ ! -x "$GENAI_BENCH" ]; then
    echo "ERROR: genai-bench not found — run: pip install genai-bench"
    exit 1
fi

# =============================================================================
# Main — parallel launch with GPU overlap detection
# =============================================================================
N=${#MODEL_CONFIGS[@]}
declare -a PIDS EXIT_CODES
declare -A CONFIG_LABELS

log "=========================================="
log "Throughput sweep — $N config(s)"
log "Scenarios:    ${TRAFFIC_SCENARIOS[*]}"
log "Concurrencies: ${CONCURRENCIES[*]}"
log "Max time/run: ${MAX_TIME_PER_RUN} min  Max requests/conc: ${MAX_REQUESTS_PER_CONC[*]}"
log "Results:      $RESULTS_BASE"
log "Logs:         $LOGS_BASE"
log "=========================================="

for i in "${!MODEL_CONFIGS[@]}"; do
    IFS='|' read -r mode hadamard rotate_v hadamard_order kv_dtype \
                    model_name gpu_devices tp_size <<< "${MODEL_CONFIGS[$i]}"
    # Strip whitespace
    mode="${mode// /}"; hadamard="${hadamard// /}"; rotate_v="${rotate_v// /}"
    hadamard_order="${hadamard_order// /}"; kv_dtype="${kv_dtype// /}"
    model_name="${model_name// /}"; gpu_devices="${gpu_devices// /}"; tp_size="${tp_size// /}"

    local_label="$(config_label "$mode" "$hadamard" "$rotate_v" "$hadamard_order" "$kv_dtype")"
    server_port=$((BASE_PORT + i))
    client_master_port=$((BASE_PORT + 100 + i))
    CONFIG_LABELS[$i]="$(basename "$model_name")/$local_label (gpu=$gpu_devices port=$server_port)"
    EXIT_CODES[$i]=-1

    log "[$((i+1))/$N] Waiting for GPU(s) [$gpu_devices]: ${CONFIG_LABELS[$i]}"
    wait_for_gpus_free "$gpu_devices" "${CONFIG_LABELS[$i]}"
    log "Launching: ${CONFIG_LABELS[$i]}"

    benchmark_single_config \
        "$mode" "$hadamard" "$rotate_v" "$hadamard_order" "$kv_dtype" \
        "$model_name" "$gpu_devices" "$tp_size" \
        "$server_port" "$client_master_port" &
    PIDS[$i]=$!

    # If next config overlaps GPUs, wait for this one to finish first
    next=$((i + 1))
    if [ "$next" -lt "$N" ]; then
        next_gpus=$(echo "${MODEL_CONFIGS[$next]}" | cut -d'|' -f7 | tr -d ' ')
        overlap=0
        IFS=',' read -ra CUR <<< "$gpu_devices"
        IFS=',' read -ra NXT <<< "$next_gpus"
        for cg in "${CUR[@]}"; do
            for ng in "${NXT[@]}"; do
                if [ "$cg" = "$ng" ]; then overlap=1; break 2; fi
            done
        done
        if [ "$overlap" -eq 1 ]; then
            log "Next config overlaps GPU(s) — waiting for current job to finish..."
            wait "${PIDS[$i]}"; EXIT_CODES[$i]=$?
            log "Cooling down 30s for GPU memory to release..."
            sleep 30 & wait $!
        else
            log "No GPU overlap — sleeping 30s before launching next config..."
            sleep 30 & wait $!
        fi
    fi
done

log "All configs launched — waiting for completion..."
OVERALL_EXIT=0
for i in "${!PIDS[@]}"; do
    if [ "${EXIT_CODES[$i]}" -eq -1 ]; then
        wait "${PIDS[$i]}"; EXIT_CODES[$i]=$?
    fi
    if [ "${EXIT_CODES[$i]}" -eq 0 ]; then
        log "✓ ${CONFIG_LABELS[$i]}"
    else
        log "✗ ${CONFIG_LABELS[$i]} (exit ${EXIT_CODES[$i]})"
        OVERALL_EXIT=1
    fi
done

log "Done. Results: $RESULTS_BASE  Exit: $OVERALL_EXIT"
exit $OVERALL_EXIT
