#!/bin/bash
# Regenerate the kimi-mtp-dataset answers with google/gemma-4-31B-it, using ALL
# GPUs. gemma-4-31B (~62.6GB) fits on one 80GB GPU, so the highest-throughput
# setup is one single-GPU vLLM server per GPU; the regenerator round-robins
# requests across all of them.
#
#   bash scripts/response_regeneration/regenerate_gemma4_31b_all_gpus.sh
#
# Keeps the prompts (human/system turns) and replaces every assistant turn with
# Gemma-4's output, so the corpus is aligned to YOUR target (not Kimi-K2.5).
# Multimodal (llava_instruct) and tool-call (continual_tool_kimi) rows are
# skipped by default (see --skip-sources).
#
# Requires a working vLLM + Gemma-4 env (transformers>=5.x, CUDA driver that runs
# the installed torch). Run inside tmux — a full regen of ~350k text rows is long.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-gemma4-spec}"
if command -v conda > /dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# --- CUDA-13 forward-compat: run the cu13 vLLM/torch on this box's 565 (CUDA 12.7)
# driver. NCCL is broken under the compat driver, so we run ONE SINGLE-GPU vLLM
# server per GPU (TP=1, DP=1, --enforce-eager) -- no NCCL, and still uses all GPUs.
COMPAT_DIR="${COMPAT_DIR:-/import/ml-sc-scratch1/chenw/cuda-compat-13.0}"
if [[ -n "${CONDA_PREFIX:-}" && -d "$COMPAT_DIR" ]]; then
    SP="$CONDA_PREFIX/lib/python3.10/site-packages"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$COMPAT_DIR:${LD_LIBRARY_PATH:-}"
    export CUDA_HOME="$SP/nvidia/cu13"
    export PATH="$SP/ninja/data/bin:$CUDA_HOME/bin:$PATH"
    export PYTHONNOUSERSITE=1
    export VLLM_USE_FLASHINFER_SAMPLER=0
fi

# ============ Configuration ============
MODEL="${MODEL:-/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it}"
INPUT="${INPUT:-/import/ml-sc-scratch5/chenw/datasets/kimi-mtp-dataset/data/train-00000-of-00001.jsonl}"
OUTFILE="${OUTFILE:-./output/kimi_regen_gemma4_31b.jsonl}"
NUM_GPUS="${NUM_GPUS:-8}"
BASE_PORT="${BASE_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# Total async workers, spread round-robin over the NUM_GPUS endpoints
# (CONCURRENCY/NUM_GPUS in flight per server). 16/server is a sane A100-80GB start.
CONCURRENCY="${CONCURRENCY:-$((NUM_GPUS * 16))}"
LIMIT="${LIMIT:-}"                 # optional: cap rows for a smoke test
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$(dirname "$OUTFILE")"
# =======================================

PIDS=()
ENDPOINTS=()
cleanup() {
    echo "Stopping ${#PIDS[@]} vLLM servers..."
    for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Launching $NUM_GPUS single-GPU Gemma-4 servers ==="
for ((g = 0; g < NUM_GPUS; g++)); do
    port=$((BASE_PORT + g))
    logf="$(dirname "$OUTFILE")/vllm_gpu${g}.log"
    echo "  GPU $g -> port $port (log: $logf)"
    CUDA_VISIBLE_DEVICES="$g" vllm serve "$MODEL" \
        --port "$port" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --enforce-eager \
        > "$logf" 2>&1 &
    PIDS+=("$!")
    ENDPOINTS+=("http://127.0.0.1:${port}/v1/chat/completions")
done

echo "=== Waiting for all servers to become healthy (weight load is slow) ==="
for ((g = 0; g < NUM_GPUS; g++)); do
    port=$((BASE_PORT + g))
    until curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; do
        # bail if that server died during startup
        kill -0 "${PIDS[$g]}" 2>/dev/null || {
            echo "!! server on GPU $g died; see $(dirname "$OUTFILE")/vllm_gpu${g}.log" >&2
            exit 1
        }
        sleep 5
    done
    echo "  GPU $g ready"
done

echo "=== Regenerating with Gemma-4 across ${NUM_GPUS} endpoints (concurrency=$CONCURRENCY) ==="
LIMIT_ARG=()
[[ -n "$LIMIT" ]] && LIMIT_ARG=(--limit "$LIMIT")
python "$SCRIPT_DIR/script.py" \
    --input-jsonl "$INPUT" \
    --endpoint "${ENDPOINTS[@]}" \
    --model "$MODEL" \
    --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_TOKENS" \
    --outfile "$OUTFILE" \
    --resume \
    "${LIMIT_ARG[@]}"

echo "=== Done. Regenerated data -> $OUTFILE ==="
echo "Use it for training: --data $OUTFILE"
