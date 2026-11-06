#!/usr/bin/env bash
# One-shot multi-GPU data regeneration: launch N independent single-GPU vLLM
# servers for a target model, then regenerate a conversations JSONL against them
# (round-robin, resumable). Handy for aligning a training set to a target's own
# outputs, and the only workable layout under CUDA forward-compat (NCCL-free).
#
# Required (env vars or edit below):
#   MODEL        target/verifier model path or HF id
#   INPUT_JSONL  local conversations JSONL to regenerate (see --input-jsonl)
#   OUTFILE      output JSONL path (failed rows -> $OUTFILE.errors.jsonl)
#
# Common options (env vars):
#   NUM_GPUS=8              GPUs / servers (default: all visible)
#   PORT_BASE=8001          first server port (uses PORT_BASE .. PORT_BASE+NUM_GPUS-1)
#   MAX_MODEL_LEN=8192      per-server context
#   CONCURRENCY=128         total in-flight requests (spread across servers)
#   MAX_TOKENS=2048         generation cap
#   GPU_MEM_UTIL=0.9        vLLM --gpu-memory-utilization
#   CONDA_ENV=gemma4-spec   conda env to activate (skipped if empty / no conda)
#   CUDA_COMPAT=/path       enable CUDA forward-compat (old <570 driver); see the
#                           gemma-4 README. Forces --enforce-eager, MAX_MODEL_LEN 4096.
#   SKIP_SOURCES / SOURCES  passed through to the client's --skip-sources / --sources
#   DRY_RUN=1               print the commands and exit (launches nothing)
#
# Example:
#   MODEL=/models/gemma-4-31B-it \
#   INPUT_JSONL=/data/kimi/train.jsonl \
#   OUTFILE=/data/kimi/train_regen.jsonl \
#   NUM_GPUS=8 CONDA_ENV=gemma4-spec CUDA_COMPAT=/opt/cuda-compat-13.0 \
#   bash scripts/response_regeneration/run_regen_multigpu.sh
set -uo pipefail

MODEL="${MODEL:?set MODEL=/path/to/target/model}"
INPUT_JSONL="${INPUT_JSONL:?set INPUT_JSONL=/path/to/conversations.jsonl}"
OUTFILE="${OUTFILE:?set OUTFILE=/path/to/output.jsonl}"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
PORT_BASE="${PORT_BASE:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
CONCURRENCY="${CONCURRENCY:-128}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
CONDA_ENV="${CONDA_ENV:-gemma4-spec}"
LOG_DIR="${LOG_DIR:-$(dirname "$OUTFILE")}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

[ "${NUM_GPUS:-0}" -ge 1 ] || { echo "NUM_GPUS must be >=1 (got '$NUM_GPUS')" >&2; exit 1; }

# --- activate conda env -----------------------------------------------------
if [ -n "$CONDA_ENV" ] && command -v conda > /dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# --- optional CUDA forward-compatibility (old <570 driver) ------------------
VLLM_EXTRA=""
if [ -n "${CUDA_COMPAT:-}" ]; then
    PREFIX="${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}"
    SP="$("$PREFIX/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    export LD_LIBRARY_PATH="$PREFIX/lib:$CUDA_COMPAT:${LD_LIBRARY_PATH:-}"
    export PATH="$SP/ninja/data/bin:$SP/nvidia/cu13/bin:$PATH"
    export CUDA_HOME="$SP/nvidia/cu13"
    export PYTHONNOUSERSITE=1 VLLM_USE_FLASHINFER_SAMPLER=0
    MAX_MODEL_LEN="${COMPAT_MAX_MODEL_LEN:-4096}"
    VLLM_EXTRA="--enforce-eager"
    echo "[regen] CUDA forward-compat on (single-GPU, --enforce-eager, max_model_len=$MAX_MODEL_LEN)"
fi

mkdir -p "$LOG_DIR" "$(dirname "$OUTFILE")"

# --- build endpoint list ----------------------------------------------------
ENDPOINTS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    ENDPOINTS+=("http://127.0.0.1:$((PORT_BASE + i))/v1/chat/completions")
done

CLIENT=(python "$REPO/scripts/response_regeneration/script_multiendpoint.py"
    --input-jsonl "$INPUT_JSONL" --outfile "$OUTFILE" --model "$MODEL"
    --endpoint "${ENDPOINTS[@]}" --concurrency "$CONCURRENCY"
    --max-tokens "$MAX_TOKENS" --resume)
[ -n "${SKIP_SOURCES:-}" ] && CLIENT+=(--skip-sources "$SKIP_SOURCES")
[ -n "${SOURCES:-}" ] && CLIENT+=(--sources "$SOURCES")

if [ -n "${DRY_RUN:-}" ]; then
    echo "[dry-run] $NUM_GPUS servers on ports $PORT_BASE..$((PORT_BASE + NUM_GPUS - 1)):"
    echo "  CUDA_VISIBLE_DEVICES=<i> vllm serve $MODEL --port <p> --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEM_UTIL $VLLM_EXTRA"
    echo "[dry-run] client:"; printf '  %q ' "${CLIENT[@]}"; echo
    exit 0
fi

# --- launch N single-GPU servers -------------------------------------------
PIDS=()
cleanup() { echo "[regen] stopping servers"; for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

echo "[regen] starting $NUM_GPUS single-GPU servers (ports $PORT_BASE..$((PORT_BASE + NUM_GPUS - 1)))"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((PORT_BASE + i))
    CUDA_VISIBLE_DEVICES=$i vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --api-key "" \
        --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" $VLLM_EXTRA \
        > "$LOG_DIR/vllm_gpu${i}.log" 2>&1 &
    PIDS+=($!)
    sleep 3   # stagger concurrent weight loads
done

echo "[regen] waiting for all servers healthy..."
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((PORT_BASE + i))
    until curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; do
        kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "[regen] server gpu$i DIED:"; tail -25 "$LOG_DIR/vllm_gpu${i}.log"; exit 1; }
        sleep 5
    done
    echo "  gpu$i (port $PORT) ready"
done

echo "[regen] regenerating -> $OUTFILE (errors -> $OUTFILE.errors.jsonl)"
"${CLIENT[@]}"
echo "[regen] done -> $OUTFILE"
