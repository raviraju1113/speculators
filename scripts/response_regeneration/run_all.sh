#!/bin/bash
#
# Run the complete response regeneration pipeline:
# start a vLLM server (with optional data/tensor parallelism), regenerate
# responses for the dataset, and stop the server.
#
# Usage examples:
#   ./run_all.sh --model "meta-llama/Llama-3.3-70B-Instruct" --dataset magpie --limit 100
#   ./run_all.sh --model "Qwen/Qwen2.5-72B-Instruct" --dp-size 4 --tp-size 2 --dataset magpie
#   ./run_all.sh --model "Qwen/Qwen2.5-72B-Instruct" --gpus 0,1,2,4 --tp-size 4 --dataset magpie
#   ./run_all.sh --model "Qwen/Qwen2.5-72B-Instruct" --dataset magpie --keep-server
#   ./run_all.sh --model "Qwen/Qwen3-8B" --tool-call-parser hermes --dataset hermes-fc
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Directory the user invoked the script from (before any `cd`). This is where
# outputs land by default so they're easy to find and persist with your work.
INVOCATION_DIR="$PWD"

# The vLLM log and regenerated output need a writable destination. Default to
# the current directory the user ran from; honor an explicit override first.
# Inside the training container /workspace/speculators is a read-only baked-in
# image layer, so fall back to the auto-mounted (and host-visible) $SLURM_TMPDIR
# then /tmp only if the current dir isn't writable.
pick_writable_dir() {
    for d in "$RESPONSE_REGEN_OUT_DIR" "$INVOCATION_DIR" "$SLURM_TMPDIR" /tmp "$SCRIPT_DIR"; do
        [ -n "$d" ] || continue
        if ( t="$d/.wtest.$$"; touch "$t" 2>/dev/null ); then
            rm -f "$d/.wtest.$$"
            printf '%s\n' "$d"
            return 0
        fi
    done
    printf '%s\n' "$SCRIPT_DIR"
}
OUT_DIR="$(pick_writable_dir)"
LOG_FILE="$OUT_DIR/vllm_server.log"

# Defaults
PORT=8000
MODEL=""
DP_SIZE=""
TP_SIZE=""
MAX_MODEL_LEN=""
REASONING_PARSER=""
GPU_MEM_UTIL=""
GPUS=""
KEEP_SERVER=false

# Parse arguments
PYTHON_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --dp-size)
            DP_SIZE="$2"
            shift 2
            ;;
        --tp-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --reasoning-parser)
            REASONING_PARSER="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEM_UTIL="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            PYTHON_ARGS+=("--model" "$2")
            shift 2
            ;;
        --keep-server)
            KEEP_SERVER=true
            shift
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --ports)
            echo "Error: $1 has been removed. Use --dp-size and --tp-size instead."
            echo "  Migration: --ports '8000,8001' becomes --dp-size 2"
            exit 1
            ;;
        *)
            PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL" ]; then
    echo "Error: --model is required."
    echo "Usage: $0 --model MODEL [--gpus GPUS] [--dp-size N] [--tp-size N] [--max-model-len N] [--reasoning-parser PARSER] [--gpu-memory-utilization F] [--dataset DATASET] [...]"
    exit 1
fi

# Default the context length. Models like Gemma expose enormous max positions
# (e.g. 262144); vLLM sizes the KV cache to fit ONE request at that length, and
# refuses to start if it can't. Response regeneration never needs that much, so
# cap it unless the caller overrides via --max-model-len or the env var. This
# keeps startup robust to a little GPU-memory noise.
if [ -z "$MAX_MODEL_LEN" ]; then
    MAX_MODEL_LEN="${RESPONSE_REGEN_MAX_MODEL_LEN:-32768}"
    echo "Note: no --max-model-len given; defaulting to $MAX_MODEL_LEN"
    echo "  (override with --max-model-len N or RESPONSE_REGEN_MAX_MODEL_LEN=N)"
fi

# Build vllm serve command
VLLM_CMD=(vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --api-key "")
[ -n "$DP_SIZE" ] && VLLM_CMD+=(--data-parallel-size "$DP_SIZE")
[ -n "$TP_SIZE" ] && VLLM_CMD+=(--tensor-parallel-size "$TP_SIZE")
[ -n "$MAX_MODEL_LEN" ] && VLLM_CMD+=(--max-model-len "$MAX_MODEL_LEN")
[ -n "$REASONING_PARSER" ] && VLLM_CMD+=(--reasoning-parser "$REASONING_PARSER")
[ -n "$GPU_MEM_UTIL" ] && VLLM_CMD+=(--gpu-memory-utilization "$GPU_MEM_UTIL")

# Surface the real root cause from the vLLM log. The fatal error (e.g. a
# ValueError about GPU memory) is often emitted by the EngineCore worker many
# lines before the generic "Engine core initialization failed" wrapper that
# ends up at the tail of the log, so a plain `tail` hides it.
report_log_error() {
    local log="$1"
    echo "  --- Likely root cause (matched error lines) ---"
    if ! grep -nE "Error|ERROR|Exception|raise [A-Z]|assert|CUDA|out of memory|OOM" "$log" \
            | grep -viE "otel\.py|sync_wrapper|core\.py:1195\] *(File|return|\^| *$)" \
            | tail -15; then
        echo "  (no obvious error lines found)"
    fi
    echo "  --- Last 20 lines of log ---"
    tail -20 "$log"
    echo "  --- Full log: $log ---"
}

# Cleanup function.
# vLLM spawns EngineCore + Worker processes as separate children. Killing only
# the `vllm serve` parent PID orphans those workers, which keep holding GPU
# memory and poison the next run's KV-cache sizing. We launch the server via
# `setsid` (below) so it leads its own process group, and here we signal the
# WHOLE group (negative PID) to reap every child.
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (process group $VLLM_PID)..."
        kill -- "-$VLLM_PID" 2>/dev/null || kill "$VLLM_PID" 2>/dev/null
        sleep 3
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "Force killing vLLM server..."
            kill -9 -- "-$VLLM_PID" 2>/dev/null || kill -9 "$VLLM_PID" 2>/dev/null
        fi
    fi
}

echo "========================================="
echo "Response Regeneration Pipeline"
echo "========================================="
echo ""

# Step 1: Start server
echo "Step 1: Starting vLLM server on port $PORT"
echo "  Model: $MODEL"
[ -n "$GPUS" ] && echo "  GPUs: $GPUS"
[ -n "$DP_SIZE" ] && echo "  Data parallel size: $DP_SIZE"
[ -n "$TP_SIZE" ] && echo "  Tensor parallel size: $TP_SIZE"
[ -n "$MAX_MODEL_LEN" ] && echo "  Max model len: $MAX_MODEL_LEN"
[ -n "$REASONING_PARSER" ] && echo "  Reasoning parser: $REASONING_PARSER"
[ -n "$GPU_MEM_UTIL" ] && echo "  GPU memory utilization: $GPU_MEM_UTIL"
echo "  Command: ${VLLM_CMD[*]}"
echo ""

# Pre-flight: refuse to launch on dirty GPUs. Leftover vLLM workers from a
# previous run keep holding GPU memory; vLLM measures device-level used memory
# to size the KV cache, so the leftovers get counted against this run and cause
# a spurious "not enough KV cache" ValueError deep into a multi-minute load.
# Fail fast (or auto-clean with RESPONSE_REGEN_KILL_STALE=1) instead.
STALE_PIDS="$(pgrep -u "$(id -u)" -f 'vllm serve|VLLM::' 2>/dev/null || true)"
if [ -n "$STALE_PIDS" ]; then
    STALE_PIDS="$(echo "$STALE_PIDS" | tr '\n' ' ')"
    if [ "${RESPONSE_REGEN_KILL_STALE:-0}" = "1" ]; then
        echo "Killing stale vLLM processes: $STALE_PIDS"
        kill -9 $STALE_PIDS 2>/dev/null || true
        sleep 3
    else
        echo "Error: stale vLLM processes are still holding the GPUs (PIDs: $STALE_PIDS)."
        echo "  They starve the KV cache and cause a spurious out-of-memory failure."
        echo "  Kill them first:  kill -9 $STALE_PIDS"
        echo "  or re-run with RESPONSE_REGEN_KILL_STALE=1 to clean them up automatically."
        exit 1
    fi
fi

# Launch under `setsid` so the server leads its own process group and cleanup()
# can reap the entire EngineCore/Worker tree, not just the CLI parent.
echo "  Log: $LOG_FILE"
if [ -n "$GPUS" ]; then
    CUDA_VISIBLE_DEVICES="$GPUS" setsid "${VLLM_CMD[@]}" > "$LOG_FILE" 2>&1 &
else
    setsid "${VLLM_CMD[@]}" > "$LOG_FILE" 2>&1 &
fi
VLLM_PID=$!

# Set up cleanup trap (unless --keep-server)
if [ "$KEEP_SERVER" = false ]; then
    trap cleanup EXIT
fi

# Step 2: Health check
echo "Step 2: Waiting for server to be ready..."
echo "  (Large models may take several minutes to load)"
ENDPOINT="http://127.0.0.1:$PORT/v1/models"
MAX_RETRIES=300  # Up to 12s per retry (2s sleep + 10s curl timeout); large models may need time for compilation
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "  vLLM server process died."
        report_log_error "$LOG_FILE"
        exit 1
    fi
    if curl -s --connect-timeout 5 --max-time 10 "$ENDPOINT" > /dev/null 2>&1; then
        echo "  Server ready (after $RETRY retries)"
        break
    fi
    RETRY=$((RETRY + 1))
    if [ $RETRY -eq $MAX_RETRIES ]; then
        echo "  Server failed to start after $MAX_RETRIES retries"
        report_log_error "$LOG_FILE"
        exit 1
    fi
    [ $((RETRY % 5)) -eq 0 ] && echo "  Still waiting... ($RETRY retries)"
    sleep 2
done
echo ""

# Step 3: Run response regeneration
echo "Step 3: Running response regeneration..."
PYTHON_ARGS+=("--endpoint" "http://127.0.0.1:$PORT/v1/chat/completions")
echo "Arguments: ${PYTHON_ARGS[*]}"
# Run from $OUT_DIR so script.py's default (relative) --outfile lands somewhere
# writable rather than the read-only checkout. An explicit absolute --outfile
# still wins. Note: $SLURM_TMPDIR is node-local — copy results to persistent
# storage before the job ends.
echo "  Output dir: $OUT_DIR"
echo ""
( cd "$OUT_DIR" && python "$SCRIPT_DIR/script.py" "${PYTHON_ARGS[@]}" )
PYTHON_EXIT_CODE=$?
echo ""

# Step 4: Cleanup
if [ "$KEEP_SERVER" = true ]; then
    echo "Keeping vLLM server running (PID $VLLM_PID)"
    echo "Stop with: kill $VLLM_PID"
fi

echo "========================================="
echo "Pipeline complete!"
echo "========================================="

exit $PYTHON_EXIT_CODE
