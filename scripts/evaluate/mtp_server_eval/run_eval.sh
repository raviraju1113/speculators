#!/usr/bin/env bash
# Unified eval against a running OpenAI-compatible speculative-decoding server.
#
# Modes (MODE=...):
#   acceptance  (default)  sequential per-benchmark eval (vLLM or SGLang)
#   throughput             GuideLLM max-rate run + acceptance.csv
#   sweep                  GuideLLM gen-len estimate + rate sweep + perf_results.csv
#
# GuideLLM flags (former evaluate.py) via env:
#   TARGET / BASE_URL, DATASET, SUBSETS, RESULT_DIR, MAX_CONCURRENCY,
#   MAX_REQUESTS, MAX_TOKENS, GEN_LEN_RATE, SWEEP_RATE, GEN_KWARGS /
#   TEMPERATURE, DATA_COLUMN_MAPPER, SPEEDBENCH_DATA_DIR
# Unset RESULT_DIR → auto <model>_TIMESTAMP (same as evaluate.py).
#
# Acceptance:
#   BACKEND=sglang BASE_URL=http://127.0.0.1:8080 ./run_eval.sh
#   NUM_SAMPLES=50 MAX_TOKENS=8192 BENCHMARKS=aime,gpqa ./run_eval.sh
#
# GuideLLM:
#   MODE=throughput BASE_URL=http://127.0.0.1:8000 ./run_eval.sh
#   MODE=sweep SUBSETS=HumanEval,qa MAX_REQUESTS=80 ./run_eval.sh
#   MODE=throughput DATASET=speedbench/qualitative ./run_eval.sh
#
# Extra args after env vars are forwarded to the underlying Python CLI.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

MODE="${MODE:-acceptance}"
BACKEND="${BACKEND:-vllm}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

guidellm_eval() {
    local mode="$1"
    shift
    local target="${TARGET:-${BASE_URL%/}/v1}"
    if [[ "$target" != */v1 ]]; then
        target="${target%/}/v1"
    fi
    local dataset="${DATASET:-}"
    local speedbench_dir="${SPEEDBENCH_DATA_DIR:-}"
    if [[ "$dataset" == speedbench/* && -z "$speedbench_dir" ]]; then
        speedbench_dir="$HERE/../speedbench_data"
    fi
    local -a cmd=(
        python "$HERE/run_guidellm_eval.py"
        "$mode"
        --target "$target"
    )
    if [[ -n "${RESULT_DIR:-}" ]]; then
        mkdir -p "$RESULT_DIR"
        cmd+=(--output-dir "$RESULT_DIR")
    fi
    if [[ -n "$dataset" ]]; then
        cmd+=(--dataset "$dataset")
    fi
    if [[ -n "${SUBSETS:-}" ]]; then
        cmd+=(--subsets "$SUBSETS")
    fi
    if [[ -n "${MAX_CONCURRENCY:-}" ]]; then
        cmd+=(--max-concurrency "$MAX_CONCURRENCY")
    fi
    if [[ -n "${MAX_REQUESTS:-}" ]]; then
        cmd+=(--max-requests "$MAX_REQUESTS")
    fi
    if [[ -n "${MAX_TOKENS:-}" ]]; then
        cmd+=(--max-tokens "$MAX_TOKENS")
    fi
    if [[ -n "${GEN_LEN_RATE:-}" ]]; then
        cmd+=(--gen-len-rate "$GEN_LEN_RATE")
    fi
    if [[ -n "${SWEEP_RATE:-}" ]]; then
        cmd+=(--sweep-rate "$SWEEP_RATE")
    fi
    local gen_kwargs="${GEN_KWARGS:-}"
    if [[ -z "$gen_kwargs" && -n "${TEMPERATURE:-}" ]]; then
        gen_kwargs="{\"temperature\": ${TEMPERATURE}}"
    fi
    if [[ -n "$gen_kwargs" ]]; then
        cmd+=(--gen-kwargs "$gen_kwargs")
    fi
    if [[ -n "${DATA_COLUMN_MAPPER:-}" ]]; then
        cmd+=(--data-column-mapper "$DATA_COLUMN_MAPPER")
    fi
    if [[ -n "$speedbench_dir" ]]; then
        cmd+=(--speedbench-data-dir "$speedbench_dir")
    fi
    echo "==> [guidellm/$mode] eval against $target"
    exec "${cmd[@]}" "$@"
}

case "$MODE" in
    throughput|sweep)
        guidellm_eval "$MODE" "$@"
        ;;
    acceptance) ;;
    *)
        echo "MODE must be acceptance, throughput, or sweep (got '$MODE')" >&2
        exit 1
        ;;
esac

NUM_SAMPLES="${NUM_SAMPLES:-20}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BENCHMARKS="${BENCHMARKS:-aime,gpqa,livecodebench}"
RESULT_DIR="${RESULT_DIR:-./results}"

case "$BACKEND" in
    sglang) EVAL=run_sglang_eval.py ;;
    vllm)   EVAL=run_vllm_eval.py ;;
    *) echo "BACKEND must be 'sglang' or 'vllm' (got '$BACKEND')" >&2; exit 1 ;;
esac

mkdir -p "$RESULT_DIR"

# Auto-prepare any missing dataset files (needs internet for some; GPQA needs
# hf login; SPEED-Bench needs SPEEDBENCH_DIR from ../prepare_speedbench.py).
for b in ${BENCHMARKS//,/ }; do
    case "$b" in
        aime)                 f=aime.jsonl ;;
        gpqa)                 f=gpqa_diamond.jsonl ;;
        livecodebench)        f=livecodebench.jsonl ;;
        gsm8k|math500|humaneval|mbpp|mt-bench|aime26|swe-bench-pro|swe-rebench|aa-lcr) f="$b.jsonl" ;;
        aa-lcr-1k|aa-lcr-2k|aa-lcr-4k|aa-lcr-8k|aa-lcr-16k|aa-lcr-32k) f="$b.jsonl" ;;
        speed-coding|speed-humanities|speed-math|speed-multilingual|speed-qa|speed-rag) f="$b.jsonl" ;;
        speed-reasoning|speed-roleplay|speed-stem|speed-summarization|speed-writing|speed-low-entropy) f="$b.jsonl" ;;
        HumanEval|math_reasoning|qa|question|rag|summarization|tool_call|translation|writing) f="$b.jsonl" ;;
        *)                    echo "warning: unknown benchmark '$b' (eval will skip it)" >&2; continue ;;
    esac
    if [[ ! -f "data/$f" ]]; then
        echo "==> data/$f missing; preparing $b ..."
        python prepare_data.py --only "$b" || true
    fi
done

echo "==> [$BACKEND] eval against $BASE_URL ($BENCHMARKS, $NUM_SAMPLES/bench, temp=$TEMPERATURE)"
exec python "$EVAL" \
    --base-url "$BASE_URL" \
    --benchmarks "$BENCHMARKS" \
    --num-samples "$NUM_SAMPLES" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --output-dir "$RESULT_DIR" \
    "$@"
