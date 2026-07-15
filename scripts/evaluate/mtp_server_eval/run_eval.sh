#!/usr/bin/env bash
# Run the MTP/EAGLE acceptance + throughput eval (LiveCodeBench + AIME +
# GPQA-Diamond) against a running OpenAI-compatible speculative-decoding server.
#
#   BACKEND=sglang BASE_URL=http://127.0.0.1:8080 ./run_eval.sh
#   BACKEND=vllm   BASE_URL=http://127.0.0.1:8000 ./run_eval.sh
#   NUM_SAMPLES=50 MAX_TOKENS=8192 ./run_eval.sh
#   BENCHMARKS=aime,gpqa ./run_eval.sh          # subset
#
# BACKEND selects how acceptance is read (SGLang windowed gauges vs vLLM
# cumulative counters) -- see the README. Prepared data ships in data/; GPQA is
# a gated HF dataset, so to regenerate it run `hf auth login` first.
set -euo pipefail

cd "$(dirname "$0")"

BACKEND="${BACKEND:-vllm}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
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

# Auto-prepare any missing dataset files (needs internet; GPQA needs hf login).
for b in ${BENCHMARKS//,/ }; do
    case "$b" in
        aime)          f=aime.jsonl ;;
        gpqa)          f=gpqa_diamond.jsonl ;;
        livecodebench) f=livecodebench.jsonl ;;
        *)             continue ;;
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
    --output-dir "$RESULT_DIR"
