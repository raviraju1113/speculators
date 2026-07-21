#!/usr/bin/env bash
# Run GLM-5.2 MTP acceptance + throughput eval against a running vLLM server.
#
# This script is for manual / interactive use. It runs the MTP server eval
# (aime, gpqa, livecodebench) against a GLM-5.2 server with native MTP enabled.
#
# Usage:
#   # Against a server already running at BASE_URL (default: http://127.0.0.1:8000)
#   ./run_glm52_eval.sh
#
#   # Against a custom server
#   BASE_URL=http://my-server:8080 ./run_glm52_eval.sh
#
#   # Run all samples (no limit), only aime + livecodebench
#   NUM_SAMPLES=0 BENCHMARKS=aime,livecodebench ./run_glm52_eval.sh
#
# For automated experiment runs, use the YAML config + run_experiments.py instead.
set -euo pipefail

cd "$(dirname "$0")"

# --- defaults (override via env) -------------------------------------------
BACKEND="${BACKEND:-vllm}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BENCHMARKS="${BENCHMARKS:-aime,gpqa,livecodebench}"
RESULT_DIR="${RESULT_DIR:-./results/glm52}"

EVAL_DIR="$(cd "$(dirname "$0")" && cd .. && pwd)"      # mtp_server_eval/

case "$BACKEND" in
    sglang) EVAL_SCRIPT=run_sglang_eval.py ;;
    vllm)   EVAL_SCRIPT=run_vllm_eval.py ;;
    *) echo "BACKEND must be 'sglang' or 'vllm' (got '$BACKEND')" >&2; exit 1 ;;
esac

mkdir -p "$RESULT_DIR"

# Auto-prepare any missing dataset files (needs internet; GPQA needs hf login).
for b in ${BENCHMARKS//,/ }; do
    case "$b" in
        aime)          f=aime.jsonl ;;
        gpqa)          f=gpqa_diamond.jsonl ;;
        livecodebench) f=livecodebench.jsonl ;;
        gsm8k)         f=gsm8k.jsonl ;;
        math500)       f=math500.jsonl ;;
        humaneval)     f=humaneval.jsonl ;;
        mbpp)          f=mbpp.jsonl ;;
        *)             continue ;;
    esac
    if [[ ! -f "${EVAL_DIR}/data/$f" ]]; then
        echo "==> data/$f missing; preparing $b ..."
        python "${EVAL_DIR}/prepare_data.py" --only "$b" || true
    fi
done

echo "==> [$BACKEND] GLM-5.2 native MTP eval against $BASE_URL"
echo "    benchmarks=$BENCHMARKS, num_samples=$NUM_SAMPLES, max_tokens=$MAX_TOKENS, temp=$TEMPERATURE"
exec python "${EVAL_DIR}/${EVAL_SCRIPT}" \
    --base-url "$BASE_URL" \
    --benchmarks "$BENCHMARKS" \
    --num-samples "$NUM_SAMPLES" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --output-dir "$RESULT_DIR"