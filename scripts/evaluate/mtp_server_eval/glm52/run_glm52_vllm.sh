#!/usr/bin/env bash
# Launch vLLM serving GLM-5.2 with native MTP speculative decoding.
#
# GLM-5.2 has MTP built into the model — no separate draft model needed.
# Uses method=mtp (not dspark), num_speculative_tokens=5.
#
# This script is for interactive/manual use. For automated experiments,
# use the YAML config + run_experiments.py instead:
#   python run_experiments.py --config glm52-eval.yaml
#
# Usage:
#   # Default: FP8 backbone + native MTP (TP=8, 1M ctx, KV cache FP8)
#   ./run_glm52_vllm.sh
#
#   # Dry run (print command without launching)
#   DRY_RUN=1 ./run_glm52_vllm.sh
#
#   # BF16 backbone, custom port, TP=4
#   MODEL=zai-org/GLM-5.2 PORT=8080 TP=4 ./run_glm52_vllm.sh
#
#   # Override speculative depth
#   NUM_SP=3 ./run_glm52_vllm.sh
set -euo pipefail

cd "$(dirname "$0")"

# --- defaults (override via env) -------------------------------------------
MODEL="${MODEL:-zai-org/GLM-5.2-FP8}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
NUM_SP="${NUM_SP:-5}"                                   # native MTP depth
GPU_MEM="${GPU_MEM:-0.9}"
CTX="${CTX:-1048576}"                                   # 1M GLM-5.2 native context
MAXMODEL="${MAXMODEL:-16384}"                           # max_model_len
KV_DTYPE="${KV_DTYPE:-fp8}"                             # FP8 KV cache
DRY_RUN="${DRY_RUN:-}"

# Native MTP: no draft model needed, just method=mtp + num_speculative_tokens
SPEC_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SP}}"

echo "============================================================"
echo "  GLM-5.2 (native MTP) vLLM launcher"
echo "============================================================"
echo "  Base model         : $MODEL"
echo "  MTP method         : mtp (built-in, no draft model)"
echo "  num_speculative_tokens: $NUM_SP"
echo "  Port               : $PORT"
echo "  TP                 : $TP"
echo "  Context            : $CTX"
echo "  max_model_len      : $MAXMODEL"
echo "  KV cache dtype     : $KV_DTYPE"
echo "============================================================"

CMD=(
    python -m vllm.entrypoints.cli.main serve
    "$MODEL"
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --gpu-memory-utilization "$GPU_MEM"
    --max-model-len "$MAXMODEL"
    --speculative-config "$SPEC_CONFIG"
    --kv-cache-dtype "$KV_DTYPE"
    --disable-chunked-prefill
    --enable-prefix-caching
    --tool-call-parser glm47
    --reasoning-parser glm45
    --enable-auto-tool-choice
)

echo "Command:"
echo " ${CMD[*]}"
echo ""

if [[ -n "$DRY_RUN" ]]; then
    echo "(dry run — not launching)"
    exit 0
fi

exec "${CMD[@]}"