#!/usr/bin/env bash
# Launch vLLM serving Kimi K3 + CSplash speculative decoding, single node
# (TP=8, no PP, no DCP) -- for an 8xB300 node, whose 275 GiB/GPU comfortably
# covers K3's ~194 GiB/GPU weight shard at TP=8 alone. A B200 node (~183-194
# GiB/GPU) does not have that margin -- see docs/experiments/kimi_k3_2node_b200_launch.md
# for the 2-node (TP=8 x PP=2) B200 recipe this was adapted from (dropping
# cross-node PP/DCP/NCCL-Gloo-fabric config and the headless/API-server split).
#
# CSplash is a natively-trained sliding-window block-parallel Kimi K3 drafter
# (not a DSpark/RadixArk retrofit) -- see the vLLM port writeup and live
# validation (AgentX + 25-benchmark acceptance-length comparison against the
# vendor's offline reference) in docs/experiments/kimi_k3_2node_b200_launch.md,
# "CSplash: a new vLLM port" / "25-benchmark live-serving validation" sections.
#
# CAVEAT: adapted from a single-node RadixArk test script on this same vLLM
# build, not independently run end-to-end on real B300 hardware in this repo
# yet -- treat as a well-reasoned starting point, not a confirmed recipe.
#
# This script is for interactive/manual use.
#
# Usage:
#   # Default: TP=8, 256K ctx, CSplash draft, FP8 KV cache
#   ./run_k3_csplash_vllm.sh
#
#   # Dry run (print command without launching)
#   DRY_RUN=1 ./run_k3_csplash_vllm.sh
#
#   # Custom port, longer context (tune up for B300's larger per-GPU memory --
#   # the 256K default is carried over from the B200 2-node work's context
#   # ceiling, not derived from B300's actual headroom)
#   PORT=8080 CTX=524288 ./run_k3_csplash_vllm.sh
#
#   # Different draft checkpoint / speculative depth
#   DRAFT_DIR=/path/to/other/draft NUM_SP=5 ./run_k3_csplash_vllm.sh
set -euo pipefail

cd "$(dirname "$0")"

# --- defaults (override via env) -------------------------------------------
MODEL="${MODEL:-/import/ml-sc-nlpcheckpoints-scratch2/raghup/checkpoints/Kimi-K3}"
DRAFT_DIR="${DRAFT_DIR:-/import/ml-sc-scratch1/chenw/CSplashKimi-Kimi-K3-2026-10-13-v2/checkpoint}"
PORT="${PORT:-8000}"
TP="${TP:-8}"                                            # single B300 node
NUM_SP="${NUM_SP:-7}"                                    # CSplash's native "7+1" block layout
GPU_MEM="${GPU_MEM:-0.88}"
CTX="${CTX:-262144}"                                      # max_model_len; see usage note above
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
KV_DTYPE="${KV_DTYPE:-fp8}"
# No node-local venv exists on a fresh B300 node by default -- the shared NFS
# copy already has the full CSplash port (model file, speculator package,
# registry/dispatch edits, HF config registration).
VENV_DIR="${VENV_DIR:-/import/ml-sc-scratch1/chenw/vllm-venv-b200}"
CACHE_DIR="${CACHE_DIR:-/scratch/$(whoami)/cache}"
DRY_RUN="${DRY_RUN:-}"

SPEC_CONFIG="{\"model\":\"${DRAFT_DIR}\",\"num_speculative_tokens\":${NUM_SP},\"method\":\"dflash\",\"draft_sample_method\":\"probabilistic\"}"

echo "============================================================"
echo "  Kimi K3 + CSplash (single-node B300) vLLM launcher"
echo "============================================================"
echo "  Target model        : $MODEL"
echo "  Draft checkpoint     : $DRAFT_DIR"
echo "  Speculative method   : dflash (routes to CSplashSpeculator)"
echo "  num_speculative_tokens: $NUM_SP"
echo "  Port                : $PORT"
echo "  TP                  : $TP"
echo "  max_model_len        : $CTX"
echo "  KV cache dtype       : $KV_DTYPE"
echo "  venv                : $VENV_DIR"
echo "============================================================"

if [[ -z "$DRY_RUN" ]]; then
    mkdir -p "$CACHE_DIR"/{vllm,hf/hub,inductor,triton,flashinfer,tilelang,tmp}
    source "$VENV_DIR/bin/activate"
fi

export CUDA_HOME=/usr/local/cuda-13.2 PATH=/usr/local/cuda-13.2/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TMPDIR="$CACHE_DIR/tmp"
# NCCL_P2P_LEVEL=NVL fixed a reproducible local-GPU-pair stall on this
# project's cp81/cp82 B200 pair specifically -- kept as a conservative
# default, unconfirmed on other hardware.
export NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=NVL
export XDG_CACHE_HOME="$CACHE_DIR" VLLM_CACHE_ROOT="$CACHE_DIR/vllm" HF_HOME="$CACHE_DIR/hf" HF_HUB_CACHE="$CACHE_DIR/hf/hub" HUGGINGFACE_HUB_CACHE="$CACHE_DIR/hf/hub" TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/inductor" TRITON_CACHE_DIR="$CACHE_DIR/triton" FLASHINFER_WORKSPACE_BASE="$CACHE_DIR/flashinfer" TILELANG_CACHE_DIR="$CACHE_DIR/tilelang"
export HF_HUB_OFFLINE=1
export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1 VLLM_ALLREDUCE_USE_FLASHINFER=1 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 VLLM_SERVER_DEV_MODE=1 NCCL_CUMEM_ENABLE=1

CMD=(
    python3 -u -m vllm.entrypoints.cli.main serve
    "$MODEL"
    --served-model-name moonshotai/Kimi-K3
    --tensor-parallel-size "$TP"
    --trust-remote-code --moe-backend auto
    --gpu-memory-utilization "$GPU_MEM"
    --no-enable-flashinfer-autotune
    --max-model-len "$CTX"
    --kv-cache-dtype "$KV_DTYPE"
    --mamba-ssm-cache-dtype bfloat16
    --attention-config '{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":true}'
    --attention-backend TOKENSPEED_MLA
    --enable-prefix-caching --enable-prompt-tokens-details
    --enable-auto-tool-choice --tool-call-parser kimi_k3 --reasoning-parser kimi_k3
    --language-model-only
    --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens 8192
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,8,18,34,53,64,75,100,128,256,512,1024,2048,4096,8192]}'
    --load-format fastsafetensors
    --speculative-config "$SPEC_CONFIG"
    --host 0.0.0.0 --port "$PORT"
)

echo "Command:"
echo " ${CMD[*]}"
echo ""

if [[ -n "$DRY_RUN" ]]; then
    echo "(dry run — not launching)"
    exit 0
fi

exec "${CMD[@]}"
