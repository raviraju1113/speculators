#!/bin/bash
# Online DSpark training for google/gemma-4-31B-it.
#
# Serves the verifier with vLLM (streaming aux hidden states) on one set of GPUs
# and trains the DSpark draft on the rest, in a single sngpu allocation.
#
# Submit (absolute path required -- sbatch copies the script to a spool dir):
#
# NOTE: sngpu forwards the job command to `sbatch --wrap`, which rejects a
# command with arguments -- so the payload must be `bash <one-script-path>`.
# `env VAR=... bash <path>` is also rejected; export the vars in a tiny wrapper
# script instead, or use the MODE presets' defaults.
#
#   mkdir -p logs
#   # overfit gate (3 of 4 GPUs on sc3-c98, ~1 h)
#   sngpu --jobname dspark_overfit --partition gpuonly --nodelist sc3-c98 \
#     --gpu 3 --gputype a100m80 --cpu 24 --mem 200000 --time 01:00:00 \
#     --output ./logs/dspark_overfit.txt \
#     -- bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/run_overfit_sc3c98.sh
#
#   # full run (all 4 GPUs on sc3-c98: 2 vLLM + 2 train)
#   sngpu --jobname dspark_full --partition gpuonly --nodelist sc3-c98 \
#     --gpu 4 --gputype a100m80 --cpu 92 --mem 900000 --time 48:00:00 \
#     --output ./logs/dspark_full.txt \
#     -- bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/run_full_sc3c98.sh
#
# Interactively (or from a wrapper) every knob is env-overridable:
#   MODE=full DATA_PATH=.../gemma4_dspark_100k EPOCHS=3 bash .../dspark_online_gemma4_31b.sh

set -euo pipefail

# ---------------- paths (absolute; see note above) ----------------
REPO="${REPO:-/import/ml-sc-scratch1/mengmengj/speculators}"
CONDA_SH="${CONDA_SH:-/import/snvm-sc-scratch1/mengmengj/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/import/ml-sc-scratch1/mengmengj/condaenvs/dspark}"
MODEL="${MODEL:-/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it}"
DATA_ROOT="${DATA_ROOT:-/import/ml-sc-scratch1/mengmengj/datasets}"
OUT_ROOT="${OUT_ROOT:-/import/ml-sc-scratch1/mengmengj/output}"

# ---------------- mode presets ----------------
MODE="${MODE:-overfit}"
case "$MODE" in
  overfit)
    # Tiny data, many epochs, high LR: the draft SHOULD memorize. Gate, not a model.
    DATA_PATH="${DATA_PATH:-$DATA_ROOT/gemma4_dspark_256}"
    EPOCHS="${EPOCHS:-50}"; LR="${LR:-1e-3}"; MAX_ANCHORS="${MAX_ANCHORS:-256}"
    VLLM_TP="${VLLM_TP:-2}"; VLLM_DP="${VLLM_DP:-1}"; TRAIN_GPUS_N="${TRAIN_GPUS_N:-1}"
    RUN_NAME="${RUN_NAME:-gemma4_31b_dspark_overfit}"
    ;;
  full)
    # Sized for sc3-c98: 4x A100-80GB, driver 595.71.05 (CUDA 13.2) -- the only
    # surveyed node whose driver supports our torch 2.11+cu130 build. Every other
    # A100/H100 box is on 565/CUDA 12.7; the 8x H200 nodes are reserved.
    # vLLM TP=2 puts ~31 GB of verifier weights on each of GPUs 0-1, leaving
    # generous KV room; GPUs 2-3 run the training ranks.
    # Throughput alternative once it's known-good: VLLM_TP=1 VLLM_DP=2 (full
    # 62.6 GB per GPU, ~17 GB KV -- fine for short prefill-only requests).
    # On an 8-GPU node use VLLM_TP=2 VLLM_DP=3 TRAIN_GPUS_N=2.
    DATA_PATH="${DATA_PATH:-$DATA_ROOT/gemma4_dspark}"
    EPOCHS="${EPOCHS:-5}"; LR="${LR:-3e-4}"; MAX_ANCHORS="${MAX_ANCHORS:-3072}"
    VLLM_TP="${VLLM_TP:-2}"; VLLM_DP="${VLLM_DP:-1}"; TRAIN_GPUS_N="${TRAIN_GPUS_N:-2}"
    RUN_NAME="${RUN_NAME:-gemma4_31b_dspark}"
    ;;
  *) echo "unknown MODE=$MODE (want: overfit|full)" >&2; exit 2 ;;
esac

# ---------------- DSpark / DFlash architecture ----------------
# Matches RedHatAI/gemma-4-31B-it-speculator.dspark so results are comparable.
# TARGET_LAYER_IDS must be IDENTICAL on the vLLM and training sides; the draft's
# fc layer is sized len(ids)*hidden_size, so a mismatch is a silent misalignment.
# launch_vllm.py appends the last layer itself (--include-last-layer, default on)
# to supply verifier_last_hidden_states -- do not add it here.
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-1 17 29 47 58}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
NUM_LAYERS="${NUM_LAYERS:-5}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
# Upstream inverted this: ALL draft layers use sliding-window attention by
# default and you opt out per layer with --full-attention-indices. RedHat's
# gemma-4 dspark config is all-sliding at 2048, i.e. exactly the default, so we
# pass only the window size and no per-layer overrides.
SLIDING_WINDOW="${SLIDING_WINDOW:-2048}"
MARKOV_RANK="${MARKOV_RANK:-256}"
MARKOV_HEAD_TYPE="${MARKOV_HEAD_TYPE:-vanilla}"
LOSS_FN="${LOSS_FN:-{\"ce\": 0.1, \"tv\": 0.9\}}"
CONFIDENCE_HEAD_ALPHA="${CONFIDENCE_HEAD_ALPHA:-1.0}"
TOTAL_SEQ_LEN="${TOTAL_SEQ_LEN:-8192}"   # must be a multiple of 128 for flex attention

# ---------------- runtime ----------------
VLLM_PORT="${VLLM_PORT:-8000}"
# Gemma-4 advertises a 262144 context; left unset, vLLM sizes the KV cache for
# that and there is no room after 62.6 GB of weights. Even 8192 is too much at
# TP=1: measured 2026-07-31, only 10.91 GiB is left for KV and 8192 needs
# 11.89 GiB (vLLM's own estimate of the feasible max was 7504).
# MUST be >= the --seq-length used by prepare_data.py (8192): that is where
# samples get truncated, so a prepared sample can be up to 8192 tokens and the
# server has to accept it. Do not size this from a sample of conversation
# lengths -- 4096 was tried and 4114-token requests returned HTTP 400.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# At TP=1 the 62.6 GB of weights leave ~10.9 GiB for KV at util 0.92, but an
# 8192 context needs ~11.9 GiB. 0.95 buys roughly +2.4 GiB, which covers it.
# If vLLM still reports insufficient KV: raise toward 0.96, add --enforce-eager
# (frees CUDA-graph memory), or use TP>=2 so the weights split across GPUs.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
HS_PATH="${HS_PATH:-$OUT_ROOT/hidden_states_$RUN_NAME}"
SAVE_PATH="${SAVE_PATH:-$OUT_ROOT/$RUN_NAME/checkpoints}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/$RUN_NAME/logs}"
# 'delete' discards each sample's hidden states after use (~zero disk). 'cache'
# reuses them from epoch 2 on, but costs ~64 KB/token: fine for the 256-sample
# gate, ~10 TB for the full 349k-conversation set. Do not flip this blindly.
ON_GENERATE="${ON_GENERATE:-delete}"

# ---------------- metric logging ----------------
# LOGGER: "" (stdout only), or wandb / tensorboard / mlflow / trackio, or a
# comma-separated list. The wandb handler forwards to wandb.init() but never sets
# `project`, so that comes from WANDB_PROJECT. Auth comes from ~/.netrc (run
# `wandb login` once on the login node) -- deliberately NOT from a key on the
# sngpu command line, which would be visible in job listings and logs.
# If compute nodes have no egress (see sngpu_recon.sh), set WANDB_MODE=offline
# and run `wandb sync "$LOG_DIR/wandb/offline-run-*"` afterwards from the login node.
LOGGER="${LOGGER:-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-dspark}"
export WANDB_MODE="${WANDB_MODE:-online}"
[ -n "${WANDB_ENTITY:-}" ] && export WANDB_ENTITY
NUM_WORKERS="${NUM_WORKERS:-8}"
# NOTE: train.py has no --max-samples any more (upstream removed it). To train on
# a subset, build a smaller prepared dataset instead and point DATA_PATH at it:
#   python scripts/prepare_data.py --model "$MODEL" --data <regen.jsonl> \
#       --seq-length 8192 --max-samples 100000 --output "$DATA_ROOT/gemma4_dspark_100k"

# ---------------- GPU split ----------------
VLLM_GPUS_N=$(( VLLM_TP * VLLM_DP ))
TOTAL_NEEDED=$(( VLLM_GPUS_N + TRAIN_GPUS_N ))
VISIBLE=$(nvidia-smi -L | wc -l)
if [ "$VISIBLE" -lt "$TOTAL_NEEDED" ]; then
  echo "!! need $TOTAL_NEEDED GPUs (vLLM $VLLM_GPUS_N + train $TRAIN_GPUS_N) but see $VISIBLE" >&2
  exit 1
fi
VLLM_GPUS=$(seq -s, 0 $(( VLLM_GPUS_N - 1 )))
TRAIN_GPUS=$(seq -s, "$VLLM_GPUS_N" $(( TOTAL_NEEDED - 1 )))

# ---------------- driver guard ----------------
# Only some nodes have a driver new enough for torch 2.11+cu130 (needs CUDA 13,
# i.e. >= ~580). As of 2026-07-31 that is sc3-c98 only; sc-c96 / sc3-c97 /
# sc-c82 / sc-c120 are on 565.57.01. SLURM exposes no driver feature to select
# on, so submit with an explicit --nodelist. Fail in seconds rather than after
# loading 62.6 GB of weights.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
DRV_MAJOR=${DRV%%.*}
if [ -z "$DRV_MAJOR" ]; then
  echo "!! cannot read the NVIDIA driver version -- is this a GPU node?" >&2; exit 1
fi
if [ "$DRV_MAJOR" -lt 580 ]; then
  echo "!! driver $DRV on $(hostname) is too old for torch 2.11+cu130 (needs >= 580)." >&2
  echo "   Resubmit with --exclude sc-c96,sc3-c97,sc-c82 (leaves sc3-c98/sc3-c81, both 595)." >&2
  exit 1
fi
echo " driver      : $DRV (OK)"

mkdir -p "$HS_PATH" "$SAVE_PATH" "$LOG_DIR"

echo "=========================================="
echo " mode        : $MODE"
echo " host        : $(hostname)"
echo " data        : $DATA_PATH"
echo " vLLM GPUs   : $VLLM_GPUS  (TP=$VLLM_TP DP=$VLLM_DP)"
echo " train GPUs  : $TRAIN_GPUS ($TRAIN_GPUS_N ranks)"
echo " layer ids   : $TARGET_LAYER_IDS"
echo " hidden st.  : $HS_PATH  (on-generate=$ON_GENERATE)"
echo " save        : $SAVE_PATH"
echo " shm         : $(df -h /dev/shm | tail -1)"
echo "=========================================="

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO"

# ---------------- NCCL ----------------
# Observed 2026-07-31 on sc3-c98: NCCL segfaults inside ncclNetPluginInit
# (plugin/net.cc:216) while loading an external network plugin -- identically in
# vLLM's TP=2 init and in a bare 2-rank all_reduce. Disabling plugin discovery
# makes NCCL fall back to its built-in transports.
# NOTE: this is needed even for single-rank training, because torchrun sets
# LOCAL_RANK and the trainer then calls init_process_group("nccl") regardless.
# Drop these if the cluster's plugin is ever fixed.
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# The crash is in net init, and Ravi's working torchrun pins rendezvous to
# loopback (--local-addr=127.0.0.1) -- a strong hint that this cluster's default
# interface selection misbehaves. Everything here is single-node, so force NCCL
# onto loopback sockets and off InfiniBand rather than letting it probe.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

# ---------------- 1. verifier server ----------------
echo "=== launching vLLM (verifier + hidden-state streaming) ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HS_PATH" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --tensor-parallel-size "$VLLM_TP" \
       --max-model-len "$MAX_MODEL_LEN" \
       --gpu-memory-utilization "$GPU_MEM_UTIL" \
       --no-enable-prefix-caching \
       --data-parallel-size "$VLLM_DP" \
       --port "$VLLM_PORT" \
    > "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "=== stopping vLLM (pid $VLLM_PID) ==="
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "waiting for vLLM health (log: $LOG_DIR/vllm.log)..."
for _ in $(seq 1 180); do
    if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
        echo "vLLM ready."; break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "!! vLLM died during startup; tail of log:" >&2
        tail -40 "$LOG_DIR/vllm.log" >&2
        exit 1
    fi
    sleep 10
done
curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null || {
    echo "!! vLLM did not become healthy in 30 min" >&2; tail -40 "$LOG_DIR/vllm.log" >&2; exit 1; }

# ---------------- 2. train ----------------
# With a single training rank, launch train.py DIRECTLY instead of through
# torchrun. torchrun sets LOCAL_RANK, which makes maybe_setup_distributed() call
# init_process_group("nccl") even for world_size=1 -- and on this cluster that
# segfaults in ncclNetPluginInit. Plain python leaves _is_distributed False and
# never touches NCCL. Multi-rank still needs torchrun (and a working NCCL).
if [ "$TRAIN_GPUS_N" -eq 1 ]; then
  echo "=== training DSpark draft (single process, no torchrun / no NCCL) ==="
  LAUNCHER=(python)
else
  echo "=== training DSpark draft (torchrun, $TRAIN_GPUS_N ranks) ==="
  # Rendezvous pinned to loopback, matching the invocation Ravi has working on
  # this cluster: --standalone alone can pick a bad interface for the c10d store.
  LAUNCHER=(torchrun --nnodes=1 --nproc_per_node "$TRAIN_GPUS_N"
            --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29500
            --local-addr=127.0.0.1)
fi

CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "${LAUNCHER[@]}" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_PATH" \
    --hidden-states-path "$HS_PATH" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$SAVE_PATH" \
    --log-dir "$LOG_DIR" \
    --run-name "$RUN_NAME" \
    --logger "$LOGGER" \
    --speculator-type dspark \
    --block-size "$BLOCK_SIZE" \
    --num-layers "$NUM_LAYERS" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --sliding-window "$SLIDING_WINDOW" \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --max-anchors "$MAX_ANCHORS" \
    --total-seq-len "$TOTAL_SEQ_LEN" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --num-workers "$NUM_WORKERS" \
    --on-missing generate \
    --on-generate "$ON_GENERATE"

echo "=== done. checkpoints -> $SAVE_PATH ==="
