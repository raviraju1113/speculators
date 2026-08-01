#!/bin/bash
# sngpu payload: DSpark Gemma-4 OVERFIT GATE on a SINGLE GPU, offline.
#
# WHY 1 GPU IS POSSIBLE HERE
#   Online needs 2 GPUs no matter how small the dataset: the verifier is 62.6 GB
#   and the training process needs ~42 GB of its own (~2.4B trainable params at
#   AdamW fp32 master weights + frozen embeddings), so ~105 GB total -- more than
#   one 80 GB A100.
#   Offline splits them in TIME instead of across devices: vLLM runs alone, dumps
#   hidden states, exits and frees the GPU; then training runs alone. Each phase
#   fits comfortably in 80 GB.
#
# ALSO: no NCCL anywhere (vLLM TP=1, training as plain `python`), and epochs after
# the dump are fast because nothing is regenerated per-epoch.
#
# DISK: hidden states are ~64.5 KB/token, ~26 MB per conversation.
#   32 samples  -> ~0.8 GB      256 samples -> ~6.6 GB
#   (Do not point this at the full 349k set: that would be ~9 TB.)
#
#   sngpu --jobname dspark_overfit_1gpu --partition gpuonly --nodelist sc3-c98 \
#     --gpu 1 --gputype a100m80 --cpu 16 --mem 200000 --time 04:00:00 \
#     --output ./logs/dspark_overfit_1gpu.txt \
#     -- bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/run_overfit_1gpu_offline.sh
#
# Override the dataset with DATA_PATH=... (default: the 256-sample split).
set -euo pipefail

REPO=/import/ml-sc-scratch1/mengmengj/speculators
CONDA_SH=/import/snvm-sc-scratch1/mengmengj/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=/import/ml-sc-scratch1/mengmengj/condaenvs/dspark
MODEL=/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it

DATA_PATH="${DATA_PATH:-/import/ml-sc-scratch1/mengmengj/datasets/gemma4_dspark_256}"
RUN_NAME="${RUN_NAME:-gemma4_31b_dspark_overfit_1gpu}"
OUT_ROOT="${OUT_ROOT:-/import/ml-sc-scratch1/mengmengj/output}"
HS_PATH="${HS_PATH:-$OUT_ROOT/hs_offline_$RUN_NAME}"
SAVE_PATH="${SAVE_PATH:-$OUT_ROOT/$RUN_NAME/checkpoints}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/$RUN_NAME/logs}"

TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-1 17 29 47 58}"
EPOCHS="${EPOCHS:-200}"     # epochs are cheap offline; this is a real memorization test
LR="${LR:-1e-3}"
MAX_ANCHORS="${MAX_ANCHORS:-256}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"   # must be >= prepare_data --seq-length
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
CONCURRENCY="${CONCURRENCY:-8}"

export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-dspark}"
LOGGER="${LOGGER:-wandb}"

mkdir -p "$HS_PATH" "$SAVE_PATH" "$LOG_DIR"

# shellcheck disable=SC1090
source "$CONDA_SH"; conda activate "$CONDA_ENV"; cd "$REPO"

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
if [ "${DRV%%.*}" -lt 580 ] 2>/dev/null; then
  echo "!! driver $DRV too old for torch 2.11+cu130; use --nodelist sc3-c98" >&2; exit 1
fi

echo "=========================================="
echo " host   : $(hostname)   driver $DRV"
echo " data   : $DATA_PATH"
echo " HS dir : $HS_PATH"
echo " epochs : $EPOCHS   lr $LR"
echo "=========================================="

# ---------------- phase 1: dump hidden states (vLLM alone) ----------------
if [ -n "$(ls -A "$HS_PATH" 2>/dev/null)" ]; then
  echo "=== hidden states already present in $HS_PATH; skipping generation ==="
else
  echo "=== phase 1: launching vLLM (TP=1, alone on this GPU) ==="
  python scripts/launch_vllm.py "$MODEL" \
      --hidden-states-path "$HS_PATH" \
      --target-layer-ids $TARGET_LAYER_IDS \
      -- --tensor-parallel-size 1 \
         --max-model-len "$MAX_MODEL_LEN" \
         --gpu-memory-utilization "$GPU_MEM_UTIL" \
         --no-enable-prefix-caching \
         --port "$VLLM_PORT" \
      > "$LOG_DIR/vllm.log" 2>&1 &
  VLLM_PID=$!
  cleanup() { kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
  trap cleanup EXIT

  echo "waiting for vLLM health..."
  for _ in $(seq 1 180); do
    curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1 && { echo "vLLM ready."; break; }
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "!! vLLM died:"; tail -40 "$LOG_DIR/vllm.log"; exit 1; }
    sleep 10
  done
  curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null || {
    echo "!! vLLM not healthy after 30 min"; tail -40 "$LOG_DIR/vllm.log"; exit 1; }

  echo "=== dumping hidden states ==="
  python scripts/data_generation_offline.py \
      --model "$MODEL" \
      --preprocessed-data "$DATA_PATH" \
      --endpoint "http://localhost:${VLLM_PORT}/v1" \
      --output "$HS_PATH" \
      --concurrency "$CONCURRENCY" \
      --validate-outputs

  echo "=== stopping vLLM to free the GPU for training ==="
  cleanup
  trap - EXIT
  # Give the driver a moment to actually release the memory before training grabs it.
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "${used:-99999}" -lt 2000 ] && break
    sleep 5
  done
  echo "GPU memory now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
  echo "hidden states on disk: $(du -sh "$HS_PATH" | cut -f1)"
fi

# ---------------- phase 2: train (no vLLM, no NCCL) ----------------
# plain `python`, NOT torchrun: torchrun sets LOCAL_RANK and the trainer would then
# call init_process_group("nccl"), which segfaults on this cluster.
echo "=== phase 2: training (offline hidden states, single process) ==="
python scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_PATH" \
    --hidden-states-path "$HS_PATH" \
    --save-path "$SAVE_PATH" \
    --log-dir "$LOG_DIR" \
    --run-name "$RUN_NAME" \
    --logger "$LOGGER" \
    --speculator-type dspark \
    --block-size 8 \
    --num-layers 5 \
    --draft-vocab-size 32000 \
    --target-layer-ids $TARGET_LAYER_IDS \
    --sliding-window 2048 \
    --markov-rank 256 \
    --markov-head-type vanilla \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' \
    --confidence-head-alpha 1.0 \
    --max-anchors "$MAX_ANCHORS" \
    --total-seq-len 8192 \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --num-workers 8 \
    --on-missing skip

echo "=== done. checkpoints -> $SAVE_PATH ==="
