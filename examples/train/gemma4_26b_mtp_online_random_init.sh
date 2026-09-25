#!/bin/bash
# Random-init Gemma4-26B-MoE MTP assistant training (baseline comparison).
#
# Same as gemma4_26b_mtp_online.sh but with --random-init flag so the
# assistant starts from scratch instead of the pretrained checkpoint.
#
# Usage:
#   bash examples/train/gemma4_26b_mtp_online_random_init.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ============ Configuration ============
# Model paths
TARGET="/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it"
ASSISTANT="/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant"

# Data (regenerated for 26B-MoE)
DATA="${DATA:-/nvmedata/data/kimi-regen-gemma4-26b-moe/train_regen.jsonl}"

# Output
OUTPUT_DIR="${OUTPUT_DIR:-./output/gemma4_26b_mtp_random_init}"

# Training config (2 GPUs via device_map auto)
GPUS="${GPUS:-0,1}"
NUM_GPUS=1
MAX_SAMPLES="${MAX_SAMPLES:-20000}"
SEQ_LENGTH=1024
EPOCHS=3
LR=6e-4
TTT_STEPS=3
BATCH_SIZE=1
GRAD_ACCUM=8  # effective batch = 1*8 = 8 per step

# =======================================

echo "=== config ==="
echo "  TARGET=$TARGET"
echo "  ASSISTANT=$ASSISTANT"
echo "  DATA=$DATA"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  GPUS=$GPUS NPROC=$NUM_GPUS"
echo "  lr=$LR epochs=$EPOCHS batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM"
echo "  ttt_steps=$TTT_STEPS max_samples=$MAX_SAMPLES"
echo "  RANDOM_INIT=yes"

mkdir -p "$OUTPUT_DIR"

echo "=== [train] training on GPUs $GPUS (random init) ==="
TORCHRUN=/root/miniconda3/envs/speculator/bin/torchrun
CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" \
    --standalone --nproc_per_node "$NUM_GPUS" \
    "$REPO_ROOT/scripts/gemma4_mtp/train_online.py" \
    --target "$TARGET" \
    --assistant "$ASSISTANT" \
    --data "$DATA" \
    --output "$OUTPUT_DIR/checkpoints" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --max-length "$SEQ_LENGTH" \
    --ttt-steps "$TTT_STEPS" \
    --max-samples "$MAX_SAMPLES" \
    --bf16 \
    --log-every 10 \
    --random-init

echo "=== [train] done. Checkpoints at $OUTPUT_DIR/checkpoints ==="
echo "Done."