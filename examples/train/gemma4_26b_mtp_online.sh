#!/bin/bash
# Online Gemma4-26B-MoE MTP assistant training.
#
# Both target and assistant are loaded on the training GPUs (no separate vLLM).
#
# Usage:
#   # Single draft (CLI flags):
#   bash examples/train/gemma4_26b_mtp_online.sh
#
#   # Multi-draft (YAML; sequential isolated runs, not mixed I/O):
#   CONFIG=examples/train/gemma4_26b_mtp_online_multi.yaml \
#     bash examples/train/gemma4_26b_mtp_online.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ============ Configuration ============
# Optional YAML (N drafts trained one-by-one). When set, CLI draft flags below are unused.
CONFIG="${CONFIG:-}"

# Model paths (single-draft mode)
TARGET="/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it"
ASSISTANT="/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant"

# Data (regenerated for 26B-MoE)
DATA="${DATA:-/nvmedata/data/kimi-regen-gemma4-26b-moe/train_regen.jsonl}"

# Output
OUTPUT_DIR="${OUTPUT_DIR:-./output/gemma4_26b_mtp_online}"

# Training config
GPUS="${GPUS:-0,1}"
NUM_GPUS=1
MAX_SAMPLES="${MAX_SAMPLES:-50000}"
SEQ_LENGTH=2048
EPOCHS=3
LR=6e-4
TTT_STEPS=3
BATCH_SIZE=1
GRAD_ACCUM=8  # effective batch = 1*8 = 8 per step

# =======================================

TORCHRUN="${TORCHRUN:-/root/miniconda3/envs/speculator/bin/torchrun}"
mkdir -p "$OUTPUT_DIR"

if [[ -n "$CONFIG" ]]; then
  echo "=== multi-draft config: $CONFIG ==="
  echo "  GPUS=$GPUS NPROC=$NUM_GPUS"
  CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" \
      --standalone --nproc_per_node "$NUM_GPUS" \
      "$REPO_ROOT/scripts/gemma4_mtp/train_online.py" \
      --config "$CONFIG"
else
  echo "=== single-draft CLI ==="
  echo "  TARGET=$TARGET"
  echo "  ASSISTANT=$ASSISTANT"
  echo "  DATA=$DATA"
  echo "  OUTPUT_DIR=$OUTPUT_DIR"
  echo "  GPUS=$GPUS NPROC=$NUM_GPUS"
  echo "  lr=$LR epochs=$EPOCHS batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM"
  echo "  ttt_steps=$TTT_STEPS max_samples=$MAX_SAMPLES"

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
      --log-every 10
fi

echo "=== [train] done ==="
echo "Done."
