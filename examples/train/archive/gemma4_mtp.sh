#!/usr/bin/env bash
# Gemma 4 MTP assistant fine-tuning: target-cache generation + training (8-GPU).
#
# Two stages:
#   1) cache: run the frozen target once over all samples, stream last_hidden +
#      shared KV to a sharded on-disk cache (async writer, large shards).
#   2) train: read the cache (no target loaded) and fine-tune the stock Gemma 4
#      assistant, exporting a stock-config checkpoint that is a drop-in
#      replacement for the assistant in vLLM.
#
# Requires transformers>=5.10.2 (ships Gemma4AssistantForCausalLM), torch>=2.5,
# and a target-regenerated conversations JSONL (see scripts/response_regeneration
# or scripts/prepare_data.py for producing training data).
#
# Usage:
#   bash examples/train/gemma4_mtp.sh              # both stages
#   STAGE=cache bash examples/train/gemma4_mtp.sh  # only generate the cache
#   STAGE=train bash examples/train/gemma4_mtp.sh  # only train (cache must exist)
#
# Override any path/hyperparam via env, e.g. LR=2e-4 EPOCHS=2 bash ...
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---- paths (override via env) ---------------------------------------------
NPROC="${NPROC:-8}"
TARGET="${TARGET:-/path/to/gemma4/target}"
ASSISTANT="${ASSISTANT:-/path/to/gemma4/assistant}"
DATA="${DATA:-./data/train_regenerated.jsonl}"
OUT_DIR="${OUT_DIR:-./out/gemma4_mtp/cache}"       # sharded target cache
CKPT_DIR="${CKPT_DIR:-./out/gemma4_mtp/ckpt}"      # exported checkpoints
MAX_LENGTH="${MAX_LENGTH:-4096}"

# ---- hyperparams ----------------------------------------------------------
EPOCHS="${EPOCHS:-10}"
LOCAL_BATCH="${LOCAL_BATCH:-2}"           # per-GPU micro-batch
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"
LR="${LR:-6e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_STEPS="${WARMUP_STEPS:-}"          # empty => 4% of total steps
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
TTT_STEPS="${TTT_STEPS:-5}"
SAVE_EVERY="${SAVE_EVERY:-0}"             # 0 = save only at end
LOG_EVERY="${LOG_EVERY:-10}"

GRAD_ACCUM=$(( GLOBAL_BATCH / (LOCAL_BATCH * NPROC) ))
if (( GRAD_ACCUM < 1 )); then GRAD_ACCUM=1; fi

STAGE="${STAGE:-all}"   # all | cache | train

echo "=== config ==="
echo "  NPROC=$NPROC TARGET=$TARGET ASSISTANT=$ASSISTANT"
echo "  DATA=$DATA OUT_DIR=$OUT_DIR CKPT_DIR=$CKPT_DIR"
echo "  lr=$LR epochs=$EPOCHS local_batch=$LOCAL_BATCH global_batch=$GLOBAL_BATCH"
echo "  grad_accum=$GRAD_ACCUM ttt_steps=$TTT_STEPS stage=$STAGE"

# ---- stage 1: prepare cache ----------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "cache" ]]; then
  if [[ -f "$OUT_DIR/manifest.json" ]]; then
    echo "=== [cache] SKIP: manifest already exists at $OUT_DIR ==="
  else
    echo "=== [cache] generating -> $OUT_DIR ==="
    torchrun --standalone --nproc_per_node "$NPROC" \
        "$REPO_ROOT/scripts/gemma4_mtp/prepare_cache.py" \
        --target "$TARGET" --data "$DATA" --out-dir "$OUT_DIR" \
        --max-length "$MAX_LENGTH" --bf16
  fi
fi

# ---- stage 2: train -------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  if [[ ! -f "$OUT_DIR/manifest.json" ]]; then
    echo "!! no cache manifest at $OUT_DIR — run STAGE=cache first." >&2
    exit 1
  fi

  if [[ -z "$WARMUP_STEPS" ]]; then
    NUM_SAMPLES=$(python -c "import json; print(json.load(open('$OUT_DIR/manifest.json'))['num_samples'])")
    WARMUP_STEPS=$(python -c "
import math
steps_per_epoch = max(1, math.ceil($NUM_SAMPLES / $GLOBAL_BATCH))
print(max(1, round(steps_per_epoch * $EPOCHS * $WARMUP_RATIO)))
")
    echo "  [train] num_samples=$NUM_SAMPLES -> warmup_steps=$WARMUP_STEPS"
  fi

  SAVE_ARG=()
  if (( SAVE_EVERY > 0 )); then SAVE_ARG=(--save-every "$SAVE_EVERY"); fi

  mkdir -p "$CKPT_DIR"
  echo "=== [train] ${NPROC}-GPU from cache -> $CKPT_DIR ==="
  torchrun --standalone --nproc_per_node "$NPROC" \
      "$REPO_ROOT/scripts/gemma4_mtp/train.py" \
      --cache-dir "$OUT_DIR" --target "$TARGET" --assistant "$ASSISTANT" \
      --output "$CKPT_DIR" --epochs "$EPOCHS" --batch-size "$LOCAL_BATCH" \
      --grad-accum "$GRAD_ACCUM" --lr "$LR" --weight-decay "$WEIGHT_DECAY" \
      --warmup-steps "$WARMUP_STEPS" --ttt-steps "$TTT_STEPS" \
      --max-length "$MAX_LENGTH" "${SAVE_ARG[@]}" \
      --bf16 --log-every "$LOG_EVERY"
  echo "=== [train] done. checkpoint at $CKPT_DIR ==="
fi
