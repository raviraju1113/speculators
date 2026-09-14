#!/bin/bash
# Simultaneous online training of FIVE draft types for Gemma4-26B-A4B (MoE),
# on the merged all-regen dataset (/nvmedata/data/merged_all_regen.jsonl:
# 26B-MoE regen + 31B regen + 31B tool regen, deduped + shuffled, 685,722 rows).
#
#   GPU0: shared vLLM hidden-states server (feeds the four generic trainers)
#   GPU1: EAGLE3  + P-EAGLE   (co-located; ~14GB + ~25GB)
#   GPU2: DSpark  + DFlash    (co-located; ~28GB + ~35GB)
#   GPU3: MTP (gemma4_mtp standalone; target+assistant share the GPU, ~78GB)
#
# All four generic trainers share one hidden-state cache (--on-generate cache):
# each sample's features are generated once, reused by every trainer and epoch.
# Budget ~26 MB/sample on disk (30k samples ~ 780 GB).
#
# Usage:  bash examples/train/gemma4_26b_penta_draft_online.sh
#         MAX_SAMPLES=50000 EPOCHS=1 bash examples/train/gemma4_26b_penta_draft_online.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ============ Configuration ============
MODEL="${MODEL:-/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it}"            # 30 layers
ASSISTANT="${ASSISTANT:-/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant}"
DATA="${DATA:-/nvmedata/data/merged_all_regen.jsonl}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/gemma4_26b_penta_draft}"
HS_PATH="${HS_PATH:-/nvmedata/chenw/hidden_states_tri_draft}"   # reuse existing cache

MAX_SAMPLES="${MAX_SAMPLES:-30000}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
MTP_SEQ_LENGTH="${MTP_SEQ_LENGTH:-2048}"
EPOCHS="${EPOCHS:-2}"
DRAFT_VOCAB_SIZE=32000
TARGET_LAYER_IDS="2 15 27"             # [2, L/2, L-3] for L=30; server appends 30
SERVER_MAX_LEN=$((SEQ_LENGTH + 256))   # prompts truncated to SEQ_LENGTH need +1 headroom

VLLM_PORT="${VLLM_PORT:-8000}"

# Per-draft hyperparams (from the respective example scripts)
EAGLE3_LR=1e-4
DSPARK_LR=3e-4; BLOCK_SIZE=8; MAX_ANCHORS=3072; DSPARK_NUM_LAYERS=3
MARKOV_RANK=256; MARKOV_HEAD_TYPE="vanilla"
DSPARK_LOSS_FN='{"ce": 0.1, "tv": 0.9}'; CONFIDENCE_HEAD_ALPHA=1.0
DFLASH_LR=3e-4; DFLASH_NUM_LAYERS=5
PEAGLE_LR=6e-4; PEAGLE_NUM_LAYERS=4; NUM_DEPTHS=4
DOWN_SAMPLE_RATIO=0.7; DOWN_SAMPLE_RATIO_MIN=0.2
# 6e-4 is the random-init recipe; it degraded the vanilla assistant in the
# tri-draft run (accept_len 4.83 -> 2.98). Fine-tuning wants a gentle lr.
MTP_LR=5e-5; TTT_STEPS=3; MTP_BATCH_SIZE=1; MTP_GRAD_ACCUM=8
# =======================================

source /root/miniconda3/etc/profile.d/conda.sh
conda activate speculator
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"

LOGS="$OUTPUT_ROOT/logs"
mkdir -p "$OUTPUT_ROOT" "$LOGS" "$HS_PATH"

echo "=== Step 1: prepare data (tokenize + loss mask + token_freq; CPU) ==="
PREP_DIR="$OUTPUT_ROOT/data_prep"
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATA" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH" \
    --output "$PREP_DIR" 2>&1 | tee "$LOGS/prepare_data.log"

echo "=== Step 2: launch shared vLLM hidden-states server (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HS_PATH" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" --max-model-len "$SERVER_MAX_LEN" \
       --gpu-memory-utilization 0.85 \
    > "$LOGS/vllm_server.log" 2>&1 &
VLLM_PID=$!

PIDS=()
cleanup() {
    echo "Stopping vLLM server and any remaining trainers..."
    kill "$VLLM_PID" "${PIDS[@]}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM /health ..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM died; see $LOGS/vllm_server.log"; exit 1; }
    sleep 5
done
echo "vLLM server ready."

echo "=== Step 3: launch the five trainers concurrently ==="

# Shared args for the four generic scripts/train.py legs
common_train_args() {
    echo --verifier-name-or-path "$MODEL" \
         --data-path "$PREP_DIR" \
         --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
         --hidden-states-path "$HS_PATH" \
         --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
         --epochs "$EPOCHS" \
         --total-seq-len "$SEQ_LENGTH" \
         --target-layer-ids $TARGET_LAYER_IDS \
         --on-missing generate \
         --on-generate cache
}

# --- EAGLE3 (GPU 1) ---
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node 1 scripts/train.py \
    $(common_train_args) \
    --save-path "$OUTPUT_ROOT/eagle3/checkpoints" \
    --speculator-type eagle3 --lr "$EAGLE3_LR" \
    > "$LOGS/eagle3.log" 2>&1 &
PIDS+=($!); echo "  eagle3  pid=${PIDS[-1]}  gpu=1"

# --- DSpark (GPU 2) ---
sleep 90
CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node 1 scripts/train.py \
    $(common_train_args) \
    --save-path "$OUTPUT_ROOT/dspark/checkpoints" \
    --speculator-type dspark --lr "$DSPARK_LR" \
    --block-size "$BLOCK_SIZE" --max-anchors "$MAX_ANCHORS" \
    --num-layers "$DSPARK_NUM_LAYERS" \
    --markov-rank "$MARKOV_RANK" --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn "$DSPARK_LOSS_FN" --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    > "$LOGS/dspark.log" 2>&1 &
PIDS+=($!); echo "  dspark  pid=${PIDS[-1]}  gpu=2"

# --- P-EAGLE (GPU 1, co-located with EAGLE3) ---
sleep 90
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node 1 scripts/train.py \
    $(common_train_args) \
    --save-path "$OUTPUT_ROOT/peagle/checkpoints" \
    --speculator-type peagle --lr "$PEAGLE_LR" \
    --num-layers "$PEAGLE_NUM_LAYERS" --num-depths "$NUM_DEPTHS" \
    --down-sample-ratio "$DOWN_SAMPLE_RATIO" \
    --down-sample-ratio-min "$DOWN_SAMPLE_RATIO_MIN" \
    --no-norm-before-residual --scheduler-type cosine \
    > "$LOGS/peagle.log" 2>&1 &
PIDS+=($!); echo "  peagle  pid=${PIDS[-1]}  gpu=1"

# --- DFlash (GPU 2, co-located with DSpark) ---
sleep 90
CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node 1 scripts/train.py \
    $(common_train_args) \
    --save-path "$OUTPUT_ROOT/dflash/checkpoints" \
    --speculator-type dflash --lr "$DFLASH_LR" \
    --block-size "$BLOCK_SIZE" --max-anchors "$MAX_ANCHORS" \
    --num-layers "$DFLASH_NUM_LAYERS" \
    > "$LOGS/dflash.log" 2>&1 &
PIDS+=($!); echo "  dflash  pid=${PIDS[-1]}  gpu=2"

# --- MTP assistant fine-tune (GPU 3; target+draft share the GPU) ---
CUDA_VISIBLE_DEVICES=3 python scripts/gemma4_mtp/train_online.py \
    --target "$MODEL" \
    --assistant "$ASSISTANT" \
    --data "$DATA" \
    --output "$OUTPUT_ROOT/mtp/checkpoints" \
    --epochs "$EPOCHS" \
    --batch-size "$MTP_BATCH_SIZE" \
    --grad-accum "$MTP_GRAD_ACCUM" \
    --lr "$MTP_LR" \
    --max-length "$MTP_SEQ_LENGTH" \
    --ttt-steps "$TTT_STEPS" \
    --max-samples "$MAX_SAMPLES" \
    --bf16 \
    --log-every 10 \
    > "$LOGS/mtp.log" 2>&1 &
PIDS+=($!); echo "  mtp     pid=${PIDS[-1]}  gpu=3"

echo "=== Waiting for all five trainers ==="
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || { echo "trainer pid=$pid FAILED"; FAIL=1; }
done

echo "=== done (fail=$FAIL). Checkpoints under $OUTPUT_ROOT/{eagle3,dspark,peagle,dflash,mtp}/checkpoints ==="
exit "$FAIL"
