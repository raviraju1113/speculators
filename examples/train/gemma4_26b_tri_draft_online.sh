#!/bin/bash
# Simultaneous online training of THREE draft types for Gemma4-26B-A4B (MoE):
#   GPU0: shared vLLM hidden-states server (target, feeds EAGLE3 + DSpark)
#   GPU1: EAGLE3  (scripts/train.py, from scratch)
#   GPU2: DSpark  (scripts/train.py, from scratch)
#   GPU3: MTP     (scripts/gemma4_mtp/train_online.py, fine-tunes the official
#                  assistant; loads its own target on the same GPU — single-GPU
#                  mode, includes the hidden-shift fix)
#
# EAGLE3 and DSpark share one hidden-state cache (--on-generate cache): the
# same sample's features are generated once and reused by both trainers and
# across epochs. Budget ~30-60 MB/sample on disk.
#
# Usage:  bash examples/train/gemma4_26b_tri_draft_online.sh
#         MAX_SAMPLES=20000 EPOCHS=3 bash examples/train/gemma4_26b_tri_draft_online.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ============ Configuration ============
MODEL="${MODEL:-/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it}"            # 30 layers
ASSISTANT="${ASSISTANT:-/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant}"
DATA="${DATA:-/nvmedata/data/kimi-regen-gemma4-26b-moe/train_regen.jsonl}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/gemma4_26b_tri_draft}"
HS_PATH="${HS_PATH:-/nvmedata/chenw/hidden_states_tri_draft}"

MAX_SAMPLES="${MAX_SAMPLES:-5000}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"       # eagle3/dspark (server max_model_len too)
MTP_SEQ_LENGTH="${MTP_SEQ_LENGTH:-2048}"
EPOCHS="${EPOCHS:-2}"
DRAFT_VOCAB_SIZE=32000
TARGET_LAYER_IDS="2 15 27"             # [2, L/2, L-3] for L=30; server appends 30

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU="0"; EAGLE3_GPU="1"; DSPARK_GPU="2"; MTP_GPU="3"

# EAGLE3
EAGLE3_LR=1e-4
# DSpark (hyperparams from dspark_qwen3_0_6b_sharegpt_online.sh)
DSPARK_LR=3e-4
BLOCK_SIZE=8
MAX_ANCHORS=3072
DSPARK_NUM_LAYERS=3
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"
DSPARK_LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0
# MTP (assistant fine-tune; recipe from gemma4_26b_mtp_online.sh)
MTP_LR=6e-4
TTT_STEPS=3
MTP_BATCH_SIZE=1
MTP_GRAD_ACCUM=8
# =======================================

# Env: conda speculator + a CUDA toolkit matching torch cu129 (flashinfer JIT).
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

echo "=== Step 2: launch shared vLLM hidden-states server (GPU $VLLM_GPU) ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPU" python scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HS_PATH" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" --max-model-len "$SEQ_LENGTH" \
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

echo "Waiting for vLLM /health (weight load takes several minutes)..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM died; see $LOGS/vllm_server.log"; exit 1; }
    sleep 5
done
echo "vLLM server ready."

echo "=== Step 3: launch the three trainers concurrently ==="

# --- EAGLE3 (GPU $EAGLE3_GPU) ---
# train.py requires torchrun (maybe_setup_distributed returns None without
# LOCAL_RANK); --standalone picks a free rendezvous port per instance.
CUDA_VISIBLE_DEVICES="$EAGLE3_GPU" torchrun --standalone --nproc_per_node 1 \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$PREP_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --hidden-states-path "$HS_PATH" \
    --save-path "$OUTPUT_ROOT/eagle3/checkpoints" \
    --speculator-type eagle3 \
    --target-layer-ids $TARGET_LAYER_IDS \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$EAGLE3_LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --on-missing generate \
    --on-generate cache \
    > "$LOGS/eagle3.log" 2>&1 &
PIDS+=($!)
echo "  eagle3 pid=${PIDS[-1]}  log=$LOGS/eagle3.log"

# Stagger starts so the shared hidden-state cache warms from one trainer first.
sleep 120

# --- DSpark (GPU $DSPARK_GPU) ---
CUDA_VISIBLE_DEVICES="$DSPARK_GPU" torchrun --standalone --nproc_per_node 1 \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$PREP_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --hidden-states-path "$HS_PATH" \
    --save-path "$OUTPUT_ROOT/dspark/checkpoints" \
    --speculator-type dspark \
    --target-layer-ids $TARGET_LAYER_IDS \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$DSPARK_LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$DSPARK_NUM_LAYERS" \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --loss-fn "$DSPARK_LOSS_FN" \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --no-sample-from-anchor \
    --on-missing generate \
    --on-generate cache \
    > "$LOGS/dspark.log" 2>&1 &
PIDS+=($!)
echo "  dspark pid=${PIDS[-1]}  log=$LOGS/dspark.log"

# --- MTP assistant fine-tune (GPU $MTP_GPU; target+draft share the GPU) ---
CUDA_VISIBLE_DEVICES="$MTP_GPU" python \
    scripts/gemma4_mtp/train_online.py \
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
    --save-every 1000 \
    > "$LOGS/mtp.log" 2>&1 &
PIDS+=($!)
echo "  mtp    pid=${PIDS[-1]}  log=$LOGS/mtp.log"

echo "=== Waiting for all three trainers ==="
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || { echo "trainer pid=$pid FAILED"; FAIL=1; }
done

echo "=== done (fail=$FAIL). Checkpoints under $OUTPUT_ROOT/{eagle3,dspark,mtp}/checkpoints ==="
exit "$FAIL"
