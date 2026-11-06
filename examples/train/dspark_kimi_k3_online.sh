#!/bin/bash
# Online DSpark training for Kimi K3 (single 8x B300 node).
#
# K3 is a ~1.5TB mxfp4 MoE that needs TP8 for the verifier, so unlike the
# Qwen3-0.6B example the verifier and trainer cannot share the node: run
# step 2 (vLLM hidden-state server) and step 3 (training) as separate
# whole-node phases, or point --vllm-endpoint at a server on another node.
#
# K3-specific differences from dspark_qwen3_0_6b_sharegpt_online.sh:
#   * --trust-remote-code everywhere (custom Kimi modeling/tokenizer code).
#   * --draft-config is REQUIRED: K3's MLA config has no head_dim and
#     7168 % 96 heads != 0, so the shaping flags would derive head_dim=74.
#     kimi_k3_dspark_draft_config.json uses 56 heads x 128.
#   * --assistant-pattern is passed explicitly: Kimi renders its chat template
#     in Python (no Jinja, no {% generation %} blocks), so HF assistant-token
#     masks are unavailable and the auto-detector may mis-parse the XTML markers.
#   * Target layers [48, 68, 88] follow the TorchSpec Kimi-K3 EAGLE3 run.
#
# Verify each step on a small MAX_SAMPLES before a full run.

set -euo pipefail

# ============ Configuration ============
MODEL="/import/ml-sc-scratch5/chenw/models/Kimi-K3-patched"
DATASET="kimi_mtp"                # lightseekorg/kimi-mtp-dataset (see configs.py)
OUTPUT_DIR="./output/dspark_kimi_k3"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=2
LR=3e-4

SPECULATOR_TYPE="dspark"
BLOCK_SIZE=8
MAX_ANCHORS=3072
DRAFT_CONFIG="examples/train/kimi_k3_dspark_draft_config.json"
DRAFT_VOCAB_SIZE=32000
# vLLM ids are +1 vs TorchSpec's 0-based layer outputs: TorchSpec's [48,68,88]
# = vLLM [49,69,89]. Capture mode must be prefix_only (leave
# VLLM_KIMI_K3_AUX_ATTN_RES_STREAM unset). Wrong convention silently costs up
# to 16x in draft accuracy — see docs/experiments/kimi_k3_draft_eval.md.
TARGET_LAYER_IDS="49 69 89"       # launch_vllm.py appends the last layer (93)

MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# Kimi K3 assistant-turn marker. The K3 tokenizer renders XTML-style turns:
#   <|open|>message role="assistant"<|sep|> ... <|close|>message<|sep|><|end_of_msg|>
# (the legacy <|im_assistant|> tokens exist in the vocab but are unused).
# Validate the produced loss masks on a few samples before a full prep run.
ASSISTANT_PATTERN='<|open|>message role="assistant"<|sep|>'

# The verifier needs all 8 GPUs (TP8); training runs as a separate phase.
NUM_TRAIN_GPUS=8

# Runtime fixes for the kimi_k3 serving env (see TRAINING.md section 7)
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_WORKSPACE_BASE=/scratch/${USER}/kimi_fi_cache
# =======================================

# Step 1: Prepare data (CPU, can run anytime)
echo "=== Step 1: Preparing data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --trust-remote-code \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH" \
    --assistant-pattern "$ASSISTANT_PATTERN"

# Step 2: Launch the TP8 verifier for hidden-state extraction (whole node)
echo "=== Step 2: Launching vLLM hidden-state server (TP8) ==="
python scripts/launch_vllm.py "$MODEL" \
    --trust-remote-code \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" \
       --tensor-parallel-size 8 \
       --max-model-len "$SEQ_LENGTH" \
       --gpu-memory-utilization 0.90 \
       --limit-mm-per-prompt '{"image": 0}' &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready (K3 load takes ~15-25 min)..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 10
done
echo "vLLM server ready."

# Step 3: Train DSpark. On a single node this phase needs the GPUs the
# verifier is holding -- generate hidden states offline first (--on-missing
# generate, then stop the server and rerun training), or run the trainer on
# a second node pointing at this server.
echo "=== Step 3: Training ==="
torchrun --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --trust-remote-code \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-config "$DRAFT_CONFIG" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --on-missing generate \
    --on-generate delete

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
