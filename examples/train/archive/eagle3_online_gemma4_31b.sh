#!/bin/bash
# Online EAGLE-3 draft training for google/gemma-4-31B-it (disaggregated).
#
# vLLM serves the backbone and generates hidden states on the inference GPUs;
# training consumes them on a separate GPU and never writes an offline dump
# (hidden states are streamed on demand and deleted after use).
#
# WHY EAGLE-3 (not MTP): gemma-4-31B-it has no native MTP head, so MTP
# finetuning does not apply. EAGLE-3 (trained from scratch) is the right choice
# here -- it matches the published RedHatAI/gemma-4-31B-it-speculator.eagle3.
# Set SPECULATOR_TYPE=dflash below to train DFlash instead.
#
# Prereqs (on the GPU machine): transformers>=5.x, a vLLM build with Gemma-4
# support, and the local model + dataset already downloaded (see paths below).
# See gemma4_31b_README.md for one-time environment setup.
set -euo pipefail

# Activate the conda env (override with CONDA_ENV=...). Skipped if conda is absent.
CONDA_ENV="${CONDA_ENV:-gemma4-spec}"
if command -v conda > /dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# ============ Configuration ============
MODEL="/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it"
# Local training data (downloaded from lightseekorg/kimi-mtp-dataset). --data
# tokenizes it on the fly, so there is no separate prepare_data.py step.
DATA="/import/ml-sc-scratch5/chenw/datasets/kimi-mtp-dataset/data/train-00000-of-00001.jsonl"
OUTPUT_DIR="./output/gemma4_31b_eagle3"
VLLM_PORT=8000

SPECULATOR_TYPE="eagle3"          # eagle3 | dflash
DRAFT_VOCAB_SIZE=32000            # reduced draft vocab (gemma vocab is large)
MAX_SAMPLES=20000                 # cap for a first run; raise/remove for a full run
SEQ_LENGTH=8192
EPOCHS=3
LR=1e-4

# ---- GPU layout ----------------------------------------------------------
# gemma-4-31B-it is ~62.6 GB (bf16); heads 32/16 => valid tensor-parallel sizes
# are {1,2,4,8} only (7 and 3 are NOT valid). Two layouts:
#
# (A) DEFAULT — safe on 40GB or 80GB: 6 inference GPUs (TP=2 x DP=3) + 1 training.
VLLM_GPUS="0,1,2,3,4,5"
VLLM_TP=2
VLLM_DP=3
TRAIN_GPUS="6"
NUM_TRAIN_GPUS=1
MAX_MODEL_LEN=8192
#
# (B) Full 7-inference + 1-training — ONLY on >=80GB GPUs (weights ~62.6GB leave
#     tight KV headroom, so lower MAX_MODEL_LEN). Uncomment to use:
# VLLM_GPUS="0,1,2,3,4,5,6"; VLLM_TP=1; VLLM_DP=7; TRAIN_GPUS="7"; MAX_MODEL_LEN=4096
# ==========================================================================

# ---- Optional: CUDA forward-compatibility (old <570 driver, datacenter GPUs) --
# Set CUDA_COMPAT=/path/to/cuda-compat-13.0 (forward-compat libcuda from NVIDIA's
# cuda-compat rpm) to run the cu13 vLLM/torch stack on a 12.7 driver. Derives the
# rest from the active conda env; no hardcoded paths. NCCL is broken under
# forward-compat, so this forces SINGLE-GPU vLLM (TP=1, DP=1, --enforce-eager).
VLLM_EXTRA=""
if [ -n "${CUDA_COMPAT:-}" ]; then
    # Derive paths from the activated conda env (gemma4-spec), not ambient `python`.
    PREFIX="${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}"
    SP="$("$PREFIX/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    export LD_LIBRARY_PATH="$PREFIX/lib:$CUDA_COMPAT:${LD_LIBRARY_PATH:-}"
    export PATH="$SP/ninja/data/bin:$SP/nvidia/cu13/bin:$PATH"
    export CUDA_HOME="$SP/nvidia/cu13"
    export PYTHONNOUSERSITE=1 VLLM_USE_FLASHINFER_SAMPLER=0
    echo "[compat] CUDA forward-compat on; NCCL broken -> single-GPU vLLM (TP=1,DP=1)"
    VLLM_GPUS="0"; VLLM_TP=1; VLLM_DP=1; TRAIN_GPUS="1"
    MAX_MODEL_LEN="${COMPAT_MAX_MODEL_LEN:-4096}"
    VLLM_EXTRA="--enforce-eager"
fi

# Step 1: launch vLLM (hidden-state extraction) on the inference GPUs
echo "=== Step 1: launching vLLM (TP=$VLLM_TP, DP=$VLLM_DP) on GPUs $VLLM_GPUS ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    -- --tensor-parallel-size "$VLLM_TP" --data-parallel-size "$VLLM_DP" \
       --max-model-len "$MAX_MODEL_LEN" --port "$VLLM_PORT" $VLLM_EXTRA &
VLLM_PID=$!
cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready (large model + graph capture is slow)..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do sleep 5; done
echo "vLLM server ready."

# Step 2: train the draft on the training GPU(s). --data tokenizes the local
# jsonl on the fly (loss-masked); hidden states stream from the live server and
# are deleted after use, so no offline hidden-state dump is produced.
echo "=== Step 2: training $SPECULATOR_TYPE on GPUs $TRAIN_GPUS ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data "$DATA" \
    --data-path "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --speculator-type "$SPECULATOR_TYPE" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --on-missing generate \
    --on-generate delete

echo "Done. Draft checkpoints saved to $OUTPUT_DIR/checkpoints/"
echo "Evaluate it with: examples/evaluate/eval_gemma4_31b.sh (point the draft at the trained checkpoint)"
