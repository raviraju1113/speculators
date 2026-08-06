#!/bin/bash
set -euo pipefail
# --- container-side environment ---
export HF_HOME=/import/snvm-sc-scratch1/mengmengj/hf_cache
export WANDB_PROJECT=gemma4-dspark
export MODE=full
export LR=1e-4
export EPOCHS=10
export RUN_NAME=gemma4_31b_dspark_bal_lr1e4
export DATA_PATH=/import/ml-sc-scratch1/mengmengj/datasets/gemma4_dspark
export VLLM_TP=1
export VLLM_DP=2
export TRAIN_GPUS_N=2
export MAX_ANCHORS=512
export ON_GENERATE=delete
export CHECKPOINT_FREQ=0.25
export SAVE_BEST=0
export ACCUM_STEPS=0
export TRAIN_DATA_RATIO=0.998
export GPU_IDS=0\,1\,2\,3
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
echo "shm: $(df -h /dev/shm | tail -1)"
_RUN=$(mktemp /tmp/dspark_driver_XXXXXX.sh)
cp /import/ml-sc-scratch1/mengmengj/speculators/examples/train/dspark_online_gemma4_31b.sh "$_RUN"
exec bash "$_RUN"
