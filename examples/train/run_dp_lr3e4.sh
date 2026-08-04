#!/bin/bash
# sngpu payload: DSpark Gemma-4 FULL RUN with vLLM data parallelism, lr 3e-4.
# 4 GPUs: 3 independent single-GPU vLLM engines + 1 training rank. No NCCL.
# Run examples/train/test_vllm_dp.sh FIRST to confirm DP works on bare metal.
#
#   sngpu --jobname dspark_dp_lr3e4 --partition gpuonly --exclude sc-c96,sc3-c97,sc-c82 \
#     --gpu 4 --gputype a100m80 --cpu 48 --mem 400000 --time 72:00:00 \
#     --output ./logs/dspark_dp_lr3e4.txt \
#     -- bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/run_dp_lr3e4.sh
set -euo pipefail

export MODE=full_dp
export LR=3e-4
export EPOCHS=2
export RUN_NAME=gemma4_31b_dspark_dp_lr3e4
export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-dspark}"
export VLLM_PORT=8000
export GPU_IDS=0,1,2,3
export ON_GENERATE=delete

# Snapshot the driver: editing a script while a job runs it makes bash resume at a
# stale byte offset and exit silently with status 0.
_SRC=/import/ml-sc-scratch1/mengmengj/speculators/examples/train/dspark_online_gemma4_31b.sh
_RUN="${SLURM_TMPDIR:-/tmp}/dspark_dp_$$.sh"
cp "$_SRC" "$_RUN"
exec bash "$_RUN"
