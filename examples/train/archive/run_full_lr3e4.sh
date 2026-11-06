#!/bin/bash
# sngpu payload: DSpark Gemma-4 FULL RUN, lr 3e-4 (upstream DFlash/DSpark example default).
# Pair with run_full_lr1e4.sh on the other node to settle the LR question in one wall-clock.
#
# Layout: vLLM TP=1 (GPU 0) + 1 training rank as plain `python` (GPU 1) -> NO NCCL.
# 2 GPUs, so two of these fit on one 4-GPU node.
#
#   sngpu --jobname dspark_lr3e4 --partition gpuonly \
#     --exclude sc-c96,sc3-c97,sc-c82 \
#     --gpu 2 --gputype a100m80 --cpu 24 --mem 200000 --time 48:00:00 \
#     --output ./logs/dspark_full_lr3e4.txt \
#     -- bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/run_full_lr3e4.sh
set -euo pipefail

export MODE=full
export LR=3e-4
export EPOCHS=2
export RUN_NAME=gemma4_31b_dspark_full_lr3e4
export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-dspark}"

# FULL corpus (349,138 samples, ~1356 tokens each = ~473M tokens/epoch).
# At the measured ~6k tok/s generation rate (TP=1) that is ~22 h/epoch, so 2
# epochs ~= 44 h -- set --time 48:00:00 and expect it may need one resubmission
# (training auto-resumes from the last checkpoint).
# Why full x2 rather than 50k x5: it sees ~946M tokens, all unique, versus 340M
# with 5x repetition. Ravi's own result was that more data with fewer epochs
# beats less data with more epochs (50k x5 > 25k x10, less overfitting).
export DATA_PATH="${DATA_PATH:-/import/ml-sc-scratch1/mengmengj/datasets/gemma4_dspark}"

# No NCCL: TP=1 vLLM, single training rank launched as plain python.
export VLLM_TP=1
export VLLM_DP=1

# Distinct ports: if both LR arms land on the SAME node they would otherwise
# both try to bind 8000 and the second vLLM would fail to start.
export VLLM_PORT=8000
export TRAIN_GPUS_N=1

# MUST stay 'delete' at this scale: hidden states measured at ~74 MB/sample,
# so caching the full set would be ~26 TB. Only the tiny overfit split can afford cache.
export ON_GENERATE=delete

# Pin GPUs explicitly. Free-memory detection races when two jobs start close
# together: a job reserves its training GPU but does not touch it until vLLM has
# finished loading (~7 min), so the other job sees it as idle and double-books it
# (observed 2026-08-01: both jobs landed a verifier and a trainer on GPU 1).
export GPU_IDS=2,3

# Run a SNAPSHOT of the driver script, not the file itself.
# bash reads scripts incrementally by byte offset, so editing the file while a job
# is executing it makes bash resume at a stale offset -- observed 2026-08-01 as a
# silent exit 0 right after "vLLM ready" (job 58650593), with no error at all.
_SRC=/import/ml-sc-scratch1/mengmengj/speculators/examples/train/dspark_online_gemma4_31b.sh
_RUN="${SLURM_TMPDIR:-/tmp}/dspark_driver_$$.sh"
cp "$_SRC" "$_RUN"
exec bash "$_RUN"
