#!/bin/bash
# sngpu payload: DSpark Gemma-4 FULL RUN with vLLM data parallelism, lr 1e-4.
# 4 GPUs: 3 independent single-GPU vLLM engines + 1 training rank. No NCCL.
# Run examples/train/test_vllm_dp.sh FIRST to confirm DP works on bare metal.
#
#   sngpu --jobname dspark_dp_lr1e4 --partition gpuonly --exclude sc-c96,sc3-c97,sc-c82 \
#     --gpu 4 --gputype a100m80 --cpu 48 --mem 400000 --time 72:00:00 \
#     --output ./logs/dspark_dp_lr1e4.txt \
#     -- bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/run_dp_lr1e4.sh
set -euo pipefail

export MODE=full_dp
export LR=1e-4
export EPOCHS=10
export RUN_NAME=gemma4_31b_dspark_dp_lr1e4
export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-dspark}"
# With DP=N, vLLM opens a TCPStore per engine on ports derived from the API port
# (API+1, API+2, ...). Reserve headroom and pick a range nothing else uses:
# 8010 collided with something on sc3-c98 and every engine died with
#   DistNetworkError: ... port: 8011 ... EADDRINUSE
# Note DP does use torch.distributed (a TCP/gloo store) even though it avoids NCCL
# collectives -- which is why the DP probe saw 0 NCCL mentions yet still needs ports.
export GPU_IDS=0,1,2,3
export ON_GENERATE=delete

# 10 epochs x 13 GB checkpoints would fill a 97%-full volume, so save every half
# epoch (a DP epoch is ~8 h, capping lost work at ~4 h) and prune to the best.
export CHECKPOINT_FREQ=0.5
export SAVE_BEST=1

# Snapshot the driver: editing a script while a job runs it makes bash resume at a
# stale byte offset and exit silently with status 0.
_SRC=/import/ml-sc-scratch1/mengmengj/speculators/examples/train/dspark_online_gemma4_31b.sh
_RUN="${SLURM_TMPDIR:-/tmp}/dspark_dp_$$.sh"
cp "$_SRC" "$_RUN"
exec bash "$_RUN"
