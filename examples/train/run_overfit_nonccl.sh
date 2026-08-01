#!/bin/bash
# sngpu payload: DSpark Gemma-4 OVERFIT GATE with **no NCCL anywhere** (2 GPUs).
#
# Why: on sc3-c98 (2026-07-31) NCCL 2.28.9+cuda13.0 segfaults inside
# ncclNetPluginInit (plugin/net.cc:216) during ncclCommInitRank. It reproduces in
# vLLM's TP=2 worker init and in a bare 2-rank all_reduce, and NCCL_NET_PLUGIN=none
# does not help. This layout sidesteps collectives entirely:
#   * vLLM TP=1 DP=1 -> uniproc executor, no NCCL
#   * 1 training rank launched as plain `python` (not torchrun) -> no LOCAL_RANK,
#     so maybe_setup_distributed() returns early and never calls
#     init_process_group("nccl")
#
# The 62.6 GB verifier fits on one 80 GB A100 with ~17 GB for KV, which is ample
# for the short prefill-only hidden-state requests this pipeline makes.
#
#   sngpu --jobname dspark_overfit_nonccl --partition gpuonly --nodelist sc3-c98 \
#     --gpu 2 --gputype a100m80 --cpu 24 --mem 200000 --time 01:00:00 \
#     --output ./logs/dspark_overfit_nonccl.txt \
#     -- bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/run_overfit_nonccl.sh
set -euo pipefail

export MODE=overfit
export RUN_NAME=gemma4_31b_dspark_overfit
export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-dspark}"
export DATA_PATH="${DATA_PATH:-/import/ml-sc-scratch1/mengmengj/datasets/gemma4_dspark_256}"

# The whole point: one GPU for vLLM, one for training, no collectives.
export VLLM_TP=1
export VLLM_DP=1
export TRAIN_GPUS_N=1

# Cache hidden states instead of discarding them. With 'delete' every one of the
# 50 overfit epochs re-derives all 256 samples through vLLM, which measured at
# ~3.2 min/epoch (~2.6 h for the full run) with the GPU work dominated by
# regeneration, not by the draft's backward pass. Caching costs ~6.6 GB here
# (64.5 KB/token) and makes epochs 2+ read from disk.
# Do NOT copy this to the full run: 100k samples would be ~2.6 TB.
export ON_GENERATE=cache

exec bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/dspark_online_gemma4_31b.sh
