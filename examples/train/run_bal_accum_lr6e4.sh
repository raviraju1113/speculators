#!/bin/bash
# sngpu payload (via submit_docker.sh): BALANCED layout + GRADIENT ACCUMULATION.
#
# Two changes vs everything before:
#
# 1. BALANCED 2 engines + 2 ranks. Measured three times: one vLLM engine and one
#    training rank both cap near ~1,250 sequences/h, so they are matched and
#    scaling either side ALONE gains nothing (3 engines + 1 rank = 1,250;
#    1 engine + 3 ranks = 1,195). 2+2 preserves the ratio and should give ~2x,
#    i.e. ~18 h/epoch instead of 36.
#
# 2. GRADIENT ACCUMULATION to DeepSpec's effective batch, which unlocks their lr.
#    ~6.9 conversations per rank-step x 2 ranks = 13.8 per global step;
#    37 x 13.8 = 511 ~= their global_batch_size 512. Only THEN is lr 6e-4 the
#    right value -- at our un-accumulated batch it would likely diverge.
#    Accumulation buys gradient quality, NOT throughput.
set -euo pipefail

export MODE=full
export VLLM_TP=1
export VLLM_DP=2          # 2 engines
export TRAIN_GPUS_N=2     # 2 ranks  -> 4 GPUs total
export ACCUM_STEPS=37     # 512 / (6.9 * 2)
export LR=6e-4            # DeepSpec's, valid once the batch matches
export EPOCHS=10
export MAX_ANCHORS=512
export ON_GENERATE=delete
export CHECKPOINT_FREQ=0.5
export SAVE_BEST=1
export RUN_NAME=gemma4_31b_dspark_bal_accum_lr6e4
export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-dspark}"
