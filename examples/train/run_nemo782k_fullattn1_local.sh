#!/bin/bash
# FROM-SCRATCH DSpark training on the nemo782k corpus (no old-corpus seed).
# Ablation partner for gemma4_31b_dspark_anchorT_nemo782k (which continued from
# the old-corpus ep14 checkpoint): at EPOCHS=16, total exposure is ~12.5M
# samples ~= the continued run's lifetime 13M (5.2M old + 7.8M new), so the
# comparison isolates whether the old-corpus phase mattered.
#
# Kept IDENTICAL to the continued run: flag=True, 4+4 layout, ACCUM 18
# (nemo conv/step), lr 6e-4 linear + 4% warmup, quarter-epoch saves, and the
# SAME d2t/t2d vocab mapping (the staged data dir carries the old-corpus
# mapping; the trainer loads it, so mapping is not a confound).
#
#   bash examples/train/run_nemo782k_scratch_local.sh
#
# SCHED_TOTAL: ~1,490 optimizer steps/epoch x 16 = 23,800 (fresh schedule,
# starts at step 0 with warmup -- no warm restart, this is a new model).
set -euo pipefail
export SAMPLE_FROM_ANCHOR=1
export FULL_ATTENTION_INDICES="${FULL_ATTENTION_INDICES:-0}"   # ablation: 1 full (layer 0) + 4 sliding
export WANDB_RUN_ID="${WANDB_RUN_ID:-fullattn1nemo}"

export MODE=full
export VLLM_TP=1
export VLLM_DP=4
export TRAIN_GPUS_N=4
export DATA_PATH=/sms-scratch/mengmengj/data/kimi_nemotron_782k
export ACCUM_STEPS=18
export SCHED_TOTAL="${SCHED_TOTAL:-38700}"   # 26 epochs x ~1,490 opt steps
export LR=6e-4
export EPOCHS="${EPOCHS:-26}"
# WSD: warmup 1 epoch -> plateau at 6e-4 through ~ep22 -> decay last ~4 epochs.
# Stop/resume anywhere on the plateau is free (no warm-restart dip); mid-run
# checkpoints carry PEAK LR, so eval them via a short decay branch or expect
# them to understate final quality vs the linear-schedule runs at the same epoch.
export SCHEDULER_TYPE="${SCHEDULER_TYPE:-wsd}"
export WSD_DECAY_RATIO="${WSD_DECAY_RATIO:-0.15}"
export MAX_ANCHORS=512
export ON_GENERATE=delete
export CHECKPOINT_FREQ=0.25
export SAVE_BEST=0        # --save-best kills sub-epoch saving (trainer.py:517)
export TRAIN_DATA_RATIO=0.998
export RUN_NAME=gemma4_31b_dspark_nemo782k_fullattn1
export JOBNAME="${JOBNAME:-dspark_fullattn1}"
export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-dspark}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/run_local_podman.sh"
