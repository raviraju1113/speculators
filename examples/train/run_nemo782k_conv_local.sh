#!/bin/bash
# FROM-SCRATCH DSpark + DFlash2 local convolution on nemo782k.
# Architecture ablation partner for gemma4_31b_dspark_nemo782k_scratch: the
# ONLY difference is DSPARK_CONV_KERNEL=2, which wraps each of the 5 draft
# layers' attention and MLP in a two-tap grouped dynamic causal convolution
# along the block axis (DFlash2, z-lab 2026-08; +72.5M params on the 5376-wide
# draft). No candidate selector, no DFlash2 loss -- Markov head, confidence
# head, loss mix, data, schedule and seed are the scratch run's.
#
#   bash examples/train/run_nemo782k_conv_local.sh
#   (SLURM: from the bastion via a sbatch wrapper, as for the other presets)
#
# Schedule mirrors the scratch/fullattn runs' FINAL config: 26 epochs,
# SCHED_TOTAL 38,700 (~1,490 optimizer steps/epoch x 26), lr 6e-4 linear +
# 4% warmup, quarter-epoch saves. Extensions are warm restarts: same RUN_NAME,
# larger EPOCHS, recomputed SCHED_TOTAL.
set -euo pipefail
export SAMPLE_FROM_ANCHOR=1
# nemoconv782k = job 319 on 097 (2026-10-13), cancelled at micro-step ~2970 with no
# checkpoint when we moved machines. A fresh start logs from step 0, and wandb
# resume=allow silently DROPS steps below the old run's last step -- so every
# from-scratch relaunch needs a NEW id (never delete the old run: ids are burned).
export WANDB_RUN_ID="${WANDB_RUN_ID:-nemoconv782k_v2}"

export DSPARK_CONV_KERNEL=2
export DSPARK_CONV_GROUP=16

export MODE=full
export VLLM_TP=1
export VLLM_DP=4
export TRAIN_GPUS_N=4
export DATA_PATH=/sms-scratch/mengmengj/data/kimi_nemotron_782k
export ACCUM_STEPS=18
export SCHED_TOTAL="${SCHED_TOTAL:-38700}"
export LR=6e-4
export EPOCHS="${EPOCHS:-26}"
export MAX_ANCHORS=512
export ON_GENERATE=delete
export CHECKPOINT_FREQ=0.25
export SAVE_BEST=0        # --save-best kills sub-epoch saving (trainer.py:517)
export TRAIN_DATA_RATIO=0.998
export RUN_NAME=gemma4_31b_dspark_nemo782k_conv
export JOBNAME="${JOBNAME:-dspark_nemo_conv}"
export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-dspark}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/run_local_podman.sh"
