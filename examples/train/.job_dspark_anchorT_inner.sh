#!/bin/bash
set -euo pipefail
export HF_HOME=/home/mengmengj/.cache/huggingface
export WANDB_PROJECT=gemma4-dspark
export WANDB_MODE=online
export WANDB_RESUME=allow
export WANDB_RUN_ID=anchort23x4v2
export VENV=/sms-scratch/mengmengj/envs/dspark
export MODE=full
export LR=6e-4
export EPOCHS=15
export RUN_NAME=gemma4_31b_dspark_accum23x4_lr6e4_anchorT
export DATA_PATH=/sms-scratch/mengmengj/data/gemma4_dspark
export MODEL=/sms-scratch/checkpoints/gemma-4-31B-it
export OUT_ROOT=/sms-scratch/mengmengj/output
export VLLM_TP=1
export VLLM_DP=4
export TRAIN_GPUS_N=4
export MAX_ANCHORS=512
export ON_GENERATE=delete
export CHECKPOINT_FREQ=0.25
export SAVE_BEST=0
export TRAIN_DATA_RATIO=0.998
export SAMPLE_FROM_ANCHOR=1
export ACCUM_STEPS=23
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
_RUN=$(mktemp /tmp/dspark_driver_XXXXXX.sh)
cp /sms-scratch/mengmengj/speculators/examples/train/dspark_online_gemma4_31b.sh "$_RUN"
exec bash "$_RUN"
