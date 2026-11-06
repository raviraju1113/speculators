#!/bin/bash
# sngpu payload: DSpark Gemma-4 FULL RUN on sc3-c98 (all 4 GPUs: 2 vLLM + 2 train).
# See run_overfit_sc3c98.sh for why this wrapper exists.
#
#   sngpu --jobname dspark_full --partition gpuonly --nodelist sc3-c98 \
#     --gpu 4 --gputype a100m80 --cpu 92 --mem 900000 --time 48:00:00 \
#     --output ./logs/dspark_full.txt \
#     -- bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/run_full_sc3c98.sh
#
# Run the overfit gate FIRST. Do not spend 48 h of queue on an unvalidated config.
set -euo pipefail

export MODE=full
export RUN_NAME=gemma4_31b_dspark
export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-dspark}"

# Start on a 100k-conversation subset, not all 349,389. With --on-generate delete
# every epoch regenerates hidden states through vLLM, so HS generation -- not the
# draft's backward pass -- sets the wall-clock. Build it once with:
#   python scripts/prepare_data.py --model <gemma4-it> --data <train_regen.jsonl> \
#       --seq-length 8192 --max-samples 100000 --output <DATA_ROOT>/gemma4_dspark_100k
# Point at the full gemma4_dspark set once a per-epoch time is known.
export DATA_PATH="${DATA_PATH:-/import/ml-sc-scratch1/mengmengj/datasets/gemma4_dspark_100k}"
export EPOCHS="${EPOCHS:-5}"
export LR="${LR:-3e-4}"

exec bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/dspark_online_gemma4_31b.sh
