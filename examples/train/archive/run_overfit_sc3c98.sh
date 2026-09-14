#!/bin/bash
# sngpu payload: DSpark Gemma-4 OVERFIT GATE on sc3-c98 (3 of 4 GPUs).
# Exists because sngpu -> `sbatch --wrap` accepts only `bash <one-path>` with no
# arguments, so env vars cannot be set on the submit line.
#
#   sngpu --jobname dspark_overfit --partition gpuonly --nodelist sc3-c98 \
#     --gpu 3 --gputype a100m80 --cpu 24 --mem 200000 --time 01:00:00 \
#     --output ./logs/dspark_overfit.txt \
#     -- bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/run_overfit_sc3c98.sh
#
# Gates to check in the log / W&B:
#   loss -> near zero, position_1_acc > 95%, confidence_pred_mean off 0.5,
#   accept_len climbing. A high plateau means target-layer-ids / mask token /
#   vocab mapping are wrong -- fix here, not in the full run.
set -euo pipefail

export MODE=overfit
export RUN_NAME=gemma4_31b_dspark_overfit
export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-dspark}"
# 256-sample split; override if you only built the 32-sample one.
export DATA_PATH="${DATA_PATH:-/import/ml-sc-scratch1/mengmengj/datasets/gemma4_dspark_256}"

exec bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/dspark_online_gemma4_31b.sh
