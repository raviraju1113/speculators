#!/bin/bash
# sngpu payload: CONTINUE the overfit gate past epoch 49 to see whether
# position_1_acc reaches the ~0.95 bar, or plateaus below it.
#
# The 50-epoch run ended at position_1_acc ~0.79-0.88 and accept_len ~3-4, still
# rising -- consistent with step-starvation (256 samples ~= 25 optimizer steps per
# epoch) rather than a defect. This settles it.
#
# Costs 1 GPU and no vLLM: the hidden states were cached during the online run
# (ON_GENERATE=cache), so training reads them from disk with --on-missing skip.
# Reuses RUN_NAME=gemma4_31b_dspark_overfit so it RESUMES from epoch 49.
#
#   sngpu --jobname dspark_overfit_ext --partition gpuonly \
#     --exclude sc-c96,sc3-c97,sc-c82 \
#     --gpu 1 --gputype a100m80 --cpu 16 --mem 200000 --time 08:00:00 \
#     --output ./logs/dspark_overfit_ext.txt \
#     -- bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/run_overfit_extend.sh
set -euo pipefail

# Same RUN_NAME as the original gate -> same SAVE_PATH -> resumes at epoch 49.
export RUN_NAME=gemma4_31b_dspark_overfit
export DATA_PATH=/import/ml-sc-scratch1/mengmengj/datasets/gemma4_dspark_256
# The cache written by the online run (21 GB for 256 samples).
export HS_PATH=/import/ml-sc-scratch1/mengmengj/output/hidden_states_gemma4_31b_dspark_overfit
export EPOCHS=200
export LR=1e-3
export MAX_ANCHORS=256
export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-dspark}"

# Snapshot the driver so later edits cannot corrupt this run mid-flight.
_SRC=/import/ml-sc-scratch1/mengmengj/speculators/examples/train/run_overfit_1gpu_offline.sh
_RUN="${SLURM_TMPDIR:-/tmp}/dspark_ext_$$.sh"
cp "$_SRC" "$_RUN"
exec bash "$_RUN"
