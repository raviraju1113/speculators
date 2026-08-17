#!/bin/bash
# CONTINUED TRAINING on the NEW 782k dataset (kimi-mtp-nemotron-stem-code-math),
# seeded from the anchorT run's epoch-14 checkpoint (copied into this run's
# checkpoints/ dir -- lineage: 15 epochs on the 349k kimi-regen corpus first).
#
#   bash examples/train/run_anchorT_nemo782k_local.sh
#
# Design (sized 2026-08-14):
#   flag=True         -- won every matched-epoch comparison; also required, the
#                        seed checkpoint is anchor-trained.
#   EPOCHS=25         -- epochs 15..24 = 10 epochs of new data (7.8M samples).
#                        Total exposure then ~13M samples, matching the DSpark
#                        paper's 1.3M x 10 budget. Stop early if val flattens;
#                        quarter-epoch checkpoints make that cheap.
#   ACCUM_STEPS=18    -- new data averages ~1,043 tok/sample (old: ~1,356), so
#                        a packed 8192 step carries ~7.3 conversations.
#                        512 / (7.3 x 4 ranks) ~= 18 keeps global batch ~512,
#                        which is what justifies lr 6e-4. (Old corpus used 23.)
#   SCHED_TOTAL=25000 -- resumed optimizer counter is 10,108; new data adds
#                        ~1,490 optimizer steps/epoch x 10. The driver's builtin
#                        formula assumes the OLD corpus's epoch length -- must
#                        override. Warm restart lands at ~3.6e-4, decaying to 0
#                        at epoch 24.
#   NO REPLAY of the old corpus for now: new mix targets math/code (our weak
#   benchmarks); revisit a 4:1 mix only if gpqa/chat regresses at checkpoint 15.
set -euo pipefail
export SAMPLE_FROM_ANCHOR=1
export WANDB_RUN_ID="${WANDB_RUN_ID:-anchortnemo782k}"

export MODE=full
export DATA_PATH=/sms-scratch/mengmengj/data/kimi_nemotron_782k
export VLLM_TP=1
export VLLM_DP=4
export TRAIN_GPUS_N=4
export ACCUM_STEPS=18
export SCHED_TOTAL=25000
export LR=6e-4
export EPOCHS=25
export MAX_ANCHORS=512
export ON_GENERATE=delete
export CHECKPOINT_FREQ=0.25
export SAVE_BEST=0
export TRAIN_DATA_RATIO=0.998
export RUN_NAME=gemma4_31b_dspark_anchorT_nemo782k
export JOBNAME="${JOBNAME:-dspark_anchorT_nemo}"
export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-dspark}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/run_local_podman.sh"
