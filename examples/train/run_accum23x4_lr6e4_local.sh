#!/bin/bash
# 8-GPU preset for the STANDALONE B200 box, mirroring run_bal_accum_lr6e4.sh
# (which was the sngpu/submit_docker payload on the cluster). Here there is no
# SLURM: this exports the run settings and hands off to run_local_podman.sh,
# which docker-runs the mengmengj-vllm-official image.
#
#   bash examples/train/run_accum23x4_lr6e4_local.sh
#
# Differences vs the cluster's 2+2 run, all deliberate:
#
# 1. BALANCED 4 engines + 4 ranks (8 GPUs). One engine and one rank each cap
#    near the same seq/h, so scale BOTH sides -- same reasoning that made 2+2
#    ~2x over 3+1, one step further.
#
# 2. ACCUM_STEPS=23, NOT 46. Global batch = 5.6 conv/rank-step x ranks x accum;
#    46 was sized for 2 ranks (512 / (5.6 x 2)). With 4 ranks, 23 keeps the SAME
#    ~512 global batch that makes LR 6e-4 valid. 46 would double it.
#
# 3. SAMPLE_FROM_ANCHOR=0: --no-sample-from-anchor, matching RedHat's shipped
#    checkpoint. (The image's algos.py patch serves either kind correctly, but
#    False is the decision as of 2026-08-06.)
set -euo pipefail
export SAMPLE_FROM_ANCHOR=0
# Pin to the original 2026-08-06 wandb run so stop/resume continues ONE graph
# (logs/wandb/latest-run -> run-20260806_114002-po2pjgk1).
export WANDB_RUN_ID="${WANDB_RUN_ID:-po2pjgk1}"

export MODE=full
export VLLM_TP=1
export VLLM_DP=4          # 4 engines
export TRAIN_GPUS_N=4     # 4 ranks  -> 8 GPUs total
export ACCUM_STEPS=23     # 512 / (5.6 x 4)
export LR=6e-4            # DeepSpec's, valid once the batch matches
# Overridable for CONTINUATION runs (e.g. EPOCHS=15 to extend the finished
# 10-epoch run). NOTE the LR schedule is rebuilt over the NEW total: resuming a
# finished 10-epoch run with EPOCHS=15 makes the restored step (~6,738) sit at
# 2/3 of the new schedule, so LR warm-restarts at ~2e-4 and decays to 0 at 15.
export EPOCHS="${EPOCHS:-10}"
export MAX_ANCHORS=512
export ON_GENERATE=delete
export CHECKPOINT_FREQ=0.25
# SAVE_BEST MUST be 0: --save-best disables sub-epoch saving entirely
# (trainer.py:517) and cost a 25 h run on 2026-08-06.
export SAVE_BEST=0
# Validation REGENERATES hidden states; 0.998 leaves ~700 val samples (enough
# for a generalization trend) instead of burning ~38% of the run validating.
export TRAIN_DATA_RATIO=0.998
export RUN_NAME=gemma4_31b_dspark_accum23x4_lr6e4
export JOBNAME="${JOBNAME:-dspark_accum23x4}"
export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-dspark}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/run_local_podman.sh"
