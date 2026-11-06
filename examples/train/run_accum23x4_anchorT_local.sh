#!/bin/bash
# ABLATION preset: identical to run_accum23x4_lr6e4_local.sh EXCEPT
# sample_from_anchor=TRUE. Purpose: measure whether the flag changes an EARLY
# checkpoint's eval numbers (the ep0-dp-run's strong k=3 aime suggested True's
# first-slot advantage matters; this isolates the flag from the other two
# variables that differed there: lr/batch regime and update count).
#
#   bash examples/train/run_accum23x4_anchorT_local.sh
#
# Everything that must stay IDENTICAL for a clean ablation is identical:
# layout (4+4), ACCUM_STEPS=23, lr 6e-4, EPOCHS=10 (same 6,738-step LR
# schedule -- do NOT shorten EPOCHS or the schedule changes; just stop the run
# once the checkpoints you want exist), data ratio, checkpoint cadence.
#
# What is deliberately DIFFERENT from the main run:
#   SAMPLE_FROM_ANCHOR=1  -> trains with --sample-from-anchor (upstream default;
#                            native k=8 at serving). Eval REQUIRES the patched
#                            vllm in mengmengj-vllm-official (algos.py derives
#                            dspark_bonus_anchor from the checkpoint config) --
#                            stock vllm decodes True checkpoints one slot off
#                            with no error (~1.8 instead of ~3.5 accept_len).
#   RUN_NAME / JOBNAME     -> own checkpoints, logs, container name.
#   WANDB_RUN_ID unset     -> fresh wandb run, NOT appended to po2pjgk1's graph.
#
# Comparison recipe once running: eval this run's checkpoint N against the main
# run's checkpoint N (same step count, same schedule) with the meeting yaml --
# k=8 for this one (native), k=7 for the main run, plus matched k=3 rows.
set -euo pipefail
export SAMPLE_FROM_ANCHOR=1
# Pin a FIXED id so stop/resume continues one graph. NOT s0v5b574: that run was
# deleted server-side, and wandb permanently refuses deleted ids (CommError
# "previously created and deleted" -- crashed both relaunches 2026-08-10).
# This id starts a fresh graph from the resume point (epoch ~8).
export WANDB_RUN_ID="${WANDB_RUN_ID:-anchort23x4v2}"

export MODE=full
export VLLM_TP=1
export VLLM_DP=4          # 4 engines
export TRAIN_GPUS_N=4     # 4 ranks  -> 8 GPUs total
export ACCUM_STEPS=23     # 512 / (5.6 x 4) -- same global batch ~512
export LR=6e-4
# Overridable for the 15-epoch continuation, mirroring the main run's two-stage
# trajectory: finish 10 (LR decays to 0), THEN relaunch with EPOCHS=15 (warm
# restart). Do not jump to 15 mid-schedule -- it would diverge from how the
# main run got its ep10-14, breaking matched-epoch comparisons.
export EPOCHS="${EPOCHS:-10}"
export MAX_ANCHORS=512
export ON_GENERATE=delete
export CHECKPOINT_FREQ=0.25
export SAVE_BEST=0        # --save-best kills sub-epoch saving (trainer.py:517)
export TRAIN_DATA_RATIO=0.998
export RUN_NAME=gemma4_31b_dspark_accum23x4_lr6e4_anchorT
export JOBNAME="${JOBNAME:-dspark_anchorT}"
export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-dspark}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/run_local_podman.sh"
