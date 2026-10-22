#!/bin/bash
# Draft-OPD full stage on one 8x B200 node (preset over run_rounds.sh).
# Submit ONLY when the sc3-cp pool is usable (sn50-hdi-0 reservation access).
#
#   cd /import/ml-sc-scratch1/mengmengj/speculators
#   PROMPTS=data/opd_prompts/<pool>.jsonl sbatch examples/train/opd/sbatch_sc3_opd_full.sh
#
# Layout per round: rollout server on the first allocated GPU (concurrency 32),
# then training with 4 hidden-state engines + 4 ranks (the SFT layout).
# Sizing (PLAN.md): one micro-step = one 8192-token packed sequence (~10-13
# rollouts), 4 ranks x ACCUM 1 ~= 45 prompts/optimizer step; 16K prompts x 8
# passes / 45 ~= 2,850 optimizer steps -> SCHED_TOTAL 2850, cosine, warmup
# 0.05, LR 5e-5 (see run_rounds.sh for why not the paper's 3e-4). Gate before
# trusting the result: val opd_acc_argmax_match stays >= 0.98 and round-to-round
# rollout accept_rate (rounds/r*/assemble.txt) RISES.
#SBATCH -N 1
#SBATCH --partition=gpuonly
#SBATCH --gres=gpu:b200m180:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=6-00:00:00
#SBATCH --signal=B:TERM@900
#SBATCH --exclude=sc3-cp78,sc3-cp79,sc3-cp80
#SBATCH -J opd_full
#SBATCH --chdir=/import/ml-sc-scratch1/mengmengj/speculators
#SBATCH -o logs/%x-%j.out

export ROUNDS="${ROUNDS:-8}"
export P="${P:-16000}"
export CONC="${CONC:-32}"
export MAX_TOKENS="${MAX_TOKENS:-2048}"
export VLLM_DP="${VLLM_DP:-4}"
export TRAIN_GPUS_N="${TRAIN_GPUS_N:-4}"
export ACCUM_STEPS="${ACCUM_STEPS:-1}"
export SCHED_TOTAL="${SCHED_TOTAL:-2850}"
export SCHEDULER_TYPE="${SCHEDULER_TYPE:-cosine}"
export LR="${LR:-5e-5}"
export KEEP_FULL="${KEEP_FULL:-1}"
export RUN_NAME="${RUN_NAME:-opd_full_${SLURM_JOB_ID}}"
export WANDB_MODE="${WANDB_MODE:-online}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
exec bash examples/train/opd/run_rounds.sh
