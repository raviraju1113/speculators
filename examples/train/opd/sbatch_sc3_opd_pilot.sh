#!/bin/bash
# Draft-OPD end-to-end pilot: one short OPD training run on a rollout dataset
# through the real driver (hidden-state vLLM server + 1 training rank, no NCCL).
# Validates the whole path -- ArrowDataset opd columns -> packing -> given
# anchors -> OPD loss -> metrics -- and gate 2 via the opd_acc_argmax_match
# metric (fraction of accepted slots where the training-side draft argmax
# equals the token the serving draft actually proposed; should be ~1.0).
#
#   cd /import/ml-sc-scratch1/mengmengj/speculators
#   DATA_PATH=output_opd/smoke_12450272/dataset sbatch -w sc3-c98 \
#     --gres=gpu:a100m80:2 examples/train/opd/sbatch_sc3_opd_pilot.sh
#SBATCH -N 1
#SBATCH --partition=gpuonly
#SBATCH --gres=gpu:a100m80:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=03:00:00
#SBATCH -J opd_pilot
#SBATCH --chdir=/import/ml-sc-scratch1/mengmengj/speculators
#SBATCH -o logs/%x-%j.out

set -uo pipefail
: "${DATA_PATH:?set DATA_PATH to a rollouts_to_dataset.py output dir}"
export DATA_PATH
export SEED_CKPT="${SEED_CKPT:-/import/ml-sc-scratch1/mengmengj/output/gemma4_31b_dspark_nemo782k_conv/checkpoints/25}"
export RUN_NAME="${RUN_NAME:-opd_pilot_${SLURM_JOB_ID}}"
export OUT_ROOT="${OUT_ROOT:-/import/ml-sc-scratch1/mengmengj/output_opd}"
export EPOCHS="${EPOCHS:-27}"        # seed is epoch 25 -> trains epoch 26 only
export LR="${LR:-3e-4}"

echo "=== job $SLURM_JOB_ID on $(hostname) $(date '+%F %T') ==="
echo "run $RUN_NAME data $DATA_PATH seed $SEED_CKPT"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

export REPO=/import/ml-sc-scratch1/mengmengj/speculators
export VENV=
export CONDA_SH=/import/snvm-sc-scratch1/mengmengj/miniconda3/etc/profile.d/conda.sh
export CONDA_ENV=/import/ml-sc-scratch1/mengmengj/condaenvs/dspark
export MODEL=/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it
# shellcheck disable=SC1090
source "$CONDA_SH" && conda activate "$CONDA_ENV"

# ---- seed the run dir once (resume path restores weights + optimizer) -------
CK="$OUT_ROOT/$RUN_NAME/checkpoints"
if ! ls -d "$CK"/[0-9]* >/dev/null 2>&1; then
  mkdir -p "$CK"
  SEED_EPOCH=$(basename "$SEED_CKPT")
  echo "seeding $CK/$SEED_EPOCH from $SEED_CKPT"
  cp -r "$SEED_CKPT" "$CK/$SEED_EPOCH.tmp" && mv "$CK/$SEED_EPOCH.tmp" "$CK/$SEED_EPOCH" || { echo "!! seed copy failed" >&2; exit 3; }
fi

# ---- recipe: conv architecture, OPD data, tiny run ---------------------------
export MODE=full
export SAMPLE_FROM_ANCHOR=1
export DSPARK_CONV_KERNEL=2
export DSPARK_CONV_GROUP=16
export ACCUM_STEPS=1              # plain train.py
export MAX_ANCHORS="${MAX_ANCHORS:-512}"
export CHECKPOINT_FREQ=1
export SAVE_BEST=0
export TRAIN_DATA_RATIO="${TRAIN_DATA_RATIO:-0.9}"
export ON_GENERATE=delete
export VLLM_TP=1
export VLLM_DP=1
export TRAIN_GPUS_N=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT=gemma4-dspark-opd
export GPU_IDS="${GPU_IDS:-}"
# A100-80: the verifier needs util 0.95 for an 8448 context at TP=1.
export GPU_MEM_UTIL=0.95

rm -rf "/dev/shm/hidden_states_${RUN_NAME}"
_RUN=$(mktemp "${TMPDIR:-/tmp}/dspark_driver_XXXXXX.sh")
cp "$REPO/examples/train/dspark_online_gemma4_31b.sh" "$_RUN"
setsid bash "$_RUN" &
DRIVER_PID=$!
trap 'kill -TERM -- -"$DRIVER_PID" 2>/dev/null' TERM INT
wait "$DRIVER_PID"; RC=$?
rm -f "$_RUN"
echo "=== driver exited rc=$RC $(date '+%F %T') ==="
echo "=== OPD metrics (last lines) ==="
grep -aoE '(opd_[a-z_]+|position_[0-7]_acc|accept_len|loss)=[0-9.e+-]+' "logs/opd_pilot-${SLURM_JOB_ID}.out" | tail -20
exit "$RC"
