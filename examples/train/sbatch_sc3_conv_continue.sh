#!/bin/bash
# Continue the nemo782k DSpark+conv run (gemma4_31b_dspark_nemo782k_conv,
# epoch 25 = checkpoint_best) for EXTRA epochs under a NEW run name -- the
# block-size ablation and its control:
#
#   block 12, gamma 12 (the experiment):
#     BLOCK_SIZE=12 DECAY_GAMMA=12 RUN_NAME=gemma4_31b_dspark_nemo782k_conv_blk12 \
#       WANDB_RUN_ID=convblk12v1 sbatch examples/train/sbatch_sc3_conv_continue.sh
#   block 8, gamma 4 (matched-exposure control: same seed, schedule, epochs):
#     BLOCK_SIZE=8 DECAY_GAMMA=4 RUN_NAME=gemma4_31b_dspark_nemo782k_conv_blk8ctrl \
#       WANDB_RUN_ID=convblk8ctrlv1 sbatch examples/train/sbatch_sc3_conv_continue.sh
#
# Mechanics (the established "continue from a checkpoint" path, as for
# anchorT_nemo782k and the fullattn extensions): the seed checkpoint dir is
# COPIED into <OUT_ROOT>/<RUN_NAME>/checkpoints/25 once, and the trainer resumes
# from it: model + optimizer + scheduler state and global_step. The draft is
# built from the CLI flags, so BLOCK_SIZE takes effect from the checkpoint's
# weights (nothing in them has a block-size dimension: convs, Markov head,
# confidence head, no slot embeddings). EPOCHS=34 trains epochs 26..33;
# SCHED_TOTAL is the optimizer-step budget of the whole 34-epoch trajectory
# (38700 * 34 / 26), so the linear schedule restarts from ~1.4e-4 and decays
# to 0 -- the usual warm-restart dip, identical for both runs.
# DECAY_GAMMA: per-position loss weight exp(-(k-1)/gamma). At gamma=4 the new
# slots 9..12 would get weights 0.14..0.06 and barely train, hence 12 for the
# block-12 run; the control keeps the recipe's 4.
# Everything else is the conv recipe (sbatch_sc3_nemo782k_conv.sh).
#SBATCH -J dspark_conv_cont
#SBATCH -N 1
#SBATCH --partition=gpuonly
#SBATCH --gres=gpu:b200m180:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
# 8 epochs at ~4.9 h/epoch (block 8); block 12 is a little slower.
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:TERM@900
#SBATCH --chdir=/import/ml-sc-scratch1/mengmengj/speculators
#SBATCH -o logs/%x-%j.out

set -uo pipefail
: "${BLOCK_SIZE:?set BLOCK_SIZE (8 or 12)}"
: "${RUN_NAME:?set RUN_NAME}"
: "${WANDB_RUN_ID:?set WANDB_RUN_ID (new id per run)}"
export BLOCK_SIZE RUN_NAME WANDB_RUN_ID
export DECAY_GAMMA="${DECAY_GAMMA:-4.0}"
export SEED_CKPT="${SEED_CKPT:-/import/ml-sc-scratch1/mengmengj/output/gemma4_31b_dspark_nemo782k_conv/checkpoints/25}"
export EPOCHS="${EPOCHS:-34}"
export SCHED_TOTAL="${SCHED_TOTAL:-50600}"

echo "=== job $SLURM_JOB_ID on $(hostname) $(date '+%F %T') ==="
echo "run $RUN_NAME: block $BLOCK_SIZE gamma $DECAY_GAMMA epochs ->$EPOCHS sched $SCHED_TOTAL seed $SEED_CKPT"
nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv,noheader
echo "out : $(df -h /import/ml-sc-scratch1 | tail -1)"

export REPO=/import/ml-sc-scratch1/mengmengj/speculators
export VENV=
export CONDA_SH=/import/snvm-sc-scratch1/mengmengj/miniconda3/etc/profile.d/conda.sh
export CONDA_ENV=/import/ml-sc-scratch1/mengmengj/condaenvs/dspark
export MODEL=/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it
export DATA_PATH=/import/ml-sc-scratch1/mengmengj/datasets/kimi_nemotron_782k
export OUT_ROOT=/import/ml-sc-scratch1/mengmengj/output

# ---- W&B gate -------------------------------------------------------------
export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-dspark}"
export WANDB_MODE=online
export WANDB_RESUME=allow
# shellcheck disable=SC1090
source "$CONDA_SH" && conda activate "$CONDA_ENV"
if ! timeout 180 python - <<'PY'
import os, wandb
v = wandb.Api(timeout=60).viewer
print(f"wandb OK on this node: entity={v.entity} user={v.username} run_id={os.environ['WANDB_RUN_ID']}", flush=True)
PY
then
  echo "!! wandb API unreachable from $(hostname) -- NOT starting training" >&2; exit 3
fi

# ---- GPU self-test and layout (as in sbatch_sc3_nemo782k_conv.sh) ----------
GOOD=(); BAD=()
for i in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do
  if CUDA_VISIBLE_DEVICES=$i timeout 120 python -c \
      'import torch; torch.cuda.init(); (torch.ones(1, device="cuda") + 1).item()' \
      >/dev/null 2>&1; then GOOD+=("$i"); else BAD+=("$i"); fi
done
echo "GPU self-test on $(hostname -s): healthy=${GOOD[*]} bad=${BAD[*]:-none}"
case "${#GOOD[@]}" in
  8) VLLM_DP_N=4 ;;
  7) VLLM_DP_N=3; echo "NOTE: 3 vLLM engines + 4 training ranks on the 7 healthy GPUs" ;;
  *) echo "!! only ${#GOOD[@]} healthy GPUs on $(hostname) -- resubmit with --exclude=$(hostname -s)" >&2; exit 4 ;;
esac
export GPU_IDS=$(IFS=,; echo "${GOOD[*]}")

# ---- seed the run dir with the checkpoint (once) ---------------------------
CK="$OUT_ROOT/$RUN_NAME/checkpoints"
if ! ls -d "$CK"/[0-9]* >/dev/null 2>&1; then
  [ -f "$SEED_CKPT/model.safetensors" ] || { echo "!! seed checkpoint missing: $SEED_CKPT" >&2; exit 3; }
  SEED_EPOCH=$(basename "$SEED_CKPT")
  echo "seeding $CK/$SEED_EPOCH from $SEED_CKPT ($(du -sh "$SEED_CKPT" | cut -f1))"
  mkdir -p "$CK"
  cp -r "$SEED_CKPT" "$CK/$SEED_EPOCH.tmp" && mv "$CK/$SEED_EPOCH.tmp" "$CK/$SEED_EPOCH" || { echo "!! seed copy failed" >&2; exit 3; }
  # the copy keeps block_size 8 in its config.json; the trainer builds the draft
  # from the CLI (--block-size) and only loads the weights, so no edit is needed.
else
  echo "run dir already has checkpoints -- resuming from the newest"
fi

# ---- recipe (conv run) + the continuation knobs ----------------------------
export MODE=full
export SAMPLE_FROM_ANCHOR=1
export DSPARK_CONV_KERNEL=2
export DSPARK_CONV_GROUP=16
export ACCUM_STEPS=18
export LR=6e-4
export MAX_ANCHORS="${MAX_ANCHORS:-512}"
export CHECKPOINT_FREQ=0.25
export SAVE_BEST=0
export TRAIN_DATA_RATIO=0.998
export ON_GENERATE=delete
export VLLM_TP=1
export VLLM_DP="$VLLM_DP_N"
export TRAIN_GPUS_N=4

rm -rf "/dev/shm/hidden_states_${RUN_NAME}"
_RUN=$(mktemp "${TMPDIR:-/tmp}/dspark_driver_XXXXXX.sh")
cp "$REPO/examples/train/dspark_online_gemma4_31b.sh" "$_RUN"
setsid bash "$_RUN" &
DRIVER_PID=$!
trap 'echo "TERM received -> forwarding to driver group $DRIVER_PID"; kill -TERM -- -"$DRIVER_PID" 2>/dev/null' TERM INT
wait "$DRIVER_PID"; RC=$?
rm -f "$_RUN"
echo "=== driver exited rc=$RC $(date '+%F %T') ==="
exit "$RC"
