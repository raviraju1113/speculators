#!/bin/bash
# SLURM wrapper for the FROM-SCRATCH nemo782k DSpark + DFlash2-conv run on the
# SambaNova sc3 B200 cluster (sc3-cp[77-108], partition gpuonly, 8x B200-180GB
# per node, driver >= 580). Counterpart of sbatch_nemo782k_conv.sh, which targets
# the lax01 fleet (podman image + /sms-scratch paths); this one runs BARE-METAL
# in the cluster conda env and uses /import paths. Same recipe, same wandb id.
#
#   cd /import/ml-sc-scratch1/mengmengj/speculators && mkdir -p logs
#   sbatch examples/train/sbatch_sc3_nemo782k_conv.sh          # any idle node
#   sbatch -w sc3-cp77 examples/train/sbatch_sc3_nemo782k_conv.sh
#
# Resume after a time-out / scancel: resubmit as-is. The driver resumes from
# the newest checkpoint under $OUT_ROOT/$RUN_NAME/checkpoints and the wandb run
# continues under the same id (WANDB_RESUME=allow).
#
# Mirrors the 2026-10-13 relaunch command verbatim:
#   MODE=full SAMPLE_FROM_ANCHOR=1 DSPARK_CONV_KERNEL=2 DSPARK_CONV_GROUP=16
#   ACCUM_STEPS=18 SCHED_TOTAL=38700 EPOCHS=26 LR=6e-4 MAX_ANCHORS=512
#   CHECKPOINT_FREQ=0.25 SAVE_BEST=0 TRAIN_DATA_RATIO=0.998 ON_GENERATE=delete
#   VLLM_TP=1 VLLM_DP=4 TRAIN_GPUS_N=4
#
# FIPS gotcha (2026-10-15, job 11151719): sc3-cp nodes run with kernel
# fips_enabled=1 (login nodes do not). The opencv-python-headless wheel that
# vllm imports bundles a RedHat FIPS build of OpenSSL 1.1.1k whose self-test
# aborts the process ("crypto/fips/fips.c:154: FATAL FIPS SELFTEST FAILURE")
# the moment `vllm serve` imports cv2. Fixed in the conda env by replacing the
# two bundled libs (opencv_python_headless.libs/libcrypto-*.so.1.1.1k and
# libssl-*.so.1.1.1k) with conda-forge OpenSSL 1.1.1w, patchelf'd to the same
# SONAMEs; originals in $CONDA_ENV/opencv_fips_orig_libs/. Reinstalling
# opencv-python-headless brings the crash back. Bare-metal NCCL works on these
# nodes (2-rank all_reduce verified on sc3-cp87), so no container is needed.
#SBATCH -J dspark_nemo_conv
#SBATCH -N 1
#SBATCH --partition=gpuonly
#SBATCH --gres=gpu:b200m180:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=128
# All node memory: transient hidden states stream through /dev/shm (RAM-backed)
# and count against the job cgroup. The nodes have 2.3 TB.
#SBATCH --mem=0
# nemo782k_scratch (identical recipe) took 132 h for 26 epochs.
#SBATCH --time=8-00:00:00
# TERM to the batch shell 15 min before the limit so the trainer can checkpoint.
#SBATCH --signal=B:TERM@900
#SBATCH --chdir=/import/ml-sc-scratch1/mengmengj/speculators
#SBATCH -o logs/%x-%j.out

set -uo pipefail
echo "=== job $SLURM_JOB_ID on $(hostname) $(date '+%F %T') ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv,noheader
echo "shm : $(df -h /dev/shm | tail -1)"
echo "out : $(df -h /import/ml-sc-scratch1 | tail -1)"

export REPO=/import/ml-sc-scratch1/mengmengj/speculators
# Empty VENV -> the driver falls through to CONDA_SH + CONDA_ENV (cluster env:
# torch 2.11+cu130, vllm 0.26.0, editable speculators from this checkout).
export VENV=
export CONDA_SH=/import/snvm-sc-scratch1/mengmengj/miniconda3/etc/profile.d/conda.sh
export CONDA_ENV=/import/ml-sc-scratch1/mengmengj/condaenvs/dspark
export MODEL=/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it
export DATA_PATH=/import/ml-sc-scratch1/mengmengj/datasets/kimi_nemotron_782k
export OUT_ROOT=/import/ml-sc-scratch1/mengmengj/output

# ---- W&B gate: refuse to train unless this node can reach the API ----------
# The driver defaults WANDB_MODE=online; without connectivity wandb.init would
# retry/hang or quietly go offline. Fail fast here instead.
export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-dspark}"
export WANDB_MODE=online
export WANDB_RESUME=allow
# Burned ids: nemoconv782k (job 319 on lax01, crashed at step 2964). Every
# from-scratch relaunch needs a NEW id; never delete the old run.
export WANDB_RUN_ID="${WANDB_RUN_ID:-nemoconv782k_v2}"
# shellcheck disable=SC1090
source "$CONDA_SH" && conda activate "$CONDA_ENV"
if ! timeout 180 python - <<'PY'
import os, wandb
api = wandb.Api(timeout=60)
v = api.viewer
print(f"wandb OK on this node: entity={v.entity} user={v.username} "
      f"project={os.environ['WANDB_PROJECT']} run_id={os.environ['WANDB_RUN_ID']}", flush=True)
PY
then
  echo "!! wandb API unreachable from $(hostname) (or ~/.netrc missing) -- NOT starting training" >&2
  exit 3
fi

# ---- GPU self-test and layout ---------------------------------------------
# 2026-10-15: on sc3-cp78/79/80 the GPU at PCI 63:00.0 (index 2) is idle
# (0 MiB, compute mode Default, no processes) yet fails cuCtxCreate (error 999)
# and shows "GPU Recovery Action: Reset" in `nvidia-smi -q` -- it needs a root
# `nvidia-smi -r`. Job 11153559 died 3 min in when DP engine 2 landed on it.
# The driver's idle-GPU check (free memory) cannot see that, so probe every
# card and hand the driver an explicit GPU_IDS of the healthy ones.
#   8 healthy -> 4 vLLM DP engines + 4 training ranks (the recipe)
#   7 healthy -> 3 vLLM DP engines + 4 training ranks: generation ~25% slower,
#                but TRAIN_GPUS_N and ACCUM_STEPS -- hence global batch and
#                SCHED_TOTAL -- are untouched, so the run stays comparable to
#                nemo782k_scratch.
#  <7 healthy -> exit 4 naming the node.
GOOD=(); BAD=()
for i in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do
  if CUDA_VISIBLE_DEVICES=$i timeout 120 python -c \
      'import torch; torch.cuda.init(); (torch.ones(1, device="cuda") + 1).item()' \
      >/dev/null 2>&1; then GOOD+=("$i"); else BAD+=("$i"); fi
done
echo "GPU self-test on $(hostname -s): healthy=${GOOD[*]} bad=${BAD[*]:-none}"
case "${#GOOD[@]}" in
  8) VLLM_DP_N=4 ;;
  7) VLLM_DP_N=3; echo "NOTE: running 3 vLLM engines + 4 training ranks on the 7 healthy GPUs" ;;
  *) echo "!! only ${#GOOD[@]} healthy GPUs on $(hostname) -- resubmit with --exclude=$(hostname -s)" >&2; exit 4 ;;
esac
export GPU_IDS=$(IFS=,; echo "${GOOD[*]}")

# ---- recipe (see header) ----------------------------------------------------
export MODE=full
export SAMPLE_FROM_ANCHOR=1
export DSPARK_CONV_KERNEL=2
export DSPARK_CONV_GROUP=16
export RUN_NAME=gemma4_31b_dspark_nemo782k_conv
export ACCUM_STEPS=18
export SCHED_TOTAL="${SCHED_TOTAL:-38700}"
export EPOCHS="${EPOCHS:-26}"
export LR=6e-4
export MAX_ANCHORS=512
export CHECKPOINT_FREQ=0.25
export SAVE_BEST=0        # --save-best disables sub-epoch saving (trainer.py:517)
export TRAIN_DATA_RATIO=0.998
export ON_GENERATE=delete
export VLLM_TP=1
export VLLM_DP="$VLLM_DP_N"   # 4, or 3 when one GPU is out (see self-test above)
export TRAIN_GPUS_N=4

# Only OUR orphan from a previous attempt on this node; never touch others'.
rm -rf "/dev/shm/hidden_states_${RUN_NAME}"

# Snapshot the driver: bash reads scripts by byte offset, so editing the file
# under a running job would kill it silently.
_RUN=$(mktemp "${TMPDIR:-/tmp}/dspark_driver_XXXXXX.sh")
cp "$REPO/examples/train/dspark_online_gemma4_31b.sh" "$_RUN"
# Own process group so scancel / the time-limit TERM reaches torchrun and the
# trainer directly (bash defers traps while a foreground child runs).
setsid bash "$_RUN" &
DRIVER_PID=$!
trap 'echo "TERM received -> forwarding to driver group $DRIVER_PID"; kill -TERM -- -"$DRIVER_PID" 2>/dev/null' TERM INT
wait "$DRIVER_PID"; RC=$?
rm -f "$_RUN"
echo "=== driver exited rc=$RC $(date '+%F %T') ==="
exit "$RC"
