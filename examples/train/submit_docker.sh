#!/bin/bash
# Submit a DSpark training run into the NGC container from a BATCH job.
#
# WHY THIS EXISTS
#   Multi-rank training (torchrun + FSDP) needs NCCL, and NCCL segfaults on bare
#   metal here (ncclNetPluginInit). Ravi's TP=4 / 4-rank runs work INSIDE the NGC
#   container, so the container is the fix. But `sngpu --image` only works with
#   --interactive -- in batch it fails (boot_docker.sh wants sudo + a TTY).
#   So: ask sngpu for a BARE node, and have the job script docker-run the image
#   itself. This is Ravi's pattern (scripts/submit_train.sh on
#   origin/ravir/eagle3_branch), reusing his exact working invocation:
#     sudo -g docker /usr/bin/cuda-docker-run-wrapper --rm --gpus device=all \
#       --net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
#       --shm-size 16G <image> bash <inner> < /dev/null
#
#   MEASURED CONTEXT (2026-08-04): generation is NOT the bottleneck -- vLLM DP=3
#   gave 1.00x (1,247 vs 1,250 steps/h). A single training rank is the constraint,
#   at 45,531 steps and ~36.4 h per epoch. Multi-rank training is the only lever
#   that addresses it, hence this script.
#
# USAGE
#   # 4 GPUs: 1 vLLM engine + 3 training ranks
#   VLLM_TP=1 VLLM_DP=1 TRAIN_GPUS_N=3 LR=1e-4 EPOCHS=10 \
#     RUN_NAME=gemma4_31b_dspark_docker_lr1e4 \
#     bash examples/train/submit_docker.sh
#
#   DRY_RUN=1 bash examples/train/submit_docker.sh    # print scripts, do not submit
set -euo pipefail

REPO="${REPO:-/import/ml-sc-scratch1/mengmengj/speculators}"
JOBNAME="${JOBNAME:-dspark_docker}"
GPU="${GPU:-4}"
GPUTYPE="${GPUTYPE:-a100m80}"
CPU="${CPU:-48}"
MEM="${MEM:-400000}"
TIME="${TIME:-256:00:00}"
NODELIST="${NODELIST:-}"
EXCLUDE="${EXCLUDE:-sc-c96,sc3-c97,sc-c82}"     # 565 drivers -- unusable
OUTPUT="${OUTPUT:-$REPO/logs/${JOBNAME}.txt}"

DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}"
SHM_SIZE="${SHM_SIZE:-16G}"
NV_VISIBLE_DEVICES="${NV_VISIBLE_DEVICES:-all}"
DOCKER_NET="${DOCKER_NET:-host}"                 # host net so localhost:PORT works

JOBSCRIPT="$REPO/examples/train/.job_${JOBNAME}.sh"
INNERSCRIPT="$REPO/examples/train/.job_${JOBNAME}_inner.sh"
mkdir -p "$REPO/logs"

# ---- SMOKE=1: validate the container path before committing days -----------
# Checks, in one ~15 min / 2 GPU job:
#   1. does `sudo -g docker cuda-docker-run-wrapper` work in BATCH (no TTY)?
#      Passwordless sudo is NOT available on the login node, so this is the one
#      assumption that could kill the whole approach.
#   2. is /import mounted inside the container (our conda env + data live there)?
#   3. is /dev/shm big enough, or did --shm-size take effect?
#   4. does a 2-rank NCCL all_reduce PASS inside the container? <- the actual point
if [ "${SMOKE:-0}" = "1" ]; then
  JOBNAME="${JOBNAME:-dspark_docker_smoke}"
  GPU=2; CPU=16; MEM=200000; TIME="00:30:00"
  OUTPUT="$REPO/logs/${JOBNAME}.txt"
  JOBSCRIPT="$REPO/examples/train/.job_${JOBNAME}.sh"
  INNERSCRIPT="$REPO/examples/train/.job_${JOBNAME}_inner.sh"
  {
    printf '#!/bin/bash\n'
    printf 'echo "=== inside container: $(hostname) ==="\n'
    printf 'nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader\n'
    printf 'echo "shm: $(df -h /dev/shm | tail -1)"\n'
    printf 'echo "/import visible: $( [ -d /import/ml-sc-scratch1/mengmengj ] && echo yes || echo NO )"\n'
    printf 'bash %q\n' "$REPO/examples/train/container_setup.sh"
  } > "$INNERSCRIPT"
  chmod +x "$INNERSCRIPT"
fi

# ---- inner script: runs INSIDE the container -------------------------------
# It just execs our normal driver. Every knob is passed through as an env var, so
# the driver stays the single source of truth for the training command.
# NOTE: inside the container with --gpus device=all the devices are visible as
# 0..N-1, so GPU_IDS is expressed in those terms.
if [ "${SMOKE:-0}" != "1" ]; then
{
  printf '#!/bin/bash\nset -euo pipefail\n'
  printf '# --- container-side environment ---\n'
  printf 'export HF_HOME=%q\n' "${HF_HOME:-/import/snvm-sc-scratch1/mengmengj/hf_cache}"
  printf 'export WANDB_PROJECT=%q\n' "${WANDB_PROJECT:-gemma4-dspark}"
  # W&B auth comes from ~/.netrc (mounted with $HOME), never a key on a command line.
  printf 'export MODE=%q\n'            "${MODE:-full}"
  printf 'export LR=%q\n'              "${LR:-1e-4}"
  printf 'export EPOCHS=%q\n'          "${EPOCHS:-10}"
  printf 'export RUN_NAME=%q\n'        "${RUN_NAME:-gemma4_31b_dspark_docker}"
  printf 'export DATA_PATH=%q\n'       "${DATA_PATH:-/import/ml-sc-scratch1/mengmengj/datasets/gemma4_dspark}"
  printf 'export VLLM_TP=%q\n'         "${VLLM_TP:-1}"
  printf 'export VLLM_DP=%q\n'         "${VLLM_DP:-1}"
  printf 'export TRAIN_GPUS_N=%q\n'    "${TRAIN_GPUS_N:-3}"
  printf 'export MAX_ANCHORS=%q\n'     "${MAX_ANCHORS:-512}"
  printf 'export ON_GENERATE=%q\n'     "${ON_GENERATE:-delete}"
  printf 'export CHECKPOINT_FREQ=%q\n' "${CHECKPOINT_FREQ:-0.5}"
  printf 'export SAVE_BEST=%q\n'       "${SAVE_BEST:-1}"
  # 0/unset -> scripts/train.py; >1 -> scripts/train_accum.py --accumulation-steps N
  printf 'export ACCUM_STEPS=%q\n'     "${ACCUM_STEPS:-0}"
  printf 'export GPU_IDS=%q\n'         "${GPU_IDS:-0,1,2,3}"
  # NCCL inside the container: the host plugin is not visible here, which is the
  # whole point. Keep the loopback pinning that Ravi has working regardless.
  printf 'export NCCL_SOCKET_IFNAME=%q\n' "${NCCL_SOCKET_IFNAME:-lo}"
  printf 'export NCCL_IB_DISABLE=%q\n'    "${NCCL_IB_DISABLE:-1}"
  printf 'export NCCL_DEBUG=%q\n'         "${NCCL_DEBUG:-WARN}"
  printf 'nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader\n'
  printf 'echo "shm: $(df -h /dev/shm | tail -1)"\n'
  # Snapshot the driver: bash reads scripts by byte offset, so editing the file
  # mid-run makes the job exit silently with status 0.
  printf '_RUN=$(mktemp /tmp/dspark_driver_XXXXXX.sh)\n'
  printf 'cp %q "$_RUN"\n' "$REPO/examples/train/dspark_online_gemma4_31b.sh"
  printf 'exec bash "$_RUN"\n'
} > "$INNERSCRIPT"
chmod +x "$INNERSCRIPT"
fi

# ---- outer script: runs on the BARE sngpu node ------------------------------
{
  printf '#!/bin/bash\nset -e\n'
  printf 'echo "host: $(hostname)"; nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1\n'
  printf 'sudo -g docker /usr/bin/cuda-docker-run-wrapper --rm \\\n'
  printf '  --gpus device=%q \\\n' "$NV_VISIBLE_DEVICES"
  printf '  --net=%q \\\n' "$DOCKER_NET"
  printf '  --ipc=host \\\n'
  printf '  --ulimit memlock=-1 \\\n'
  printf '  --ulimit stack=67108864 \\\n'
  printf '  --shm-size %q \\\n' "$SHM_SIZE"
  printf '  %q \\\n' "$DOCKER_IMAGE"
  printf '  bash %q < /dev/null\n' "$INNERSCRIPT"
} > "$JOBSCRIPT"
chmod +x "$JOBSCRIPT"

SNGPU_ARGS=(--jobname "$JOBNAME" --partition gpuonly --time "$TIME"
            --cpu "$CPU" --mem "$MEM" --gpu "$GPU" --gputype "$GPUTYPE"
            --output "$OUTPUT")
[ -n "$NODELIST" ] && SNGPU_ARGS+=(--nodelist "$NODELIST")
[ -n "$EXCLUDE" ]  && SNGPU_ARGS+=(--exclude "$EXCLUDE")
SNGPU_ARGS+=(--filepath "$JOBSCRIPT")

echo "==============================================="
echo " jobname : $JOBNAME    gpus: $GPU    time: $TIME"
echo " layout  : vLLM TP=${VLLM_TP:-1} DP=${VLLM_DP:-1} + ${TRAIN_GPUS_N:-3} training ranks"
echo " lr      : ${LR:-1e-4}   epochs: ${EPOCHS:-10}   run: ${RUN_NAME:-gemma4_31b_dspark_docker}"
echo " docker  : $DOCKER_IMAGE  (net=$DOCKER_NET shm=$SHM_SIZE ipc=host)"
echo " log     : $OUTPUT"
echo "==============================================="
echo "--- outer ---"; cat "$JOBSCRIPT"
echo "--- inner ---"; cat "$INNERSCRIPT"
echo "--- sngpu ---"; printf 'sngpu'; printf ' %q' "${SNGPU_ARGS[@]}"; echo

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "(DRY_RUN=1 -- scripts written, not submitted)"; exit 0
fi
sngpu "${SNGPU_ARGS[@]}"
echo "Submitted. tail -f $OUTPUT"
