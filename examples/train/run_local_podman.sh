#!/bin/bash
# Launch DSpark training inside the NGC container on the STANDALONE 8x B200 box.
#
# This replaces submit_docker.sh on machines with no SLURM/sngpu: podman runs
# straight on the node (rootless, GPU access via CDI -- /var/run/cdi/nvidia.yaml
# exists here, so --device nvidia.com/gpu=all works without sudo).
#
# WHY A CONTAINER AT ALL: the host has no conda/torch, only system python. The
# NGC image supplies the CUDA userspace; our actual python stack lives in a
# PERSISTENT venv on /home (local disk, 709G free) so pip installs survive
# container restarts. /dev/shm comes from the host via --ipc=host (1.5T RAM).
#
# ONE-TIME SETUP (build the venv, ~10 min):
#   SETUP=1 bash examples/train/run_local_podman.sh
#
# SANITY CHECK (GPUs + mounts + 2-rank NCCL all_reduce, ~2 min):
#   SMOKE=1 bash examples/train/run_local_podman.sh
#
# 8-GPU TRAINING RUN (defaults: 4 vLLM DP engines + 4 training ranks):
#   JOBNAME=dspark_accum23x4 RUN_NAME=gemma4_31b_dspark_accum23x4_lr6e4 \
#     LR=6e-4 EPOCHS=10 ACCUM_STEPS=23 CHECKPOINT_FREQ=0.25 SAVE_BEST=0 \
#     TRAIN_DATA_RATIO=0.998 bash examples/train/run_local_podman.sh
#
#   ACCUM_STEPS sizing: global batch = 5.62 conv/step * TRAIN_GPUS_N * ACCUM_STEPS.
#   The cluster's 2-rank runs used 46; at 4 ranks, 23 keeps the SAME global batch
#   (~517 conversations, matching DeepSpec's 512).
#
# The run detaches (nohup + podman). Follow it with:
#   tail -f logs/<JOBNAME>.txt
# Stop it with:
#   podman stop -t 30 <JOBNAME>
set -euo pipefail

REPO="${REPO:-/sms-scratch/mengmengj/speculators}"
# Image follows johnl's pattern: official vllm/vllm-openai base supplies
# torch+vllm+CUDA (and carries our algos.py sample_from_anchor patch, baked in
# at build -- see /sms-scratch/mengmengj/docker/Dockerfile). The venv on
# /sms-scratch is created with --system-site-packages, so it reuses the image's
# torch/vllm and adds ONLY speculators, hs_connectors, and wandb.
# (Fallback stack, known-good but unpatched: IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
#  VENV=/home/mengmengj/envs/dspark -- a fat venv with its own pip vllm/torch.)
IMAGE="${IMAGE:-localhost/mengmengj-vllm-official}"
VENV="${VENV:-/sms-scratch/mengmengj/envs/dspark}"
JOBNAME="${JOBNAME:-dspark_local}"
LOGFILE="$REPO/logs/${JOBNAME}.txt"
mkdir -p "$REPO/logs"

# Rootless podman: in-container root == host user, so files written to the
# mounts below land owned by us. --net=host so localhost:PORT reaches vLLM.
PODMAN_ARGS=(
  --rm --name "$JOBNAME"
  --device nvidia.com/gpu=all
  --net=host --ipc=host
  # SELinux is Enforcing and denies container_t readdir on the NFS /sms-scratch
  # (files stat fine, directories won't list -> ModuleNotFoundError for anything
  # in the venv). Verified 2026-08-06: default DENIED, label=disable OK. The
  # sudo-less per-container fix; the system-wide one is `setsebool -P virt_use_nfs 1`.
  --security-opt label=disable
  # Rootless podman caps a container at 2048 pids/threads by default. vLLM DP=4
  # (engines x workers x gloo/OMP threads) blows past that during
  # ProcessGroupGloo init -> "RuntimeError: Resource temporarily unavailable"
  # (observed 2026-08-06, first 8-GPU launch). Lift it, and raise nofile too --
  # every DP engine opens sockets/fds for its TCPStore plus hidden-state files.
  --pids-limit=-1
  # Rootless podman cannot raise limits past the process's HARD limit (crun:
  # setrlimit EPERM). SSH shells here have 524288, but SLURM batch jobs often
  # inherit a lower one from slurmd -- so request min(hard, 524288) dynamically.
  --ulimit "nofile=$(v=$(ulimit -Hn); if [ "$v" = unlimited ] || [ "$v" -gt 524288 ]; then v=524288; fi; echo "$v:$v")"
  --ulimit memlock=-1 --ulimit stack=67108864
  # Persist torch.compile / flashinfer / vllm caches across container restarts
  # (first-time CUDA-graph + compile on B200 took ~10 min; cached it's ~1 min).
  -v /home/mengmengj/container_cache:/root/.cache
  -v /sms-scratch:/sms-scratch
  -v /home/mengmengj:/home/mengmengj
  -e HOME=/home/mengmengj
  -w "$REPO"
)

# ---- SETUP=1: build the thin venv on /sms-scratch -----------------------------
# --system-site-packages so the image's torch/vllm/flashinfer are picked up and
# not duplicated (colleague's recommended pattern for these boxes).
if [ "${SETUP:-0}" = "1" ]; then
  podman run "${PODMAN_ARGS[@]}" "$IMAGE" bash -c "
    set -euo pipefail
    export PIP_CACHE_DIR=/home/mengmengj/.cache/pip
    python3 -m venv --system-site-packages $VENV
    source $VENV/bin/activate
    pip install -U pip
    # hs_connectors is a uv workspace member -- must be installed BEFORE speculators
    pip install -e $REPO/hs_connectors
    pip install -e $REPO
    pip install wandb
    python - <<'PY'
import torch, vllm, transformers, speculators.models.dspark as d, hs_connectors
print('torch', torch.__version__, '| vllm', vllm.__version__,
      '| transformers', transformers.__version__)
print('dspark:', d.__file__)
import vllm.transformers_utils.configs.speculators.algos as algos
src = open(algos.__file__).read()
assert 'sample_from_anchor' in src, 'algos.py patch MISSING from image!'
print('algos.py sample_from_anchor patch: present')
PY
  "
  echo "=== venv built at $VENV ==="
  exit 0
fi

# ---- SMOKE=1: GPUs, mounts, shm, and a 2-rank NCCL all_reduce ----------------
if [ "${SMOKE:-0}" = "1" ]; then
  podman run "${PODMAN_ARGS[@]}" "$IMAGE" bash -c "
    set -euo pipefail
    echo '=== GPUs inside container ==='
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    echo \"shm: \$(df -h /dev/shm | tail -1)\"
    echo \"model dir: \$( [ -f /sms-scratch/checkpoints/gemma-4-31B-it/config.json ] && echo ok || echo MISSING )\"
    echo \"data dir : \$( [ -f /sms-scratch/mengmengj/data/gemma4_dspark/dataset_info.json ] && echo ok || echo MISSING )\"
    source $VENV/bin/activate
    python -c 'import torch, vllm, speculators; print(\"imports ok | cuda:\", torch.cuda.is_available(), torch.cuda.get_device_name(0))'
    echo '=== 2-rank NCCL all_reduce (the DSpark multi-rank prerequisite) ==='
    cat > /tmp/nccl_test.py <<'PY'
import os, torch, torch.distributed as dist
rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)          # BEFORE init_process_group, or both ranks hit device 0
dist.init_process_group('nccl')
t = torch.ones(8, device='cuda') * (rank + 1)
dist.all_reduce(t)
print(f'rank {rank}: all_reduce -> {t[0].item()} (want 3.0)', flush=True)
dist.destroy_process_group()
PY
    torchrun --nnodes=1 --nproc_per_node 2 \
      --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29777 --local-addr=127.0.0.1 \
      /tmp/nccl_test.py
    echo '=== SMOKE PASSED ==='
  " 2>&1 | tee "$LOGFILE"
  exit 0
fi

# ---- training run -------------------------------------------------------------
# Inner script: exports every knob, then execs a SNAPSHOT of the driver (bash
# reads scripts by byte offset; editing a running script kills the job silently).
INNERSCRIPT="$REPO/examples/train/.job_${JOBNAME}_inner.sh"
{
  printf '#!/bin/bash\nset -euo pipefail\n'
  printf 'export HF_HOME=%q\n'        "${HF_HOME:-/home/mengmengj/.cache/huggingface}"
  printf 'export WANDB_PROJECT=%q\n'  "${WANDB_PROJECT:-gemma4-dspark}"
  # W&B: online needs ~/.netrc (run `wandb login` once on the host -- $HOME is
  # mounted). Without it, default to offline; `wandb sync` the run dir later.
  if [ -f /home/mengmengj/.netrc ]; then
    printf 'export WANDB_MODE=%q\n' "${WANDB_MODE:-online}"
  else
    printf 'export WANDB_MODE=%q\n' "${WANDB_MODE:-offline}"
  fi
  # Continue the SAME wandb graph across stop/resume: the trainer's wandb.init()
  # passes no id, so without these a relaunch creates a second run. Set
  # WANDB_RUN_ID to the original run's id (see logs/wandb/latest-run) in the
  # preset. Metrics are logged at step=global_step, which resume restores, so
  # the curve continues; wandb drops the rewound sub-checkpoint steps silently.
  printf 'export WANDB_RESUME=%q\n' "${WANDB_RESUME:-allow}"
  [ -n "${WANDB_RUN_ID:-}" ] && printf 'export WANDB_RUN_ID=%q\n' "$WANDB_RUN_ID"
  printf 'export VENV=%q\n'            "$VENV"
  printf 'export MODE=%q\n'            "${MODE:-full}"
  printf 'export LR=%q\n'              "${LR:-6e-4}"
  printf 'export EPOCHS=%q\n'          "${EPOCHS:-10}"
  printf 'export RUN_NAME=%q\n'        "${RUN_NAME:-gemma4_31b_dspark_local}"
  printf 'export DATA_PATH=%q\n'       "${DATA_PATH:-/sms-scratch/mengmengj/data/gemma4_dspark}"
  printf 'export MODEL=%q\n'           "${MODEL:-/sms-scratch/checkpoints/gemma-4-31B-it}"
  # /sms-scratch is the SHARED filesystem (visible from other nodes; /home is
  # node-local). Checkpoints must be reachable for eval elsewhere. 13 GB/save --
  # prune old epoch dirs periodically, this disk is shared and fills up.
  printf 'export OUT_ROOT=%q\n'        "${OUT_ROOT:-/sms-scratch/mengmengj/output}"
  # 8-GPU balanced layout: engines and ranks each cap near the same seq/h, so
  # scale BOTH (measured on the cluster: 2+2 was ~2x over 3+1).
  printf 'export VLLM_TP=%q\n'         "${VLLM_TP:-1}"
  printf 'export VLLM_DP=%q\n'         "${VLLM_DP:-4}"
  printf 'export TRAIN_GPUS_N=%q\n'    "${TRAIN_GPUS_N:-4}"
  printf 'export MAX_ANCHORS=%q\n'     "${MAX_ANCHORS:-512}"
  printf 'export ON_GENERATE=%q\n'     "${ON_GENERATE:-delete}"
  printf 'export CHECKPOINT_FREQ=%q\n' "${CHECKPOINT_FREQ:-0.25}"
  printf 'export SAVE_BEST=%q\n'       "${SAVE_BEST:-0}"
  printf 'export TRAIN_DATA_RATIO=%q\n' "${TRAIN_DATA_RATIO:-0.998}"
  # 0 = train with --no-sample-from-anchor (matches RedHat's shipped checkpoint
  # and unpatched vLLM serving). The image's algos.py patch derives the serving
  # flag from the checkpoint config, so BOTH kinds of checkpoint serve correctly
  # -- but 0 is the default per 2026-08-06 decision.
  printf 'export SAMPLE_FROM_ANCHOR=%q\n' "${SAMPLE_FROM_ANCHOR:-0}"
  printf 'export ACCUM_STEPS=%q\n'     "${ACCUM_STEPS:-23}"
  [ -n "${SCHED_TOTAL:-}" ] && printf 'export SCHED_TOTAL=%q\n' "$SCHED_TOTAL"
  [ -n "${GPU_IDS:-}" ]     && printf 'export GPU_IDS=%q\n' "$GPU_IDS"
  [ -n "${FULL_ATTENTION_INDICES:-}" ] && printf 'export FULL_ATTENTION_INDICES=%q\n' "$FULL_ATTENTION_INDICES"
  # Single node: keep NCCL on loopback and off IB, same as the cluster runs.
  # Engine init (compile + CUDA-graph capture) took 621 s cold on B200; vLLM's
  # default API-server readiness timeout is 600 s and killed the 2026-08-06 run
  # FIVE SECONDS before the engines came up. Give it a real margin.
  printf 'export VLLM_ENGINE_READY_TIMEOUT_S=%q\n' "${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
  printf 'export NCCL_SOCKET_IFNAME=%q\n' "${NCCL_SOCKET_IFNAME:-lo}"
  printf 'export NCCL_IB_DISABLE=%q\n'    "${NCCL_IB_DISABLE:-1}"
  printf 'export NCCL_DEBUG=%q\n'         "${NCCL_DEBUG:-WARN}"
  printf '_RUN=$(mktemp /tmp/dspark_driver_XXXXXX.sh)\n'
  printf 'cp %q "$_RUN"\n' "$REPO/examples/train/dspark_online_gemma4_31b.sh"
  printf 'exec bash "$_RUN"\n'
} > "$INNERSCRIPT"
chmod +x "$INNERSCRIPT"

echo "==============================================="
echo " jobname : $JOBNAME"
echo " layout  : vLLM TP=${VLLM_TP:-1} DP=${VLLM_DP:-4} + ${TRAIN_GPUS_N:-4} training ranks"
echo " lr      : ${LR:-6e-4}  epochs: ${EPOCHS:-10}  accum: ${ACCUM_STEPS:-23}  ratio: ${TRAIN_DATA_RATIO:-0.998}"
echo " run     : ${RUN_NAME:-gemma4_31b_dspark_local}"
echo " image   : $IMAGE"
echo " log     : $LOGFILE"
echo "==============================================="
echo "--- inner ---"; cat "$INNERSCRIPT"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "(DRY_RUN=1 -- scripts written, not started)"; exit 0
fi

# APPEND so a resume keeps the history of previous launches in one file.
echo "=== launch $(date '+%F %T') (job $JOBNAME) ===" >> "$LOGFILE"
nohup podman run "${PODMAN_ARGS[@]}" "$IMAGE" bash "$INNERSCRIPT" \
  >> "$LOGFILE" 2>&1 &
echo "Started (podman container '$JOBNAME', pid $!)."
echo "  follow : tail -f $LOGFILE"
echo "  stop   : podman stop -t 30 $JOBNAME"
