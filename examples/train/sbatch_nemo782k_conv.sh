#!/bin/bash
# SLURM wrapper for the FROM-SCRATCH nemo782k DSpark + DFlash2-conv run
# (architecture ablation partner of sbatch_nemo782k_scratch.sh). Submit from
# the bastion:
#
#   sbatch [-w trn-b200x8-XXX] \
#     /sms-scratch/mengmengj/speculators/examples/train/sbatch_nemo782k_conv.sh
#
# Self-sufficient on any node (podman runtime fallback for batch jobs, image
# auto-load, wandb-offline warning, off-SLURM-squatter wait). Holds the
# allocation via podman wait; scancel -> graceful stop (5 min for checkpoint).
#SBATCH -N1
# No node exclusion (2026-09-27, user request): take any free node. The sick
# list (097/099 full /tmp, 101 bad GPUs) is NOT applied here -- if the job
# lands on one of them, the image load / GPU checks below will fail loudly.
#SBATCH --gres=gpu:b200:8
#SBATCH -J dspark_nemo_conv
#SBATCH --chdir=/sms-scratch/mengmengj/speculators
#SBATCH -o logs/%x-%j.out

set -uo pipefail
echo "=== job $SLURM_JOB_ID on $(hostname) $(date '+%F %T') ==="

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [ ! -d "$XDG_RUNTIME_DIR" ]; then
  export XDG_RUNTIME_DIR="/tmp/podman-run-$(id -u)"
  mkdir -p "$XDG_RUNTIME_DIR"
  echo "no user session; using XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
fi
mkdir -p /home/mengmengj/container_cache
echo "disk: $(df -h /home/mengmengj | tail -1)"
if ! podman image exists localhost/mengmengj-vllm-official; then
  echo "image missing on $(hostname); loading (~25 GB, needs free space on /home)"
  podman load -i /sms-scratch/mengmengj/docker/mengmengj-vllm-official.tar || {
    echo "!! podman load failed on $(hostname) (disk full?) -- exiting so SLURM frees the node" >&2
    exit 1
  }
fi
[ -f "$HOME/.netrc" ] || echo "NOTE: no ~/.netrc on $(hostname) -> wandb OFFLINE (sync later)"

for i in $(seq 1 72); do
  FREE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1<1000' | wc -l)
  [ "$FREE" -ge 8 ] && break
  echo "only $FREE/8 GPUs idle (off-SLURM work present); waiting ($i/72)"; sleep 300
done

bash examples/train/run_nemo782k_conv_local.sh

sleep 60
podman wait dspark_nemo_conv &
trap 'echo "scancel -> graceful stop"; podman stop -t 300 dspark_nemo_conv' TERM INT
wait $!
echo "=== container exited $(date '+%F %T') ==="
