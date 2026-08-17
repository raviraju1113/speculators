#!/bin/bash
# SLURM wrapper for the nemo782k continued-training run. Submit from bastion:
#
#   sbatch [-w trn-b200x8-XXX] \
#     /sms-scratch/mengmengj/speculators/examples/train/sbatch_nemo782k.sh
#
# Self-sufficient on any node: sets up a podman runtime dir when there is no
# systemd user session (batch jobs have none unless linger is enabled -- the
# missing /run/user/1025 killed the first submission), loads the image if
# absent, warns if wandb will be offline, holds the allocation via podman wait,
# and converts scancel into a graceful stop (5 min for checkpoint writing).
#SBATCH -N1
#SBATCH --exclude=trn-b200x8-101
#SBATCH --gres=gpu:b200:8
#SBATCH -J dspark_anchorT_nemo
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
podman image exists localhost/mengmengj-vllm-official || \
  podman load -i /sms-scratch/mengmengj/docker/mengmengj-vllm-official.tar
[ -f "$HOME/.netrc" ] || echo "NOTE: no ~/.netrc on $(hostname) -> wandb OFFLINE (sync later)"

# don't stack onto off-SLURM squatters: need 8 idle GPUs (like the driver checks,
# but wait instead of dying; up to 6h)
for i in $(seq 1 72); do
  FREE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1<1000' | wc -l)
  [ "$FREE" -ge 8 ] && break
  echo "only $FREE/8 GPUs idle (off-SLURM work present); waiting ($i/72)"; sleep 300
done

bash examples/train/run_anchorT_nemo782k_local.sh

sleep 60
podman wait dspark_anchorT_nemo &
trap 'echo "scancel -> graceful stop"; podman stop -t 300 dspark_anchorT_nemo' TERM INT
wait $!
echo "=== container exited $(date '+%F %T') ==="
