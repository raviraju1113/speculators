#!/bin/bash
# SLURM wrapper for the sample_from_anchor=TRUE ablation (lax01-b200 cluster).
# Submit FROM THE BASTION (la-dc); compute nodes cannot reach the SLURM
# controller directly:
#
#   sbatch --chdir=/sms-scratch/mengmengj/speculators \
#     /sms-scratch/mengmengj/speculators/examples/train/sbatch_anchorT.sh
#
# The job self-heals node-local prerequisites (podman image, cache dir), then
# runs the containerized training via the anchorT preset. The batch script
# stays alive via `podman wait`, so:
#   - the GPU allocation is held exactly as long as the container runs
#   - `scancel <jobid>` sends TERM here, which podman-stops the container with
#     a 5 min grace period (lets the trainer write its `interrupted` checkpoint)
# No `loginctl enable-linger` needed on this path: slurmd owns the job, not a
# login session.
#SBATCH -N1
#SBATCH --nodelist=trn-b200x8-100
#SBATCH --gres=gpu:b200:8
#SBATCH -J dspark_anchorT
#SBATCH -o logs/%x-%j.out

set -uo pipefail

echo "=== job $SLURM_JOB_ID on $(hostname) $(date '+%F %T') ==="

# --- node-local prerequisites (any fresh node) -------------------------------
mkdir -p /home/mengmengj/container_cache
# rootless podman needs a runtime dir; batch jobs have no login session, so
# /run/user/$UID may not exist -- fall back to /tmp.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [ ! -d "$XDG_RUNTIME_DIR" ]; then
  export XDG_RUNTIME_DIR="/tmp/podman-run-$(id -u)"
  mkdir -p "$XDG_RUNTIME_DIR"
fi
if ! podman image exists localhost/mengmengj-vllm-official; then
  echo "loading image from shared tar (~2-3 min)..."
  podman load -i /sms-scratch/mengmengj/docker/mengmengj-vllm-official.tar
fi
# W&B: needs ~/.netrc on THIS node (node-local /home). Without it the launcher
# falls back to WANDB_MODE=offline; `wandb sync` the run dir later, or copy:
#   scp trn-b200x8-098:~/.netrc ~/ && chmod 600 ~/.netrc
[ -f "$HOME/.netrc" ] || echo "NOTE: no ~/.netrc on $(hostname) -> wandb offline"

# --- launch ------------------------------------------------------------------
cd /sms-scratch/mengmengj/speculators
bash examples/train/run_accum23x4_anchorT_local.sh

# --- hold the allocation; graceful stop on scancel ---------------------------
podman wait dspark_anchorT &
trap 'echo "scancel received -> podman stop"; podman stop -t 300 dspark_anchorT' TERM INT
wait $!
echo "=== container exited, job done $(date '+%F %T') ==="
