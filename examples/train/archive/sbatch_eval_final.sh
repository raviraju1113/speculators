#!/bin/bash
# SLURM wrapper: eval the accum23x4 run's FINAL checkpoint (epoch 9) on 097.
# Submit from the bastion:
#
#   sbatch --chdir=/sms-scratch/mengmengj/speculators \
#     /sms-scratch/mengmengj/speculators/examples/train/sbatch_eval_final.sh
#
# Requests 1 GPU (evals are single-GPU; leaves the rest of 101 usable). The
# podman run is FOREGROUND, so the job holds its allocation exactly as long as
# the eval runs and `scancel` kills it cleanly.
#SBATCH -N1
#SBATCH --nodelist=trn-b200x8-097
#SBATCH --gres=gpu:b200:1
#SBATCH -J dspark_eval_final
#SBATCH -o logs/%x-%j.out

set -uo pipefail
echo "=== job $SLURM_JOB_ID on $(hostname) $(date '+%F %T') ==="

mkdir -p /home/mengmengj/container_cache
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
[ -d "$XDG_RUNTIME_DIR" ] || { export XDG_RUNTIME_DIR="/tmp/podman-run-$(id -u)"; mkdir -p "$XDG_RUNTIME_DIR"; }
podman image exists localhost/mengmengj-vllm-official || \
  podman load -i /sms-scratch/mengmengj/docker/mengmengj-vllm-official.tar

# SLURM jobs inherit slurmd's (often low) hard limit; rootless podman cannot
# exceed it, so request exactly what we have (capped at our usual 524288).
NOFILE=$(ulimit -Hn); [ "$NOFILE" = unlimited ] || [ "$NOFILE" -gt 524288 ] && NOFILE=524288
echo "nofile hard limit here: $(ulimit -Hn) -> requesting $NOFILE"
# Use the GPU SLURM assigned us (it sets CUDA_VISIBLE_DEVICES for the job).
# Exposing ONLY that device makes it index 0 inside the container, which is
# what the yaml's `gpus: "0"` expects -- and keeps us off other jobs' GPUs.
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"; GPU_ID="${GPU_ID:-0}"
echo "SLURM-assigned GPU(s): ${CUDA_VISIBLE_DEVICES:-unset} -> exposing gpu=$GPU_ID"
podman run --rm --name dspark_eval_final --security-opt label=disable \
  --device "nvidia.com/gpu=$GPU_ID" --net=host --ipc=host \
  --pids-limit=-1 --ulimit "nofile=$NOFILE:$NOFILE" \
  -v /home/mengmengj/container_cache:/root/.cache \
  -v /sms-scratch:/sms-scratch -v /home/mengmengj:/home/mengmengj \
  -e HOME=/home/mengmengj \
  localhost/mengmengj-vllm-official \
  bash -c 'source /sms-scratch/mengmengj/envs/dspark/bin/activate &&
           cd /sms-scratch/mengmengj/speculators/scripts/evaluate/experiments &&
           python run_experiments.py --config gemma4-31b-dspark-final.yaml'

echo "=== eval done $(date '+%F %T') ==="
