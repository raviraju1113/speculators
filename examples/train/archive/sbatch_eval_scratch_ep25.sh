#!/bin/bash
# Eval nemo782k checkpoints (ep24 k8/k3, ep20 k8). Bare-metal host venv, 1 GPU.
# Submit: sbatch /sms-scratch/mengmengj/speculators/examples/train/sbatch_eval_nemo.sh
#SBATCH -N1
#SBATCH --exclude=trn-b200x8-101,trn-b200x8-097
#SBATCH --gres=gpu:b200:1
#SBATCH -J eval_scratch_ep25
#SBATCH --chdir=/sms-scratch/mengmengj/speculators
#SBATCH -o logs/%x-%j.out

set -uo pipefail
echo "=== job $SLURM_JOB_ID on $(hostname) $(date '+%F %T') ==="
source /sms-scratch/mengmengj/envs/dspark-host/bin/activate

G=""
for i in $(seq 1 72); do
  G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
      awk -F', ' '$1>=0 && $2<1000{print $1; exit}')
  [ -n "$G" ] && break
  echo "GPUs busy with off-SLURM work; waiting ($i/72)"; sleep 300
done
[ -z "$G" ] && { echo "!! no free GPU after 6h"; exit 1; }
export CUDA_VISIBLE_DEVICES="$G"
echo "using GPU $G"

cd scripts/evaluate/experiments
python run_experiments.py --config gemma4-31b-dspark-scratch-ep25.yaml
echo "=== eval done $(date '+%F %T') ==="
