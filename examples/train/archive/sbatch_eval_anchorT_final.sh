#!/bin/bash
# Eval anchorT ep9+ep14 (k8/k7/k3) on any healthy node, queued via SLURM.
# Submit from the bastion:
#   sbatch /sms-scratch/mengmengj/speculators/examples/train/sbatch_eval_anchorT_final.sh
#
# Defensive bits, both learned the hard way:
#  - excludes 101 (2 GPUs off the bus; NVML/Triton both broken until reboot)
#  - if the SLURM-assigned GPU is held by off-SLURM work (jupyter/ray squatters
#    are invisible to the scheduler), WAIT up to 6h for it to free instead of
#    dying instantly like job 127 did.
#SBATCH -N1
#SBATCH --exclude=trn-b200x8-101
#SBATCH --gres=gpu:b200:1
#SBATCH -J eval_anchorT_final
#SBATCH --chdir=/sms-scratch/mengmengj/speculators
#SBATCH -o logs/%x-%j.out

set -uo pipefail
echo "=== job $SLURM_JOB_ID on $(hostname) $(date '+%F %T') ==="
source /sms-scratch/mengmengj/envs/dspark-host/bin/activate

G=""
for i in $(seq 1 72); do
  G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
      awk -F', ' '$2<1000{print $1; exit}')
  [ -n "$G" ] && break
  echo "visible GPU(s) busy with off-SLURM work; waiting ($i/72, 5 min each)"
  sleep 300
done
if [ -z "$G" ]; then echo "!! no free GPU after 6h; giving up"; exit 1; fi
export CUDA_VISIBLE_DEVICES="$G"
echo "using GPU $G"

cd scripts/evaluate/experiments
python run_experiments.py --config gemma4-31b-dspark-anchorT-final.yaml
echo "=== eval done $(date '+%F %T') ==="
