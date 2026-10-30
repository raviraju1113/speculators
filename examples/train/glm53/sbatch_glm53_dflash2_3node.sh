#!/usr/bin/env bash
# Full GLM-5.3 DFlash2 run (Kimi K3 recipe, off-policy gemma data) on 3 B200 nodes.
#
#   REPO=/import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2/code_<sha> \
#     sbatch -w sc3-cp81,sc3-cp82,sc3-cp108 examples/train/glm53/sbatch_glm53_dflash2_3node.sh
#
# CAVEAT: srun here fans out from the first node; sc3-cp81 has a stale local
# slurm.conf and cannot reach some nodes (jobs 14626017, 14630799). If the first
# node is not known-good, use launch_glm53_dflash2_salloc.sh instead.
#
# Node order = role (run_glm53_dflash2_3node.sh): the first two nodes serve the
# verifier, the third trains. Resume: resubmit as-is; train.py picks up the newest
# checkpoint under SAVE_PATH and wandb continues the same run id.
#SBATCH -J glm53_dflash2
#SBATCH -p gpuonly
#SBATCH --reservation=sn50-hdi-0
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:b200m180:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=10-00:00:00
#SBATCH --signal=B:TERM@900
#SBATCH -o /import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2/logs/%x-%j.out
set -uo pipefail
: "${REPO:?export REPO=<frozen code snapshot>}"
export REPO
export RUN_NAME="${RUN_NAME:-glm53_dflash2_gemma782k_kimirecipe}"
export RUN_DIR="${RUN_DIR:-/import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2/runs/${RUN_NAME}_${SLURM_JOB_ID}}"
echo "=== job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST repo $REPO run_dir $RUN_DIR ==="
srun --ntasks-per-node=1 --kill-on-bad-exit=0 bash "$REPO/examples/train/glm53/run_glm53_dflash2_3node.sh" &
STEP=$!
trap 'echo "TERM -> step $STEP"; kill -TERM "$STEP" 2>/dev/null' TERM INT
wait "$STEP"; RC=$?
echo "=== exited rc=$RC $(date '+%F %T') ==="
exit "$RC"
