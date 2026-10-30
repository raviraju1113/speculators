#!/usr/bin/env bash
# Start the 3-node GLM-5.3 DFlash2 run inside an existing allocation, with the
# srun client detached on the login node.
#
# Why not sbatch_glm53_dflash2_3node.sh: in a batch job srun fans out from the
# first node, and sc3-cp81 carries a stale local /etc/slurm/slurm.conf
# (2026-10-03) -- job 14626017/14630799 died with "fwd_tree_thread: can't find
# address for host sc3-cp86/cp84". From the login node (shared, current conf)
# the same srun works.
#
#   salloc -p gpuonly --reservation=sn50-hdi-0 -w <verifier0>,<verifier1>,<trainer> -N3 \
#     --ntasks-per-node=1 --gres=gpu:b200m180:8 --exclusive --cpus-per-task=128 \
#     --mem=0 --time=10-00:00:00 -J glm53_dflash2 --no-shell
#   JOBID=<id> REPO=<frozen snapshot> bash examples/train/glm53/launch_glm53_dflash2_salloc.sh
#
# Resume after a crash: rerun the same command (fresh RUN_DIR per launch; the
# trainer resumes from the newest checkpoint and the same wandb run id).
set -euo pipefail
: "${JOBID:?salloc job id}"
: "${REPO:?frozen code snapshot}"
export REPO
export RUN_NAME="${RUN_NAME:-glm53_dflash2_gemma782k_kimirecipe}"
export RUN_DIR="${RUN_DIR:-/import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2/runs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"
LOG="/import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2/logs/${RUN_NAME}_srun_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
setsid nohup srun --jobid="$JOBID" --overlap -N3 -n3 --ntasks-per-node=1 --kill-on-bad-exit=0 \
    bash "$REPO/examples/train/glm53/run_glm53_dflash2_3node.sh" > "$LOG" 2>&1 < /dev/null &
echo "srun pid $! | run_dir $RUN_DIR | log $LOG"
