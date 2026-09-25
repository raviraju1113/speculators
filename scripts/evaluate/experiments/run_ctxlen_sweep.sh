#!/bin/bash
# Long-context sweep runner. Launch inside tmux so it survives disconnects:
#   tmux new-window -t ravir -n ctxsweep 'bash .../run_ctxlen_sweep.sh; exec bash'
set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda activate speculators
export PATH="/home/ravira/miniconda3/envs/speculators/bin:$PATH"   # runner calls a bare `vllm`
cd /sms-scratch/ravira/speculators/scripts/evaluate/experiments
LOG=/sms-scratch/ravira/speculators/scripts/evaluate/experiments/ctxsweep_run.log
echo "=== started $(date -Is) ===" | tee -a "$LOG"
python run_experiments.py --config gemma4-31b-mamba2-ctxlen-sweep.yaml 2>&1 | tee -a "$LOG"
echo "=== finished $(date -Is) rc=$? ===" | tee -a "$LOG"
