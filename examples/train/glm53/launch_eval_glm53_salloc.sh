#!/usr/bin/env bash
# Start run_eval_glm53_1node.sh inside an existing 1-node allocation, with the srun
# client detached on the login node (survives the launching shell).
#
#   salloc -p gpuonly --reservation=sn50-hdi-0 -w <node> -N1 --ntasks-per-node=1 \
#     --gres=gpu:b200m180:8 --exclusive --cpus-per-task=128 --mem=0 \
#     --time=2-00:00:00 -J glm53_dflash2_eval --no-shell
#   JOBID=<id> CONFIG=examples/train/glm53/eval_glm53_dflash2_fp8_v033.yaml \
#     bash examples/train/glm53/launch_eval_glm53_salloc.sh [--only a,b]
set -euo pipefail
: "${JOBID:?salloc job id}"
: "${CONFIG:?eval config yaml}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
LOG="/import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2/logs/eval_$(basename "$CONFIG" .yaml)_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
setsid nohup srun --jobid="$JOBID" --overlap -N1 -n1 /usr/bin/bash \
    "$REPO/examples/train/glm53/run_eval_glm53_1node.sh" "$(realpath "$CONFIG")" "$@" \
    > "$LOG" 2>&1 < /dev/null &
echo "srun pid $! | log $LOG"
