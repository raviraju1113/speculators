#!/bin/bash
# Does vLLM DATA parallelism work on bare metal (i.e. without NCCL)?
#
# WHY THIS MATTERS
#   Hidden-state generation is a large share of a full run: ~22 h/epoch at TP=1
#   (measured ~6k tok/s over ~473M tokens). TENSOR parallelism would cut that but
#   needs NCCL, which segfaults here (ncclNetPluginInit). DATA parallelism instead
#   runs N independent single-GPU engines -- no model-level collectives -- so it may
#   give the same throughput win with no container and no NCCL.
#   DP=2 -> ~2x, DP=3 -> ~3x generation.
#
#   Unknown: vLLM runs a DP coordinator, and it is not obvious whether that
#   initializes a torch.distributed process group. This script answers it.
#
#   sngpu --jobname vllm_dp_test --partition gpuonly --exclude sc-c96,sc3-c97,sc-c82 \
#     --gpu 2 --gputype a100m80 --cpu 16 --mem 200000 --time 00:30:00 \
#     --output ./logs/vllm_dp_test.txt \
#     -- bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/test_vllm_dp.sh
set -uo pipefail

REPO=/import/ml-sc-scratch1/mengmengj/speculators
CONDA_SH=/import/snvm-sc-scratch1/mengmengj/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=/import/ml-sc-scratch1/mengmengj/condaenvs/dspark
MODEL=/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it
PORT="${PORT:-8123}"
DP="${DP:-2}"
LOG=/tmp/vllm_dp_test_$$.log
HS=/tmp/vllm_dp_hs_$$

# shellcheck disable=SC1090
source "$CONDA_SH"; conda activate "$CONDA_ENV"; cd "$REPO"
mkdir -p "$HS"

echo "host: $(hostname)  driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "visible GPUs: $(nvidia-smi -L | wc -l)   testing DP=$DP (TP=1)"
echo "vLLM log -> $LOG"
echo

python scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HS" \
    --target-layer-ids 1 17 29 47 58 \
    -- --tensor-parallel-size 1 \
       --data-parallel-size "$DP" \
       --max-model-len 8448 \
       --gpu-memory-utilization 0.95 \
       --no-enable-prefix-caching \
       --port "$PORT" \
    > "$LOG" 2>&1 &
PID=$!
cleanup() { kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; rm -rf "$HS"; }
trap cleanup EXIT

echo "waiting up to 25 min for health..."
OK=0
for _ in $(seq 1 150); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then OK=1; break; fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "!! vLLM exited early"; break; fi
  sleep 10
done

echo
echo "=================== VERDICT ==================="
if [ "$OK" = 1 ]; then
  echo ">>> vLLM DP=$DP CAME UP on bare metal."
  echo "    Generation can be scaled without NCCL and without the container:"
  echo "      MODE=full_dp (VLLM_TP=1 VLLM_DP=3 TRAIN_GPUS_N=1) on 4 GPUs -> ~3x."
  echo "-- GPU memory (expect $DP verifier copies) --"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
else
  echo ">>> vLLM DP=$DP FAILED to come up."
fi
echo "-- was NCCL involved at all? --"
n=$(grep -cE "ncclCommInitRank|ncclNetPluginInit" "$LOG" 2>/dev/null); n=${n:-0}
echo "NCCL init mentions in the vLLM log: $n"
if [ "${n:-0}" -gt 0 ]; then
  echo "   -> DP does touch NCCL here; expect the same segfault as TP>=2."
  grep -m3 -E "ncclCommInitRank|ncclNetPluginInit|Segfault" "$LOG" 2>/dev/null
else
  echo "   -> no NCCL init seen, consistent with independent single-GPU engines."
fi
echo "-- last errors, if any --"
grep -nE "Error|ERROR|Traceback|Segfault" "$LOG" 2>/dev/null | grep -v 'File "' | tail -5
echo "==============================================="
