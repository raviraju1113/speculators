#!/bin/bash
# AA-LCR paired context-length bins (1k..112k, Ravi's prepare_aa_lcr_sweep.py
# design: same questions at every length) x three servings:
#   k=8 spec, k=3 spec, no-draft baseline  (scratch checkpoint_best)
# 128k is NOT possible: AA-LCR's longest row is ~120k gemma tokens; top bin is
# 112k (n=37) and 96k (n=67) -- bins above 64k are not fully paired (n<100).
# Submit: sbatch /sms-scratch/mengmengj/speculators/examples/train/sbatch_lcr_bins.sh
#SBATCH -N1
#SBATCH --exclude=trn-b200x8-101,trn-b200x8-099,trn-b200x8-097
#SBATCH --gres=gpu:b200:2
#SBATCH -J lcr_bins
#SBATCH --chdir=/sms-scratch/mengmengj/speculators
#SBATCH -o logs/%x-%j.out

set -uo pipefail
echo "=== job $SLURM_JOB_ID on $(hostname) $(date '+%F %T') ==="
source /sms-scratch/mengmengj/envs/dspark-host/bin/activate

IDLE=""
for i in $(seq 1 72); do
  IDLE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
         awk -F', ' '$2<1000{print $1}' | head -2 | paste -sd,)
  N=$(echo "$IDLE" | awk -F, 'NF>0{print NF}'); N=${N:-0}
  [ "$N" -ge 2 ] && break
  echo "fewer than 2 idle GPUs; waiting ($i/72)"; sleep 300
done
N=$(echo "$IDLE" | awk -F, 'NF>0{print NF}'); N=${N:-0}
[ "$N" -lt 2 ] && { echo "!! never got 2 idle GPUs; aborting"; exit 1; }
export CUDA_VISIBLE_DEVICES="$IDLE"
echo "using GPUs $IDLE"

# CKPT/TAG/SKIP_BASE overridable: submit with e.g.
#   CKPT=.../fullattn2/checkpoints/15 TAG=fullattn2_ep15 SKIP_BASE=1 sbatch ...
# (baseline pass is checkpoint-independent -- run it once, skip elsewhere)
# METHOD/K_MAIN for non-dspark drafts, e.g. RedHat's all-full-attention DFlash
# (native k=7, no anchor sampling):
#   CKPT=/sms-scratch/checkpoints/gemma-4-31B-it-speculator.dflash \
#   TAG=redhat_dflash METHOD=dflash K_MAIN=7 SKIP_BASE=1 sbatch ...
CKPT="${CKPT:-/sms-scratch/mengmengj/output/gemma4_31b_dspark_nemo782k_scratch/checkpoints/checkpoint_best}"
TAG="${TAG:-scratch_best}"
METHOD="${METHOD:-dspark}"
K_MAIN="${K_MAIN:-8}"
BENCHES="aa-lcr-1k,aa-lcr-2k,aa-lcr-4k,aa-lcr-8k,aa-lcr-16k,aa-lcr-32k,aa-lcr-64k,aa-lcr-96k,aa-lcr-112k"
PORT=8600

run_pass () {  # $1 = tag (k8|k3|base); remaining args = extra vllm serve args
  local tag=$1; shift
  vllm serve /sms-scratch/checkpoints/gemma-4-31B-it \
    --host 127.0.0.1 --port $PORT --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.92 --max-model-len 131072 \
    --enforce-eager --no-enable-prefix-caching \
    --chat-template /sms-scratch/checkpoints/gemma-4-31B-it/chat_template.jinja \
    "$@" > "logs/lcr_bins_server_${tag}.log" 2>&1 &
  local srv=$!
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $srv 2>/dev/null || { echo "!! $tag server died"; tail -20 "logs/lcr_bins_server_${tag}.log"; return 1; }
    sleep 15
  done
  echo "--- $tag pass $(date '+%H:%M') ---"
  python scripts/evaluate/mtp_server_eval/run_vllm_eval.py \
    --base-url "http://127.0.0.1:$PORT" \
    --benchmarks "$BENCHES" --num-samples 0 --max-tokens 512 --temperature 0.0 \
    --output-dir "scripts/evaluate/experiments/results/lcr_bins_${TAG}_${tag}"
  kill $srv 2>/dev/null; wait $srv 2>/dev/null; sleep 10
}

run_pass "k${K_MAIN}" --speculative-config "{\"model\": \"$CKPT\", \"num_speculative_tokens\": ${K_MAIN}, \"method\": \"$METHOD\"}"
# SKIP_K3=1 for add-on passes (e.g. matched-k7 reruns) where k3 already exists
[ "${SKIP_K3:-0}" = "1" ] || run_pass k3 --speculative-config "{\"model\": \"$CKPT\", \"num_speculative_tokens\": 3, \"method\": \"$METHOD\"}"
[ "${SKIP_BASE:-0}" = "1" ] || run_pass base
echo "=== lcr bins done $(date '+%F %T') ==="
