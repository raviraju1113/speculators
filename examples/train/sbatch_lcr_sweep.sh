#!/bin/bash
# AA-LCR long-context sweep (16k/32k/64k/full~106k) for scratch checkpoint_best
# at k=8, PLUS a no-draft baseline pass for speedup denominators.
# Submit: sbatch /sms-scratch/mengmengj/speculators/examples/train/sbatch_lcr_sweep.sh
#
# TP=2 (2 GPUs): at ~120k context the KV does not fit beside the 62.6GB weights
# on one B200. NOTE tok/s here is a TP=2 series -- NOT comparable with the TP=1
# numbers elsewhere; compare only within this sweep (spec vs baseline).
#SBATCH -N1
#SBATCH --exclude=trn-b200x8-101,trn-b200x8-099,trn-b200x8-097
#SBATCH --gres=gpu:b200:2
#SBATCH -J lcr_sweep
#SBATCH --chdir=/sms-scratch/mengmengj/speculators
#SBATCH -o logs/%x-%j.out

set -uo pipefail
echo "=== job $SLURM_JOB_ID on $(hostname) $(date '+%F %T') ==="
source /sms-scratch/mengmengj/envs/dspark-host/bin/activate

# need 2 idle GPUs (off-SLURM squatters invisible to the scheduler)
IDLE=""
for i in $(seq 1 72); do
  IDLE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
         awk -F', ' '$2<1000{print $1}' | head -2 | paste -sd,)
  N=$(echo "$IDLE" | awk -F, 'NF>0{print NF}'); N=${N:-0}
  [ "$N" -ge 2 ] && break
  echo "fewer than 2 idle GPUs; waiting ($i/72)"; sleep 300
done
N=$(echo "$IDLE" | awk -F, 'NF>0{print NF}'); N=${N:-0}
if [ "$N" -lt 2 ]; then
  echo "!! never got 2 idle GPUs (off-SLURM work on this node's allocation); aborting"
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$IDLE"
echo "using GPUs $IDLE"

CKPT=/sms-scratch/mengmengj/output/gemma4_31b_dspark_nemo782k_scratch/checkpoints/checkpoint_best
PROMPTS=/sms-scratch/mengmengj/data/aa_lcr/prompts
FILES="$PROMPTS/lcr_16384.jsonl,$PROMPTS/lcr_32768.jsonl,$PROMPTS/lcr_65536.jsonl,$PROMPTS/lcr_full.jsonl"
PORT=8600

run_pass () {  # $1 = spec|base, $2 = extra server args
  local tag=$1; shift
  vllm serve /sms-scratch/checkpoints/gemma-4-31B-it \
    --host 127.0.0.1 --port $PORT --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.92 --max-model-len 131072 \
    --enforce-eager --no-enable-prefix-caching \
    --chat-template /sms-scratch/checkpoints/gemma-4-31B-it/chat_template.jinja \
    "$@" > "logs/lcr_sweep_server_${tag}.log" 2>&1 &
  local srv=$!
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $srv 2>/dev/null || { echo "!! $tag server died"; tail -20 "logs/lcr_sweep_server_${tag}.log"; return 1; }
    sleep 15
  done
  echo "--- $tag pass ---"
  (cd scripts/evaluate/mtp_server_eval && python sweep_context_len.py \
     --base-url "http://127.0.0.1:$PORT" \
     --model /sms-scratch/checkpoints/gemma-4-31B-it \
     --prompts-files "$FILES" --requests 30 --max-tokens 512 \
     --output-dir "../experiments/results/lcr_sweep_${tag}")
  kill $srv 2>/dev/null; wait $srv 2>/dev/null; sleep 10
}

run_pass spec --speculative-config "{\"model\": \"$CKPT\", \"num_speculative_tokens\": 8, \"method\": \"dspark\"}"
run_pass base
echo "=== lcr sweep done $(date '+%F %T') ==="
