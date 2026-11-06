#!/bin/bash
# Context-length sweep for the scratch checkpoint_best (ep25) at k=8.
# Submit: sbatch /sms-scratch/mengmengj/speculators/examples/train/sbatch_ctx_sweep.sh
#
# Serves the verifier+draft at max-model-len 34816 (sweep goes to 32k prompt +
# 512 gen) and measures accept_len / accept_rate / decode tok/s per prompt-length
# bucket {1k..32k}. Draft trained at total_seq_len 8192 with a 2048 sliding
# window -- beyond ~8k is extrapolation; this measures the degradation curve.
#SBATCH -N1
#SBATCH --exclude=trn-b200x8-101,trn-b200x8-099
#SBATCH --gres=gpu:b200:1
#SBATCH -J ctx_sweep
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
  echo "GPUs busy with off-SLURM work; waiting ($i/72)"; sleep 300
done
[ -z "$G" ] && { echo "!! no free GPU after 6h"; exit 1; }
export CUDA_VISIBLE_DEVICES="$G"
echo "using GPU $G"

CKPT=/sms-scratch/mengmengj/output/gemma4_31b_dspark_nemo782k_scratch/checkpoints/checkpoint_best
PORT=8600
vllm serve /sms-scratch/checkpoints/gemma-4-31B-it \
  --host 127.0.0.1 --port $PORT --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 --max-model-len 34816 \
  --speculative-config "{\"model\": \"$CKPT\", \"num_speculative_tokens\": 8, \"method\": \"dspark\"}" \
  --enforce-eager --no-enable-prefix-caching \
  --chat-template /sms-scratch/checkpoints/gemma-4-31B-it/chat_template.jinja \
  > logs/ctx_sweep_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT TERM INT

for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "!! server died:"; tail -20 logs/ctx_sweep_server.log; exit 1; }
  sleep 15
done
echo "server healthy; sweeping..."

cd scripts/evaluate/mtp_server_eval
python sweep_context_len.py --base-url "http://127.0.0.1:$PORT" \
  --model /sms-scratch/checkpoints/gemma-4-31B-it \
  --lengths 1024,2048,4096,8192,16384,32768 \
  --requests 24 --max-tokens 512 \
  --output-dir ../experiments/results/ctx_sweep_scratch_ep25_k8
echo "=== sweep done $(date '+%F %T') ==="
