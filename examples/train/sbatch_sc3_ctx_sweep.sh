#!/bin/bash
# sc3-cluster twin of sbatch_ctx_sweep.sh: context-length sweep (1k..32k
# prompt + 512 gen) at k=8 for one draft checkpoint, 1 B200. Measures
# accept_len / accept_rate / decode tok/s per prompt-length bucket. The draft
# trains at total_seq_len 8192 with a 2048 sliding window, so beyond ~8k this
# is the extrapolation curve. Results land in
# scripts/evaluate/experiments/results/ctx_sweep_<TAG>_k<K>, comparable with
# ctx_sweep_scratch_ep25_k8 (same harness, TP=1).
#
#   cd /import/ml-sc-scratch1/mengmengj/speculators
#   sbatch examples/train/sbatch_sc3_ctx_sweep.sh                  # conv ep25, k=8
#   CKPT=<dir> TAG=<name> K=8 sbatch examples/train/sbatch_sc3_ctx_sweep.sh
#SBATCH -N 1
#SBATCH --partition=gpuonly
#SBATCH --gres=gpu:b200m180:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --exclude=sc3-cp78,sc3-cp79,sc3-cp80
#SBATCH -J ctx_sweep
#SBATCH --chdir=/import/ml-sc-scratch1/mengmengj/speculators
#SBATCH -o logs/%x-%j.out

# shellcheck disable=SC1091
source /import/ml-sc-scratch1/mengmengj/speculators/examples/train/sc3_eval_prelude.sh

CKPT="${CKPT:-/import/ml-sc-scratch1/mengmengj/output/gemma4_31b_dspark_nemo782k_conv/checkpoints/25}"
TAG="${TAG:-conv_ep25}"
K="${K:-8}"
[ -d "$CKPT" ] || { echo "!! checkpoint missing: $CKPT" >&2; exit 3; }
check_conv_servable "$CKPT" || exit $?
# Nodes are shared with other users' 1-GPU jobs: derive the port from the job id.
PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 10000 ))
OUT="$REPO/scripts/evaluate/experiments/results/ctx_sweep_${TAG}_k${K}"

vllm serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 --max-model-len 34816 \
  --speculative-config "{\"model\": \"$CKPT\", \"num_speculative_tokens\": $K, \"method\": \"dspark\"}" \
  --enforce-eager --no-enable-prefix-caching \
  --chat-template "$CHAT_TEMPLATE" \
  > "logs/ctx_sweep_server_${TAG}_${SLURM_JOB_ID}.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT TERM INT

for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "!! server died:"; tail -20 "logs/ctx_sweep_server_${TAG}_${SLURM_JOB_ID}.log"; exit 1; }
  sleep 15
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "!! server not healthy after 30 min" >&2; exit 1; }
echo "server healthy on :$PORT; sweeping -> $OUT"

cd "$REPO/scripts/evaluate/mtp_server_eval"
python sweep_context_len.py --base-url "http://127.0.0.1:$PORT" \
  --model "$MODEL" \
  --lengths 1024,2048,4096,8192,16384,32768 \
  --requests 24 --max-tokens 512 \
  --output-dir "$OUT"
RC=$?
echo "=== sweep done rc=$RC $(date '+%F %T') ==="
exit $RC
