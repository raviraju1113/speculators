#!/bin/bash
# Draft-OPD phase 1 smoke test (gate 1): serve the target + DSpark draft with the
# rollout logger on, push N prompts through, assemble and validate the records.
# 1 GPU (A100-80 is enough at 2k context, 4 concurrent seqs).
#
#   cd /import/ml-sc-scratch1/mengmengj/speculators
#   sbatch -w sc3-c98 --gres=gpu:a100m80:1 examples/train/opd/sbatch_sc3_rollout_smoke.sh
#   CKPT=<draft dir> K=8 N=40 PROMPTS=<jsonl> sbatch ...
# B300 (sc3-c130/131, reservation sn50-hdi-0; also the env/arch smoke for OPD):
#   K=12 sbatch -w sc3-c130 --reservation sn50-hdi-0 --gres=gpu:b300m288:1 \
#     examples/train/opd/sbatch_sc3_rollout_smoke.sh      # seed = blk12 ep33 default
#SBATCH -N 1
#SBATCH --partition=gpuonly
#SBATCH --gres=gpu:a100m80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH -J opd_smoke
#SBATCH --chdir=/import/ml-sc-scratch1/mengmengj/speculators
#SBATCH -o logs/%x-%j.out

# shellcheck disable=SC1091
source /import/ml-sc-scratch1/mengmengj/speculators/examples/train/sc3_eval_prelude.sh

CKPT="${CKPT:-/import/ml-sc-scratch1/mengmengj/output/gemma4_31b_dspark_nemo782k_conv_blk12/checkpoints/33}"
K="${K:-12}"
N="${N:-40}"
CONC="${CONC:-4}"
MAXTOK="${MAXTOK:-512}"
PROMPTS="${PROMPTS:-$REPO/scripts/evaluate/eval_datasets/gsm8k.jsonl}"
OUT="${OUT:-$REPO/output_opd/smoke_${SLURM_JOB_ID}}"
mkdir -p "$OUT/engine_log"
check_conv_servable "$CKPT" || exit $?
# env/arch facts for the log (B300 is sm_103: a torch/vllm built for B200 only
# works there via forward-compatible PTX; kernels built sm_100a-only will fail)
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'archs', torch.cuda.get_arch_list()); print('device', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 10000 ))

# The logger lives in the engine-core process, which inherits the env.
export DSPARK_ROLLOUT_LOG="$OUT/engine_log"
vllm serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 --max-model-len 2048 --max-num-seqs "$CONC" \
  --speculative-config "{\"model\": \"$CKPT\", \"num_speculative_tokens\": $K, \"method\": \"dspark\"}" \
  --max-num-batched-tokens "${MAX_BATCHED:-16384}" \
  --enforce-eager --no-enable-prefix-caching \
  --no-async-scheduling \
  --chat-template "$CHAT_TEMPLATE" > "$OUT/server.log" 2>&1 &
# --max-num-batched-tokens: k=12 drives vLLM 0.26's scheduled-token budget
# negative at the default (the k=12 eval hit it first).
# --no-async-scheduling is REQUIRED for the logger: only the sync path feeds the
# real draft ids back to the scheduler (the logger refuses to start otherwise).
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT TERM INT
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "!! server died:"; grep -E 'Error|error|Traceback' "$OUT/server.log" | head -20; exit 1; }
  sleep 15
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "!! server not healthy" >&2; exit 1; }
grep -m1 'rollout logging to' "$OUT/server.log" || { echo "!! logger did not start (no 'rollout logging to' line)" >&2; exit 7; }

echo "=== rollout: $N prompts, k=$K, concurrency $CONC, max_tokens $MAXTOK ==="
python scripts/opd/rollout.py --base-url "http://127.0.0.1:$PORT" --model "$MODEL" \
  --prompts "$PROMPTS" --limit "$N" --concurrency "$CONC" --max-tokens "$MAXTOK" \
  --temperature 0 --tag smoke --out "$OUT"
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 5

echo "=== assemble + validate ==="
python scripts/opd/assemble_rollouts.py --log-dir "$OUT/engine_log" --responses "$OUT/responses.jsonl" \
  --out "$OUT/rollouts.jsonl" --draft-ckpt "$CKPT"
RC=$?
echo "=== smoke done rc=$RC $(date '+%F %T') -> $OUT ==="
exit $RC
