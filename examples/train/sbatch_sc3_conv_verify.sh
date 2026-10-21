#!/bin/bash
# Verify the vLLM DFlash2-conv serving patch on a real GPU before any eval of a
# conv checkpoint (gate job; the eval jobs are submitted --dependency=afterok).
#   1. Triton kernel parity vs the speculators training conv (test_dflash2_conv_parity.py)
#   2. serve the checkpoint at k=8 and confirm the conv layers were built
#      (log line "DFlash2 local conv enabled") and all conv weights loaded
#   3. a 5-prompt AIME smoke eval: accept_len must clear a floor that a draft
#      with silently dropped convs could not (scratch ep25 gets ~6.0 at k=8).
# Exit codes: 5 patch missing, 6 parity, 7 conv layers not built, 8 eval
# produced no summary, 9 accept_len below floor.
#
#   cd /import/ml-sc-scratch1/mengmengj/speculators
#   V=$(sbatch --parsable examples/train/sbatch_sc3_conv_verify.sh)
#   CONFIG=gemma4-31b-dspark-conv-ep25.yaml sbatch --dependency=afterok:$V examples/train/sbatch_sc3_eval.sh
#SBATCH -N 1
#SBATCH --partition=gpuonly
#SBATCH --gres=gpu:b200m180:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --exclude=sc3-cp78,sc3-cp79,sc3-cp80
#SBATCH -J conv_verify
#SBATCH --chdir=/import/ml-sc-scratch1/mengmengj/speculators
#SBATCH -o logs/%x-%j.out

# shellcheck disable=SC1091
source /import/ml-sc-scratch1/mengmengj/speculators/examples/train/sc3_eval_prelude.sh

CKPT="${CKPT:-/import/ml-sc-scratch1/mengmengj/output/gemma4_31b_dspark_nemo782k_conv/checkpoints/25}"
K="${K:-8}"
FLOOR="${FLOOR:-3.5}"
[ -d "$CKPT" ] || { echo "!! checkpoint missing: $CKPT" >&2; exit 3; }
check_conv_servable "$CKPT" || exit $?

echo "=== 1. kernel parity (cuda) ==="
python examples/train/vllm_patches/test_dflash2_conv_parity.py "$CKPT" --device cuda || { echo "!! parity failed" >&2; exit 6; }

echo "=== 2. serve k=$K ==="
PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 10000 ))
LOG="logs/conv_verify_server_${SLURM_JOB_ID}.log"
vllm serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 --max-model-len 4096 \
  --speculative-config "{\"model\": \"$CKPT\", \"num_speculative_tokens\": $K, \"method\": \"dspark\"}" \
  --enforce-eager --no-enable-prefix-caching \
  --chat-template "$CHAT_TEMPLATE" > "$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT TERM INT
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "!! server died:"; grep -E 'Error|error|Traceback' "$LOG" | head -20; exit 1; }
  sleep 15
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "!! server not healthy after 30 min" >&2; exit 1; }
if grep -q 'DFlash2 local conv enabled' "$LOG"; then
  grep -m1 'DFlash2 local conv enabled' "$LOG"
else
  echo "!! conv layers were NOT built (no 'DFlash2 local conv enabled' in $LOG)" >&2; exit 7
fi

echo "=== 3. smoke eval: aime x5 at k=$K ==="
OUT="scripts/evaluate/experiments/results/conv_verify_${SLURM_JOB_ID}_k${K}_aime5"
python scripts/evaluate/mtp_server_eval/run_vllm_eval.py \
  --base-url "http://127.0.0.1:$PORT" \
  --benchmarks aime --num-samples 5 --max-tokens 1024 --temperature 0.0 \
  --output-dir "$OUT"
S=$(ls "$OUT"/*summary*.json 2>/dev/null | head -1)
[ -n "$S" ] || { echo "!! no summary json in $OUT" >&2; exit 8; }
python - "$S" "$FLOOR" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); floor = float(sys.argv[2])
found = []
def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (int, float)) and "accept_len" in k and "std" not in k:
                found.append((path + "/" + k, float(v)))
            walk(v, path + "/" + k)
    elif isinstance(o, list):
        for i, v in enumerate(o): walk(v, f"{path}[{i}]")
walk(d)
print("accept_len fields:", found)
if not found:
    print("!! no accept_len field in summary"); sys.exit(8)
best = max(v for _, v in found)
print(f"max accept_len {best:.3f} vs floor {floor}")
sys.exit(0 if best >= floor else 9)
PY
RC=$?
echo "=== verify done rc=$RC $(date '+%F %T') ==="
exit $RC
