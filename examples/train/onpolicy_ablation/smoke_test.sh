#!/usr/bin/env bash
# Phase 0: end-to-end smoke test of the ablation pipeline on 4 GPUs.
#   regen 64 prompts -> prep both arms -> 30 DSpark steps on 3 ranks (NCCL) ->
#   serve the checkpoint with dspark spec decoding and read vLLM's acceptance log.
#
#   sbatch -p gpuonly -w sc-c96 --gres=gpu:a100m80:4 -c 32 --mem 200G -t 3:00:00 \
#     -o <abl>/logs/smoke-%j.out examples/train/onpolicy_ablation/smoke_test.sh
set -euo pipefail
ABL="${ABL:-/import/ml-sc-scratch1/ravir/spec_decoding_work/onpolicy_ablation}"
REPO="${REPO:-/import/ml-sc-scratch1/ravir/spec_decoding_work/speculators}"
PY_ENV="${PY_ENV:-/import/ml-sc-scratch1/ravir/miniconda3/envs/dspark_cu12}"
MODEL="${MODEL:-/import/ml-sc-scratch1/mengmengj/models/llama3.1-8b-instruct}"
D="$REPO/examples/train/onpolicy_ablation"
mkdir -p "$ABL/smoke"
source "$(dirname "$PY_ENV")/../etc/profile.d/conda.sh"; conda activate "$PY_ENV"
nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda ok:', torch.cuda.is_available(), torch.cuda.device_count())"

echo "=== 1. regen 64 prompts (GPU 0) ==="
[ "$(wc -l < "$ABL/smoke/on_policy_smoke.jsonl" 2>/dev/null || echo 0)" -ge 64 ] || \
CUDA_VISIBLE_DEVICES=0 LIMIT=64 NUM_GPUS=1 CONCURRENCY=64 \
    OUTFILE="$ABL/smoke/on_policy_smoke.jsonl" LOG_DIR="$ABL/smoke/regen_logs" \
    bash "$D/run_regen.sh"
wc -l "$ABL/smoke/on_policy_smoke.jsonl"

echo "=== 2. on-arm prep only ==="
python "$REPO/scripts/prepare_data.py" --model "$MODEL" \
    --data "$ABL/smoke/on_policy_smoke.jsonl" --seq-length 8192 \
    --output "$ABL/smoke/prep_on" --overwrite

echo "=== 3. off-arm: 30 steps, vLLM GPU0, train GPUs 1,2,3 ==="
SMOKE=1 ARM=off DATA_JSONL="$ABL/data/off_policy.jsonl" VLLM_GPU=0 TRAIN_GPUS=1,2,3 bash "$D/train_arm.sh"
CKPT=$(ls -d "$ABL"/ckpt/smoke_off_s0/*/ 2>/dev/null | sort | tail -1)
echo "checkpoint: $CKPT"; ls "$CKPT"
python - "$CKPT" <<'PY'
import json, sys
c = json.load(open(sys.argv[1].rstrip('/') + '/config.json'))
sc = c.get('speculators_config', {})
print('sample_from_anchor', c.get('sample_from_anchor'), 'speculative_tokens',
      [p.get('speculative_tokens') for p in sc.get('proposal_methods', [])],
      'aux', c.get('target_layer_ids') or c.get('aux_hidden_state_layer_ids'))
PY

echo "=== 4. serve with dspark spec decoding (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" --port 8300 --max-model-len 8192 \
    --speculative-config "{\"method\":\"dspark\",\"model\":\"${CKPT%/}\",\"num_speculative_tokens\":7}" \
    > "$ABL/smoke/serve.log" 2>&1 &
SP=$!; trap 'kill $SP 2>/dev/null || true' EXIT
until curl -sf localhost:8300/health >/dev/null; do
    kill -0 $SP || { tail -40 "$ABL/smoke/serve.log"; exit 1; }; sleep 5; done
for q in "Explain TCP slow start." "Write a python function for quicksort." \
         "What is 17*23? Show steps." "Summarize the causes of World War I."; do
    curl -s localhost:8300/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":256,\"temperature\":0}" \
      | python -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'][:120].replace(chr(10),' '))"
done
sleep 12
grep -E "SpecDecoding metrics|Per-position" "$ABL/smoke/serve.log" | tail -4
echo "=== SMOKE OK ==="
