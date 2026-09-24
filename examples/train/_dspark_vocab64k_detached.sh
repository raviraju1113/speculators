#!/bin/bash
# Experiment 1: does a larger draft vocab close DSpark's multilingual /
# structured-text gap?
#
# Diagnosis being tested (docs/experiments/gemma4_26b_moe_results.md §7/§9):
# DSpark drafts from a 32k pruned vocab while the target has 262k, so any
# out-of-vocab token is an unreachable draft — a guaranteed rejection. Scaling
# data 30k -> 400k moved multilingual only 0.97x -> 1.06x while math gained
# ~0.5, which says the blocker is vocabulary, not data volume.
#
# Fast read first: 100k samples (~12h) rather than 400k (~2 days/epoch). If
# multilingual moves, rerun at full scale.
set -euo pipefail
cd /nvmedata/chenw/speculators
source /root/miniconda3/etc/profile.d/conda.sh
conda activate speculator
export CUDA_HOME=/usr/local/cuda-12.9 PATH="/usr/local/cuda-12.9/bin:$PATH"

VOCAB="${VOCAB:-65536}"
SAMPLES="${SAMPLES:-100000}"
OUT="output/gemma4_26b_dspark_vocab${VOCAB}"
HS=/nvmedata/chenw/hidden_states_vocab_exp
mkdir -p "$OUT/logs" "$OUT/pids" "$HS"

if [[ ! -d "$OUT/data_prep" ]]; then
  echo "=== preparing $SAMPLES samples (token_freq drives the d2t/t2d maps) ==="
  python scripts/prepare_data.py \
      --model /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it \
      --data /nvmedata/data/merged_all_regen.jsonl \
      --max-samples "$SAMPLES" --seq-length 4096 \
      --output "$OUT/data_prep" 2>&1 | tail -3
fi

setsid nohup python scripts/launch_vllm.py /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it \
    --hidden-states-path "$HS" --target-layer-ids 2 15 27 \
    -- --port 8000 --max-model-len 4352 --gpu-memory-utilization 0.85 \
    >> "$OUT/logs/vllm_server.log" 2>&1 < /dev/null &
echo $! > "$OUT/pids/server.pid"; echo "server pid=$(cat "$OUT/pids/server.pid")"
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 10; done
echo "server ready"

CUDA_VISIBLE_DEVICES=1,2,3 setsid nohup torchrun --standalone --nproc_per_node 3 \
    scripts/train.py \
    --verifier-name-or-path /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it \
    --data-path "$OUT/data_prep" \
    --vllm-endpoint "http://localhost:8000/v1" \
    --hidden-states-path "$HS" \
    --save-path "$OUT/dspark/checkpoints" \
    --speculator-type dspark --lr 3e-4 \
    --target-layer-ids 2 15 27 \
    --draft-vocab-size "$VOCAB" \
    --epochs 3 --early-stop-patience 1 --total-seq-len 4096 \
    --block-size 8 --max-anchors 3072 --num-layers 3 \
    --markov-rank 256 --markov-head-type vanilla \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' --confidence-head-alpha 1.0 \
    --no-sample-from-anchor --checkpoint-freq 0.25 \
    --on-missing generate --on-generate delete \
    >> "$OUT/logs/dspark.log" 2>&1 < /dev/null &
echo $! > "$OUT/pids/trainer.pid"; echo "trainer pid=$(cat "$OUT/pids/trainer.pid") vocab=$VOCAB samples=$SAMPLES"
