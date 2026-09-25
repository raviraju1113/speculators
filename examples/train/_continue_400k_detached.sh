#!/bin/bash
# Continuation of the 400k DSpark run: 4 more epochs with early stopping.
#
# NOT a plain resume: the first run used the default linear LR decay and ended
# at lr ~6.5e-07, so resuming into --save-path would restore that dead schedule
# and learn nothing. Instead we load the trained weights with --from-pretrained
# into a FRESH save-path, restarting the schedule at a lower peak LR (1e-4 vs
# the original 3e-4) with cosine decay — standard continued-training practice.
#
# NOTE: --from-pretrained carries the architecture, so model-definition flags
# like --num-layers must NOT be passed alongside it (train.py rejects them).
#
# Early stopping (--early-stop-patience 1) decides the real epoch count: it
# stops one epoch after validation loss stops improving. NOTE: best_val_loss
# starts fresh in the new save-path, so the reference is this run's epoch 0,
# not the previous run's 0.495 — compare against that manually.
set -euo pipefail
cd /nvmedata/chenw/speculators
source /root/miniconda3/etc/profile.d/conda.sh
conda activate speculator
export CUDA_HOME=/usr/local/cuda-12.9 PATH="/usr/local/cuda-12.9/bin:$PATH"

PREV=output/gemma4_26b_dspark_400k
OUT=output/gemma4_26b_dspark_400k_cont
HS=/nvmedata/chenw/hidden_states_400k_transit
mkdir -p "$OUT/logs" "$OUT/pids" "$HS"

setsid nohup python scripts/launch_vllm.py /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it \
    --hidden-states-path "$HS" --target-layer-ids 2 15 27 \
    -- --port 8000 --max-model-len 4352 --gpu-memory-utilization 0.85 \
    >> "$OUT/logs/vllm_server.log" 2>&1 < /dev/null &
echo $! > "$OUT/pids/server.pid"; echo "server pid=$(cat "$OUT/pids/server.pid")"

echo "waiting for /health ..."
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 10; done
echo "server ready"

CUDA_VISIBLE_DEVICES=1,2,3 setsid nohup torchrun --standalone --nproc_per_node 3 \
    scripts/train.py \
    --verifier-name-or-path /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it \
    --from-pretrained "$PREV/dspark/checkpoints/checkpoint_best" \
    --data-path "$PREV/data_prep" \
    --vllm-endpoint "http://localhost:8000/v1" \
    --hidden-states-path "$HS" \
    --save-path "$OUT/dspark/checkpoints" \
    --speculator-type dspark --lr 1e-4 --scheduler-type cosine \
    --target-layer-ids 2 15 27 --draft-vocab-size 32000 \
    --epochs 4 --early-stop-patience 1 --total-seq-len 4096 \
    --block-size 8 --max-anchors 3072 \
    --markov-rank 256 --markov-head-type vanilla \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' --confidence-head-alpha 1.0 \
    --no-sample-from-anchor --checkpoint-freq 0.25 \
    --on-missing generate --on-generate delete \
    >> "$OUT/logs/dspark.log" 2>&1 < /dev/null &
echo $! > "$OUT/pids/trainer.pid"; echo "trainer pid=$(cat "$OUT/pids/trainer.pid") (detached)"
