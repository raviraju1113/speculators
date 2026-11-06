#!/bin/bash
# Detached resume of the 400k DSpark run: setsid puts the server and trainer in
# their own sessions/process groups so a dying parent shell (or a harness task
# teardown) cannot SIGTERM them. PIDs land in $OUT/pids/ for a clean stop.
set -euo pipefail
cd /nvmedata/chenw/speculators
source /root/miniconda3/etc/profile.d/conda.sh
conda activate speculator
export CUDA_HOME=/usr/local/cuda-12.9 PATH="/usr/local/cuda-12.9/bin:$PATH"

OUT=output/gemma4_26b_dspark_400k
HS=/nvmedata/chenw/hidden_states_400k_transit
mkdir -p "$OUT/logs" "$OUT/pids" "$HS"

setsid nohup python scripts/launch_vllm.py /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it \
    --hidden-states-path "$HS" --target-layer-ids 2 15 27 \
    -- --port 8000 --max-model-len 4352 --gpu-memory-utilization 0.85 \
    >> "$OUT/logs/vllm_server.log" 2>&1 < /dev/null &
echo $! > "$OUT/pids/server.pid"
echo "server pid=$(cat "$OUT/pids/server.pid") (detached)"

echo "waiting for /health ..."
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
    --target-layer-ids 2 15 27 --draft-vocab-size 32000 \
    --epochs 3 --early-stop-patience 1 --total-seq-len 4096 \
    --block-size 8 --max-anchors 3072 --num-layers 3 \
    --markov-rank 256 --markov-head-type vanilla \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' --confidence-head-alpha 1.0 \
    --no-sample-from-anchor --checkpoint-freq 0.25 \
    --on-missing generate --on-generate delete \
    >> "$OUT/logs/dspark.log" 2>&1 < /dev/null &
echo $! > "$OUT/pids/trainer.pid"
echo "trainer pid=$(cat "$OUT/pids/trainer.pid") (detached, resumes from epoch 0)"
