#!/bin/bash
# DSpark with the FULL 262k verifier vocab (no d2t/t2d pruning).
#
# Why: our 32k-vocab DSpark reaches only 12.2% of target ids (measured: 82.8%
# of sc1_delta tokens, 54.3% of multilingual), and every unreachable token is a
# guaranteed rejection. Measured cost/benefit of that pruning on sc1_delta:
# it saves ~7% of draft cost (omega 1.18 vs 1.35) but destroys ~51% of
# acceptance (AL 2.48 vs stock DFlash's 5.03). Stock DFlash - same parallel
# block architecture, full vocab - gets 2.14x where ours gets 1.14x.
#
# Memory: the 262k output head makes loss activations ~8x larger than at 32k
# (first attempt OOMed in backward trying to allocate 12 GiB on top of 58 GiB).
# max_anchors is halved 3072 -> 1536 to compensate; PYTORCH_CUDA_ALLOC_CONF
# reduces fragmentation. Loss is already the fused (chunked) implementation.
#
# train.py falls back to the full verifier vocab only when it can generate
# NO mappings, so this data_prep has d2t.npy/t2d.npy removed and token_freq.pt
# renamed; passing no --draft-vocab-size is not sufficient on its own.
set -euo pipefail
cd /nvmedata/chenw/speculators
source /root/miniconda3/etc/profile.d/conda.sh
conda activate speculator
export CUDA_HOME=/usr/local/cuda-12.9 PATH="/usr/local/cuda-12.9/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=output/gemma4_26b_dspark_fullvocab
HS=/nvmedata/chenw/hidden_states_fullvocab
mkdir -p "$OUT/logs" "$OUT/pids" "$HS"

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
    --epochs 3 --early-stop-patience 1 --total-seq-len 4096 \
    --block-size 8 --max-anchors 1536 --num-layers 3 \
    --markov-rank 256 --markov-head-type vanilla \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' --confidence-head-alpha 1.0 \
    --no-sample-from-anchor --checkpoint-freq 0.25 \
    --on-missing generate --on-generate delete \
    >> "$OUT/logs/dspark.log" 2>&1 < /dev/null &
echo $! > "$OUT/pids/trainer.pid"; echo "trainer pid=$(cat "$OUT/pids/trainer.pid") FULL VOCAB"
