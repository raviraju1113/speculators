#!/bin/bash
# DSpark warm-started from the STOCK DFlash backbone.
#
# Rationale (docs/experiments/gemma4_26b_moe_results.md §9-§11): our
# from-scratch DSpark matches stock DFlash on public benchmarks (1.94x vs
# 1.93x) but collapses on production traffic (1.14x vs 2.14x on sc1_delta).
# Both the architecture explanation (§9 - DFlash is the same parallel block
# draft and does fine) and the vocabulary explanation (§11 - doubling to 64k
# changed nothing) were refuted, leaving TRAINING BREADTH as the cause.
# Rather than try to reproduce that breadth with our narrow kimi-regen mix,
# inherit it: DSpark == "DFlash backbone + Markov head + confidence head"
# (its own docstring), so initialize the backbone from stock DFlash and train
# only to fit the two new heads.
#
# Init checkpoint built by converting the vLLM-native stock DFlash to
# speculators format, then relabelling it as dspark with markov/confidence
# enabled (those two heads load as freshly initialized).
#
# Differences from previous DSpark runs, all inherited from DFlash:
#   aux layers [2,7,12,18,23,28] (6, not our usual [2,15,27])
#   block_size 16 -> native k = 15 (not 8 -> 7)
#   full 262k draft vocab (not 32k)
# Memory: 2,779M-param draft, so max_anchors is cut to 512 (the 1536 run
# OOMed at 69.8 GiB) and expandable_segments is on.
set -euo pipefail
cd /nvmedata/chenw/speculators
source /root/miniconda3/etc/profile.d/conda.sh
conda activate speculator
export CUDA_HOME=/usr/local/cuda-12.9 PATH="/usr/local/cuda-12.9/bin:$PATH"
# NB: expandable_segments must NOT be exported globally - the vLLM
# hidden-states KV connector refuses to start with it ("KV connector
# ExampleHiddenStatesConnector is incompatible with ... expandable_segments").
# It is set inline on the trainer only, below.

INIT=output/dspark_from_dflash_init
OUT=output/gemma4_26b_dspark_from_dflash
HS=/nvmedata/chenw/hidden_states_from_dflash
AUX="2 7 12 18 23 28"
mkdir -p "$OUT/logs" "$OUT/pids" "$HS"

# data_prep without d2t/t2d so the full-vocab path is kept (the init ckpt is
# already 262k; a cached 32k mapping would fight it).
if [[ ! -d "$OUT/data_prep" ]]; then
  cp -r output/gemma4_26b_dspark_fullvocab/data_prep "$OUT/data_prep"
fi

setsid nohup python scripts/launch_vllm.py /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it \
    --hidden-states-path "$HS" --target-layer-ids $AUX \
    -- --port 8000 --max-model-len 4352 --gpu-memory-utilization 0.85 \
    >> "$OUT/logs/vllm_server.log" 2>&1 < /dev/null &
echo $! > "$OUT/pids/server.pid"; echo "server pid=$(cat "$OUT/pids/server.pid") aux=[$AUX]"
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 10; done
echo "server ready"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=1,2,3 setsid nohup torchrun --standalone --nproc_per_node 3 \
    scripts/train.py \
    --verifier-name-or-path /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it \
    --from-pretrained "$INIT" \
    --data-path "$OUT/data_prep" \
    --vllm-endpoint "http://localhost:8000/v1" \
    --hidden-states-path "$HS" \
    --save-path "$OUT/dspark/checkpoints" \
    --speculator-type dspark --lr 1e-4 --scheduler-type cosine \
    --target-layer-ids $AUX \
    --epochs 2 --early-stop-patience 1 --total-seq-len 4096 \
    --max-anchors 512 \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' --confidence-head-alpha 1.0 \
    --checkpoint-freq 0.25 \
    --on-missing generate --on-generate delete \
    >> "$OUT/logs/dspark.log" 2>&1 < /dev/null &
echo $! > "$OUT/pids/trainer.pid"; echo "trainer pid=$(cat "$OUT/pids/trainer.pid") DSPARK-FROM-DFLASH"
