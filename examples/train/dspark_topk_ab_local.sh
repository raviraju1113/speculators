#!/usr/bin/env bash
# A/B test of DSA-style top-k context selection on the best 26B DSpark draft.
#
# Both arms continue training output/gemma4_26b_dspark_from_dflash (5 layers,
# layer 4 = full attention, block 16, full vocab) for the same number of steps on
# the same data, same lr, same seed:
#   A (dense): plain continued training.
#   B (topk) : same + --topk-context 512 on layer 4, 1000 dense warm-up steps for
#              the indexer, then sparse. Compare val/accept_len_epoch and watch
#              topk_recall in B.
# Each arm has its own hidden-state server (26B-A4B MoE, 1 GPU) and 1 trainer GPU:
#   A: server GPU0 :8010, trainer GPU1     B: server GPU2 :8011, trainer GPU3
# Waits for a given PID (e.g. a running eval) and for all GPUs to be free first.
set -euo pipefail
cd /nvmedata/chenw/speculators
source /root/miniconda3/etc/profile.d/conda.sh
conda activate speculator
export CUDA_HOME=/usr/local/cuda-12.9 PATH="/usr/local/cuda-12.9/bin:$PATH"
export PYTHONUNBUFFERED=1
# expandable_segments only for the trainers: vLLM's hidden-states KV connector rejects it.
TRAINER_ALLOC=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it
INIT=output/gemma4_26b_dspark_from_dflash/dspark/checkpoints/checkpoint_best
DATA=output/gemma4_26b_dspark_from_dflash/data_prep
AUX="2 7 12 18 23 28"
OUT="${OUT:-output/gemma4_26b_dspark_topk_ab}"
MAX_STEPS="${MAX_STEPS:-8000}"
WAIT_PID="${WAIT_PID:-}"
mkdir -p "$OUT/logs" "$OUT/pids"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/logs/launcher.log"; }

if [[ -n "$WAIT_PID" ]]; then
  log "waiting for pid $WAIT_PID to exit"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
log "waiting for all 4 GPUs to drop below 2 GiB used"
while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2048 {f=1} END {exit !f}'; do sleep 60; done
log "GPUs free; starting"

start_server() {  # gpu port hs_dir
  CUDA_VISIBLE_DEVICES=$1 setsid nohup python scripts/launch_vllm.py "$MODEL" \
      --hidden-states-path "$3" --target-layer-ids $AUX \
      -- --port "$2" --max-model-len 4352 --gpu-memory-utilization 0.85 \
      >> "$OUT/logs/vllm_$2.log" 2>&1 < /dev/null &
  echo $!
}
wait_health() { until curl -sf "http://localhost:$1/health" >/dev/null 2>&1; do sleep 10; done; }

set_common_args() {  # port hs_dir save_dir -> fills COMMON (array: quoting-safe)
  COMMON=( --verifier-name-or-path "$MODEL"
           --from-pretrained "$INIT"
           --data-path "$DATA"
           --vllm-endpoint "http://localhost:$1/v1"
           --hidden-states-path "$2"
           --save-path "$3"
           --speculator-type dspark --lr 1e-4 --scheduler-type constant
           --target-layer-ids $AUX
           --epochs 1 --max-steps "$MAX_STEPS" --train-data-ratio 0.99
           --total-seq-len 4096 --max-anchors 512
           --loss-fn '{"ce": 0.1, "tv": 0.9}' --confidence-head-alpha 1.0
           --checkpoint-freq 0.25
           --on-missing generate --on-generate delete )
}

HS_A=/nvmedata/chenw/hidden_states_topk_ab_A; HS_B=/nvmedata/chenw/hidden_states_topk_ab_B
mkdir -p "$HS_A" "$HS_B"
SA=$(start_server 0 8010 "$HS_A"); echo "$SA" > "$OUT/pids/server_A.pid"
SB=$(start_server 2 8011 "$HS_B"); echo "$SB" > "$OUT/pids/server_B.pid"
log "servers A=$SA (gpu0:8010) B=$SB (gpu2:8011); waiting for /health"
wait_health 8010; wait_health 8011
log "servers ready"
set_common_args 8010 "$HS_A" "$OUT/A_dense/checkpoints"; COMMON_A=("${COMMON[@]}")
set_common_args 8011 "$HS_B" "$OUT/B_topk/checkpoints";  COMMON_B=("${COMMON[@]}")

env $TRAINER_ALLOC CUDA_VISIBLE_DEVICES=1 setsid nohup torchrun --standalone --nproc_per_node 1 --master_port 29510 \
    scripts/train.py "${COMMON_A[@]}" \
    >> "$OUT/logs/A_dense.log" 2>&1 < /dev/null &
TA=$!; echo "$TA" > "$OUT/pids/trainer_A.pid"
env $TRAINER_ALLOC CUDA_VISIBLE_DEVICES=3 setsid nohup torchrun --standalone --nproc_per_node 1 --master_port 29511 \
    scripts/train.py "${COMMON_B[@]}" \
    --topk-context 512 --topk-local-window 128 --topk-warmup-steps 1000 \
    --indexer-loss-weight 1.0 \
    >> "$OUT/logs/B_topk.log" 2>&1 < /dev/null &
TB=$!; echo "$TB" > "$OUT/pids/trainer_B.pid"
log "trainers A=$TA (gpu1) B=$TB (gpu3); max_steps=$MAX_STEPS"

wait "$TA" && log "A finished rc=0" || log "A finished rc=$?"
wait "$TB" && log "B finished rc=0" || log "B finished rc=$?"
kill "$SA" "$SB" 2>/dev/null || true
log "servers stopped"
{
  echo "== val metrics =="
  for arm in A_dense B_topk; do
    echo "-- $arm"; grep -aoE "val/(accept_len|accept_rate|full_acc|loss|indexer_kl|topk_recall|topk_density)_epoch=[0-9.]+" "$OUT/logs/$arm.log" | tail -8
  done
} | tee -a "$OUT/logs/launcher.log"
