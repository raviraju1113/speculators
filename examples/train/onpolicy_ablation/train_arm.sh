#!/usr/bin/env bash
# Phase 1b: train one arm of the on- vs off-policy DSpark ablation (Llama-3.1-8B).
#
#   ARM=off VLLM_GPU=0 TRAIN_GPUS=1,2,3 bash examples/train/onpolicy_ablation/train_arm.sh
#   ARM=on  VLLM_GPU=4 TRAIN_GPUS=5,6,7 PORT=8100 MASTER_PORT=29600 bash ...
#
# The two arms must differ ONLY in DATA_JSONL. Every other knob below is shared;
# override one for both arms or for neither. GPUs are pinned explicitly because on
# IDC concurrent jobs can see (and land on) each other's cards.
#
# Smoke test: SMOKE=1 ARM=off ... (256 samples, 30 steps, no wandb)
set -euo pipefail

ARM="${ARM:?set ARM=off|on}"
ABL="${ABL:-/import/ml-sc-scratch1/ravir/spec_decoding_work/onpolicy_ablation}"
REPO="${REPO:-/import/ml-sc-scratch1/ravir/spec_decoding_work/speculators}"
PY_ENV="${PY_ENV:-/import/ml-sc-scratch1/ravir/miniconda3/envs/dspark_cu12}"
MODEL="${MODEL:-/import/ml-sc-scratch1/mengmengj/models/llama3.1-8b-instruct}"
# prompt-matched files from finalize_arms.py (same prompt ids, same order)
DATA_JSONL="${DATA_JSONL:-$ABL/data/${ARM}_policy.final.jsonl}"
SEED="${SEED:-0}"
TAG="${TAG:-${ARM}_s${SEED}}"

VLLM_GPU="${VLLM_GPU:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-1,2,3}"
PORT="${PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29500}"

# --- shared recipe (identical across arms) ---
SEQ_LEN=8192
TARGET_LAYER_IDS="1 8 15 22 29"   # z-lab LLaMA3.1-8B DFlash; same on vLLM + train side
# mask token = <|reserved_special_token_0|> (as z-lab). Llama has no mask token, and
# the auto-fallback picks pad=128009=<|eot_id|>, a real token in every response.
EPOCHS="${EPOCHS:-3}"
LR="${LR:-3e-4}"
MAX_ANCHORS="${MAX_ANCHORS:-512}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-0.25}"
LOGGER="${LOGGER:-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-dspark-onpolicy-ablation}"

if [ -n "${SMOKE:-}" ]; then
    # tensorboard hparams logging rejects list-valued args (target_layer_ids)
    TAG="smoke_${TAG}"; EPOCHS=1; LOGGER=wandb; export WANDB_MODE=offline
    EXTRA_TRAIN=(--max-steps 30)
    EXTRA_PREP=(--max-samples 256)
else
    EXTRA_TRAIN=(); EXTRA_PREP=()
fi
PREP="$ABL/prepared/$TAG"
SAVE="$ABL/ckpt/$TAG"
LOGS="$ABL/logs/$TAG"
# hidden states are produced and consumed per step (--on-generate delete): keep
# them on node-local scratch, not NFS
HS="${HS:-${SLURM_TMPDIR:-/tmp}/hs_$TAG}"
mkdir -p "$LOGS" "$HS"

source "$(dirname "$PY_ENV")/../etc/profile.d/conda.sh"
conda activate "$PY_ENV"
cd "$REPO"
echo "[$TAG] $(hostname) $(date '+%F %T') git=$(git rev-parse --short HEAD)$(git diff --quiet || echo +dirty)"

# 1. tokenize (both arms through the same prepare_data call)
if [ ! -f "$PREP/token_freq.pt" ]; then
    python scripts/prepare_data.py --model "$MODEL" --data "$DATA_JSONL" \
        --seq-length "$SEQ_LEN" --seed "$SEED" --output "$PREP" "${EXTRA_PREP[@]}" \
        2>&1 | tee "$LOGS/prepare.log"
fi

# 2. verifier hidden-state server. --max-model-len must be STRICTLY > seq len.
CUDA_VISIBLE_DEVICES="$VLLM_GPU" setsid python scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HS" --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$PORT" --max-model-len $((SEQ_LEN + 256)) \
       --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
    > "$LOGS/vllm.log" 2>&1 &
VLLM_PID=$!
cleanup() { kill -- -"$VLLM_PID" 2>/dev/null || true; rm -rf "$HS"; }
trap cleanup EXIT INT TERM
until curl -sf "http://localhost:$PORT/health" >/dev/null; do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM died"; tail -30 "$LOGS/vllm.log"; exit 1; }
    sleep 5
done
echo "[$TAG] vLLM ready on GPU $VLLM_GPU"

# 3. train. torchrun only when >1 rank (it forces NCCL init even for world_size=1).
NPROC=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
if [ "$NPROC" -gt 1 ]; then
    LAUNCH=(torchrun --standalone --master-port "$MASTER_PORT" --nproc_per_node "$NPROC")
else
    LAUNCH=(python)
fi
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "${LAUNCH[@]}" scripts/train.py \
    --verifier-name-or-path "$MODEL" --data-path "$PREP" \
    --vllm-endpoint "http://localhost:$PORT/v1" --hidden-states-path "$HS" \
    --on-missing generate --on-generate delete \
    --speculator-type dspark --block-size 8 --num-layers 5 --no-sample-from-anchor \
    --draft-vocab-size 32000 --target-layer-ids $TARGET_LAYER_IDS \
    --mask-token-id 128002 \
    --markov-rank 256 --markov-head-type vanilla \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' --confidence-head-alpha 1.0 \
    --max-anchors "$MAX_ANCHORS" --total-seq-len "$SEQ_LEN" \
    --epochs "$EPOCHS" --lr "$LR" --seed "$SEED" \
    --checkpoint-freq "$CHECKPOINT_FREQ" \
    --logger "$LOGGER" --log-dir "$LOGS" --run-name "llama31_8b_dspark_$TAG" \
    --save-path "$SAVE" "${EXTRA_TRAIN[@]}" \
    2>&1 | tee "$LOGS/train.log"
echo "[$TAG] done $(date '+%F %T')"
