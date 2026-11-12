#!/usr/bin/env bash
# GLM-5.3 DFlash2 draft training, PLAN_multidraft.md layout A:
#   ROLE=verifier  one B300 (8 x 288 GB): mooncake_master :50051 + vLLM 0.26 TP8 (no pipeline
#                  stage, no vllm_pp_aux plugin) with the Mooncake hidden-state producer.
#                  Validated by probe_b300_verifier.sh (jobs 16963612 / 17136435).
#   ROLE=trainer   one B200 (8 x 180 GB): 8-rank DDP trainer pulling hidden states over TCP.
# The two roles are separate Slurm jobs (different GPU types) that hand-shake through RUN_DIR
# (ip_verifier, TRAIN_DONE, VLLM_DIED). Launch with sbatch_glm53_dflash2_layoutA.sh.
# Everything else (env, code snapshot, trainer flags) is ravir's run_glm53_dflash2_3node.sh.
#
#   RUN_NAME / RUN_DIR   run id (= wandb run id, resume key) / shared dir
#   REPO                 frozen code snapshot (never the live worktree)
#   DATA_PATH            prepared dataset dir (trainer waits for its token_freq.pt)
#   DRAFT_CONFIG, TARGET_LAYER_IDS (vLLM ids), BLOCK_SIZE, LR, EPOCHS, SCHEDULER (wsd),
#   WSD_DECAY_RATIO, WARMUP_RATIO, CHECKPOINT_FREQ, FROM_PRETRAINED, VLLM_EXTRA_ARGS, SMOKE=1
set -uo pipefail
: "${ROLE:?ROLE=verifier|trainer}"; : "${RUN_NAME:?}"; : "${RUN_DIR:?}"; : "${REPO:?frozen snapshot}"
VENV="${VENV:-/import/ml-sc-scratch1/ravir/envs/glm53_dflash2}"
MODEL="${MODEL:-/import/ml-sc-nlpcheckpoints-scratch3/mengmengj/glm-5.3-bf16}"
DATA_PATH="${DATA_PATH:-/import/ml-sc-scratch1/mengmengj/datasets/glm53_nemotron_782k_mling50k_s16k}"
SAVE_PATH="${SAVE_PATH:-/import/ml-sc-scratch1/mengmengj/speculators/output/glm53/$RUN_NAME/checkpoints}"
DRAFT_CONFIG="${DRAFT_CONFIG:-/import/ml-sc-scratch1/mengmengj/speculators/examples/train/glm53/glm53_dflash2_incoai_shape_L5.json}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-6 20 34 48 62 76}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
NUM_CAPTURED=$(( $(wc -w <<< "$TARGET_LAYER_IDS") + 1 ))
VLLM_PORT=8000; MC_PORT=50051; SEQ_LEN=16384

mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/${ROLE}_$(hostname -s).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $ROLE $(hostname -s) $(date '+%F %T') job ${SLURM_JOB_ID:-} run $RUN_NAME repo $REPO ==="

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$REPO/src:$REPO/hs_connectors/src"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
CACHE="${NODE_CACHE:-/tmp/${USER}_glm53_cache}"
export FLASHINFER_WORKSPACE_BASE="$CACHE" VLLM_CACHE_ROOT="$CACHE/vllm" TRITON_CACHE_DIR="$CACHE/triton" \
       TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" XDG_CACHE_HOME="$CACHE/xdg" TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"
python - "$REPO" <<'PY' || exit 2
import os, sys, hs_connectors, speculators
for m in (hs_connectors, speculators):
    assert os.path.realpath(m.__file__).startswith(os.path.realpath(sys.argv[1])), m.__file__
print("code from", sys.argv[1])
PY

# B300s have no brcm0; their mlnx0 is on the same 10.17.16.0/21 network as the B200s' brcm0.
IFACE=$(ip -4 -o addr show brcm0 >/dev/null 2>&1 && echo brcm0 || echo mlnx0)
MY_IP=$(ip -4 -o addr show "$IFACE" | awk '{print $4}' | cut -d/ -f1)
export MOONCAKE_LOCAL_HOSTNAME="$MY_IP"
echo "$IFACE: me=$MY_IP"

children=()
cleanup() {
    echo "=== $ROLE cleanup $(date '+%F %T') ==="
    [ "$ROLE" = trainer ] && touch "$RUN_DIR/TRAIN_DONE"
    for p in "${children[@]}"; do kill -TERM -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null; done
    sleep 10
    for p in "${children[@]}"; do kill -KILL -- -"$p" 2>/dev/null; done
}
trap cleanup EXIT
trap 'exit 143' TERM INT

# ---- verifier (B300) --------------------------------------------------------
if [ "$ROLE" = verifier ]; then
    export NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=NVL NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_RPC_TIMEOUT=600000
    # The writer's segment holds the in-flight samples (~86 KB/token, 1.4 GB for a 16k row).
    export HS_MOONCAKE_GLOBAL_SEGMENT_SIZE="${HS_MOONCAKE_GLOBAL_SEGMENT_SIZE:-128G}"
    export HS_MOONCAKE_LOCAL_BUFFER_SIZE="${HS_MOONCAKE_LOCAL_BUFFER_SIZE:-16G}"
    setsid mooncake_master --rpc_port "$MC_PORT" > "$RUN_DIR/mooncake_master.log" 2>&1 &
    children+=($!)
    ARGS=(--tensor-parallel-size 8 --host 0.0.0.0 --port "$VLLM_PORT"
          --max-model-len $((SEQ_LEN + 1024)) --max-num-batched-tokens 32768
          --max-num-seqs 256 --gpu-memory-utilization "${GPU_MEM_UTIL:-0.80}" --block-size 64
          --no-enable-prefix-caching --enforce-eager)
    # shellcheck disable=SC2206
    ARGS+=(${VLLM_EXTRA_ARGS:-})
    cd "$REPO"
    setsid python scripts/launch_vllm.py "$MODEL" \
        --hidden-states-backend mooncake --mooncake-master "$MY_IP:$MC_PORT" \
        --mooncake-protocol tcp --target-layer-ids $TARGET_LAYER_IDS \
        -- "${ARGS[@]}" > "$RUN_DIR/vllm.log" 2>&1 &
    children+=($!)
    echo "vLLM pid ${children[1]} (log vllm.log); waiting for /v1/models"
    T0=$(date +%s)
    until curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null; do
        kill -0 "${children[1]}" 2>/dev/null || { echo "!! vLLM died"; tail -60 "$RUN_DIR/vllm.log"; touch "$RUN_DIR/VLLM_DIED"; exit 1; }
        [ $(( $(date +%s) - T0 )) -gt 3600 ] && { echo "!! vLLM not ready after 60 min"; touch "$RUN_DIR/VLLM_DIED"; exit 1; }
        sleep 15
    done
    echo "vLLM ready after $(( $(date +%s) - T0 )) s"
    echo "$MY_IP" > "$RUN_DIR/ip_verifier"      # published only once the server answers
    while kill -0 "${children[1]}" 2>/dev/null && [ ! -e "$RUN_DIR/TRAIN_DONE" ]; do sleep 20; done
    [ -e "$RUN_DIR/TRAIN_DONE" ] && { echo "trainer done; stopping vLLM"; exit 0; }
    echo "!! vLLM exited; tail:"; tail -60 "$RUN_DIR/vllm.log"; touch "$RUN_DIR/VLLM_DIED"; exit 1
fi

# ---- trainer (B200) ---------------------------------------------------------
echo "waiting for the prepared data ($DATA_PATH/token_freq.pt) and the verifier (ip_verifier)"
for _ in $(seq 1 1440); do   # up to 12 h: covers the data prep + the 1.5 TB weight load
    [ -e "$RUN_DIR/VLLM_DIED" ] && { echo "!! verifier died"; exit 1; }
    [ -s "$RUN_DIR/ip_verifier" ] && [ -e "$DATA_PATH/token_freq.pt" ] && break
    sleep 30
done
VIP=$(cat "$RUN_DIR/ip_verifier") || { echo "!! no verifier"; exit 3; }
echo "verifier at $VIP; data ready $(date '+%T')"
until curl -sf "http://$VIP:$VLLM_PORT/v1/completions" -H 'Content-Type: application/json' \
        -d '{"model":"'"$MODEL"'","prompt":"1+1=","max_tokens":1}' >/dev/null 2>&1; do
    [ -e "$RUN_DIR/VLLM_DIED" ] && { echo "!! verifier died"; exit 1; }; sleep 15
done
echo "verifier answers a completion"

# Readers mount no segment (see ravir's launcher: 1 GB reader segments lost ~5% of samples).
export HS_MOONCAKE_GLOBAL_SEGMENT_SIZE="${TRAIN_MC_SEGMENT:-0}" HS_MOONCAKE_LOCAL_BUFFER_SIZE="${TRAIN_MC_BUFFER:-2G}"
PROBE_ARGS=(); [[ "${VLLM_EXTRA_ARGS:-}" == *dummy* ]] || PROBE_ARGS+=(--verifier "$MODEL")
python "$REPO/examples/train/glm53/probe_hidden_states.py" --data-path "$DATA_PATH" \
    --vllm-endpoint "http://$VIP:$VLLM_PORT/v1" --mooncake-master "$VIP:$MC_PORT" \
    --num-layers "$NUM_CAPTURED" --hidden-size 6144 "${PROBE_ARGS[@]+"${PROBE_ARGS[@]}"}" \
    || { echo "!! hidden-state probe failed"; exit 4; }

export NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export WANDB_PROJECT="${WANDB_PROJECT:-glm53_dflash2}" WANDB_ENTITY="${WANDB_ENTITY:-mmj710-stanford-university}"
export WANDB_MODE=online WANDB_RESUME=allow WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_NAME}"
timeout 180 python -c "import wandb; v=wandb.Api(timeout=60).viewer; print('wandb OK', v.username, v.entity)" \
    || { echo "!! wandb unreachable"; exit 3; }

EXTRA=()
[ "${SMOKE:-0}" = 1 ] && EXTRA+=(--max-steps "${MAX_STEPS:-30}")
[ -n "${SCHEDULER_TOTAL_STEPS:-}" ] && EXTRA+=(--scheduler-total-steps "$SCHEDULER_TOTAL_STEPS")
if [ -n "${FROM_PRETRAINED:-}" ]; then DRAFT_INIT=(--from-pretrained "$FROM_PRETRAINED"); else DRAFT_INIT=(--draft-config "$DRAFT_CONFIG"); fi
cd "$REPO"
python -m torch.distributed.run --nnodes=1 --nproc_per_node 8 \
    --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29500 --local-addr=127.0.0.1 \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_PATH" \
    --vllm-endpoint "http://$VIP:$VLLM_PORT/v1" \
    --hidden-states-backend mooncake --mooncake-master "$VIP:$MC_PORT" --mooncake-protocol tcp \
    --on-missing generate --on-generate delete --request-timeout 600 \
    --num-workers "${NUM_WORKERS:-16}" --prefetch-factor 4 \
    --save-path "$SAVE_PATH" --log-dir "$RUN_DIR/train_logs" \
    --run-name "$RUN_NAME" --logger wandb \
    --speculator-type dflash2 \
    "${DRAFT_INIT[@]}" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --block-size "$BLOCK_SIZE" --max-anchors 512 --total-seq-len "$SEQ_LEN" \
    --optimizer adamw --lr "${LR:-2e-4}" --weight-decay 0.0 \
    --scheduler-type "${SCHEDULER:-wsd}" --scheduler-warmup-ratio "${WARMUP_RATIO:-0.04}" \
    --scheduler-wsd-decay-ratio "${WSD_DECAY_RATIO:-0.15}" \
    --loss-fn kl_div --per-position-loss-weight fixed-exp-decay --dflash-decay-gamma 4 \
    --epochs "${EPOCHS:-5}" --checkpoint-freq "${CHECKPOINT_FREQ:-0.5}" \
    --train-data-ratio 0.998 \
    "${EXTRA[@]+"${EXTRA[@]}"}"
RC=$?
echo "=== trainer exited rc=$RC $(date '+%F %T') ==="
exit "$RC"
