#!/usr/bin/env bash
# GLM-5.3 bf16 DFlash2 draft, Kimi K3 recipe, 3 x 8 B200 (sc3-cp, 100 GbE brcm0).
# Runs once PER NODE (srun --ntasks-per-node=1); the role comes from SLURM_NODEID:
#
#   node 0  vLLM node-rank 0: API server :8000, PP stage 0 (layers 0-38), TP8
#   node 1  vLLM node-rank 1: headless, PP stage 1 (layers 39-77 + lm_head +
#           extract_hidden_states drafter); its TP rank 0 writes every sample's
#           hidden states into Mooncake
#   node 2  mooncake_master :50051 + 8-rank DDP trainer, pulling hidden states
#           over TCP on brcm0
#
# Why this layout: README_dflash2_kimi_recipe.md ("Verifier topology"). The
# vllm_pp_aux plugin (VLLM_PP_AUX_PATCH=1) carries stage-0 aux layers to stage 1.
#
#   REPO      frozen code snapshot (required; never the live worktree)
#   RUN_DIR   per-run dir for logs / IP handshake (required, fresh per launch)
#   SMOKE=1   probe hidden states, then train MAX_STEPS (default 30) and exit
#   VLLM_EXTRA_ARGS  extra `vllm serve` flags for both verifier nodes
set -uo pipefail

: "${REPO:?set REPO to a frozen code snapshot}"
: "${RUN_DIR:?set RUN_DIR}"
NODE_ID="${SLURM_NODEID:?run under srun}"
VENV="${VENV:-/import/ml-sc-scratch1/ravir/envs/glm53_dflash2}"
MODEL="${MODEL:-/import/ml-sc-nlpcheckpoints-scratch3/mengmengj/glm-5.3-bf16}"
WORK="${WORK:-/import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2}"
DATA_PATH="${DATA_PATH:-$WORK/data/gemma782k_glm53_s16k}"
RUN_NAME="${RUN_NAME:-glm53_dflash2_gemma782k_kimirecipe}"
SAVE_PATH="${SAVE_PATH:-$WORK/output/$RUN_NAME/checkpoints}"
TARGET_LAYER_IDS="7 20 44 57 71"   # Kimi [7,23,51,67,83]/93 rescaled to 78 layers, vLLM ids
VLLM_PORT=8000
MC_PORT=50051
SEQ_LEN=16384

mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/node${NODE_ID}_$(hostname -s).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== node $NODE_ID $(hostname -s) $(date '+%F %T') job ${SLURM_JOB_ID:-} ==="

# ---- env --------------------------------------------------------------------
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$REPO/src:$REPO/hs_connectors/src"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
CACHE="$WORK/cache"
export FLASHINFER_WORKSPACE_BASE="$CACHE" VLLM_CACHE_ROOT="$CACHE/vllm" \
       TRITON_CACHE_DIR="$CACHE/triton" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" \
       XDG_CACHE_HOME="$CACHE/xdg" TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"
python - "$REPO" <<'PY' || exit 2
import os, sys, hs_connectors, speculators
for m in (hs_connectors, speculators):
    assert os.path.realpath(m.__file__).startswith(os.path.realpath(sys.argv[1])), m.__file__
print("code from", sys.argv[1])
PY

MY_IP=$(ip -4 addr show brcm0 | grep -oP 'inet \K[\d.]+')
echo "$MY_IP" > "$RUN_DIR/ip_node${NODE_ID}"
peer_ip() {  # wait for node $1 to publish its brcm0 IP
    for _ in $(seq 1 120); do [ -s "$RUN_DIR/ip_node$1" ] && { cat "$RUN_DIR/ip_node$1"; return; }; sleep 2; done
    echo "!! node $1 never published its IP" >&2; exit 3
}
HEAD_IP=$(peer_ip 0); TRAIN_IP=$(peer_ip 2)
echo "brcm0: me=$MY_IP head=$HEAD_IP train=$TRAIN_IP"
export MOONCAKE_LOCAL_HOSTNAME="$MY_IP"

children=()
cleanup() {
    echo "=== node $NODE_ID cleanup $(date '+%F %T') ==="
    for p in "${children[@]}"; do kill -TERM -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null; done
    sleep 10
    for p in "${children[@]}"; do kill -KILL -- -"$p" 2>/dev/null; done
}
trap cleanup EXIT
trap 'exit 143' TERM INT

# ---- verifier nodes ---------------------------------------------------------
if [ "$NODE_ID" -le 1 ]; then
    # Inter-node traffic is brcm0 sockets only (no IB). NCCL_NET_PLUGIN=none: the
    # cluster's external net plugin segfaults in init. P2P_LEVEL=NVL: the
    # intra-node P2P stall seen on cp81/cp82 in the Kimi K3 launch.
    export NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1 NCCL_NET=Socket \
           NCCL_SOCKET_IFNAME=brcm0 GLOO_SOCKET_IFNAME=brcm0 \
           NCCL_SOCKET_NTHREADS=8 NCCL_NSOCKS_PERTHREAD=4 NCCL_P2P_LEVEL=NVL \
           NCCL_DEBUG="${NCCL_DEBUG:-WARN}" VLLM_HOST_IP="$MY_IP"
    export VLLM_PP_AUX_PATCH=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_RPC_TIMEOUT=600000
    # Only stage-1 TP rank 0 opens a store client; its segment holds the
    # in-flight samples (~74 KB/token, 1.2 GB for a 16k row).
    export HS_MOONCAKE_GLOBAL_SEGMENT_SIZE="${HS_MOONCAKE_GLOBAL_SEGMENT_SIZE:-128G}"
    export HS_MOONCAKE_LOCAL_BUFFER_SIZE="${HS_MOONCAKE_LOCAL_BUFFER_SIZE:-16G}"
    ARGS=(--tensor-parallel-size 8 --pipeline-parallel-size 2
          --nnodes 2 --node-rank "$NODE_ID" --master-addr "$HEAD_IP"
          --host 0.0.0.0 --port "$VLLM_PORT"
          --max-model-len $((SEQ_LEN + 1024)) --max-num-batched-tokens 32768
          --max-num-seqs 256 --gpu-memory-utilization 0.90
          --no-enable-prefix-caching --enforce-eager)
    [ "$NODE_ID" = 1 ] && ARGS+=(--headless)
    # e.g. VLLM_EXTRA_ARGS="--load-format dummy": structural smoke without the 10 min load
    # shellcheck disable=SC2206
    ARGS+=(${VLLM_EXTRA_ARGS:-})
    cd "$REPO"
    setsid python scripts/launch_vllm.py "$MODEL" \
        --hidden-states-backend mooncake --mooncake-master "$TRAIN_IP:$MC_PORT" \
        --mooncake-protocol tcp --target-layer-ids $TARGET_LAYER_IDS \
        -- "${ARGS[@]}" > "$RUN_DIR/vllm_node${NODE_ID}.log" 2>&1 &
    children+=($!)
    echo "vLLM node-rank $NODE_ID pid ${children[0]} (log vllm_node${NODE_ID}.log)"
    # Stay up until the trainer finishes, this vLLM dies, or the other verifier
    # node's vLLM dies (the surviving PP stage would otherwise hang forever).
    while kill -0 "${children[0]}" 2>/dev/null && [ ! -e "$RUN_DIR/TRAIN_DONE" ] \
          && [ ! -e "$RUN_DIR/VLLM_DIED" ]; do sleep 20; done
    [ -e "$RUN_DIR/TRAIN_DONE" ] && { echo "trainer done; stopping vLLM"; exit 0; }
    [ -e "$RUN_DIR/VLLM_DIED" ] && kill -0 "${children[0]}" 2>/dev/null \
        && { echo "!! the other verifier node died; stopping"; exit 1; }
    echo "!! vLLM exited; tail:"; tail -60 "$RUN_DIR/vllm_node${NODE_ID}.log"
    touch "$RUN_DIR/VLLM_DIED"
    exit 1
fi

# ---- trainer node -----------------------------------------------------------
setsid mooncake_master --rpc_port "$MC_PORT" > "$RUN_DIR/mooncake_master.log" 2>&1 &
children+=($!)
echo "mooncake_master pid ${children[0]}"

echo "waiting for the verifier at $HEAD_IP:$VLLM_PORT (1.5 TB load, expect 10-25 min)"
for i in $(seq 1 240); do
    [ -e "$RUN_DIR/VLLM_DIED" ] && { echo "!! a verifier node died"; touch "$RUN_DIR/TRAIN_DONE"; exit 1; }
    if curl -sf "http://$HEAD_IP:$VLLM_PORT/v1/models" >/dev/null 2>&1 &&
       curl -sf "http://$HEAD_IP:$VLLM_PORT/v1/completions" -H 'Content-Type: application/json' \
            -d '{"model":"'"$MODEL"'","prompt":"1+1=","max_tokens":1}' >/dev/null 2>&1; then
        echo "verifier answers a completion after $((i * 15)) s"; break
    fi
    sleep 15
done

export MOONCAKE_LOCAL_HOSTNAME="$MY_IP"
# Trainer-side clients (one per dataloader worker) only read: small segments;
# the local buffer must hold one whole sample.
export HS_MOONCAKE_GLOBAL_SEGMENT_SIZE="${TRAIN_MC_SEGMENT:-1G}" HS_MOONCAKE_LOCAL_BUFFER_SIZE="${TRAIN_MC_BUFFER:-2G}"

python "$REPO/examples/train/glm53/probe_hidden_states.py" --data-path "$DATA_PATH" \
    --vllm-endpoint "http://$HEAD_IP:$VLLM_PORT/v1" --mooncake-master "$TRAIN_IP:$MC_PORT" \
    --num-layers 6 --hidden-size 6144 || { echo "!! hidden-state probe failed"; touch "$RUN_DIR/TRAIN_DONE"; exit 4; }

# All 8 trainer ranks are on this node: NCCL on loopback (see dspark_online_gemma4_31b.sh).
export NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export WANDB_PROJECT="${WANDB_PROJECT:-glm53_dflash2}" WANDB_ENTITY="${WANDB_ENTITY:-ravi-raju0594-personal}"
export WANDB_MODE=online WANDB_RESUME=allow WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_NAME}"
timeout 180 python -c "import wandb; print('wandb OK', wandb.Api(timeout=60).viewer.username)" \
    || { echo "!! wandb unreachable"; touch "$RUN_DIR/TRAIN_DONE"; exit 3; }

EXTRA=()
[ "${SMOKE:-0}" = 1 ] && EXTRA+=(--max-steps "${MAX_STEPS:-30}")
cd "$REPO"
python -m torch.distributed.run --nnodes=1 --nproc_per_node 8 \
    --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29500 --local-addr=127.0.0.1 \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_PATH" \
    --vllm-endpoint "http://$HEAD_IP:$VLLM_PORT/v1" \
    --hidden-states-backend mooncake --mooncake-master "$TRAIN_IP:$MC_PORT" --mooncake-protocol tcp \
    --on-missing generate --on-generate delete --request-timeout 600 \
    --num-workers "${NUM_WORKERS:-16}" --prefetch-factor 4 \
    --save-path "$SAVE_PATH" --log-dir "$RUN_DIR/train_logs" \
    --run-name "$RUN_NAME" --logger wandb \
    --speculator-type dflash2 \
    --draft-config "$REPO/examples/train/glm53/glm53_dflash2_draft_config.json" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --block-size 7 --max-anchors 512 --total-seq-len "$SEQ_LEN" \
    --optimizer adamw --lr 5e-5 --weight-decay 0.0 \
    --scheduler-type cosine --scheduler-warmup-ratio 0.04 \
    --loss-fn kl_div --per-position-loss-weight fixed-exp-decay --dflash-decay-gamma 4 \
    --epochs "${EPOCHS:-10}" --checkpoint-freq "${CHECKPOINT_FREQ:-0.5}" \
    --train-data-ratio 0.998 \
    "${EXTRA[@]+"${EXTRA[@]}"}"
RC=$?
echo "=== trainer exited rc=$RC $(date '+%F %T') ==="
touch "$RUN_DIR/TRAIN_DONE"
exit "$RC"
