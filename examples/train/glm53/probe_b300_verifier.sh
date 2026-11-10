#!/usr/bin/env bash
# Probe: GLM-5.3 bf16 verifier on ONE B300 (TP8, no pipeline stage) feeding DFlash2
# hidden states through Mooncake -- PLAN_multidraft.md layout A, phase-1 step 1.
# Same software ravir's 3-node B200 runs use today (frozen snapshot + venv, vLLM 0.26,
# hs_connectors Mooncake connector), minus the second node and the vllm_pp_aux plugin.
#
#   1. mooncake_master on this node, vLLM TP8 with the Mooncake hidden-state producer
#   2. probe_hidden_states.py --verifier: shape / NaN / zero-layer / lm_head top-1 gate
#   3. load test (--concurrency 32 then 64): NaN + agreement under load; stream rate
#      is read from vLLM's "Avg prompt throughput" lines during the load test
#   4. ping a B200 trainer node's brcm0 IP (cross-node Mooncake path for layout A)
#
# Runs as one sngpu job on a B300 (see PROBE_SUBMIT below). Logs under $LOG_DIR/probe_b300_<job>/.
set -uo pipefail

SNAP="${SNAP:-/import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2/code_295f434}"
VENV="${VENV:-/import/ml-sc-scratch1/ravir/envs/glm53_dflash2}"
MODEL="${MODEL:-/import/ml-sc-nlpcheckpoints-scratch3/mengmengj/glm-5.3-bf16}"
DATA_PATH="${DATA_PATH:-/import/ml-sc-scratch1/mengmengj/datasets/glm53_dryrun_20k_s16k}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-6 20 34 48 62 76}"   # incoai taps, vLLM ids (ravir's A1)
LOG_DIR="${LOG_DIR:-/import/ml-sc-scratch1/mengmengj/speculators/data/glm53/logs}"
RUN_DIR="$LOG_DIR/probe_b300_${SLURM_JOB_ID:-manual}"
B200_PEER_IP="${B200_PEER_IP:-10.17.16.80}"   # sc3-cp108 brcm0 (ravir's A0 trainer node)
VLLM_PORT=8000; MC_PORT=50051; SEQ_LEN=16384
NUM_CAPTURED=$(( $(wc -w <<< "$TARGET_LAYER_IDS") + 1 ))

mkdir -p "$RUN_DIR"; exec > >(tee -a "$RUN_DIR/probe.log") 2>&1
echo "=== probe_b300_verifier $(hostname -s) $(date '+%F %T') job ${SLURM_JOB_ID:-} snap $SNAP ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | head -8

source "$VENV/bin/activate"
export PYTHONPATH="$SNAP/src:$SNAP/hs_connectors/src"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
CACHE="/tmp/${USER}_glm53_cache"
export FLASHINFER_WORKSPACE_BASE="$CACHE" VLLM_CACHE_ROOT="$CACHE/vllm" TRITON_CACHE_DIR="$CACHE/triton" \
       TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" XDG_CACHE_HOME="$CACHE/xdg" TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"
python -c "import vllm, hs_connectors, speculators; print('vllm', vllm.__version__, 'code', speculators.__file__)" || exit 2

# B300 nodes have no brcm0; mlnx0 is on the same 10.17.16.0/21 network as the B200s' brcm0.
MY_IP=$(ip -4 -o addr show mlnx0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
[ -z "$MY_IP" ] && MY_IP=127.0.0.1
export MOONCAKE_LOCAL_HOSTNAME="$MY_IP"
echo "mooncake host $MY_IP; ping B200 peer $B200_PEER_IP:"; ping -c 3 -W 2 "$B200_PEER_IP" | tail -2

export NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=NVL NCCL_DEBUG=WARN
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_RPC_TIMEOUT=600000
export HS_MOONCAKE_GLOBAL_SEGMENT_SIZE="${HS_MOONCAKE_GLOBAL_SEGMENT_SIZE:-128G}"
export HS_MOONCAKE_LOCAL_BUFFER_SIZE="${HS_MOONCAKE_LOCAL_BUFFER_SIZE:-16G}"

children=()
cleanup() { echo "cleanup $(date '+%T')"; for p in "${children[@]}"; do kill -TERM -- -"$p" 2>/dev/null; done; sleep 10; for p in "${children[@]}"; do kill -KILL -- -"$p" 2>/dev/null; done; }
trap cleanup EXIT; trap 'exit 143' TERM INT

setsid mooncake_master --rpc_port "$MC_PORT" > "$RUN_DIR/mooncake_master.log" 2>&1 &
children+=($!)

cd "$SNAP"
ARGS=(--tensor-parallel-size 8 --host 0.0.0.0 --port "$VLLM_PORT"
      --max-model-len $((SEQ_LEN + 1024)) --max-num-batched-tokens 32768
      --max-num-seqs 256 --gpu-memory-utilization 0.90 --block-size 64
      --no-enable-prefix-caching --enforce-eager)
# shellcheck disable=SC2206
ARGS+=(${VLLM_EXTRA_ARGS:-})
setsid python scripts/launch_vllm.py "$MODEL" \
    --hidden-states-backend mooncake --mooncake-master "$MY_IP:$MC_PORT" \
    --mooncake-protocol tcp --target-layer-ids $TARGET_LAYER_IDS \
    -- "${ARGS[@]}" > "$RUN_DIR/vllm.log" 2>&1 &
children+=($!)
echo "vLLM pid ${children[1]} (log $RUN_DIR/vllm.log); waiting for /v1/models"
T0=$(date +%s)
until curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null; do
    kill -0 "${children[1]}" 2>/dev/null || { echo "!! vLLM died"; tail -60 "$RUN_DIR/vllm.log"; exit 1; }
    [ $(( $(date +%s) - T0 )) -gt 3600 ] && { echo "!! vLLM not ready after 60 min"; exit 1; }
    sleep 15
done
echo "vLLM ready after $(( $(date +%s) - T0 )) s"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -8

PROBE=(python examples/train/glm53/probe_hidden_states.py --data-path "$DATA_PATH"
       --vllm-endpoint "http://127.0.0.1:$VLLM_PORT/v1" --mooncake-master "$MY_IP:$MC_PORT"
       --num-layers "$NUM_CAPTURED" --hidden-size 6144 --verifier "$MODEL")
echo "--- gate: rows 0 1 2 160 ---"; "${PROBE[@]}" --rows 0 1 2 160; RC_GATE=$?
for C in 32 64; do
    echo "--- load test concurrency $C, 512 random rows --- $(date '+%T')"
    T1=$(date +%s); "${PROBE[@]}" --concurrency "$C" --random-rows 512; RC_LOAD=$?; T2=$(date +%s)
    echo "load test c=$C rc=$RC_LOAD wall $((T2 - T1)) s"
    grep "Avg prompt throughput" "$RUN_DIR/vllm.log" | tail -n 40 | grep -oP "prompt throughput: \K[0-9.]+" \
        | awk '$1>0{s+=$1;n++; if($1>m)m=$1} END{if(n) printf "  vLLM prompt tok/s during window: mean %.0f max %.0f over %d samples\n", s/n, m, n}'
done
echo "RESULT gate_rc=$RC_GATE load_rc=$RC_LOAD $(date '+%F %T')"
exit $(( RC_GATE != 0 || RC_LOAD != 0 ))
