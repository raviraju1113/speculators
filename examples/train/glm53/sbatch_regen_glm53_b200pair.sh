#!/usr/bin/env bash
# Regenerate a prompt pool with GLM-5.3 on a PAIR of 8x B200 nodes: one vLLM
# server with tensor-parallel 8 inside each node and pipeline-parallel 2 across
# the pair, on a Ray cluster, plus the same resumable client as run_regen_glm53.sh.
#
# Why a pair: GLM-5.3 bf16 is ~1.5 TB of weights and a B200 node has
# 8 x 183 GB = 1.46 TB, so one node cannot hold the model. Two nodes hold it with
# ~1.2 TB left for KV cache (1.6M tokens measured, vs 0.75M on one B300 node),
# and this job is KV-bound, so a pair should beat a single B300 node.
#
# Why two single-node jobs instead of one 2-node job: srun inside a multi-node
# allocation fails here with "fwd_tree_thread: can't find address for host
# sc3-cpNN, check slurm.conf" (the compute nodes' Slurm node table cannot reach
# peers). So head and worker are separate jobs that find each other through a
# rendezvous file on NFS: the head writes its Ray address to $ADDR_FILE, the
# worker polls for it. The head is submitted with --dependency=after:<worker>
# so it starts only once the worker is running; each side waits at most
# JOIN_WAIT_MIN for the other, then gives up and frees its node.
#
# Submit from a login node (every env var of run_regen_glm53.sh is honoured):
#   INPUT_JSONL=... OUTFILE=... bash examples/train/glm53/sbatch_regen_glm53_b200pair.sh
#   NODES=sc3-cp79,sc3-cp84 ...   # pin the two nodes instead of letting Slurm pick
#   EXCLUDE_NODES=sc3-cp80 ...     # never place head or worker on these nodes
#   EVAL_CONFIG=glm53-mtp-eval.yaml JOBNAME=glm53_mtp_eval TIME=1-00:00:00 bash ...   # eval mode (see below)
#   LIMIT=300 TIME=2:00:00 OUTFILE=$PWD/data/glm53/smoke_b200pair.jsonl bash examples/train/glm53/sbatch_regen_glm53_b200pair.sh
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
MODEL="${MODEL:-/import/ml-sc-nlpcheckpoints-scratch3/mengmengj/glm-5.3-bf16}"
INPUT_JSONL="${INPUT_JSONL:-$REPO/data/glm53/prompts_nemo782k_shardB.jsonl}"
OUTFILE="${OUTFILE:-$REPO/data/glm53/nemo782k_regen_glm53_shardB.jsonl}"
REASONING_EFFORT="${REASONING_EFFORT:-max}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
CONCURRENCY="${CONCURRENCY:-256}"
TP="${TP:-8}"
PP="${PP:-2}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
EXPERT_PARALLEL="${EXPERT_PARALLEL:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"      # what each node registers with ray (TP*PP may span nodes, e.g. TP=16 PP=1)
PORT="${PORT:-8000}"
RAY_PORT="${RAY_PORT:-6379}"
LOG_DIR="${LOG_DIR:-$(dirname "$OUTFILE")/logs}"
CONDA_ENV="${CONDA_ENV:-dspark}"
CONDA_BASE="${CONDA_BASE:-/import/snvm-sc-scratch1/mengmengj/miniconda3}"
CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-/import/ml-sc-scratch1/mengmengj/condaenvs}"
CACHE_BASE="${CACHE_BASE:-/import/ml-sc-scratch1/mengmengj/cache}"
JOIN_WAIT_MIN="${JOIN_WAIT_MIN:-60}"      # how long head/worker wait for each other
export REPO MODEL INPUT_JSONL OUTFILE REASONING_EFFORT TEMPERATURE TOP_P MAX_TOKENS MAX_MODEL_LEN \
       CONCURRENCY TP PP KV_CACHE_DTYPE GPU_MEM_UTIL EXPERT_PARALLEL BLOCK_SIZE PORT RAY_PORT LOG_DIR \
       CONDA_ENV CONDA_BASE CONDA_ENVS_DIR CACHE_BASE JOIN_WAIT_MIN GPUS_PER_NODE
[ -n "${LIMIT:-}" ] && export LIMIT
# EVAL_CONFIG=<yaml[,yaml]> (relative to scripts/evaluate/experiments): instead of the
# regen client, run run_experiments.py on the pair (it launches vllm serve itself; the
# yaml's extra_args must carry the PP2/ray/block-size flags). ONLY= narrows experiments.
[ -n "${EVAL_CONFIG:-}" ] && export EVAL_CONFIG
[ -n "${ONLY:-}" ] && export ONLY

case "$REASONING_EFFORT" in low|high|max) ;; *) echo "REASONING_EFFORT must be low|high|max" >&2; exit 2;; esac
SELF="$REPO/examples/train/glm53/sbatch_regen_glm53_b200pair.sh"

# ---------------------------------------------------------------- submit ----
if [ -z "${ROLE:-}" ]; then
    mkdir -p "$LOG_DIR"
    JOBNAME="${JOBNAME:-glm53_regen_pair}"
    TIME="${TIME:-5-00:00:00}"
    PAIR_TAG="${JOBNAME}_$(date +%m%d_%H%M%S)"
    ADDR_FILE="$LOG_DIR/ray_head_${PAIR_TAG}.addr"
    COMMON=(--partition="${PARTITION:-gpuonly}" --reservation="${RESERVATION:-sn50-hdi-0}" --nodes=1 --ntasks=1
            --gres="gpu:${GPUTYPE:-b200m180}:8" --cpus-per-task="${CPU:-64}" --mem="${MEM:-500G}" --exclusive --time="$TIME")
    # EXCLUDE_NODES=sc3-cp80,... keeps known-bad nodes (e.g. NVLink faults) out of both jobs
    [ -n "${EXCLUDE_NODES:-}" ] && COMMON+=(--exclude="$EXCLUDE_NODES")
    NODE_ARR=()
    [ -n "${NODES:-}" ] && IFS=, read -r -a NODE_ARR <<< "$NODES"
    [ "${#NODE_ARR[@]}" -eq 0 ] || [ "${#NODE_ARR[@]}" -eq "$PP" ] || { echo "NODES must list $PP nodes" >&2; exit 1; }
    echo "[submit] pair $PAIR_TAG: $PP nodes${NODES:+ pinned to $NODES}, rendezvous $ADDR_FILE"
    [ -n "${DRY_RUN:-}" ] && { printf '[dry-run] sbatch '; printf '%q ' "${COMMON[@]}"; echo "... $SELF"; exit 0; }
    WORKER_JOBIDS=""
    for i in $(seq 1 $((PP - 1))); do
        PIN=(); [ "${#NODE_ARR[@]}" -gt 0 ] && PIN=(--nodelist="${NODE_ARR[$i]}")
        jid=$(ROLE=worker PAIR_TAG="$PAIR_TAG" ADDR_FILE="$ADDR_FILE" sbatch --parsable --export=ALL "${COMMON[@]}" "${PIN[@]}" \
              --job-name="${JOBNAME}_w" --output="$LOG_DIR/${JOBNAME}_worker_%j.out" "$SELF")
        echo "[submit] worker -> job $jid"
        WORKER_JOBIDS="${WORKER_JOBIDS:+$WORKER_JOBIDS,}$jid"
    done
    PIN=(); [ "${#NODE_ARR[@]}" -gt 0 ] && PIN=(--nodelist="${NODE_ARR[0]}")
    jid=$(ROLE=head PAIR_TAG="$PAIR_TAG" ADDR_FILE="$ADDR_FILE" WORKER_JOBIDS="$WORKER_JOBIDS" sbatch --parsable --export=ALL "${COMMON[@]}" "${PIN[@]}" \
          --dependency="after:${WORKER_JOBIDS//,/:}" --job-name="$JOBNAME" --output="$LOG_DIR/${JOBNAME}_head_%j.out" "$SELF")
    echo "[submit] head -> job $jid (starts after worker(s) $WORKER_JOBIDS start; cancels them on exit)"
    exit 0
fi

# --------------------------------------------------------------- in-job -----
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENVS_DIR/$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 RAY_DISABLE_USAGE_STATS=1
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
# JIT/compile caches live on NODE-LOCAL disk, warmed from the shared NFS cache.
# Sharing $CACHE_BASE/triton between 16 workers on two nodes (plus the B300
# server) gave "OSError: [Errno 116] Stale file handle" on .cubin files during
# warm-up and hung the engine; NFS is now read-only for these.
LOCAL_CACHE="${SLURM_TMPDIR:-/tmp}/cache_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_CACHE" "$LOG_DIR"
for d in triton inductor vllm .cache/flashinfer; do
    [ -d "$CACHE_BASE/$d" ] || continue
    mkdir -p "$LOCAL_CACHE/$d"
    rsync -a --ignore-errors "$CACHE_BASE/$d/" "$LOCAL_CACHE/$d/" 2>/dev/null || true
done
export FLASHINFER_WORKSPACE_BASE="$LOCAL_CACHE"
export VLLM_CACHE_ROOT="$LOCAL_CACHE/vllm"
export TRITON_CACHE_DIR="$LOCAL_CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE/inductor"
export XDG_CACHE_HOME="$LOCAL_CACHE/xdg"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"
# This node's hostname IP and the interface carrying it: Ray, Gloo and NCCL's
# bootstrap socket bind there; NCCL payload goes over the mlx5 HCAs.
NODE_IP=$(getent hosts "$(hostname -s)" | awk '{print $1}' | head -1)
[ -n "$NODE_IP" ] || NODE_IP=$(hostname -I | awk '{print $1}')
NODE_IF=$(ip -o -4 addr show | awk -v ip="$NODE_IP/" 'index($4, ip)==1 {print $2; exit}')
export NODE_IP NODE_IF VLLM_HOST_IP="$NODE_IP" NCCL_SOCKET_IFNAME="$NODE_IF" GLOO_SOCKET_IFNAME="$NODE_IF"
RAY_TMP="/tmp/ray_${SLURM_JOB_ID}"   # short: Ray AF_UNIX socket paths are capped at 107 bytes
mkdir -p "$RAY_TMP"
echo "[$ROLE $(hostname -s)] job $SLURM_JOB_ID pair=$PAIR_TAG ip=$NODE_IP if=$NODE_IF tp=$TP pp=$PP ep=$EXPERT_PARALLEL caches=$LOCAL_CACHE ($(du -sh "$LOCAL_CACHE" | cut -f1))"

if [ "$ROLE" = "worker" ]; then
    # Wait for the head's address, join its Ray cluster, stay alive while it runs.
    for _ in $(seq 1 $((JOIN_WAIT_MIN * 6))); do
        [ -s "$ADDR_FILE" ] && break
        sleep 10
    done
    [ -s "$ADDR_FILE" ] || { echo "[worker] no head address in $ADDR_FILE after $JOIN_WAIT_MIN min, giving up"; exit 1; }
    RAY_ADDR=$(cat "$ADDR_FILE")
    echo "[worker] head at $RAY_ADDR"
    for _ in $(seq 1 30); do
        if ray start --address="$RAY_ADDR" --num-gpus="$GPUS_PER_NODE" --num-cpus="${SLURM_CPUS_PER_TASK:-32}" \
               --node-ip-address="$NODE_IP" --temp-dir="$RAY_TMP" > "$LOG_DIR/ray_worker_${SLURM_JOB_ID}.log" 2>&1; then
            echo "[worker] joined $RAY_ADDR"
            trap 'ray stop --force >/dev/null 2>&1 || true' EXIT INT TERM
            while ray status --address "$RAY_ADDR" > /dev/null 2>&1; do sleep 30; done
            echo "[worker] head gone, exiting"; exit 0
        fi
        sleep 10
    done
    echo "[worker] could not join $RAY_ADDR"; tail -20 "$LOG_DIR/ray_worker_${SLURM_JOB_ID}.log"; exit 1
fi

# ---- head ----
RAY_ADDR="$NODE_IP:$RAY_PORT"
echo "[regen] model=$MODEL effort=$REASONING_EFFORT temp=$TEMPERATURE top_p=$TOP_P max_tokens=$MAX_TOKENS max_model_len=$MAX_MODEL_LEN kv=$KV_CACHE_DTYPE block=$BLOCK_SIZE concurrency=$CONCURRENCY"
echo "[regen] $INPUT_JSONL -> $OUTFILE"
cleanup() {
    echo "[regen] cleanup: stopping server and ray, cancelling workers ${WORKER_JOBIDS:-}"
    [ -n "${SERVER_PID:-}" ] && { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
    ray stop --force > /dev/null 2>&1 || true
    rm -f "$ADDR_FILE"
    # shellcheck disable=SC2086
    [ -n "${WORKER_JOBIDS:-}" ] && scancel ${WORKER_JOBIDS//,/ } 2>/dev/null || true
}
trap cleanup EXIT INT TERM
ray start --head --node-ip-address="$NODE_IP" --port="$RAY_PORT" --num-gpus="$GPUS_PER_NODE" \
    --num-cpus="${SLURM_CPUS_PER_TASK:-32}" --temp-dir="$RAY_TMP" --disable-usage-stats > "$LOG_DIR/ray_head_${SLURM_JOB_ID}.log" 2>&1
echo "$RAY_ADDR" > "$ADDR_FILE"
WANT_GPUS=$((TP * PP))
echo "[regen] ray head up at $RAY_ADDR; waiting up to $JOIN_WAIT_MIN min for $WANT_GPUS GPUs"
HAVE=0
for _ in $(seq 1 $((JOIN_WAIT_MIN * 6))); do
    HAVE=$(ray status --address "$RAY_ADDR" 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ GPU' | awk -F/ '{print int($2)}' || echo 0)
    [ "${HAVE:-0}" -ge "$WANT_GPUS" ] && break
    sleep 10
done
[ "${HAVE:-0}" -ge "$WANT_GPUS" ] || { echo "[regen] ray has only ${HAVE:-0}/$WANT_GPUS GPUs:"; ray status --address "$RAY_ADDR" || true; exit 1; }
echo "[regen] ray up with $HAVE GPUs"
export RAY_ADDRESS="$RAY_ADDR"   # without this vLLM starts its own local Ray (8 GPUs) instead of joining the pair

if [ -n "${EVAL_CONFIG:-}" ]; then
    echo "[eval] run_experiments.py on the pair: configs=$EVAL_CONFIG only=${ONLY:-all} (RAY_ADDRESS=$RAY_ADDRESS)"
    cd "$REPO/scripts/evaluate/experiments"
    ONLY_ARGS=(); [ -n "${ONLY:-}" ] && ONLY_ARGS=(--only "$ONLY")
    RC=0
    for cfg in ${EVAL_CONFIG//,/ }; do
        EVAL_LOG="$LOG_DIR/eval_pair_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}.log"
        echo "[eval] $cfg -> $EVAL_LOG"
        python run_experiments.py --config "$cfg" --port "$PORT" "${ONLY_ARGS[@]+"${ONLY_ARGS[@]}"}" 2>&1 | tee -a "$EVAL_LOG"
        RC=${PIPESTATUS[0]}
    done
    echo "[eval] done rc=$RC"
    exit "$RC"
fi

# --block-size 64: with PP the two stages auto-picked different KV block sizes
# (FLASHINFER_MLA_SPARSE -> 32 on stage 1, DEEPSEEK_V32_INDEXER -> 64 on stage 0)
# and the engine died with "No common block size for 32"; 64 is what the
# single-node B300 server ends up with, so force it.
SERVE=(vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --api-key ""
    --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" --distributed-executor-backend ray
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$CONCURRENCY" --gpu-memory-utilization "$GPU_MEM_UTIL"
    --kv-cache-dtype "$KV_CACHE_DTYPE" --block-size "$BLOCK_SIZE" --served-model-name glm-5.3 --trust-remote-code)
[ "$EXPERT_PARALLEL" = "1" ] && SERVE+=(--enable-expert-parallel)
EXTRA_BODY=$(printf '{"temperature": %s, "top_p": %s, "chat_template_kwargs": {"reasoning_effort": "%s", "clear_thinking": true}}' \
    "$TEMPERATURE" "$TOP_P" "$REASONING_EFFORT")
CLIENT=(python "$REPO/scripts/response_regeneration/script_multiendpoint.py"
    --input-jsonl "$INPUT_JSONL" --outfile "$OUTFILE" --model glm-5.3
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions"
    --concurrency "$CONCURRENCY" --max-tokens "$MAX_TOKENS"
    --extra-body "$EXTRA_BODY" --skip-sources "" --resume)
[ -n "${LIMIT:-}" ] && CLIENT+=(--limit "$LIMIT")

SERVER_LOG="$LOG_DIR/vllm_glm53_pair_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}.log"
echo "[regen] server log: $SERVER_LOG"
"${SERVE[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[regen] waiting for the server (1.5 TB of weights over 2 nodes; expect 10-25 min)"
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "[regen] server DIED:"; tail -60 "$SERVER_LOG"; exit 1; }
    sleep 15
done
echo "[regen] server ready"
"${CLIENT[@]}" 2>&1 | tee -a "$LOG_DIR/client_pair_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}.log"
echo "[regen] done -> $OUTFILE ($(wc -l < "$OUTFILE") rows; errors: $(wc -l < "$OUTFILE.errors.jsonl" 2>/dev/null || echo 0))"
