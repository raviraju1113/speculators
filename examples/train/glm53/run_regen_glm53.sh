#!/usr/bin/env bash
# Regenerate a prompt pool with GLM-5.3 on ONE node: a single tensor-parallel
# vLLM server over all GPUs + the resumable multi-endpoint client.
#
# Runs on the allocated node (submit with sngpu_regen_glm53_sc3-c130.sh, or
# inside an interactive allocation). Everything is an env var:
#
#   MODEL          target checkpoint (default: GLM-5.3 bf16 on nlpcheckpoints-scratch3)
#   INPUT_JSONL    prompt pool from extract_prompts.py
#   OUTFILE        regenerated conversations JSONL (resumable; errors -> $OUTFILE.errors.jsonl)
#   REASONING_EFFORT  low | high | max (GLM-5.3 cannot switch thinking off; default low)
#   TEMPERATURE / TOP_P  sampling (default: the model's generation_config, 1.0 / 0.95)
#   MAX_TOKENS     per-reply cap (default 12288: thinking + answer)
#   MAX_MODEL_LEN  server context (default 24576; must exceed longest prompt + MAX_TOKENS)
#   CONCURRENCY    in-flight requests (default 256)
#   TP             tensor-parallel size (default: all visible GPUs)
#   KV_CACHE_DTYPE auto (bf16) by default; fp8_e4m3 trades fidelity for cache
#   SPEC_MTP       1 -> serve with GLM's own MTP head as speculative decoder
#                  (lossless; faster at low concurrency, little gain at high)
#   LIMIT          cap rows (throughput probe / smoke test)
#   CONDA_ENV      conda env with vLLM >= 0.26 (default: dspark)
#   DRY_RUN=1      print the commands and exit
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
MODEL="${MODEL:-/import/ml-sc-nlpcheckpoints-scratch3/mengmengj/glm-5.3-bf16}"
INPUT_JSONL="${INPUT_JSONL:-$REPO/data/glm53/prompts_nemo782k.jsonl}"
OUTFILE="${OUTFILE:-$REPO/data/glm53/nemo782k_regen_glm53.jsonl}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-12288}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
CONCURRENCY="${CONCURRENCY:-256}"
TP="${TP:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-$(dirname "$OUTFILE")/logs}"
CONDA_ENV="${CONDA_ENV:-dspark}"
CONDA_BASE="${CONDA_BASE:-/import/snvm-sc-scratch1/mengmengj/miniconda3}"
CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-/import/ml-sc-scratch1/mengmengj/condaenvs}"

case "$REASONING_EFFORT" in low|high|max) ;; *) echo "REASONING_EFFORT must be low|high|max" >&2; exit 2;; esac
[ "${TP:-0}" -ge 1 ] || { echo "no GPUs visible (TP=$TP)" >&2; exit 1; }

# The template adds "<|system|>Reasoning Effort: <level>" and always opens the
# reply with <think>; clear_thinking=true drops earlier turns' reasoning from
# the context, which is how GLM is served in chat (README of the checkpoint).
# No --reasoning-parser on the server: the raw reply "reasoning</think>answer"
# is what the draft must learn to speculate, so it is stored whole.
EXTRA_BODY=$(printf '{"temperature": %s, "top_p": %s, "chat_template_kwargs": {"reasoning_effort": "%s", "clear_thinking": true}}' \
    "$TEMPERATURE" "$TOP_P" "$REASONING_EFFORT")

SERVE=(vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --api-key ""
    --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$CONCURRENCY" --gpu-memory-utilization "$GPU_MEM_UTIL"
    --kv-cache-dtype "$KV_CACHE_DTYPE" --served-model-name glm-5.3
    --trust-remote-code)
if [ "${SPEC_MTP:-0}" = "1" ]; then
    SERVE+=(--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}')
fi

# --model is the name the server exposes (--served-model-name), not the path:
# job 13948923 got 2000x HTTP 404 "model does not exist" with the path.
CLIENT=(python "$REPO/scripts/response_regeneration/script_multiendpoint.py"
    --input-jsonl "$INPUT_JSONL" --outfile "$OUTFILE" --model glm-5.3
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions"
    --concurrency "$CONCURRENCY" --max-tokens "$MAX_TOKENS"
    --extra-body "$EXTRA_BODY" --skip-sources "" --resume)
[ -n "${LIMIT:-}" ] && CLIENT+=(--limit "$LIMIT")

echo "[regen] model=$MODEL tp=$TP effort=$REASONING_EFFORT temp=$TEMPERATURE top_p=$TOP_P max_tokens=$MAX_TOKENS"
echo "[regen] $INPUT_JSONL -> $OUTFILE"
if [ -n "${DRY_RUN:-}" ]; then
    printf '[dry-run] server: '; printf '%q ' "${SERVE[@]}"; echo
    printf '[dry-run] client: '; printf '%q ' "${CLIENT[@]}"; echo
    exit 0
fi

# --- env -------------------------------------------------------------------
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENVS_DIR/$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
# $HOME has a 10 GB quota and is full: every JIT/compile cache goes to scratch
# (persistent, so FlashInfer's B300 kernels are built once, not per job).
# FlashInfer crashed the first probe with "Disk quota exceeded" in
# ~/.cache/flashinfer during warm-up.
CACHE_BASE="${CACHE_BASE:-/import/ml-sc-scratch1/mengmengj/cache}"
export FLASHINFER_WORKSPACE_BASE="$CACHE_BASE"            # -> $CACHE_BASE/.cache/flashinfer
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_BASE/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_BASE/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_BASE/inductor}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_BASE/xdg}"
export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
mkdir -p "$LOG_DIR" "$(dirname "$OUTFILE")" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$CACHE_BASE/.cache"

# --- server ----------------------------------------------------------------
SERVER_LOG="$LOG_DIR/vllm_glm53_$(date +%Y%m%d_%H%M%S).log"
echo "[regen] server log: $SERVER_LOG"
"${SERVE[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() { echo "[regen] stopping server ($SERVER_PID)"; kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[regen] waiting for the server (1.5 TB of weights; expect 10-20 min)"
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "[regen] server DIED:"; tail -40 "$SERVER_LOG"; exit 1; }
    sleep 15
done
echo "[regen] server ready"

# --- client ----------------------------------------------------------------
"${CLIENT[@]}" 2>&1 | tee -a "$LOG_DIR/client_$(date +%Y%m%d_%H%M%S).log"
echo "[regen] done -> $OUTFILE ($(wc -l < "$OUTFILE") rows; errors: $(wc -l < "$OUTFILE.errors.jsonl" 2>/dev/null || echo 0))"
