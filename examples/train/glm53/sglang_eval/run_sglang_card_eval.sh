#!/usr/bin/env bash
# Phase 0 serving check: reproduce the incoai GLM-5.3-DFlash2 card on SGLang, then run our
# draft the same way. One 8xB200 node (one srun task), FP8 GLM-5.3, TP8. Card recipe:
# --speculative-algorithm DFLASH, FA4 draft attention; eval = the card's 5 tasks at
# T 1.0 / top-p 0.95 / reasoning_effort max / 4096 max tokens (50 prompts each, seed 42).
#
#   srun --jobid=<alloc> --overlap -N1 -n1 /usr/bin/bash \
#     examples/train/glm53/sglang_eval/run_sglang_card_eval.sh <out_dir> <name>=<draft_dir> [...]
set -uo pipefail
OUT="${1:?out dir}"; shift
ENV_DIR="${ENV_DIR:-/import/ml-sc-scratch1/ravir/envs/sglang_dflash2}"
MODEL="${MODEL:-/import/ml-sc-nlpcheckpoints-scratch3/mengmengj/glm-5.3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${BENCH:-gsm8k,math500,humaneval,mbpp,mt-bench}"
PORT=30000

export PATH="$ENV_DIR/bin:$PATH" PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
CACHE="${SLURM_TMPDIR:-/tmp}/sglang_cache_${SLURM_JOB_ID:-local}"
mkdir -p "$CACHE" "$OUT"
export TMPDIR="${SLURM_TMPDIR:-/tmp}" XDG_CACHE_HOME="$CACHE/xdg" TRITON_CACHE_DIR="$CACHE/triton" \
       TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" FLASHINFER_WORKSPACE_BASE="$CACHE"

for spec in "$@"; do
  name="${spec%%=*}"; draft="${spec#*=}"
  d="$OUT/$name"; mkdir -p "$d"
  echo "=== $name: draft=$draft $(date '+%F %T')"
  attn=(--speculative-draft-attention-backend fa4)
  for attempt in 1 2; do
    python -m sglang.launch_server --model-path "$MODEL" --tp-size 8 --trust-remote-code \
      --host 127.0.0.1 --port "$PORT" --context-length 32768 --mem-fraction-static 0.85 \
      --enable-metrics --speculative-algorithm DFLASH --speculative-draft-model-path "$draft" \
      "${attn[@]}" > "$d/server_attempt$attempt.log" 2>&1 &
    SPID=$!
    ok=0
    for _ in $(seq 1 240); do
      curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ok=1; break; }
      kill -0 $SPID 2>/dev/null || break
      sleep 10
    done
    [ $ok = 1 ] && break
    echo "!! server failed to start (attempt $attempt, attn=${attn[*]}); tail:"; tail -5 "$d/server_attempt$attempt.log"
    kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
    attn=()                                    # retry with SGLang's default draft attention
  done
  if [ $ok != 1 ]; then echo "!! $name: giving up"; continue; fi
  echo "server up (attn=${attn[*]:-default}) $(date '+%T')"
  python "$HERE/run_sglang_eval_glm.py" --base-url "http://127.0.0.1:$PORT" --benchmarks "$BENCH" \
    --num-samples 50 --max-tokens 4096 --temperature 1.0 --top-p 0.95 \
    --extra-body '{"chat_template_kwargs": {"reasoning_effort": "max"}}' --output-dir "$d" 2>&1 | tee "$d/eval.log"
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
  sleep 20
done
echo "[sglang card eval] done $(date)"
