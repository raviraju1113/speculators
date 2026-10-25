#!/usr/bin/env bash
#
# Step 2 — regenerate assistant responses with the TARGET model itself.
#
# This is the only step that makes a corpus "on-policy": the prompts are left
# alone and every assistant turn is rewritten in-context by the target. The
# matched control (step 3) keeps the original other-model responses for the
# same ids, so policy is the single variable between the two arms.
#
# Two endpoints are round-robined. Point them at whichever GPUs are free;
# throughput is ~110-140 rows/min for the pair on A100s.
#
# -------------------------------------------------------------------------
# THREE THINGS THAT COST US HOURS -- do not re-learn them
# -------------------------------------------------------------------------
# 1. NEVER reuse a server that was launched for TRAINING. A hidden-states
#    server runs at --max-model-len $((SEQ+256)) because that is all training
#    needs. Generation needs prompt + max_tokens, so long rows fail with
#    "maximum context length is 4352". Worse, the failures are not random:
#    they are the long multi-turn chat rows (sharegpt, ultrachat), so the
#    corpus silently skews against its own control. This script always starts
#    its own plain servers at CTX.
#
# 2. PRE-FILTER BY LENGTH. ~19% of a typical mix exceeds the 4096-token
#    training window. Those rows error out on the context cap and, when they
#    do fit, their prefill dominates GPU time -- while being dropped by data
#    prep anyway. Filtering took throughput from 12 to 108 rows/min and the
#    error rate from 47% to under 1%.
#
# 3. REGENERATING A MULTI-TURN ROW INFLATES ITS OWN CONTEXT. Each assistant
#    turn is rewritten in context, so the target's output accumulates. A row
#    comfortably inside the cap as written can exceed it once regenerated.
#    The residual ~1% of failures are almost all many-turn sharegpt. Raising
#    CTX does not fix it (prep truncates to SEQ regardless); the real fix
#    would be turn-level truncation before regeneration.
#
# Usage:
#   ./02_regenerate.sh <input.jsonl> <output.jsonl> [gpu_a] [gpu_b]
#
set -euo pipefail

IN=${1:?usage: 02_regenerate.sh <input.jsonl> <output.jsonl> [gpu_a] [gpu_b]}
OUT_JSONL=${2:?usage: 02_regenerate.sh <input.jsonl> <output.jsonl> [gpu_a] [gpu_b]}
GPU_A=${3:-0}
GPU_B=${4:-2}

MODEL=${MODEL:-/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it}
REPO=${REPO:-/nvmedata/chenw/speculators}
RUNDIR=${RUNDIR:-$REPO/output/onpolicy_regen}
SEQ=${SEQ:-4096}
MAX_TOKENS=${MAX_TOKENS:-2048}
CTX=${CTX:-8192}            # >= longest kept prompt + MAX_TOKENS
CONCURRENCY=${CONCURRENCY:-128}
# 15000 chars ~= 6122 tokens at the densest observed ratio (2.45 ch/tok, p10 of
# a 1500-row tokenizer calibration) -- just under CTX-MAX_TOKENS, so context
# errors go to zero; ~4048 at the median 3.705 ch/tok, i.e. the prep cutoff.
MAX_CHARS=${MAX_CHARS:-15000}

cd "$REPO"
mkdir -p "$RUNDIR/logs" "$RUNDIR/pids"

FILTERED="${IN%.jsonl}_lenfiltered.jsonl"
if [ ! -f "$FILTERED" ]; then
  echo "=== length-filtering $IN (<= $MAX_CHARS chars) ==="
  python - "$IN" "$FILTERED" "$MAX_CHARS" <<'PYEOF'
import json, sys, collections
src, dst, thresh = sys.argv[1], sys.argv[2], int(sys.argv[3])
kept = total = 0
by_kept, by_drop = collections.Counter(), collections.Counter()
with open(dst, 'w') as w:
    for line in open(src):
        total += 1
        d = json.loads(line)
        turns = d.get('messages') or d.get('conversations') or []
        n = sum(len(t.get('content') or t.get('value') or '') for t in turns)
        s = (d.get('source') or (d.get('id') or '_').rsplit('_', 1)[0])
        if n <= thresh:
            w.write(line); kept += 1; by_kept[s] += 1
        else:
            by_drop[s] += 1
print(f'kept {kept}/{total} = {kept/total:.1%}')
print('kept by source:   ', dict(by_kept.most_common()))
print('dropped by source:', dict(by_drop.most_common()))
print('NOTE: this filter is source-skewed -- long-form sources (code, math) '
      'lose the most. Both arms inherit the same skew, so the PAIR stays '
      'matched, but report per-domain rather than an aggregate.')
PYEOF
else
  echo "=== reusing existing $FILTERED ==="
fi

start_server() {   # $1=gpu  $2=port
  local gpu=$1 port=$2
  if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
    echo "WARNING: port $port already serving -- NOT reusing it (see note 1); pick another port" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup python -m vllm.entrypoints.cli.main serve "$MODEL" \
      --port "$port" --max-model-len "$CTX" --gpu-memory-utilization 0.85 \
      >> "$RUNDIR/logs/server_gpu$gpu.log" 2>&1 &
  echo $! > "$RUNDIR/pids/server_gpu$gpu.pid"
  echo "  waiting for :$port (GPU $gpu)..."
  for _ in $(seq 1 180); do curl -sf "http://localhost:$port/health" >/dev/null 2>&1 && break; sleep 10; done
  curl -sf "http://localhost:$port/health" >/dev/null 2>&1 || { echo "FATAL: :$port never came up"; exit 1; }
  echo "  :$port ready"
}

echo "=== starting generation servers at ctx $CTX $(date -u) ==="
start_server "$GPU_A" 8000
start_server "$GPU_B" 8002

echo "=== regenerating $(date -u) ==="
cd scripts/response_regeneration
setsid nohup python script.py \
    --endpoint http://127.0.0.1:8000/v1/chat/completions \
               http://127.0.0.1:8002/v1/chat/completions \
    --input-jsonl "$FILTERED" \
    --outfile "$OUT_JSONL" \
    --model "$MODEL" \
    --concurrency "$CONCURRENCY" --max-tokens "$MAX_TOKENS" --resume \
    >> "$RUNDIR/logs/regen.log" 2>&1 &
echo $! > "$RUNDIR/pids/regen.pid"
echo "regen pid=$(cat "$RUNDIR/pids/regen.pid")  -> $OUT_JSONL"
echo
echo "Watch:    tail -f $RUNDIR/logs/regen.log"
echo "Progress: wc -l $OUT_JSONL"
echo "Errors:   wc -l ${OUT_JSONL}.errors.jsonl   (failed rows go here, never into the output)"
echo
echo "--resume is safe across an input change: the skip key is the row's stable"
echo "\`id\`, not its file position, so you may re-filter and restart."
