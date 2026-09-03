#!/usr/bin/env bash
# AgentX (SemiAnalysis InferenceX) agentic trace-replay eval.
#
# Replays real Claude-Code agentic traces against a running OpenAI-compatible
# server at fixed concurrency, measuring speculative-decoding value under
# realistic long-context, multi-user load -- the fourth benchmark alongside the
# static prompt sets (aime, gpqa, livecodebench) handled by run_eval.sh.
#
# Unlike the internal sweep this was migrated from, it does NOT launch or manage
# the server: point it at a server you started yourself (spec on OR off), exactly
# like run_eval.sh. To compare, run it twice (baseline server vs spec server) and
# diff the matrices -- see the baseline-vs-spec workflow in the README.
#
# It sweeps one axis: concurrency (USERS_LIST). Per level it runs a replay cell,
# reads acceptance off /metrics (SGLang windowed gauges or vLLM cumulative
# counters, per BACKEND), and writes a row to matrix.tsv.
#
#   BACKEND=vllm  BASE_URL=http://127.0.0.1:8000 ./run_agentx.sh
#   USERS_LIST="1 8 16" DURATION=1800 ./run_agentx.sh
#
# Requires network access: clones SemiAnalysis InferenceX (trace-replay client)
# and downloads the traces dataset on first run.
set -euo pipefail

cd "$(dirname "$0")"

# --- settings (override via env) -------------------------------------------
BACKEND="${BACKEND:-vllm}"                  # sglang | vllm (acceptance reader)
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
USERS_LIST="${USERS_LIST:-1 8 16}"          # concurrency levels to sweep
DURATION="${DURATION:-300}"                 # replay seconds per cell (1800 for a real run)
TEMPERATURE="${TEMPERATURE:-0}"             # 0 = greedy/argmax acceptance
MAX_CONTEXT="${MAX_CONTEXT:-128000}"        # traces longer than this are filtered out
HF_DATASET="${HF_DATASET:-semianalysisai/cc-traces-weka-042026}"
RESULT_DIR="${RESULT_DIR:-./results/agentx}"
POLL_INTERVAL="${POLL_INTERVAL:-0.25}"
# InferenceX trace-replay client checkout.
AGENTX_DIR="${AGENTX_DIR:-./.agentx/InferenceX}"
AGENTX_BRANCH="${AGENTX_BRANCH:-chore/agentx-integration}"
AGENTX_REPO="${AGENTX_REPO:-https://github.com/SemiAnalysisAI/InferenceX.git}"
REPLAY="$AGENTX_DIR/utils/trace-replay/trace_replay_tester.py"

case "$BACKEND" in sglang|vllm) ;; *) echo "BACKEND must be sglang|vllm" >&2; exit 1;; esac
mkdir -p "$RESULT_DIR"
MATRIX_HEADER="users\tdecode_tok_s\taccept_len\taccept_rate\tout_tok_s"

setup_agentx() {
    if [[ ! -f "$REPLAY" ]]; then
        echo "==> cloning InferenceX ($AGENTX_BRANCH) into $AGENTX_DIR"
        git clone --recurse-submodules -b "$AGENTX_BRANCH" "$AGENTX_REPO" "$AGENTX_DIR"
    fi
    [[ -f "$REPLAY" ]] || { echo "!! trace_replay_tester.py missing at $REPLAY" >&2; exit 1; }
    python -m pip install -q -r "$AGENTX_DIR/utils/trace-replay/requirements.txt"
    hf download --repo-type dataset "$HF_DATASET" >/dev/null || true
}

# vLLM cumulative counters -> "drafts draft_tokens accepted" (summed over labels).
scrape_vllm() {
    curl -sf "$BASE_URL/metrics" 2>/dev/null | awk '
        /^vllm:spec_decode_num_drafts(_total)?[{ ]/       {d  += $NF}
        /^vllm:spec_decode_num_draft_tokens(_total)?[{ ]/ {dt += $NF}
        /^vllm:spec_decode_num_accepted_tokens(_total)?[{ ]/ {a += $NF}
        END {print (d+0), (dt+0), (a+0)}'
}

# Run one replay cell at concurrency=$1; write result.row + rebuild matrix.
run_cell() {
    local users="$1" out_dir="$RESULT_DIR/users${users}"
    mkdir -p "$out_dir"
    local log="$out_dir/replay.log"
    echo "==> [users=$users] replay dur=${DURATION}s backend=$BACKEND -> $out_dir"

    # Acceptance: the replay client is a black-box OpenAI consumer, so read it off
    # /metrics. SGLang exposes windowed gauges (poll + average); vLLM exposes
    # cumulative counters (before/after delta) -- mirrors run_{sglang,vllm}_eval.py.
    local accpid="" acc_file="$out_dir/accept.samples" vllm_before=""
    if [[ "$BACKEND" == "sglang" ]]; then
        : > "$acc_file"
        ( while :; do
            curl -sf "$BASE_URL/metrics" 2>/dev/null \
              | awk '/^sglang:spec_accept_length[{ ]/{print "L",$NF} /^sglang:spec_accept_rate[{ ]/{print "R",$NF}' \
              >> "$acc_file" 2>/dev/null || true
            sleep "$POLL_INTERVAL"
          done ) &
        accpid=$!
    else
        vllm_before="$(scrape_vllm)"
    fi

    python "$REPLAY" \
        --api-endpoint "$BASE_URL" \
        --hf-dataset "$HF_DATASET" \
        --output-dir "$out_dir/trace_replay" \
        --metrics-output-prefix "$out_dir/metrics" \
        --start-users "$users" --max-users "$users" \
        --test-duration "$DURATION" --recycle --warmup-enabled \
        --max-concurrent-requests 0 \
        --max-context "$MAX_CONTEXT" \
        --advance-min 0.0 --advance-max 0.7 \
        --temperature "$TEMPERATURE" \
        --seed 42 2>&1 | tee "$log" || echo "!! [users=$users] replay errored (see $log)"

    # Wall-clock aggregate output tok/s from the client summary (reference only).
    local out_tps
    out_tps=$({ grep -oE '[0-9,]+ output tok/s' "$log" || true; } | tail -1 | tr -d ', ' | sed 's/outputtok\/s//')

    # Pooled DECODE tok/s -- the spec-decode-relevant number: per successful
    # request decode_time = ttlt - ttft; pool as sum(output)/sum(decode_time).
    local decode_tps
    decode_tps=$(python - "$out_dir/trace_replay/detailed_results.csv" <<'PY'
import csv, sys
num = den = 0.0
try:
    with open(sys.argv[1]) as f:
        for r in csv.DictReader(f):
            if r.get('success') != 'True':
                continue
            try:
                out = float(r['output_tokens_actual']); dt = float(r['ttlt']) - float(r['ttft'])
            except (KeyError, ValueError):
                continue
            if out > 0 and dt > 0:
                num += out; den += dt
    print(f"{num/den:.1f}" if den > 0 else "NA")
except FileNotFoundError:
    print("NA")
PY
) || true

    # Acceptance (NA for baseline / no spec).
    local acc_len="NA" acc_rate="NA"
    if [[ "$BACKEND" == "sglang" ]]; then
        kill "$accpid" 2>/dev/null || true
        local acc
        acc=$(python3 - "$acc_file" <<'PY'
import sys
L=[]; R=[]
try:
    for ln in open(sys.argv[1]):
        p=ln.split()
        if len(p)!=2: continue
        tag,v=p
        try: v=float(v)
        except ValueError: continue
        (L if tag=='L' else R).append(v)
except FileNotFoundError:
    pass
def avg_windows(xs):
    xs=[x for x in xs if x>0]
    if not xs: return None
    w=[xs[0]]
    for x in xs[1:]:
        if x!=w[-1]: w.append(x)
    return sum(w)/len(w)
al=avg_windows(L); ar=avg_windows(R)
print(f"{al:.3f}" if al is not None else "NA", f"{ar:.4f}" if ar is not None else "NA")
PY
) || true
        acc_len=${acc%% *}; acc_rate=${acc##* }
    else
        local vllm_after; vllm_after="$(scrape_vllm)"
        acc=$(python3 - "$vllm_before" "$vllm_after" <<'PY'
import sys
b = [float(x) for x in sys.argv[1].split()]
a = [float(x) for x in sys.argv[2].split()]
d_drafts, d_dtok, d_acc = (a[0]-b[0]), (a[1]-b[1]), (a[2]-b[2])
if d_drafts <= 0:
    print("NA NA")
else:
    al = 1 + d_acc / d_drafts
    ar = d_acc / d_dtok if d_dtok else None
    print(f"{al:.3f}", f"{ar:.4f}" if ar is not None else "NA")
PY
) || true
        acc_len=${acc%% *}; acc_rate=${acc##* }
    fi
    : "${acc_len:=NA}" "${acc_rate:=NA}" "${decode_tps:=NA}" "${out_tps:=NA}"

    echo "==> [users=$users] decode_tok_s=${decode_tps}  accept_len=${acc_len}  accept_rate=${acc_rate}  (wall-clock out_tok_s=${out_tps})"
    echo -e "${users}\t${decode_tps}\t${acc_len}\t${acc_rate}\t${out_tps}" > "$out_dir/result.row"

    {
      echo -e "$MATRIX_HEADER"
      for u in $USERS_LIST; do
        if [[ -f "$RESULT_DIR/users${u}/result.row" ]]; then
          cat "$RESULT_DIR/users${u}/result.row"
        fi
      done
    } > "$RESULT_DIR/matrix.tsv"
}

echo "==> AgentX trace-replay against $BASE_URL (backend=$BACKEND, users: $USERS_LIST, ${DURATION}s/cell)"
setup_agentx
for users in $USERS_LIST; do
    run_cell "$users"
done

echo "================================================================"
echo "==> AgentX results (concurrency x metrics; decode_tok_s = spec-decode signal):"
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$RESULT_DIR/matrix.tsv"
else
    cat "$RESULT_DIR/matrix.tsv"
fi
echo "==> per-cell results under $RESULT_DIR"
