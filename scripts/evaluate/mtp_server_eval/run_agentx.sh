#!/usr/bin/env bash
# AgentX (SemiAnalysis InferenceX) agentic trace-replay eval.
#
# Replays real Claude-Code agentic traces against a running OpenAI-compatible
# server at fixed concurrency, measuring speculative-decoding value under
# realistic long-context, multi-user load -- the fourth benchmark alongside the
# static prompt sets (aime, gpqa, livecodebench) handled by run_eval.sh.
#
# The replay client is aiperf's `--scenario inferencex-agentx-mvp`, which is the
# current upstream AgentX implementation: it bundles the scenario's locked replay
# rules (preserve trace timing, no early stop, cache-bust the first-turn prefix,
# >=900s duration) and stamps `submission_valid` onto its output.
#
# Like run_eval.sh, it does NOT launch or manage the server: point it at a server
# you started yourself (spec on OR off). To compare, run it twice (baseline
# server vs spec server) and diff the matrices -- or let the YAML runner do it,
# see `eval.mode: agentx` in ../experiments/.
#
# It sweeps one axis: concurrency (USERS_LIST). Per level it runs a replay cell,
# reads acceptance off /metrics (SGLang windowed gauges or vLLM cumulative
# counters, per BACKEND), and writes a row to matrix.tsv.
#
#   BACKEND=vllm  BASE_URL=http://127.0.0.1:8000 ./run_agentx.sh
#   USERS_LIST="1 8 16" DURATION=1800 ./run_agentx.sh
#
# NOTE ON WHAT IS MEASURED: the trace corpus carries no prompt *text* -- only
# per-request token counts and KV block hashes. aiperf synthesizes prompts that
# reproduce each trace's length and prefix-sharing structure. So AgentX measures
# the serving regime of agentic load (long context, high prefix reuse, short
# outputs, concurrency), not draft quality on real text. Read decode_tok_s and
# its scaling with concurrency as the headline; cross-check absolute acceptance
# against the real-text benchmarks in run_eval.sh.
#
# Requires network access on first run: aiperf downloads the traces dataset from
# HuggingFace (public, no auth) and caches it.
set -euo pipefail

# Resolve output/input paths against the caller's cwd *before* cd'ing to the
# script dir (which we do so the helper scripts next to us are on hand). Without
# this a relative RESULT_DIR would silently land under mtp_server_eval/.
INVOCATION_DIR="$PWD"
cd "$(dirname "$0")"

abspath() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *)  printf '%s\n' "$INVOCATION_DIR/$1" ;;
    esac
}

# --- settings (override via env) -------------------------------------------
BACKEND="${BACKEND:-vllm}"                  # sglang | vllm (acceptance reader)
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
USERS_LIST="${USERS_LIST:-1 8 16}"          # concurrency levels to sweep
DURATION="${DURATION:-1800}"                # replay seconds per cell (scenario minimum 900)
TEMPERATURE="${TEMPERATURE:-0}"             # 0 = greedy/argmax acceptance
MAX_CONTEXT="${MAX_CONTEXT:-128000}"        # traces longer than this are filtered out
# Date-pinned corpus alias; the rolling `..._with_subagents` alias advances when a
# new drop lands, and two runs on different drops are not comparable.
PUBLIC_DATASET="${PUBLIC_DATASET:-semianalysis_cc_traces_weka_062126}"
RESULT_DIR="${RESULT_DIR:-./results/agentx}"
POLL_INTERVAL="${POLL_INTERVAL:-0.25}"
# Model name the server advertises, and the tokenizer aiperf rebuilds prompts
# with. MODEL is usually the served path/alias; TOKENIZER defaults to it.
MODEL="${MODEL:-}"
TOKENIZER="${TOKENIZER:-${MODEL}}"
# aiperf lives in its own venv so it cannot disturb the serving env's vllm/torch:
#   python3.11 -m venv /sms-scratch/ravira/.venv-aiperf
#   /sms-scratch/ravira/.venv-aiperf/bin/pip install aiperf
AIPERF_BIN="${AIPERF_BIN:-/sms-scratch/ravira/.venv-aiperf/bin/aiperf}"

# GPU telemetry is off by default: aiperf polls a DCGM exporter on :9400, and a
# host whose exporter omits fields aiperf requires (e.g. `hostname`) floods the
# log with pydantic validation errors. Nothing here uses the telemetry, and this
# eval's numbers come from the server's own /metrics. Set GPU_TELEMETRY=1 to
# re-enable, or to a value aiperf accepts (`pynvml`, `amdsmi`, a DCGM URL).
GPU_TELEMETRY="${GPU_TELEMETRY:-}"
case "$GPU_TELEMETRY" in
    ""|0|no|false) GPU_TELEMETRY_ARGS=(--no-gpu-telemetry) ;;
    1|yes|true)    GPU_TELEMETRY_ARGS=(--gpu-telemetry) ;;
    *)             GPU_TELEMETRY_ARGS=(--gpu-telemetry "$GPU_TELEMETRY") ;;
esac

# The scenario rejects durations under 900s. Allow a short smoke run, but only
# via --unsafe-override, which stamps submission_valid=false so nobody mistakes
# a plumbing check for a comparable result.
MIN_VALID_DURATION=900
UNSAFE_ARGS=()
if (( DURATION < MIN_VALID_DURATION )); then
    UNSAFE_ARGS=(--unsafe-override)
    echo "!! DURATION=${DURATION}s is below the AgentX minimum of ${MIN_VALID_DURATION}s."
    echo "!! Passing --unsafe-override: results will be stamped submission_valid=false."
    echo "!! Use this for plumbing validation only, never for reported numbers."
fi

case "$BACKEND" in sglang|vllm) ;; *) echo "BACKEND must be sglang|vllm" >&2; exit 1;; esac
[[ -n "$MODEL" ]] || { echo "!! set MODEL to the model name/path the server serves" >&2; exit 1; }
RESULT_DIR="$(abspath "$RESULT_DIR")"
mkdir -p "$RESULT_DIR"
MATRIX_HEADER="users\tdecode_tok_s\taccept_len\taccept_rate\tout_tok_s\tvalid"

setup_agentx() {
    if [[ ! -x "$AIPERF_BIN" ]]; then
        echo "!! aiperf not found at $AIPERF_BIN" >&2
        echo "   create it with:" >&2
        echo "     python3.11 -m venv $(dirname "$(dirname "$AIPERF_BIN")")" >&2
        echo "     $(dirname "$AIPERF_BIN")/pip install aiperf" >&2
        echo "   or point AIPERF_BIN at an existing install." >&2
        exit 1
    fi
    echo "==> aiperf $("$AIPERF_BIN" --version 2>/dev/null || echo '?') at $AIPERF_BIN"
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

    # aiperf's own --server-metrics scrape is on by default and would also hit
    # $BASE_URL/metrics; harmless alongside the reads above (both are GETs).
    "$AIPERF_BIN" profile \
        --scenario inferencex-agentx-mvp \
        --url "$BASE_URL" \
        --endpoint-type chat \
        --model "$MODEL" \
        --tokenizer "$TOKENIZER" \
        --public-dataset "$PUBLIC_DATASET" \
        --max-context-length "$MAX_CONTEXT" \
        --concurrency "$users" \
        --benchmark-duration "$DURATION" \
        --streaming \
        --use-server-token-count \
        --extra-inputs ignore_eos:true \
        --extra-inputs "temperature:$TEMPERATURE" \
        --cache-bust first_turn_prefix \
        --artifact-dir "$out_dir/aiperf" \
        --random-seed 42 \
        --ui simple \
        "${GPU_TELEMETRY_ARGS[@]}" \
        "${UNSAFE_ARGS[@]}" 2>&1 | tee "$log" || echo "!! [users=$users] replay errored (see $log)"

    # Pooled DECODE tok/s (excludes TTFT) + wall-clock out tok/s + validity stamp.
    local cell decode_tps out_tps valid
    cell=$(python3 agentx_metrics.py "$out_dir/aiperf") || cell=""
    decode_tps=$(cut -f1 <<<"$cell"); out_tps=$(cut -f2 <<<"$cell"); valid=$(cut -f3 <<<"$cell")

    # Acceptance (NA for baseline / no spec).
    local acc_len="NA" acc_rate="NA" acc
    if [[ "$BACKEND" == "sglang" ]]; then
        kill "$accpid" 2>/dev/null || true
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
    : "${acc_len:=NA}" "${acc_rate:=NA}" "${decode_tps:=NA}" "${out_tps:=NA}" "${valid:=NA}"

    echo "==> [users=$users] decode_tok_s=${decode_tps}  accept_len=${acc_len}  accept_rate=${acc_rate}  (wall-clock out_tok_s=${out_tps}, submission_valid=${valid})"
    echo -e "${users}\t${decode_tps}\t${acc_len}\t${acc_rate}\t${out_tps}\t${valid}" > "$out_dir/result.row"

    # `if` (not `[[ ... ]] && cat`): on every pass but the last, the not-yet-run
    # levels make the final test false, which under `set -e` would abort the whole
    # sweep after the first cell.
    { echo -e "$MATRIX_HEADER"
      for u in $USERS_LIST; do
        if [[ -f "$RESULT_DIR/users${u}/result.row" ]]; then
            cat "$RESULT_DIR/users${u}/result.row"
        fi
      done
    } > "$RESULT_DIR/matrix.tsv"
}

echo "==> AgentX trace-replay against $BASE_URL (backend=$BACKEND, users: $USERS_LIST, ${DURATION}s/cell)"
echo "==> model=$MODEL corpus=$PUBLIC_DATASET max_context=$MAX_CONTEXT"
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
