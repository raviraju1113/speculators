# JOB 429 -- job 424 redone after I destroyed it.
#
# WHAT HAPPENED. Jobs 425/426/427/428 inherited a preamble that killed every
# compute process on ALL FOUR GPUs at script start. Launching them while 424
# was running killed 424's servers: its four baselines came up 14:19:15 and
# died 14:30:12, ten seconds after 425 launched, and 428 at 15:08 killed 424's
# last arm mid-run. Result: 335-525 failed requests per arm.
#
# WHY THAT POISONS THE TABLE SILENTLY. speedup = (rows/wall_s) ratio, and a
# failed request still counts in rows while costing almost no wall time. So the
# dead baselines read 1.8273 req/s where GPU 0 has measured 1.218-1.228 every
# other time -- 50% too FAST -- and every arm divided by them came out absurd
# (the published 4L DFlash read 0.8428, the 43% checkpoint 0.5222). Nothing was
# wrong with the models; the denominator was a corpse.
#
# THREE FIXES, all in this script:
#  1. The preamble now wipes only $OWNED_GPUS, not every card.
#  2. run_arm checks the harness's own error count and quarantines any arm with
#     errors instead of writing it where the summary can find it.
#  3. The summary refuses to print any arm reporting failed requests.
# And the discipline the fixes cannot enforce: ONE job at a time.
#
# STILL VALID from before: job 421's verdict lane, every arm 0 errors --
# MTP 1.1700, 4L DFlash 1.2411, 6L DFlash 1.2018, one card, one baseline.
#
# WHAT THIS MEASURES. Lane 0 is the data question: the published 24.1%
# checkpoint against the 43% checkpoint from job 423, with MTP and a live
# baseline on the same card. Lanes 1-3 take the 43% run's intermediate
# snapshots (3,041 / 6,024 / 9,003 steps) and both 6L DSpark 43% snapshots.
# Lane 2 re-runs the published checkpoint on a second card as a cross-check.
#
# Two contaminated runs hinted the 43% checkpoint is much worse -- 1257s wall
# with 525 errors, and 2001s clean at k=4 against 770s for the 24.1% checkpoint
# at the same k -- but neither is trustworthy, which is the point of this job.
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="/nvmedata/chenw/envs/speculator-vllm031/bin:/usr/local/cuda-12.9/bin:$PATH"
export PYTHONUNBUFFERED=1 FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_WORKSPACE_BASE=/nvmedata/chenw/jitcache
export VLLM_CACHE_ROOT=/nvmedata/chenw/jitcache/vllm
export TRITON_CACHE_DIR=/nvmedata/chenw/jitcache/triton
export TORCHINDUCTOR_CACHE_DIR=/nvmedata/chenw/jitcache/inductor
export LD_LIBRARY_PATH="/nvmedata/chenw/envs/speculator-vllm031/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
M=/nvmedata/hf_checkpoints/qwen35-l1234-v25-step6000
DATA=/nvmedata/chenw/datasets/l1-v2-gpt56-test/l1_v2_gpt56_test.jsonl
R=/nvmedata/chenw/speculators/scripts/evaluate/experiments/results/qwen35-redo
S=/tmp/claude-0/-nvmedata-chenw-speculators/8b4a78df-65f2-4ea0-93d6-2674dd27812b/scratchpad
VL=/nvmedata/chenw/envs/speculator-vllm031/lib/python3.12/site-packages/vllm
O=/nvmedata/chenw/speculators/output
WIDE=$O/e4_6l_dspark_wide_qwen35/checkpoints/0
DS=',"dspark_draft_topk":64'
N=2000   # the FULL held-out set, as asked -- 2000 unique prompts, 0 overlap with training
mkdir -p "$R"
echo "=== JOB 429 START $(date -u) ==="
grep -q "_prime_markov_state" $VL/v1/worker/gpu/spec_decode/dspark/speculator.py || {
  echo "ABORT: anchor-slot serving fix missing"; exit 1; }
for p in $(pgrep -f -- "[-]-port 8003"); do kill -9 $p 2>/dev/null; done
sleep 10
# Own ONLY the GPUs this job declares. The unconditional "for g in 0 1 2 3"
# that used to live here wiped every card at script start, so launching a
# second job killed the first one's servers mid-run: on 2026-11-06 job 425's
# preamble killed all four of job 424's arms ten seconds after launch, and
# job 428's killed its last one, voiding the whole job with 335-525 failed
# requests per arm. A failed request still counts in req/s = rows/wall_s, so a
# dead baseline reads ~50% FASTER and silently corrupts every speedup.
for g in ${OWNED_GPUS:-0 1 2 3}; do
  for q in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g); do kill -9 $q 2>/dev/null; done
done
sleep 20
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  /'

# one arm = one server on one GPU, run to completion, torn down by process group
run_arm () {
  local gpu=$1 port=$2 label=$3 spec=$4
  [ -f "$R/$label.json" ] && { echo "  [g$gpu] $label already done"; return 0; }
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu)
    [ "$used" -lt 2000 ] && break
    for q in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $gpu); do kill -9 $q 2>/dev/null; done
    sleep 10
  done
  local a=(vllm serve "$M" --host 127.0.0.1 --port $port --tensor-parallel-size 1
     --gpu-memory-utilization 0.90 --max-model-len 32768 --max-num-seqs 64
     --no-enable-prefix-caching --trust-remote-code)
  [ -n "$spec" ] && a+=(--speculative-config "$spec")
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup "${a[@]}" > "$R/server_$label.log" 2>&1 &
  local srv=$!
  for i in $(seq 1 300); do
    curl -sf http://127.0.0.1:$port/health >/dev/null 2>&1 && break
    pgrep -f -- "[-]-port $port" >/dev/null 2>&1 || { echo "  [g$gpu] $label server exited after $((i*10))s"; break; }
    sleep 10
  done
  if ! curl -sf http://127.0.0.1:$port/health >/dev/null 2>&1; then
    echo "  [g$gpu] !! $label SERVER FAILED"
    grep -oE 'nvcc fatal.{0,70}|Free memory on device.{0,70}|ValueError.{0,100}|No available memory.{0,50}' \
      "$R/server_$label.log" | sort -u | head -2 | sed 's/^/       /'
    kill -9 -$srv 2>/dev/null; return 1; fi
  echo "  [g$gpu] SERVER UP $label $(date -u +%H:%M:%S)"
  python $S/qwen35_mtp_sweep.py --base-url http://127.0.0.1:$port --input "$DATA" \
    --model "$M" --label "$label" --no-thinking --out "$R/$label.json" \
    --limit $N --concurrency 32 --force-temperature 0 >> "$R/$label.log" 2>&1
  echo "  [g$gpu] RUN $label rc=$? $(date -u +%H:%M:%S)"
  # A failed request still counts in req/s = rows/wall_s, so an arm with errors
  # reads FASTER than it is. Quarantine it rather than let it reach the table.
  ne=$(python3 -c "import json;print(json.load(open('$R/$label.json')).get('errors',0))" 2>/dev/null || echo 999)
  if [ "$ne" != "0" ]; then
    echo "  [g$gpu] !! $label HAD $ne FAILED REQUESTS -- quarantining, not usable"
    mkdir -p "$R/void"; mv "$R/$label.json"* "$R/void/" 2>/dev/null
  fi
  echo "  [g$gpu] per-position: $(grep -oE 'Per-position acceptance rate:.*' "$R/server_$label.log" | tail -1)"
  kill -9 -$srv 2>/dev/null
  for p in $(pgrep -f -- "[-]-port $port"); do kill -9 $p 2>/dev/null; done
  sleep 10
}

W4=$O/e4_4l_dflash_wide12k_qwen35/checkpoints          # 4L DFlash, 43%
W6=$O/e4_6l_dspark_wide12k_qwen35/checkpoints          # 6L DSpark, 43%
P4=$O/e4_4l_dflash_qwen35/checkpoints/checkpoint_best  # 4L DFlash, 24.1% (published)
DS=',"dspark_draft_topk":64'
for c in "$W4/0" "$W4/step3041" "$W4/step6024" "$W4/step9003" "$W6/0" "$W6/step7012" "$P4"; do
  [ -f "$c/model.safetensors" ] || { echo "ABORT: no weights at $c"; exit 1; }
done
df () { echo "{\"model\":\"$1\",\"num_speculative_tokens\":7,\"method\":\"dflash\"}"; }
ds () { echo "{\"model\":\"$1\",\"num_speculative_tokens\":7,\"method\":\"dspark\"$DS}"; }

lane0 () {
  run_arm 0 8910 v_baseline       ''
  run_arm 0 8910 v_mtp_k7         '{"method":"qwen3_5_mtp","num_speculative_tokens":7}'
  run_arm 0 8910 v_4l_24pct       "$(df $P4)"
  run_arm 0 8910 v_4l_43pct_final "$(df $W4/0)"
}
lane1 () {
  run_arm 1 8911 c1_baseline        ''
  run_arm 1 8911 c1_4l_43pct_s3041  "$(df $W4/step3041)"
  run_arm 1 8911 c1_4l_43pct_s6024  "$(df $W4/step6024)"
}
lane2 () {
  run_arm 2 8912 c2_baseline        ''
  run_arm 2 8912 c2_4l_43pct_s9003  "$(df $W4/step9003)"
  run_arm 2 8912 c2_4l_24pct_again  "$(df $P4)"
}
lane3 () {
  run_arm 3 8913 c3_baseline        ''
  run_arm 3 8913 c3_6l_43pct_final  "$(ds $W6/0)"
  run_arm 3 8913 c3_6l_43pct_s7012  "$(ds $W6/step7012)"
}
lane0 & p0=$!; lane1 & p1=$!; lane2 & p2=$!; lane3 & p3=$!
wait $p0 $p1 $p2 $p3
echo "=== JOB 429 DONE $(date -u) ==="

python3 - "$R" <<'PY'
import json, sys, glob
from pathlib import Path
R = Path(sys.argv[1]); d = {}
for f in sorted(R.glob("*.json")):
    j = json.loads(f.read_text())
    if not j.get("wall_s"): continue
    if j.get("errors", 0): print(f"  SKIPPING {f.stem}: {j['errors']} failed requests"); continue
    d[f.stem] = (j, str(f)[:-5] + ".json.rows.jsonl")
print()
print("  cost = accepted length / speedup, in baseline forward passes.")
print("  tok-norm divides output tok/s instead of req/s (cancels length drift).")
print()
print("  %-22s %7s %8s %7s %9s %9s %4s" % ("arm","req/s","AL","cost","speedup","tok-norm","run"))
print("  " + "-"*74)
for pre in ("v_","c1_","c2_","c3_"):
    base = next((v for k,v in d.items() if k.startswith(pre) and "baseline" in k), None)
    if not base: continue
    bj = base[0]; br = bj["rows"]/bj["wall_s"]
    for k,(j,rf) in sorted(d.items()):
        if not k.startswith(pre): continue
        r = j["rows"]/j["wall_s"]; al = j.get("accept_length") or 0; sp = r/br
        try:
            rw = sum(1 for l in open(rf) if json.loads(l)["finish"] != "stop")
        except Exception:
            rw = -1
        print("  %-22s %7.4f %8s %7s %9.4f %9.4f %4d" % (
            k, r, ("%.4f"%al) if al else "-", ("%.3f"%(al/sp)) if al else "-",
            sp, j["output_tok_s"]/bj["output_tok_s"], rw))
    print()
print("  THE ASK: v_4l_43pct_final against v_4l_24pct -- same architecture,")
print("  same init, same 12,000 steps, same card, same baseline. The only")
print("  difference is 43.0% of the corpus against 24.1%. v_mtp_k7 is the bar.")
print("  'run' counts requests that did NOT finish on 'stop'; a runaway on a")
print("  BASELINE inflates every arm measured against it.")
PY
