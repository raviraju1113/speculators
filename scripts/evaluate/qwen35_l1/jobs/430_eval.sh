# JOB 430 -- job 429 redone after I voided it the same way I voided 424.
#
# WHAT HAPPENED. I believed I had killed the n=1200 copy of 429 before
# relaunching it at n=2000. I had not: `ps` showed pid 1735383 (started
# 15:18:43) still alive at 17:20, reparented to PPID 1, running
# `--label v_4l_43pct_final --limit 1200` against port 8910, while the n=2000
# copy (pid 1746926) ran `--label v_mtp_k7 --limit 2000` against THE SAME PORT.
# I had killed the parent shell, not the process group, so the lane subshells
# survived -- the identical pid-vs-process-group mistake as the orphaned
# EngineCore. Consequences: all four n=2000 baselines lost 735-749 of 2000
# requests and were quarantined by the error gate; v_mtp_k7's server never
# started ("Engine core initialization failed"); and four n=1200 result files
# were written into the same directory, so the surviving table mixed sample
# sizes.
#
# THE FIX THAT MATTERS is not another kill loop -- it is making a second copy
# impossible to start. This script takes an exclusive flock and refuses to run
# if any sweep client or vLLM server is alive anywhere on the box.
#
# WHAT SURVIVED 429, and it is the real finding: vLLM's own per-draft-step
# acceptance, read out of the server log, does not care about failed client
# requests or contention. Every wide-prep (43% coverage, window 16384)
# checkpoint has a first-guess accuracy of 0.25-0.30 against 0.842 for the
# published 24.1% checkpoint -- 4L DFlash step3041 0.291, step9003 0.250,
# final 0.258; 6L DSpark final 0.297. Collapsed at every snapshot from step
# 3,041 on, so it was never healthy. Coverage and window moved together in
# those runs, so the honest statement is "the wide-prep recipe produces broken
# drafts", not "more data hurts". One wide-prep run at window 8192 would
# separate them; this job does not attempt it.
#
# SO THIS JOB stops spending cards on the collapsed checkpoints (one
# confirmation arm, not five) and spends them on what is actually open:
#   lane 0  the headline at n=2000 -- baseline, MTP k=7, published 24.1%
#           checkpoint, and one 43% arm, all on GPU 0 against one baseline.
#   lane 1  the single-card k sweep that job 421 could not settle because it
#           was spread across four cards with a 7% spread between them.
#   lane 2  the k tails, plus 6L DFlash at n=2000.
#   lane 3  the deferred on-policy block-7 DSpark eval, plus the 6L DSpark 43%
#           confirmation.
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
R=/nvmedata/chenw/speculators/scripts/evaluate/experiments/results/qwen35-430
S=/tmp/claude-0/-nvmedata-chenw-speculators/8b4a78df-65f2-4ea0-93d6-2674dd27812b/scratchpad
VL=/nvmedata/chenw/envs/speculator-vllm031/lib/python3.12/site-packages/vllm
O=/nvmedata/chenw/speculators/output
WIDE=$O/e4_6l_dspark_wide_qwen35/checkpoints/0
DS=',"dspark_draft_topk":64'
N=2000   # the FULL held-out set, as asked -- 2000 unique prompts, 0 overlap with training
echo "=== JOB 430 START $(date -u) ==="
# ---- SINGLE INSTANCE. This is the gate whose absence voided 424 and 429. ----
exec 9>/nvmedata/chenw/speculators/scripts/evaluate/experiments/.qwen35_eval.lock
flock -n 9 || { echo "ABORT: another qwen35 eval job holds the lock"; exit 1; }
stray=$(pgrep -cf "qwen35_mtp_sweep[.]py" || true)
[ "${stray:-0}" -gt 0 ] && { echo "ABORT: $stray sweep client(s) still alive -- kill their PROCESS GROUPS first"; exit 1; }
stray=$(pgrep -cf "vllm serve" || true)
[ "${stray:-0}" -gt 0 ] && { echo "ABORT: $stray vllm server(s) still alive"; exit 1; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" -gt 0 ] && { echo "ABORT: $busy compute process(es) on the GPUs"; exit 1; }
[ -d "$R" ] && { echo "ABORT: $R exists -- a fresh dir per job, so no arm is ever skipped as 'already done' at the wrong sample size"; exit 1; }
mkdir -p "$R"
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

W4=$O/e4_4l_dflash_wide12k_qwen35/checkpoints          # 4L DFlash, 43% coverage, window 16384
W6=$O/e4_6l_dspark_wide12k_qwen35/checkpoints          # 6L DSpark, 43% coverage, window 16384
P4=$O/e4_4l_dflash_qwen35/checkpoints/checkpoint_best  # 4L DFlash, 24.1% -- the published one
P6=$O/e4_6l_dflash_qwen35/checkpoints/checkpoint_best  # 6L DFlash, 24.1%
OP=$O/onpolicy_qwen35/checkpoints_block7/checkpoint_best
DS=',"dspark_draft_topk":64'
for c in "$W4/0" "$W6/0" "$P4" "$P6" "$OP"; do
  [ -f "$c/model.safetensors" ] || { echo "ABORT: no weights at $c"; exit 1; }
done
df () { echo "{\"model\":\"$1\",\"num_speculative_tokens\":${2:-7},\"method\":\"dflash\"}"; }
ds () { echo "{\"model\":\"$1\",\"num_speculative_tokens\":${2:-7},\"method\":\"dspark\"$DS}"; }

# Lane 0 is the only lane that answers the headline. Everything in it shares
# one card and one baseline, which is the only comparison that has ever held up.
lane0 () {
  run_arm 0 8910 v_baseline       ''
  run_arm 0 8910 v_mtp_k7         '{"method":"qwen3_5_mtp","num_speculative_tokens":7}'
  run_arm 0 8910 v_4l_24pct_k7    "$(df $P4)"
  run_arm 0 8910 v_4l_43pct_k7    "$(df $W4/0)"
}
# The k sweep job 421 could not settle: four cards, 7% apart, so a 0.2%
# difference between k=5 and k=8 meant nothing. All on one card this time.
lane1 () {
  run_arm 1 8911 k_baseline       ''
  run_arm 1 8911 k_4l_24pct_k5    "$(df $P4 5)"
  run_arm 1 8911 k_4l_24pct_k7    "$(df $P4 7)"
  run_arm 1 8911 k_4l_24pct_k9    "$(df $P4 9)"
}
lane2 () {
  run_arm 2 8912 t_baseline       ''
  run_arm 2 8912 t_4l_24pct_k3    "$(df $P4 3)"
  run_arm 2 8912 t_4l_24pct_k11   "$(df $P4 11)"
  run_arm 2 8912 t_6l_24pct_k7    "$(df $P6 7)"
}
lane3 () {
  run_arm 3 8913 o_baseline       ''
  run_arm 3 8913 o_onpolicy_b7_k7 "$(ds $OP)"
  run_arm 3 8913 o_6l_43pct_k7    "$(ds $W6/0)"
}
lane0 & p0=$!; lane1 & p1=$!; lane2 & p2=$!; lane3 & p3=$!
wait $p0 $p1 $p2 $p3
echo "=== JOB 430 DONE $(date -u) ==="

python3 - "$R" <<'PY'
import json, re, sys
from pathlib import Path
R = Path(sys.argv[1]); d = {}
for f in sorted(R.glob("*.json")):
    j = json.loads(f.read_text())
    if not j.get("wall_s"):
        continue
    if j.get("errors", 0):
        print(f"  SKIPPING {f.stem}: {j['errors']} failed requests"); continue
    d[f.stem] = (j, str(f)[:-5] + ".json.rows.jsonl")

# Every arm must have measured the same prompts. 429 mixed 1200 and 2000 in one
# directory because two jobs wrote into it; this refuses to print such a table.
sizes = {k: v[0]["rows"] for k, v in d.items()}
if len(set(sizes.values())) > 1:
    print("  REFUSING TO PRINT: arms have different sample sizes:")
    for k, n in sorted(sizes.items()): print(f"    {k}: {n}")
    sys.exit(1)
print(f"\n  all arms n={next(iter(sizes.values()), 0)}")
print("  'right/step' = tokens guessed right per step (vLLM's accepted length).")
print("  'first' = how often the draft's FIRST guess is right, from the server's")
print("  own per-step counters -- immune to client errors and to contention.")
print("  'cost' = tokens right per step / times faster, in baseline forward passes.")
print()
print("  %-22s %7s %10s %6s %7s %9s %9s %4s" % (
    "arm", "req/s", "right/step", "first", "cost", "x faster", "tok-norm", "run"))
print("  " + "-" * 86)
for pre in ("v_", "k_", "t_", "o_"):
    base = next((v for k, v in d.items() if k.startswith(pre) and "baseline" in k), None)
    if not base:
        continue
    bj = base[0]; br = bj["rows"] / bj["wall_s"]
    for k, (j, rf) in sorted(d.items()):
        if not k.startswith(pre):
            continue
        r = j["rows"] / j["wall_s"]; al = j.get("accept_length") or 0; sp = r / br
        try:
            txt = (R / f"server_{k}.log").read_text()
            m = re.findall(r"Per-position acceptance rate: ([0-9.]+)", txt)
            first = float(m[-1]) if m else 0.0
        except Exception:
            first = 0.0
        try:
            rw = sum(1 for l in open(rf) if json.loads(l)["finish"] != "stop")
        except Exception:
            rw = -1
        print("  %-22s %7.4f %10s %6s %7s %9.4f %9.4f %4d" % (
            k, r, ("%.4f" % al) if al else "-", ("%.3f" % first) if first else "-",
            ("%.3f" % (al / sp)) if al else "-", sp,
            j["output_tok_s"] / bj["output_tok_s"], rw))
    print()
print("  Lane v_ is the headline: MTP k=7 against the published 24.1% 4L DFlash,")
print("  one card, one baseline, at the full 2000 prompts. v_4l_43pct_k7 is the")
print("  one confirmation arm for the collapsed wide-prep checkpoints -- expect a")
print("  first-guess accuracy near 0.26 and a times-faster below 1.")
print("  Lane k_ is the k sweep on ONE card, which is the only way it means")
print("  anything: the cards are 7% apart, so job 421's cross-card sweep could")
print("  not resolve the 0.2% it was looking for.")
print("  'run' counts requests that did NOT finish on 'stop'. A runaway on a")
print("  BASELINE inflates every arm measured against it -- that happened twice")
print("  on GPU 2 at row 1035, so check this column before believing lane t_.")
PY
