# JOB 421 -- DFlash is ahead of MTP. Confirm it on ONE card, then find its k.
#
# WHAT 420 FOUND (n=1200, per-GPU baselines):
#   g0_mtp_k7          AL 4.8234  speedup 1.1769   <- the bar
#   g2_4l_dflash_12k   AL 4.8608  speedup 1.2609   <- +7.1% over MTP
#   g1_6l_dflash_12k   AL 4.9612  speedup 1.2411   <- +5.5% over MTP
#   g0_dspark_12k      AL 4.6435  speedup 1.1432
#   g0_wide_fulldata   AL 3.6745  speedup 0.9532   <- full data COLLAPSED
# The DFlash cells beat MTP by more than the MTP-vs-DSpark gap we had been
# grinding on. 4L DFlash ran on GPU 2, whose baseline (1.2212) is within 0.26%
# of GPU 0's (1.2180), so that comparison is already near same-card -- but
# "near" is not "same", and this is the number we would publish. Hence lane 0:
# baseline, MTP and BOTH DFlash depths on one card, one denominator.
#
# The rest of the machine sweeps k. DFlash drops DSpark's Markov head, so its
# draft cost per step is lower (4L: 3.855 vs DSpark 4.062 in units of a
# baseline forward pass, derived as AL/speedup). Cheaper drafting moves the k
# cost cliff out, so k=7 is probably no longer the optimum -- that is the whole
# reason to sweep it.
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
R=/nvmedata/chenw/speculators/scripts/evaluate/experiments/results/qwen35-dflash-k
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARNESS=$S/../qwen35_mtp_sweep.py
VL=/nvmedata/chenw/envs/speculator-vllm031/lib/python3.12/site-packages/vllm
O=/nvmedata/chenw/speculators/output
WIDE=$O/e4_6l_dspark_wide_qwen35/checkpoints/0
DS=',"dspark_draft_topk":64'
N=1200
mkdir -p "$R"
echo "=== JOB 421 START $(date -u) ==="
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
  python $HARNESS --base-url http://127.0.0.1:$port --input "$DATA" \
    --model "$M" --label "$label" --no-thinking --out "$R/$label.json" \
    --limit $N --concurrency 32 --force-temperature 0 >> "$R/$label.log" 2>&1
  echo "  [g$gpu] RUN $label rc=$? $(date -u +%H:%M:%S)"
  echo "  [g$gpu] per-position: $(grep -oE 'Per-position acceptance rate:.*' "$R/server_$label.log" | tail -1)"
  kill -9 -$srv 2>/dev/null
  for p in $(pgrep -f -- "[-]-port $port"); do kill -9 $p 2>/dev/null; done
  sleep 10
}

F4=$O/e4_4l_dflash_qwen35/checkpoints/checkpoint_best
F6=$O/e4_6l_dflash_qwen35/checkpoints/checkpoint_best
for c in "$F4" "$F6"; do
  [ -f "$c/model.safetensors" ] || { echo "ABORT: no weights at $c"; exit 1; }
done
df () { echo "{\"model\":\"$1\",\"num_speculative_tokens\":$2,\"method\":\"dflash\"}"; }

# Lane 0 is the verdict: the bar and both challengers on one card, one baseline.
lane0 () {
  run_arm 0 8910 v_baseline   ''
  run_arm 0 8910 v_mtp_k7     '{"method":"qwen3_5_mtp","num_speculative_tokens":7}'
  run_arm 0 8910 v_4l_dflash_k7 "$(df $F4 7)"
  run_arm 0 8910 v_6l_dflash_k7 "$(df $F6 7)"
}
# Lanes 1-3 sweep k. Each lane re-measures its own card's baseline, because the
# ~7% spread between cards makes a borrowed denominator worthless.
lane1 () {
  run_arm 1 8911 k1_baseline ''
  run_arm 1 8911 k1_4l_dflash_k9 "$(df $F4 9)"
  run_arm 1 8911 k1_6l_dflash_k9 "$(df $F6 9)"
}
lane2 () {
  run_arm 2 8912 k2_baseline ''
  run_arm 2 8912 k2_4l_dflash_k8 "$(df $F4 8)"
  run_arm 2 8912 k2_4l_dflash_k5 "$(df $F4 5)"
}
lane3 () {
  run_arm 3 8913 k3_baseline ''
  run_arm 3 8913 k3_6l_dflash_k8 "$(df $F6 8)"
  run_arm 3 8913 k3_6l_dflash_k5 "$(df $F6 5)"
}
lane0 & p0=$!
lane1 & p1=$!
lane2 & p2=$!
lane3 & p3=$!
wait $p0 $p1 $p2 $p3
echo "=== JOB 421 DONE $(date -u) ==="

python3 - "$R" <<'PY'
import json, sys
from pathlib import Path
R = Path(sys.argv[1])
d = {}
for f in sorted(R.glob("*.json")):
    j = json.loads(f.read_text())
    if j.get("wall_s"): d[f.stem] = j
print()
print("  Every arm is divided by the baseline measured on its OWN card.")
print("  'cost' = AL / speedup: the draft work per step, in units of one")
print("  baseline forward pass. Lower is cheaper.")
print()
print("  %-20s %7s %8s %7s %9s" % ("arm", "req/s", "AL", "cost", "speedup"))
print("  " + "-"*58)
for pre in ("v_", "k1_", "k2_", "k3_"):
    base = next((v for k, v in d.items() if k.startswith(pre) and "baseline" in k), None)
    if not base: continue
    br = base["rows"] / base["wall_s"]
    for k, j in sorted(d.items()):
        if not k.startswith(pre): continue
        r = j["rows"] / j["wall_s"]
        al = j.get("accept_len") or 0
        sp = r / br
        print("  %-20s %7.4f %8s %7s %9.4f" % (
            k, r, ("%.4f" % al) if al else "-",
            ("%.3f" % (al / sp)) if al else "-", sp))
    print()
print("  THE VERDICT is lane v_: v_4l_dflash_k7 and v_6l_dflash_k7 against")
print("  v_mtp_k7, same card, same baseline. Job 420 put 4L DFlash 7.1% over")
print("  MTP on a card whose baseline was within 0.26% of MTP's -- this lane")
print("  removes that last 0.26% of doubt.")
print("  The k1/k2/k3 lanes say whether k=7 is still the right k now that the")
print("  Markov head is gone and each draft token costs less.")
PY
