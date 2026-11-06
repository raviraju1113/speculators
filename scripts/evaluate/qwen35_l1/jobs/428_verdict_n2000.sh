# JOB 428 -- the verdict lane at the FULL 2000 prompts, not 1200.
#
# WHY. Repeated no-draft baselines on GPU 0 read 1.2180 (job 420), 1.2184
# (421), 1.2272 (411) and 1.2282 (418): 0.03% reproducibility within a job and
# ~0.75% drift across jobs. The MTP-vs-DFlash gap is 6.1%, about 8x that, so
# n=1200 settles it. But the k-sweep differences are 0.2-2% -- the same size as
# the drift -- so n=1200 cannot rank k, and the same doubt applies to any
# 24.1%-vs-43% gap that turns out to be small. This re-runs the four decisive
# arms over all 2000 held-out prompts.
#
# The eval split is clean: all 2000 prompts are unique and ZERO of them appear
# among the 266,940 training prompts (sha1 over the concatenated user turns).
#
# Only lane 0 here. The snapshot curve (424 lanes 1-3) stays at n=1200, where
# it is only being used to find the SHAPE of the degradation, not to rank
# neighbours.
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
R=/nvmedata/chenw/speculators/scripts/evaluate/experiments/results/qwen35-verdict-n2000
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARNESS=$S/../qwen35_mtp_sweep.py
VL=/nvmedata/chenw/envs/speculator-vllm031/lib/python3.12/site-packages/vllm
O=/nvmedata/chenw/speculators/output
WIDE=$O/e4_6l_dspark_wide_qwen35/checkpoints/0
DS=',"dspark_draft_topk":64'
N=2000
mkdir -p "$R"
echo "=== JOB 428 START $(date -u) ==="
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
echo "  waiting for GPU 0 to go idle (job 424 lane 0 has 4 arms)"
for i in $(seq 1 120); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$used" -lt 2000 ] && break
  sleep 30
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
[ "$used" -lt 2000 ] || { echo "ABORT: GPU 0 still busy ($used MiB)"; exit 1; }
echo "  GPU 0 idle $(date -u)"
lane0
echo "=== JOB 428 DONE $(date -u) ==="

python3 - "$R" <<'PY'
import json, sys, glob
from pathlib import Path
R = Path(sys.argv[1]); d = {}
for f in sorted(R.glob("*.json")):
    j = json.loads(f.read_text())
    if j.get("wall_s"): d[f.stem] = (j, str(f)[:-5] + ".json.rows.jsonl")
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
