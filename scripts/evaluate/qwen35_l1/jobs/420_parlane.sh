# JOB 420 -- the eval lane, rebuilt to use all four GPUs.
#
# WHY THIS SHAPE. Jobs 418/419 measured three servers running at once:
#   GPU 0  1.2282 req/s solo (and 1.2272 in job 411 yesterday -- 0.1% apart)
#   GPU 1  1.3241  (+7.8%)
#   GPU 2  1.2262  (-0.2%)   <- ran CONCURRENTLY with GPUs 1 and 3
#   GPU 3  1.3133  (+6.9%)
# Two findings, and they are not the same finding:
#  1. Concurrency is free. GPU 2 under three-way load reproduced GPU 0's solo
#     number to 0.2%, so serial lanes were costing 75% of the machine for
#     nothing. (My probe script printed "contention is REAL" off the magnitude
#     of the deviation; the deviation was POSITIVE. That verdict was wrong.)
#  2. The GPUs are not interchangeable. A reproducible ~7% spread separates
#     {0,2} from {1,3} on identical A100 80GB PCIe cards -- same clocks, power
#     and PCIe gen, differing only in NUMA affinity. 7% would swamp what we
#     measure: MTP 1.1803 vs DSpark 1.1533 is a 2.3% gap.
# Therefore: parallelise, but every arm is divided by a baseline measured on ITS
# OWN GPU. The three arms that answer the actual question are pinned together on
# GPU 0 so they share one denominator and one card.
#
# Cleanup uses kill -9 on the process GROUP. Killing only the pid matching
# --port leaves a VLLM::EngineCore child holding 71 of 79 GiB, which is how job
# 416 lost six of its seven arms.
set -uo pipefail
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
R=/nvmedata/chenw/speculators/scripts/evaluate/experiments/results/qwen35-parlane
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARNESS=$S/../qwen35_mtp_sweep.py
VL=/nvmedata/chenw/envs/speculator-vllm031/lib/python3.12/site-packages/vllm
O=/nvmedata/chenw/speculators/output
WIDE=$O/e4_6l_dspark_wide_qwen35/checkpoints/0
DS=',"dspark_draft_topk":64'
N=1200
mkdir -p "$R"
echo "=== JOB 420 START $(date -u) ==="
grep -q "_prime_markov_state" $VL/v1/worker/gpu/spec_decode/dspark/speculator.py || {
  echo "ABORT: anchor-slot serving fix missing"; exit 1; }
[ -f "$WIDE/model.safetensors" ] || { echo "ABORT: the wide checkpoint is missing"; exit 1; }
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

# GPU 0 carries the question: one baseline, MTP, the full-data arm and the
# quarter-data arm, all on one card so they share a denominator.
lane0 () {
  run_arm 0 8910 g0_baseline ''
  run_arm 0 8910 g0_mtp_k7       '{"method":"qwen3_5_mtp","num_speculative_tokens":7}'
  run_arm 0 8910 g0_wide_fulldata "{\"model\":\"$WIDE\",\"num_speculative_tokens\":7,\"method\":\"dspark\"$DS}"
  run_arm 0 8910 g0_dspark_12k    "{\"model\":\"$O/e4_6l_s12k_qwen35/checkpoints/checkpoint_best\",\"num_speculative_tokens\":7,\"method\":\"dspark\"$DS}"
}
# the architecture cells, one per remaining GPU, each with its own baseline
lane1 () {
  run_arm 1 8911 g1_baseline ''
  run_arm 1 8911 g1_6l_dflash_12k "{\"model\":\"$O/e4_6l_dflash_qwen35/checkpoints/checkpoint_best\",\"num_speculative_tokens\":7,\"method\":\"dflash\"}"
}
lane2 () {
  run_arm 2 8912 g2_baseline ''
  run_arm 2 8912 g2_4l_dflash_12k "{\"model\":\"$O/e4_4l_dflash_qwen35/checkpoints/checkpoint_best\",\"num_speculative_tokens\":7,\"method\":\"dflash\"}"
}
lane3 () {
  run_arm 3 8913 g3_baseline ''
  run_arm 3 8913 g3_4l_dspark_12k "{\"model\":\"$O/e4_4l_van_qwen35/checkpoints/checkpoint_best\",\"num_speculative_tokens\":7,\"method\":\"dspark\"$DS}"
}
lane0 & p0=$!
lane1 & p1=$!
lane2 & p2=$!
lane3 & p3=$!
wait $p0 $p1 $p2 $p3
echo "=== JOB 420 DONE $(date -u) ==="

python3 - "$R" <<'PY'
import json
from pathlib import Path
R = Path("/nvmedata/chenw/speculators/scripts/evaluate/experiments/results/qwen35-parlane")
d = {}
for f in sorted(R.glob("g*.json")):
    j = json.loads(f.read_text())
    if j.get("wall_s"): d[f.stem] = j
print()
print("  Each arm is divided by the baseline measured on ITS OWN GPU -- the cards")
print("  differ by ~7%, so a cross-GPU ratio would be meaningless.")
print()
print("  %-22s %7s %8s %8s %9s" % ("arm", "req/s", "AL", "AR@1", "speedup"))
print("  " + "-"*60)
for g in "0123":
    base = d.get(f"g{g}_baseline")
    if not base: continue
    br = base["rows"] / base["wall_s"]
    for k, j in d.items():
        if not k.startswith(f"g{g}_"): continue
        r = j["rows"] / j["wall_s"]
        al = j.get("accept_len") or j.get("AL") or 0
        a1 = j.get("ar1") or j.get("AR@1") or 0
        print("  %-22s %7.4f %8s %8s %9.4f" % (
            k, r, ("%.4f" % al) if al else "-", ("%.4f" % a1) if a1 else "-", r / br))
    print()
print("  THE ASK: g0_wide_fulldata (99.8% of the corpus, 27,920 steps) against")
print("  g0_dspark_12k (24.1%, 12,000 steps) -- same architecture, same init,")
print("  same card, same baseline. g0_mtp_k7 is the bar to clear.")
PY
