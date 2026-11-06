# JOB 427 -- the single-card k sweep I should have run in the first place.
#
# Job 421 spread k=5,8 on GPU 2, k=9 on GPU 1 and k=7 alone on GPU 0, so the k
# comparison was CROSS-CARD -- the exact flaw the per-GPU baselines exist to
# prevent. Identical A100s in this host differ by ~7%, and the whole k effect
# looked like 1-2%, so that sweep settled nothing. Within GPU 2 alone, k=5 read
# 1.2126 and k=8 read 1.2153: 0.2% apart. Accepted length climbs steadily with
# k (4.2514 -> 4.8608 -> 5.1062 -> 5.2837) and cost climbs with it
# (3.506 -> 3.917 -> 4.202 -> 4.323), so the curve is broad and flat, but where
# its peak sits is unmeasured.
#
# This runs every k on ONE card against ONE baseline. Subject: the published
# 24.1% checkpoint, because that is the artifact people will serve and its k
# belongs on its model card. If job 424 shows the 43% checkpoint wins, this
# sweep gets re-run on that one.
#
# It gates on ONE GPU going idle, not on job 424 finishing: 424's lanes are
# unbalanced (4 arms on GPU 0, 2 on GPU 3), so a card frees ~30 min early.
# Concurrency is free -- measured, a card under three-way load reproduced its
# solo number to 0.2% -- but the baseline here is taken on the same card in the
# same window as its arms, so any residual effect divides out.
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
R=/nvmedata/chenw/speculators/scripts/evaluate/experiments/results/qwen35-onpolicy
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARNESS=$S/../qwen35_mtp_sweep.py
VL=/nvmedata/chenw/envs/speculator-vllm031/lib/python3.12/site-packages/vllm
O=/nvmedata/chenw/speculators/output
WIDE=$O/e4_6l_dspark_wide_qwen35/checkpoints/0
DS=',"dspark_draft_topk":64'
N=1200
mkdir -p "$R"
echo "=== JOB 427 START $(date -u) ==="
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

GPU=${GPU:-1}
PORT=$((8920+GPU))
P4=$O/onpolicy_qwen35/checkpoints_block7/checkpoint_best
[ -f "$P4/model.safetensors" ] || { echo "ABORT: no weights at $P4"; exit 1; }

echo "  waiting for GPU $GPU to go idle (job 424 lane $GPU has the fewest arms)"
for i in $(seq 1 120); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
  [ "$used" -lt 2000 ] && break
  sleep 30
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
[ "$used" -lt 2000 ] || { echo "ABORT: GPU $GPU still busy ($used MiB) after 1h"; exit 1; }
echo "  GPU $GPU idle $(date -u)"

run_arm $GPU $PORT s_baseline ''
for k in 3 4 5 6; do
  run_arm $GPU $PORT s_4l_k$k "{\"model\":\"$P4\",\"num_speculative_tokens\":$k,\"method\":\"dspark\",\"dspark_draft_topk\":64}"
done
echo "=== JOB 427 DONE $(date -u) ==="

python3 - "$R" <<'PYX'
import json, sys
from pathlib import Path
R=Path(sys.argv[1]); d={}
for f in sorted(R.glob("s_*.json")):
    j=json.loads(f.read_text())
    if j.get("wall_s"): d[f.stem]=(j,str(f)[:-5]+".json.rows.jsonl")
b=d.get("s_baseline")
if not b: print("  no baseline"); raise SystemExit
bj=b[0]; br=bj["rows"]/bj["wall_s"]
print()
print("  Every row on ONE card against ONE baseline. 6L on-policy DSpark, block_size 7 (native depth 6).")
print()
print("  %-14s %7s %8s %7s %9s %9s %4s" % ("k","req/s","AL","cost","speedup","tok-norm","run"))
print("  "+"-"*62)
rows=[]
for k,(j,rf) in sorted(d.items(), key=lambda x: (len(x[0]), x[0])):
    r=j["rows"]/j["wall_s"]; al=j.get("accept_length") or 0; sp=r/br
    try: rw=sum(1 for l in open(rf) if json.loads(l)["finish"]!="stop")
    except Exception: rw=-1
    print("  %-14s %7.4f %8s %7s %9.4f %9.4f %4d" % (
        k,r,("%.4f"%al) if al else "-",("%.3f"%(al/sp)) if al else "-",
        sp,j["output_tok_s"]/bj["output_tok_s"],rw))
    if al: rows.append((sp,k,al))
if rows:
    best=max(rows)
    print()
    print(f"  best k on this card: {best[1]} at {best[0]:.4f} (accepted length {best[2]:.4f})")
    print(f"  MTP k=7 served 1.1700 on GPU 0 in job 421.")
PYX
