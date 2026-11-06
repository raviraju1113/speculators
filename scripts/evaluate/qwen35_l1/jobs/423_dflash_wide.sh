# JOB 423 -- 4L DFlash, the arm that beats MTP, on the wide data.
#
# WHY THIS ARM. Measured in job 421, GPU 0, one shared baseline, 1200 requests
# each, zero runaways, 100% valid output:
#   no draft                        AL -        cost -       1.0000
#   target MTP          k=7         AL 4.8234   cost 4.123   1.1700
#   DFlash 6-layer      k=7         AL 4.9576   cost 4.125   1.2018
#   DFlash 4-layer      k=7         AL 4.8608   cost 3.917   1.2411  <- this one
# 4L DFlash is 6.1% faster than MTP and wins on COST, not accuracy: it guesses
# slightly worse than the 6-layer and barely better than MTP, but each draft
# step is cheaper because DFlash has no Markov head. It is also the smaller
# model, 1.712B against 1.912B.
#
# WHAT IT HAS NEVER SEEN. It was trained on the 8192 prep at 12,000 steps =
# 24.1% of the corpus. Job 422 is testing the data question on 6L DSpark, which
# is the architecture that LOST; this run tests it on the one that won.
#
#   prep      window  epoch    12,000 steps covers
#   e4_full     8192   24,089   24.1%   <- the published 1.2411 checkpoint
#   e4_wide    16384   27,920   43.0%   <- this run
#
# Only the data changes. Steps stay at the 12,000 that produced 1.2411, because
# a single epoch at 27,920 steps collapsed the DSpark arm to 0.9532 -- below the
# no-draft baseline -- with every row fresh, so the optimizer budget is the
# mechanism and it is not architecture-specific until proven otherwise.
#
# CHECKPOINT FREQUENCY, corrected. trainer.py computes the save interval from
# len(train_loader), i.e. the FULL epoch (27,920), not from --max-steps. So
# --checkpoint-freq 0.25 on a 12,000-step run saves exactly once, at step 6,980.
# 0.107 gives an interval of 2,987 -> saves at 2,987 / 5,974 / 8,961 / 11,948,
# which is the curve we actually want to serve-eval. (Job 422 is already past
# that decision and will yield two snapshots, not four.)
#
# The watcher is still required: saves are named by EPOCH index, so with
# --epochs 1 all four write to checkpoints/0 and overwrite each other.
set -uo pipefail
export PATH="/root/miniconda3/envs/speculator/bin:$PATH"
PY=/root/miniconda3/envs/speculator/bin/python
export CUDA_HOME=/usr/local/cuda-12.9 PATH="/usr/local/cuda-12.9/bin:$PATH"
export PYTHONUNBUFFERED=1 FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_WORKSPACE_BASE=/nvmedata/chenw/jitcache
export VLLM_CACHE_ROOT=/nvmedata/chenw/jitcache/vllm
export TRITON_CACHE_DIR=/nvmedata/chenw/jitcache/triton
export TORCHINDUCTOR_CACHE_DIR=/nvmedata/chenw/jitcache/inductor
M=/nvmedata/hf_checkpoints/qwen35-l1234-v25-step6000
WD=/nvmedata/chenw/speculators/output/e4_wide_qwen35
O=/nvmedata/chenw/speculators/output
MAIN=/nvmedata/chenw/speculators
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2}
HS_GPU=${HS_GPU:-3}
NPROC=$(echo $TRAIN_GPUS | tr ',' '\n' | wc -l)
SEQ=16384
STEPS=12000
INIT=$O/e4_init_4l_dflash
STYPE=dflash
R=0.992
OD=$O/e4_4l_dflash_wide12k_qwen35
cd $MAIN
echo "=== JOB 423 START $(date -u) ==="
[ -d "$WD/data_prep" ] || { echo "FATAL: $WD/data_prep missing"; exit 1; }
[ -d "$INIT" ] || { echo "FATAL: init $INIT missing"; exit 1; }

# Gate on the GPUs actually being idle, not on another job printing DONE --
# a finished script can still have a vLLM child holding 71 GiB.
echo "  checking the GPUs are clear"
for i in $(seq 1 60); do         # 422 finished 06:10; its orphan is cleared
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$busy" -eq 0 ] && break
  sleep 60
done
for p in $(pgrep -f -- "[-]-port 8003"); do kill -9 $p 2>/dev/null; done
sleep 15
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" -eq 0 ] || { echo "ABORT: $busy process(es) still on the GPUs after 15h"; \
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  /'; exit 1; }
echo "  GPUs clear $(date -u)"

# Confirm the coverage claim from the real sampler before spending 7.5 hours.
COV=$($PY - "$WD" "$NPROC" "$SEQ" "$STEPS" <<'PYX'
import sys, numpy as np
sys.path.insert(0, "/nvmedata/chenw/speculators/src")
from datasets import load_from_disk
from speculators.train.distributed_batch_sampler import MultipackDistributedBatchSamplerV2
wd, nproc, seq, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
d = load_from_disk(wd + "/data_prep")
s = np.array(d["seq_len"]); tr = s[: int(len(s) * 0.992)]
n = len(MultipackDistributedBatchSamplerV2(batch_max_length=seq, lengths=tr,
        num_replicas=nproc, rank=0)._generate_batches(0))
print(f"{n} {100.0*min(1.0, steps/n):.1f}")
PYX
) || { echo "FATAL: could not compute the epoch length"; exit 1; }
EPOCH=${COV%% *}; PCT=${COV##* }
echo "  epoch at window $SEQ on $NPROC ranks = $EPOCH steps"
echo "  this run: $STEPS steps = $PCT% of the corpus (the 1.1432 arm saw 24.1%)"
[ "$STEPS" -lt "$EPOCH" ] || { echo "FATAL: $STEPS is a full epoch -- that is what collapsed"; exit 1; }

start_hs () {
  curl -sf http://127.0.0.1:8003/health >/dev/null 2>&1 && return 0
  CUDA_VISIBLE_DEVICES=$HS_GPU setsid nohup vllm serve "$M" \
    --host 0.0.0.0 --port 8003 --max-model-len $SEQ --gpu-memory-utilization 0.85 \
    --speculative_config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [2, 6, 10, 14, 18, 22, 26, 30, 32]}}}' \
    --trust-remote-code > $S_DIR/423_hs.log 2>&1 &
  HS_PG=$!
  for i in $(seq 1 300); do curl -sf http://127.0.0.1:8003/health >/dev/null 2>&1 && break; sleep 10; done
  curl -sf http://127.0.0.1:8003/health >/dev/null 2>&1 || {
    echo "FATAL: :8003 never came up at --max-model-len $SEQ"
    grep -oE 'ValueError.{0,120}|Free memory.{0,80}' $S_DIR/422_hs.log | head -3 | sed 's/^/    /'
    return 1; }
  echo "  :8003 up at --max-model-len $SEQ $(date -u)"
}

# ---- the snapshot watcher: the whole point of this run's bookkeeping ----
# checkpoints/0 is overwritten by every save. Copy it aside, named by the
# global_step the trainer logged, as soon as its weights stop changing.
snapshot_watcher () {
  local ck=$OD/checkpoints seen="" 
  while :; do
    if [ -f "$ck/0/model.safetensors" ]; then
      local sz1 sz2
      sz1=$(stat -c %s%Y "$ck/0/model.safetensors" 2>/dev/null)
      sleep 45
      sz2=$(stat -c %s%Y "$ck/0/model.safetensors" 2>/dev/null)
      if [ -n "$sz1" ] && [ "$sz1" = "$sz2" ] && [ "$sz1" != "$seen" ]; then
        local st
        st=$(grep -oE 'global_step[=: ]+[0-9]+' $OD/logs/train.log 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
        st=${st:-unknown}
        if [ ! -d "$ck/step$st" ]; then
          cp -r "$ck/0" "$ck/step$st" 2>/dev/null \
            && echo "  SNAPSHOT step$st saved $(date -u +%H:%M:%S)"
        fi
        seen=$sz1
      fi
    fi
    sleep 60
  done
}

run_train () {
  mkdir -p $OD/logs
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$TRAIN_GPUS \
    torchrun --standalone --nproc_per_node $NPROC --master_port 29903 scripts/train.py \
      --verifier-name-or-path "$M" \
      --from-pretrained "$INIT" \
      --data-path "$WD/data_prep" \
      --train-data-ratio $R \
      --vllm-endpoint http://localhost:8003/v1 \
      --hidden-states-path /nvmedata/chenw/hs_qwen35 \
      --save-path "$OD/checkpoints" \
      --speculator-type $STYPE --lr 1e-4 --scheduler-type cosine \
      --target-layer-ids 2 6 10 14 18 22 26 30 \
      --epochs 1 --max-steps $STEPS --scheduler-total-steps $STEPS \
      --total-seq-len $SEQ --max-anchors 512 \
      --loss-fn '{"ce": 0.1, "tv": 0.9}' --confidence-head-alpha 1.0 \
      --per-position-loss-weight fixed-exp-decay \
      --checkpoint-freq 0.107 --on-missing generate --on-generate delete \
      >> $OD/logs/train.log 2>&1
}

S_DIR=${S_DIR:-$S}   # hidden-state server log dir
start_hs || exit 1
rm -rf $OD; mkdir -p $OD/logs
snapshot_watcher & WATCH=$!
# Record memory from INSIDE the run; nvidia-smi after torchrun exits reads ~1 MiB.
( sleep 900; echo "  live mem/util $(date -u +%H:%M:%S):"; \
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/    /' ) &
run_train; rc=$?
kill $WATCH 2>/dev/null
echo "=== 422 train rc=$rc $(date -u) ==="
grep -oE 'global_step[=: ]+[0-9]+' $OD/logs/train.log | tail -1
echo "  checkpoints that survived:"
ls -d $OD/checkpoints/*/ 2>/dev/null | sed 's/^/    /'
kill -9 -${HS_PG:-0} 2>/dev/null
for p in $(pgrep -f -- "[-]-port 8003"); do kill -9 $p 2>/dev/null; done
# Confirm the cards actually drained; a surviving EngineCore kills the next job.
for i in $(seq 1 18); do
  left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$left" -eq 0 ] && break
  for q in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $q 2>/dev/null; done
  sleep 10
done
echo "  GPUs after teardown:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/    /'
echo "=== JOB 423 DONE $(date -u) ==="
echo "  next: serve-eval every step* snapshot at k=7 on ONE card against one"
echo "  baseline and MTP. The 24.1% arm served 1.1432; if 43% beats it, data"
echo "  helps once the step budget is held fixed, and the 417 collapse was"
echo "  purely the step count."
