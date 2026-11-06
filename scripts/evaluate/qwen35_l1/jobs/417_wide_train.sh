# JOB 417 -- train on the recovered corpus at a 16,384 window.
#
# 415 re-prepped at 16384 to get back the 59,321 rows the 8192 window dropped.
# The packing sampler TRUNCATES anything longer than batch_max_length, and
# batch_max_length IS --total-seq-len, so the recovered tails are only actually
# seen if training also runs at 16384. Hence this job rather than pointing 414
# at the new prep.
#
# TWO THINGS THAT MUST CHANGE WITH THE WINDOW, both easy to miss:
#  1. the hidden-states server's --max-model-len. At 8192 it would refuse the
#     very rows this job exists to use, and the run would fail per-row at
#     generate time rather than at startup.
#  2. memory. Measured during 414 (6-layer DSpark, 8192, 3 ranks):
#     48,441 MiB of 81,920 per training rank. Of that, roughly 36 GB is static
#     -- 2.54B params as bf16 weights plus fp32 master and Adam moments -- and
#     does NOT scale with sequence length; about 12 GB does. Doubling the window
#     predicts ~60 GB, inside 80 with room. That is an estimate, so stage 1
#     below spends 200 steps proving it before stage 2 commits ~7 hours.
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
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARNESS=$S/../qwen35_mtp_sweep.py
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2}
HS_GPU=${HS_GPU:-3}
NPROC=$(echo $TRAIN_GPUS | tr ',' '\n' | wc -l)
SEQ=16384
INIT=$O/e4_init_6l_van
STYPE=dspark
cd $MAIN
echo "=== JOB 417 START $(date -u) ==="

[ -d "$WD/data_prep" ] || { echo "FATAL: $WD/data_prep missing -- run 415 first"; exit 1; }
[ -d "$INIT" ] || { echo "FATAL: init $INIT missing"; exit 1; }
kill $(pgrep -f -- "[-]-port 8003") 2>/dev/null || true
sleep 20
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" -eq 0 ] || { echo "ABORT: $busy process(es) still on the GPUs"; \
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  /'; exit 1; }

STEPS=$($PY - "$WD" "$NPROC" "$SEQ" <<'PY'
import sys, numpy as np
sys.path.insert(0, "/nvmedata/chenw/speculators/src")
from datasets import load_from_disk
from speculators.train.distributed_batch_sampler import MultipackDistributedBatchSamplerV2
d = load_from_disk(sys.argv[1] + "/data_prep"); nproc = int(sys.argv[2]); bml = int(sys.argv[3])
s = np.clip(np.array(d["seq_len"]), 0, bml)
tr = s[: int(len(s) * 0.992)]
print(len(MultipackDistributedBatchSamplerV2(batch_max_length=bml, lengths=tr,
      num_replicas=nproc, rank=0)._generate_batches(0)))
PY
) || { echo "FATAL: could not compute the epoch length"; exit 1; }
case "$STEPS" in ''|*[!0-9]*) echo "FATAL: bad step count '$STEPS'"; exit 1;; esac

RATIO=$($PY - "$WD" <<'PY'
import sys
from pathlib import Path
from datasets import load_from_disk
O = Path(sys.argv[1])
ntr, nva = len(load_from_disk(str(O/"prep_train"))), len(load_from_disk(str(O/"prep_val")))
total = ntr + nva
for extra in range(0, 1001):
    s = "%.10f" % ((ntr + extra/1000.0) / total)
    if 0.0 < float(s) < 1.0 and int(total*float(s)) == ntr:
        print("%s %d %d" % (s, ntr, nva)); break
else:
    sys.exit(1)
PY
) || { echo "FATAL: could not reproduce the split boundary"; exit 1; }
set -- $RATIO; R=$1; PTR=$2; PVA=$3
echo "  wide corpus: $PTR train + $PVA val rows, ratio=$R"
echo "  one full epoch at $SEQ on $NPROC ranks = $STEPS steps"

# The hidden-states server MUST span the new window, or the long rows this job
# exists to train on cannot have their hidden states generated at all.
start_hs () {
  curl -sf http://127.0.0.1:8003/health >/dev/null 2>&1 && return 0
  CUDA_VISIBLE_DEVICES=$HS_GPU setsid nohup python -m vllm.entrypoints.cli.main serve "$M" \
    --host 0.0.0.0 --port 8003 --max-model-len $SEQ --gpu-memory-utilization 0.85 \
    --speculative_config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [2, 6, 10, 14, 18, 22, 26, 30, 32]}}}' \
    --kv_transfer_config '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"shared_storage_path": "/nvmedata/chenw/hs_qwen35_shared"}}' \
    > $O/hs_417.log 2>&1 &
  for i in $(seq 1 300); do curl -sf http://127.0.0.1:8003/health >/dev/null 2>&1 && break; sleep 10; done
  curl -sf http://127.0.0.1:8003/health >/dev/null 2>&1 || {
    echo "FATAL: :8003 never came up at --max-model-len $SEQ"
    grep -oE "AssertionError.*|ValueError.*|No available memory.*|RuntimeError.*" $O/hs_417.log | sort -u | head -3
    return 1; }
  echo "  :8003 up at --max-model-len $SEQ $(date -u)"
}

run_train () {
  local od=$1 steps=$2 sched=$3
  mkdir -p $od/logs
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$TRAIN_GPUS \
    torchrun --standalone --nproc_per_node $NPROC --master_port 29901 scripts/train.py \
      --verifier-name-or-path "$M" \
      --from-pretrained "$INIT" \
      --data-path "$WD/data_prep" \
      --train-data-ratio $R \
      --vllm-endpoint http://localhost:8003/v1 \
      --hidden-states-path /nvmedata/chenw/hs_qwen35 \
      --save-path "$od/checkpoints" \
      --speculator-type $STYPE --lr 1e-4 --scheduler-type cosine \
      --target-layer-ids 2 6 10 14 18 22 26 30 \
      --epochs 1 --max-steps $steps --scheduler-total-steps $sched \
      --total-seq-len $SEQ --max-anchors 512 \
      --loss-fn '{"ce": 0.1, "tv": 0.9}' --confidence-head-alpha 1.0 \
      --per-position-loss-weight fixed-exp-decay \
      --checkpoint-freq 0.25 --on-missing generate --on-generate delete \
      >> $od/logs/train.log 2>&1
}

# ---------- stage 1: does 16384 fit? 200 steps, discarded ----------
echo "=== stage 1: 200-step memory probe at $SEQ $(date -u) ==="
start_hs || exit 1
PROBE=$O/e4_wide_probe_qwen35
rm -rf $PROBE; run_train $PROBE 200 $STEPS
rc=$?
peak=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -3 | tr '\n' ' ')
echo "  probe rc=$rc  gpu mem now: $peak"
if [ $rc -ne 0 ]; then
  echo "  PROBE FAILED at $SEQ:"
  grep -oE "OutOfMemoryError.*|Error.*|Traceback.*" $PROBE/logs/train.log | sort -u | head -4
  echo "  NOT spending 7 hours on it. Options, in order: --total-seq-len 12288"
  echo "  (p90 of prompt+response is 11,932, so it recovers most of the loss),"
  echo "  or gradient checkpointing, or 2 ranks with a bigger window."
  exit 1
fi
grep -oE "global_step=[0-9]+" $PROBE/logs/train.log | tail -1
echo "  probe passed; the window fits"
rm -rf $PROBE

# ---------- stage 2: the real full-epoch run on the recovered corpus ----------
OD=$O/e4_6l_dspark_wide_qwen35
echo "=== stage 2: full epoch at $SEQ, $STEPS steps $(date -u) ==="
run_train $OD $STEPS $STEPS
rc=$?
echo "  wide full-epoch TRAIN rc=$rc $(date -u)"
[ $rc -ne 0 ] && grep -oE "OutOfMemoryError.*|Error.*|Traceback.*" $OD/logs/train.log | sort -u | head -4
ls -d $OD/checkpoints/*/ 2>/dev/null | sed 's/^/    /'
kill $(pgrep -f -- "[-]-port 8003") 2>/dev/null || true
for p in $(pgrep -f -- "[s]cripts/train.py --verifier-name-or-path"); do kill -9 $p 2>/dev/null; done
sleep 25
echo "=== JOB 417 DONE $(date -u) ==="
echo "  next: serve-eval $OD/checkpoints at k=7 against baseline + MTP k=7."
echo "  Serving is unaffected by the training window, so this lands in the same"
echo "  lane as 416 and is directly comparable to the 8192 full-epoch arm."
cd $S && setsid nohup bash 416_fulleval.sh > $S/416.log 2>&1 < /dev/null &
echo "  416 launched: serve-eval the wide checkpoints against MTP, plus the grid cells $(date -u)"
