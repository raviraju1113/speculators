#!/bin/bash
# sc3-cluster twin of sbatch_lcr_bins.sh: AA-LCR paired context-length bins
# (1k..112k, same questions at every length) x three servings for one draft:
#   k=K_MAIN spec, k=3 spec, no-draft baseline.  2 B200 (TP=2: at ~120k the KV
# does not fit beside the 62.6 GB weights on one card). tok/s here is a TP=2
# series -- compare only within this sweep (spec vs baseline), never with the
# TP=1 numbers from sbatch_sc3_eval.sh / the ctx sweep.
# Results: scripts/evaluate/experiments/results/lcr_bins_<TAG>_{k8,k3,base};
# plot with scripts/evaluate/experiments/plot_lcr_bins.py (expects these names).
#
#   cd /import/ml-sc-scratch1/mengmengj/speculators
#   sbatch examples/train/sbatch_sc3_lcr_bins.sh                    # conv ep25
#   CKPT=<dir> TAG=<name> SKIP_BASE=1 sbatch examples/train/sbatch_sc3_lcr_bins.sh
#   (the baseline pass is checkpoint-independent: run it once, SKIP_BASE=1 after)
#
# DATA: needs scripts/evaluate/mtp_server_eval/data/aa-lcr-{1k..112k}.jsonl,
# which are NOT in git and not yet on this cluster. Copy them from the lax01
# repo (from sc-vnc10, which can reach that host):
#   rsync -avP 63.208.233.129:/sms-scratch/mengmengj/speculators/scripts/evaluate/mtp_server_eval/data/aa-lcr-\*.jsonl \
#         /import/ml-sc-scratch1/mengmengj/speculators/scripts/evaluate/mtp_server_eval/data/
# (or regenerate: prepare_aa_lcr_sweep.py from /sms-scratch/mengmengj/data/aa_lcr).
#SBATCH -N 1
#SBATCH --partition=gpuonly
#SBATCH --gres=gpu:b200m180:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --exclude=sc3-cp78,sc3-cp79,sc3-cp80
#SBATCH -J lcr_bins
#SBATCH --chdir=/import/ml-sc-scratch1/mengmengj/speculators
#SBATCH -o logs/%x-%j.out

# shellcheck disable=SC1091
source /import/ml-sc-scratch1/mengmengj/speculators/examples/train/sc3_eval_prelude.sh

CKPT="${CKPT:-/import/ml-sc-scratch1/mengmengj/output/gemma4_31b_dspark_nemo782k_conv/checkpoints/25}"
TAG="${TAG:-conv_ep25}"
METHOD="${METHOD-dspark}"    # unset = dspark; set-but-EMPTY (METHOD= ...) = omit, let vLLM infer it
                             # (google assistant MTP). NOTE ${VAR-default}, not ${VAR:-default}:
                             # the latter also replaces an empty value (burned one run).
TP="${TP:-2}"                # 2 x B200; use TP=4 with --gres=gpu:a100m80:4 on A100-80 nodes
K_MAIN="${K_MAIN:-8}"
# fullattn1 k8 OOMed at engine init with 0.92 on lax01 (post-KV warmup wanted
# 3.6 GiB more); GPU_UTIL=0.88 if that recurs.
GPU_UTIL="${GPU_UTIL:-0.92}"
BENCHES="aa-lcr-1k,aa-lcr-2k,aa-lcr-4k,aa-lcr-8k,aa-lcr-16k,aa-lcr-32k,aa-lcr-64k,aa-lcr-96k,aa-lcr-112k"
PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 10000 ))

[ -d "$CKPT" ] || { echo "!! checkpoint missing: $CKPT" >&2; exit 3; }
check_conv_servable "$CKPT" || exit $?
# The bin files come from the lax01 repo by rsync (header); wait for them
# (up to DATA_WAIT_H hours) so the job can be queued before the copy lands.
for _ in $(seq 1 $(( ${DATA_WAIT_H:-8} * 12 ))); do
  MISSING=$(for b in ${BENCHES//,/ }; do [ -f "$REPO/scripts/evaluate/mtp_server_eval/data/$b.jsonl" ] || echo -n "$b.jsonl "; done)
  [ -z "$MISSING" ] && break
  echo "$(date '+%T') waiting for AA-LCR data: $MISSING"; sleep 300
done
[ -z "$MISSING" ] || { echo "!! AA-LCR data still missing after ${DATA_WAIT_H:-8} h: $MISSING" >&2; exit 3; }

run_pass () {  # $1 = tag (k8|k3|base); remaining args = extra vllm serve args
  local tag=$1; shift
  local log="logs/lcr_bins_server_${TAG}_${tag}_${SLURM_JOB_ID}.log"
  vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_UTIL" --max-model-len 131072 \
    --enforce-eager --no-enable-prefix-caching \
    --chat-template "$CHAT_TEMPLATE" \
    "$@" > "$log" 2>&1 &
  local srv=$!
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $srv 2>/dev/null || { echo "!! $tag server died"; tail -20 "$log"; return 1; }
    sleep 15
  done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "!! $tag server not healthy after 30 min" >&2; kill $srv 2>/dev/null; return 1; }
  echo "--- $tag pass $(date '+%H:%M') ---"
  python scripts/evaluate/mtp_server_eval/run_vllm_eval.py \
    --base-url "http://127.0.0.1:$PORT" \
    --benchmarks "$BENCHES" --num-samples 0 --max-tokens 512 --temperature 0.0 \
    --output-dir "scripts/evaluate/experiments/results/lcr_bins_${TAG}_${tag}"
  local rc=$?
  kill $srv 2>/dev/null; wait $srv 2>/dev/null; sleep 10
  return $rc
}

spec_cfg () {  # $1 = k; omits "method" when METHOD is empty (vLLM infers it, e.g. google assistant MTP)
  if [ -n "$METHOD" ]; then
    echo "{\"model\": \"$CKPT\", \"num_speculative_tokens\": $1, \"method\": \"$METHOD\"}"
  else
    echo "{\"model\": \"$CKPT\", \"num_speculative_tokens\": $1}"
  fi
}
run_pass "k${K_MAIN}" --speculative-config "$(spec_cfg "$K_MAIN")"
[ "${SKIP_K3:-0}" = "1" ] || run_pass k3 --speculative-config "$(spec_cfg 3)"
[ "${SKIP_BASE:-0}" = "1" ] || run_pass base
echo "=== lcr bins done $(date '+%F %T') ==="
