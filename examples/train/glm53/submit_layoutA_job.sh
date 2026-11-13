#!/usr/bin/env bash
# Submit ONE layout-A job (ROLE=verifier|trainer) without touching other runs' handshake files.
#   ROLE=trainer RUN_NAME=x NODE=sc3-cp100 [VERIFIER_DIR=...] [NUM_WORKERS=8 PREFETCH=2 ...] bash submit_layoutA.sh
set -euo pipefail
: "${ROLE:?}" "${RUN_NAME:?}" "${NODE:?}"
export ROLE RUN_NAME
export REPO="${REPO:-/import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2/code_295f434}"
export RUN_DIR="${RUN_DIR:-/import/ml-sc-scratch1/mengmengj/speculators/output/glm53/$RUN_NAME/run}"
LOGS=/import/ml-sc-scratch1/mengmengj/speculators/data/glm53/logs
mkdir -p "$RUN_DIR"
if [ "$ROLE" = verifier ]; then rm -f "$RUN_DIR/ip_verifier" "$RUN_DIR/TRAIN_DONE" "$RUN_DIR/VLLM_DIED"; GRES=gpu:b300m288:8; else rm -f "$RUN_DIR/TRAIN_DONE"; GRES=gpu:b200m180:8; fi
SCRIPT=/import/ml-sc-scratch1/mengmengj/speculators/examples/train/glm53/run_glm53_dflash2_layoutA.sh
sbatch --parsable -p gpuonly --reservation="${RESERVATION:-sn50-hdi-1}" -N 1 --exclusive --mem=0 --time="${TIME:-10-00:00:00}" \
  --signal=B:TERM@900 --export=ALL -J "${RUN_NAME}_${ROLE}" -w "$NODE" --gres=$GRES --cpus-per-task=128 \
  -o "$LOGS/${RUN_NAME}_${ROLE}_%j.out" "$SCRIPT"
