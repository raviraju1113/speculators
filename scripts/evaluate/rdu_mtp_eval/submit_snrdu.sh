#!/bin/bash
# Submit the full 25-benchmark RDU MTP k=5 eval. Resume is automatic.
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
PEF="${PEF:-/import/snvm-sc-podscratch4/weip/gemma4/0828_mtp_mattf/apps_persistent/gemma4_31b_full_layers_tp16_ssss_cg_ss_kv_ss_tg_parallel_sdk_bf16/coe_pef_bsBS_max8_ssSS_CG_max131072_SS_KV_max131072_SS_TG_max131072/gemma4_31b_full_layers_TP16_ssSS_CG_SS_KV_SS_TG_parallel_sdk_bf16_CoE_ckpt_sharing_BSBS_max8_SSSS_CG_max131072_SS_KV_max131072_SS_TG_max131072/gemma4_31b_full_layers_TP16_ssSS_CG_SS_KV_SS_TG_parallel_sdk_bf16_CoE_ckpt_sharing_BSBS_max8_SSSS_CG_max131072_SS_KV_max131072_SS_TG_max131072.pef}"
LOG_DIR="${LOG_DIR:-$EVAL_DIR/logs}"
TIMEOUT="${TIMEOUT:-7-00:00:00}"
HOST_MEM="${HOST_MEM:-512G}"
QOS="${QOS:-5}"
JOBNAME="${JOBNAME:-gemma4-rdu-mtp-eval-chenw}"
# Optional pin, e.g. NODELIST=sc3-s339 ./submit_snrdu.sh
NODELIST="${NODELIST:-}"
RESERVATION="${RESERVATION:-}"
# Drain/down 16-tile nodes make Slurm report ReqNodeNotAvail.
EXCLUDE="${EXCLUDE:-sc3-s220,sc3-s240}"

mkdir -p "$LOG_DIR"
cd "$EVAL_DIR"

EXTRA=()
if [[ -n "$NODELIST" ]]; then
  EXTRA+=(--nodelist "$NODELIST")
fi
if [[ -n "$RESERVATION" ]]; then
  EXTRA+=(--reservation "$RESERVATION")
fi
if [[ -n "$EXCLUDE" ]]; then
  EXTRA+=(--sbatch-directive "#SBATCH --exclude=${EXCLUDE}")
fi

exec snrdu run \
  --allow-local-lib-python \
  --pef "$PEF" \
  --timeout "$TIMEOUT" \
  --host-mem "$HOST_MEM" \
  --qos "$QOS" \
  --jobname "$JOBNAME" \
  "${EXTRA[@]}" \
  -o "$LOG_DIR/snrdu.log" \
  -- bash "$EVAL_DIR/run_rdu_eval.sh"
