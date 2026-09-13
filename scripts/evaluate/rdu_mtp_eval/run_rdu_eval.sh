#!/bin/bash
# snrdu payload: Gemma4-31B MTP k=5 CoE PEF on the GPU full benchmark list.
set -euxo pipefail

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-/import/snvm-sc-scratch2/weip/sambaflow_oA0pU8PUsZ}"
PEF="${PEF:-/import/snvm-sc-podscratch4/weip/gemma4/0828_mtp_mattf/apps_persistent/gemma4_31b_full_layers_tp16_ssss_cg_ss_kv_ss_tg_parallel_sdk_bf16/coe_pef_bsBS_max8_ssSS_CG_max131072_SS_KV_max131072_SS_TG_max131072/gemma4_31b_full_layers_TP16_ssSS_CG_SS_KV_SS_TG_parallel_sdk_bf16_CoE_ckpt_sharing_BSBS_max8_SSSS_CG_max131072_SS_KV_max131072_SS_TG_max131072/gemma4_31b_full_layers_TP16_ssSS_CG_SS_KV_SS_TG_parallel_sdk_bf16_CoE_ckpt_sharing_BSBS_max8_SSSS_CG_max131072_SS_KV_max131072_SS_TG_max131072.pef}"
CKPT="${CKPT:-/import/mlcp-sc-nlp/gemma-4/gemma-4-31b-it-pad5632-kv8-prefix}"
ASSISTANT="${ASSISTANT:-/import/ml-sc-nlpcheckpoints-scratch3/weip/gemma-4-31b-it-assistant-pad5632-prefix-split}"
RESULT_DIR="${RESULT_DIR:-$EVAL_DIR/../experiments/results/gemma4-31b-rdu-k5}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
BENCHMARKS="${BENCHMARKS:-}"

export SOFTWARE_HOME="$INSTALL_ROOT"
export INSTALL_ROOT
export ARTIFACT_ROOT="$RESULT_DIR"
export PATH="/opt/sambanova/bin:/usr/local/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL_ROOT/bazel-install/lib:/opt/sambaflow/runtime/lib:/opt/sambaflow/runtime/graph/lib:/opt/sambanova/lib:/opt/sambanova/llvm19/lib:/opt/sambanova/llvm19/lib/x86_64-unknown-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_HOME="${HF_HOME:-/import/snvm-sc-cache-scratch1/snvm_rgrs/cache_editable/huggingface/}"
export ARG_LOAD=DDR
export PROG_LOAD=DDR
export PROG_UNLOAD=DDR
export OUTPUT_FOLDER="${SLURM_TMPDIR:-$RESULT_DIR/tmp}"
export PEF CKPT ASSISTANT RESULT_DIR
mkdir -p "$OUTPUT_FOLDER" "$RESULT_DIR"

ulimit -l unlimited || true

source "$INSTALL_ROOT/model_zoo/sambanova_modelzoo/models/gemma4/venv/bin/activate"
cd "$INSTALL_ROOT/model_zoo/tests/accuracy"

EXTRA=()
if [[ -n "$BENCHMARKS" ]]; then
  EXTRA+=(--benchmarks "$BENCHMARKS")
fi

PYTHONPATH="$EVAL_DIR:$INSTALL_ROOT/model_zoo/accuracy:${PYTHONPATH:-}" python -u \
  "$EVAL_DIR/rdu_mtp_eval.py" \
  --pef "$PEF" \
  --ckpt "$CKPT" \
  --assistant "$ASSISTANT" \
  --output-dir "$RESULT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --max-tokens "$MAX_TOKENS" \
  "${EXTRA[@]}"
