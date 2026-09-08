#!/usr/bin/env bash
# Box-specific submit wrapper: convert downloaded OpenCodeInstruct parquet shards
# into a conversation JSONL and regenerate them with Gemma-4-31B on sc-c96.
#
# Submit:
#   sngpu --nodelist sc-c96 --gputype a100m80 --gpu 8 --cpu 32 --mem 128000 \
#     --bash /import/ml-sc-scratch1/chenw/speculators/scripts/response_regeneration/submit_regen_opencodeinstruct_sc-c96.sh
#
# Optional overrides:
#   MODEL=/path/to/gemma-4-31B-it \
#   INPUT_DIR=/path/to/parquet_dir \
#   INPUT_JSONL=/path/to/input.jsonl \
#   OUTFILE=/path/to/output.jsonl \
#   LIMIT=1000 \
#   DRY_RUN=1 \
#   bash scripts/response_regeneration/submit_regen_opencodeinstruct_sc-c96.sh
set -euo pipefail

export MODEL="${MODEL:-/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it}"
export INPUT_DIR="${INPUT_DIR:-/import/ml-sc-scratch5/chenw/datasets/OpenCodeInstruct/data}"
export INPUT_JSONL="${INPUT_JSONL:-/import/ml-sc-scratch5/chenw/datasets/OpenCodeInstruct/open_code_instruct_conversations.jsonl}"
export OUTFILE="${OUTFILE:-/import/ml-sc-scratch5/chenw/datasets/OpenCodeInstruct/open_code_instruct_regen_gemma4_31b.jsonl}"
export LIMIT="${LIMIT:-}"

export NUM_GPUS="${NUM_GPUS:-8}"
export CONDA_ENV="${CONDA_ENV:-gemma4-spec}"
export CUDA_COMPAT="${CUDA_COMPAT:-/import/ml-sc-scratch1/chenw/cuda-compat-13.0}"
export COMPAT_MAX_MODEL_LEN="${COMPAT_MAX_MODEL_LEN:-32768}"
export MAX_TOKENS="${MAX_TOKENS:-4096}"
export CONCURRENCY="${CONCURRENCY:-64}"

REPO="${REPO:-/import/ml-sc-scratch1/chenw/speculators}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "$INPUT_JSONL" || "${FORCE_RECONVERT:-0}" == "1" ]]; then
  echo "[regen-opencode] preparing raw OpenCodeInstruct data -> $INPUT_JSONL"
  if [[ -n "$LIMIT" ]]; then
    "$PYTHON_BIN" "$REPO/scripts/response_regeneration/prepare_opencodeinstruct.py" \
      --input-dir "$INPUT_DIR" \
      --output-jsonl "$INPUT_JSONL" \
      --limit "$LIMIT"
  else
    "$PYTHON_BIN" "$REPO/scripts/response_regeneration/prepare_opencodeinstruct.py" \
      --input-dir "$INPUT_DIR" \
      --output-jsonl "$INPUT_JSONL"
  fi
else
  echo "[regen-opencode] reusing existing prepared input JSONL: $INPUT_JSONL"
fi

echo "[regen-opencode] starting Gemma-4 regeneration"
exec bash "$REPO/scripts/response_regeneration/run_regen_multigpu.sh"
