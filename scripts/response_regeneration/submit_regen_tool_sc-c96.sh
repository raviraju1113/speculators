#!/usr/bin/env bash
# Regenerate ONLY the tool-call source (continual_tool_kimi) with the Gemma-4
# target, so the tool-call assistant turns are in gemma-4's format — useful when
# evaluating on tool/agentic workloads (AgentX). Output goes to a SEPARATE file;
# merge it into the text training set afterwards.
#
# Submit:
#   sngpu --jobname regen_tool --partition gpuonly --nodelist sc-c96 \
#     --gpu 8 --gputype a100m80 --cpu 32 --mem 128000 \
#     --output ./logs/regen_tool.txt --time 8:00:00 \
#     -- bash /import/ml-sc-scratch1/chenw/speculators/scripts/response_regeneration/submit_regen_tool_sc-c96.sh
set -euo pipefail

export MODEL="${MODEL:-/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it}"
export INPUT_JSONL="${INPUT_JSONL:-/import/ml-sc-scratch5/chenw/datasets/kimi-mtp-dataset/data/train-00000-of-00001.jsonl}"
export OUTFILE="${OUTFILE:-/import/ml-sc-scratch5/chenw/datasets/kimi-regen-gemma4-31b/train_regen_tool.jsonl}"

export NUM_GPUS="${NUM_GPUS:-8}"
export CONDA_ENV="${CONDA_ENV:-gemma4-spec}"
export CUDA_COMPAT="${CUDA_COMPAT:-/import/ml-sc-scratch1/chenw/cuda-compat-13.0}"
export COMPAT_MAX_MODEL_LEN="${COMPAT_MAX_MODEL_LEN:-32768}"
export MAX_TOKENS="${MAX_TOKENS:-4096}"
export CONCURRENCY="${CONCURRENCY:-64}"

# Only regenerate the tool-call source. continual_tool_kimi is in the client's
# DEFAULT skip list, so we must (a) override the skip list to NOT drop it, and
# (b) allowlist just that source.
export SKIP_SOURCES="${SKIP_SOURCES:-llava_instruct}"
export SOURCES="${SOURCES:-continual_tool_kimi}"

REPO="${REPO:-/import/ml-sc-scratch1/chenw/speculators}"
exec bash "$REPO/scripts/response_regeneration/run_regen_multigpu.sh"
