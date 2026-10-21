#!/usr/bin/env bash
# Phase 1a: on-policy regeneration of the ablation prompts with Llama-3.1-8B-Instruct.
# One single-GPU vLLM server per GPU, client fans out (run_regen_multigpu.sh).
# Sampling = the model's generation_config (T=0.6, top_p=0.9) since the client sets none.
#
#   DRY_RUN=1 bash examples/train/onpolicy_ablation/run_regen.sh       # print only
#   sbatch -p gpuonly -w sc-c96 --gres=gpu:a100m80:8 -c 32 --mem 200G -t 12:00:00 \
#     -o <abl>/logs/regen-%j.out examples/train/onpolicy_ablation/run_regen.sh
#
# Smoke test: LIMIT=64 NUM_GPUS=1 OUTFILE=<tmp>/on_policy_smoke.jsonl bash run_regen.sh
set -euo pipefail

ABL="${ABL:-/import/ml-sc-scratch1/ravir/spec_decoding_work/onpolicy_ablation}"
export REPO="${REPO:-/import/ml-sc-scratch1/ravir/spec_decoding_work/speculators}"
export MODEL="${MODEL:-/import/ml-sc-scratch1/mengmengj/models/llama3.1-8b-instruct}"
export INPUT_JSONL="${INPUT_JSONL:-$ABL/data/off_policy.jsonl}"
export OUTFILE="${OUTFILE:-$ABL/data/on_policy.jsonl}"
export LOG_DIR="${LOG_DIR:-$ABL/logs/regen}"
export CONDA_ENV="${CONDA_ENV:-/import/ml-sc-scratch1/ravir/miniconda3/envs/dspark_cu12}"
export NUM_GPUS="${NUM_GPUS:-8}"
# longest prompt is ~1.7k tokens, so 6144 new tokens fits an 8192 context
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_TOKENS="${MAX_TOKENS:-6144}"
export CONCURRENCY="${CONCURRENCY:-512}"
# the off-policy file has no `source` field; keep every row
export SKIP_SOURCES="${SKIP_SOURCES:-}"

if [ -n "${LIMIT:-}" ]; then
    head -n "$LIMIT" "$INPUT_JSONL" > "$OUTFILE.input_head"
    export INPUT_JSONL="$OUTFILE.input_head"
fi
exec bash "$REPO/scripts/response_regeneration/run_regen_multigpu.sh"
