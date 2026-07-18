#!/usr/bin/env bash
# Box-specific submit wrapper: regenerate the kimi training data with the
# Gemma-4-31B target on sc-c96 (8x A100-80GB, old 565 driver -> CUDA forward-compat).
#
# It just sets the paths + env, then runs run_regen_multigpu.sh (which starts one
# single-GPU vLLM server per GPU and fans the client out across them). All the
# CUDA forward-compat wiring is applied by run_regen_multigpu.sh when CUDA_COMPAT
# is set. Multimodal/tool rows are dropped automatically (client --skip-sources
# default), so only the text conversations are regenerated.
#
# Preview first (login node, no GPUs needed — prints commands and exits):
#   DRY_RUN=1 bash scripts/response_regeneration/submit_regen_sc-c96.sh
#
# Submit the real job:
#   sngpu --nodelist sc-c96 --gputype a100m80 --gpu 8 --cpu 32 --mem 128000 \
#     --bash /import/ml-sc-scratch1/chenw/speculators/scripts/response_regeneration/submit_regen_sc-c96.sh
set -euo pipefail

# --- paths (override via env if needed) ---
export MODEL="${MODEL:-/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it}"
export INPUT_JSONL="${INPUT_JSONL:-/import/ml-sc-scratch5/chenw/datasets/kimi-mtp-dataset/data/train-00000-of-00001.jsonl}"
export OUTFILE="${OUTFILE:-/import/ml-sc-scratch5/chenw/datasets/kimi-mtp-dataset/train_regen_gemma4.jsonl}"

# --- run config ---
export NUM_GPUS="${NUM_GPUS:-8}"
export CONDA_ENV="${CONDA_ENV:-gemma4-spec}"
# Forward-compat libcuda for this box's 565/CUDA-12.7 driver (see gemma4 README §0b).
# run_regen_multigpu.sh forces single-GPU serving + --enforce-eager when this is set
# (NCCL is broken under forward-compat, so N independent 1-GPU servers is the layout).
export CUDA_COMPAT="${CUDA_COMPAT:-/import/ml-sc-scratch1/chenw/cuda-compat-13.0}"

# Reasonable throughput knobs for 8 servers (override as needed).
export CONCURRENCY="${CONCURRENCY:-256}"
export MAX_TOKENS="${MAX_TOKENS:-2048}"

# Absolute repo path — a Slurm/sngpu batch job copies this wrapper to a spool dir
# (/var/spool/slurmd/...), so a $(dirname "$BASH_SOURCE") relative reference would
# not find run_regen_multigpu.sh. Override REPO if the checkout moves.
REPO="${REPO:-/import/ml-sc-scratch1/chenw/speculators}"
exec bash "$REPO/scripts/response_regeneration/run_regen_multigpu.sh"
