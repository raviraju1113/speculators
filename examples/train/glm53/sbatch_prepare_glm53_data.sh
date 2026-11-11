#!/usr/bin/env bash
# Full GLM-5.3 data prep on a dedicated node (the login node ran the 777k-row prep at ~10 rows/s
# under load 200+). CPU only; an idle A100 node is fine. Usage:
#   sbatch --nodelist=<node> examples/train/glm53/sbatch_prepare_glm53_data.sh
# Env (all optional): REGEN_JSONL OUTPUT SEQ_LENGTH KEEP_TRUNCATED WORKERS (see prepare_glm53_data.sh)
#SBATCH --job-name=glm53_prep
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --time=08:00:00
#SBATCH --output=/import/ml-sc-scratch1/mengmengj/speculators/data/glm53/logs/glm53_prep_%j.out
set -uo pipefail
cd /import/ml-sc-scratch1/mengmengj/speculators
export REGEN_JSONL="${REGEN_JSONL:-data/glm53/nemo782k_regen_glm53_all.jsonl}"
export OUTPUT="${OUTPUT:-/import/ml-sc-scratch1/mengmengj/datasets/glm53_nemotron_782k_s16k}"
export SEQ_LENGTH="${SEQ_LENGTH:-16384}" KEEP_TRUNCATED="${KEEP_TRUNCATED:-1}" WORKERS="${WORKERS:-80}"
echo "$(date) host=$(hostname) REGEN_JSONL=$REGEN_JSONL OUTPUT=$OUTPUT SEQ_LENGTH=$SEQ_LENGTH KEEP_TRUNCATED=$KEEP_TRUNCATED WORKERS=$WORKERS"
time bash examples/train/glm53/prepare_glm53_data.sh
echo "$(date) PREP DONE rc=$?"
