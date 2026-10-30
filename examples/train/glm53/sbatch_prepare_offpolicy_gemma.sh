#!/usr/bin/env bash
# CPU batch wrapper for prepare_offpolicy_gemma.sh (782k rows, ~1 h at 44 workers).
#   sbatch examples/train/glm53/sbatch_prepare_offpolicy_gemma.sh
#SBATCH -J glm53_prep_gemma
#SBATCH -p cpuonly
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH -o /import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2/logs/%x-%j.out
set -euo pipefail
export WORKERS="${WORKERS:-44}"
exec bash "${SLURM_SUBMIT_DIR:-.}/examples/train/glm53/prepare_offpolicy_gemma.sh"
