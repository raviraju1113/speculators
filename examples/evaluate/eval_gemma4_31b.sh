#!/bin/bash
# Evaluate a draft for google/gemma-4-31B-it: baseline (backbone alone) vs
# speculative decoding, then print the speedup table. Works with the published
# speculator or a freshly-trained checkpoint (set DRAFT=...).
#
# Drives the config-driven experiment runner (serve -> eval -> compare) so all
# the server launch/cleanup is handled for you.
#
#   bash examples/evaluate/eval_gemma4_31b.sh                               # published eagle3
#   DRAFT=./output/gemma4_31b_eagle3/checkpoints/checkpoint_best bash examples/evaluate/eval_gemma4_31b.sh
#   DRAFT=RedHatAI/gemma-4-31B-it-speculator.dflash bash examples/evaluate/eval_gemma4_31b.sh
set -euo pipefail

# Activate the conda env (override with CONDA_ENV=...). Skipped if conda is absent.
CONDA_ENV="${CONDA_ENV:-gemma4-spec}"
if command -v conda > /dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

MODEL="${MODEL:-/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it}"
DRAFT="${DRAFT:-RedHatAI/gemma-4-31B-it-speculator.eagle3}"
GPUS="${GPUS:-0,1}"                 # gemma-4-31B needs TP>=2 on 40GB (fits/tight on 80GB)
TP="${TP:-2}"
BENCHMARKS="${BENCHMARKS:-aime, gpqa, livecodebench}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/gemma4_31b_eval}"
MAX_MODEL_LEN=8192
EXTRA_ARGS="[]"

# Optional CUDA forward-compatibility (old <570 driver). Set CUDA_COMPAT=/path to
# enable; derives the env from the active conda env. NCCL is broken under compat,
# so force single-GPU serving (TP=1, --enforce-eager).
if [ -n "${CUDA_COMPAT:-}" ]; then
    # Derive paths from the activated conda env (gemma4-spec), not ambient `python`.
    PREFIX="${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}"
    SP="$("$PREFIX/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    export LD_LIBRARY_PATH="$PREFIX/lib:$CUDA_COMPAT:${LD_LIBRARY_PATH:-}"
    export PATH="$SP/ninja/data/bin:$SP/nvidia/cu13/bin:$PATH"
    export CUDA_HOME="$SP/nvidia/cu13"
    export PYTHONNOUSERSITE=1 VLLM_USE_FLASHINFER_SAMPLER=0
    echo "[compat] CUDA forward-compat on; forcing single-GPU (TP=1, --enforce-eager)"
    GPUS="${GPUS%%,*}"; TP=1; MAX_MODEL_LEN="${COMPAT_MAX_MODEL_LEN:-4096}"
    EXTRA_ARGS='["--enforce-eager"]'
fi

REPO="$(cd "$(dirname "$0")/../.." && pwd)"

CFG="$(mktemp --suffix=.yaml)"
trap 'rm -f "$CFG"' EXIT
cat > "$CFG" <<YAML
backbone: ${MODEL}
gpus: "${GPUS}"
server:
  tensor_parallel_size: ${TP}
  gpu_memory_utilization: 0.9
  max_model_len: ${MAX_MODEL_LEN}
  extra_args: ${EXTRA_ARGS}
eval:
  backend: vllm
  benchmarks: [${BENCHMARKS}]
  num_samples: ${NUM_SAMPLES}
output_dir: ${OUTPUT_DIR}
experiments:
  - name: baseline
  - name: draft
    draft: ${DRAFT}
YAML

echo "=== Config ==="
cat "$CFG"
echo "=== Running (baseline vs draft) ==="
cd "$REPO/scripts/evaluate/experiments"
python run_experiments.py --config "$CFG"
