#!/usr/bin/env bash
# Run one GLM-5.3 acceptance-eval config on the node this runs on (one srun task
# inside an 8-GPU allocation). run_experiments.py starts and stops `vllm serve`
# per experiment itself.
#
# Uses Mengmeng's evaluation harness read-only (scripts/evaluate in her checkout:
# the one her GLM-5.3 DFlash2 numbers came from) with her vllm033 env; results go
# to the config's absolute output_dir. JIT caches are node-local (NFS stale handles).
#
#   srun --jobid=<alloc> --overlap -N1 -n1 /usr/bin/bash \
#     examples/train/glm53/run_eval_glm53_1node.sh <config.yaml> [--only a,b]
set -euo pipefail
CONFIG="$(realpath "${1:?eval config yaml}")"
shift
HARNESS="${HARNESS:-/import/ml-sc-scratch1/mengmengj/speculators/scripts/evaluate/experiments}"
ENV_DIR="${ENV_DIR:-/import/ml-sc-scratch1/mengmengj/condaenvs/vllm033}"

export PATH="$ENV_DIR/bin:$PATH"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
LOCAL_CACHE="${SLURM_TMPDIR:-/tmp}/glm53_eval_cache_${SLURM_JOB_ID:-local}"
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export VLLM_CACHE_ROOT="$LOCAL_CACHE/vllm" TRITON_CACHE_DIR="$LOCAL_CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE/inductor" XDG_CACHE_HOME="$LOCAL_CACHE/xdg"
export FLASHINFER_WORKSPACE_BASE="$LOCAL_CACHE"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"

OUT=$(python -c 'import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$CONFIG")
mkdir -p "$OUT"
echo "[eval $(hostname -s)] config=$CONFIG out=$OUT vllm=$(python -c 'import vllm;print(vllm.__version__)') $(date)"
cd "$OUT"
python "$HARNESS/run_experiments.py" --config "$CONFIG" "$@"
python "$HARNESS/tabulate_results.py" --dir "$OUT" --baseline baseline || true
echo "[eval] done $(date)"
