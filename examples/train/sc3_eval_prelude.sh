#!/bin/bash
# Shared prelude for the sc3-cluster eval wrappers (sbatch_sc3_eval.sh,
# sbatch_sc3_ctx_sweep.sh, sbatch_sc3_lcr_bins.sh). `source` it after the
# SBATCH header. It:
#   1. activates the eval conda env (EVAL_ENV; a clone of the training env so
#      serving patches never touch the env a training job is running in),
#   2. self-tests every GPU SLURM gave us (the fleet has cards that are idle
#      yet fail cuCtxCreate -- "GPU Recovery Action: Reset"; see HANDOFF.md §0),
#   3. defines check_conv_servable <ckpt>: refuses to serve a conv checkpoint
#      on a vllm whose gemma4_dspark.py has no DFlash2 conv support, because
#      load_weights() there silently DROPS unknown keys and the draft would run
#      without its trained convs -- wrong numbers, no error.
# Paths on this cluster (Oct 2026): model and data from chenw's scratch and
# ours; no /sms-scratch.
set -uo pipefail

export REPO="${REPO:-/import/ml-sc-scratch1/mengmengj/speculators}"
export MODEL="${MODEL:-/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-$MODEL/chat_template.jinja}"
export EVAL_ENV="${EVAL_ENV:-/import/snvm-sc-scratch1/mengmengj/condaenvs/dspark-host}"
CONDA_SH=/import/snvm-sc-scratch1/mengmengj/miniconda3/etc/profile.d/conda.sh

echo "=== job ${SLURM_JOB_ID:-?} on $(hostname) $(date '+%F %T') ==="
echo "fips=$(cat /proc/sys/crypto/fips_enabled 2>/dev/null) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# shellcheck disable=SC1090
source "$CONDA_SH" && conda activate "$EVAL_ENV" || { echo "!! cannot activate $EVAL_ENV" >&2; exit 2; }
python -c "import vllm, speculators" 2>/dev/null || { echo "!! $EVAL_ENV lacks vllm/speculators" >&2; exit 2; }
echo "env: $EVAL_ENV (vllm $(python -c 'import vllm;print(vllm.__version__)'))"

# NCCL for TP>1 serving on the A100 nodes (sc3-c98/c81): the host net plugin
# breaks init (2-rank all_reduce fails by default, passes with these -- probed
# 2026-10-22; the training driver sets the same). Harmless at TP=1.
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

# --- GPU self-test on the allocated cards (indices as SLURM exposes them) ----
BAD=""
for i in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do
  CUDA_VISIBLE_DEVICES=$i timeout 120 python -c \
    'import torch; torch.cuda.init(); (torch.ones(1, device="cuda") + 1).item()' \
    >/dev/null 2>&1 || BAD="$BAD $i"
done
if [ -n "$BAD" ]; then
  echo "!! GPU(s)$BAD of this allocation on $(hostname -s) fail CUDA init -- resubmit with --exclude=$(hostname -s)" >&2
  exit 4
fi
echo "all $(nvidia-smi -L | wc -l) allocated GPUs pass CUDA init"

# --- conv-servability gate --------------------------------------------------
check_conv_servable () {  # $1 = draft checkpoint dir
  local ckpt=$1
  [ -f "$ckpt/config.json" ] || { echo "!! no config.json in $ckpt" >&2; return 1; }
  local k
  k=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('conv_kernel_size', 0))" "$ckpt/config.json")
  if [ "$k" != "0" ]; then
    local f
    # speculators DSpark checkpoints are served by Qwen3DSparkModel (qwen3_dspark.py),
    # whatever the target model -- the conv patch lives there, not in gemma4_dspark.py.
    f=$(python -c "import vllm,os; print(os.path.join(os.path.dirname(vllm.__file__),'model_executor','models','qwen3_dspark.py'))")
    if ! grep -q 'DFlash2DSparkDecoderLayer' "$f"; then
      echo "!! $ckpt has conv_kernel_size=$k but $f has no DFlash2 conv support." >&2
      echo "   vllm would silently drop the conv weights (load_weights ignores unknown keys)" >&2
      echo "   and report numbers for a draft missing its trained convs. Apply the" >&2
      echo "   gemma4_dspark conv patch to $EVAL_ENV first (examples/train/vllm_patches/README.md)." >&2
      return 5
    fi
    echo "conv checkpoint (kernel $k): vllm gemma4_dspark.py has conv support -- OK"
  fi
  return 0
}

cd "$REPO" || exit 2
