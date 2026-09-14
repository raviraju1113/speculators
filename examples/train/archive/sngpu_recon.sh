#!/bin/bash
# Node recon for DSpark Gemma-4 training. Run as an sngpu batch job:
#
#   mkdir -p logs
#   sngpu --jobname recon --partition gpuonly --gpu 1 --gputype a100m80 \
#     --cpu 8 --mem 64000 --time 00:10:00 --output ./logs/recon.txt \
#     -- bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/sngpu_recon.sh
#
# Answers, in one shot, the questions that gate the real runs:
#   1. driver >= 570?            -> can we use plain multi-GPU vLLM (no cuda-compat)
#   2. /dev/shm size?            -> do we need Ravi's docker --shm-size/--ipc=host route
#   3. env sane on the node?     -> torch sees GPUs, vllm imports, dspark importable
# NOTE: absolute paths only -- sbatch copies this script to a spool dir, so
# $(dirname "$BASH_SOURCE") does NOT point at the repo.

set -uo pipefail

REPO=/import/ml-sc-scratch1/mengmengj/speculators
CONDA_ENV="${CONDA_ENV:-/import/ml-sc-scratch1/mengmengj/condaenvs/dspark}"
CONDA_SH="${CONDA_SH:-/import/snvm-sc-scratch1/mengmengj/miniconda3/etc/profile.d/conda.sh}"

echo "=================== HOST ==================="
hostname; date

echo "=================== DRIVER / GPUS ==================="
nvidia-smi || echo "!! nvidia-smi missing"
echo "--- driver line ---"
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv 2>/dev/null

echo "=================== SHARED MEMORY ==================="
# The DataLoader ships hidden-state tensors between workers through /dev/shm.
# Gemma-4-31B streams ~64 KB per token (5 aux layers + last, bf16), so a small
# /dev/shm is a hard failure, not a slowdown.
df -h /dev/shm
echo "total RAM:"; free -g | head -2

echo "=================== CPU ==================="
nproc

echo "=================== ENV ==================="
# shellcheck disable=SC1090
source "$CONDA_SH" && conda activate "$CONDA_ENV" || echo "!! conda activate failed"
which python
python - <<'PY'
import importlib
for mod in ("torch", "transformers", "vllm", "datasets", "hs_connectors"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:14s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{mod:14s} IMPORT FAILED: {type(e).__name__}: {e}")

try:
    import torch
    print("cuda_available:", torch.cuda.is_available(), "| device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device 0:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch cuda probe failed:", e)

try:
    import speculators.models.dspark as d
    print("dspark module:", d.__file__)
except Exception as e:
    print("dspark IMPORT FAILED:", type(e).__name__, e)
PY

echo "=================== W&B REACHABILITY ==================="
# Compute nodes often have no outbound internet even when the login node does.
# If this fails, run training with WANDB_MODE=offline and `wandb sync` afterwards
# from the login node.
python -c "import wandb; print('wandb', wandb.__version__)" 2>&1 | tail -1
curl -sS -m 15 -o /dev/null -w "api.wandb.ai -> %{http_code} (any HTTP code = reachable)\n" \
  https://api.wandb.ai || echo "!! api.wandb.ai UNREACHABLE from this node -> use WANDB_MODE=offline"
if [ -f "$HOME/.netrc" ] && grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
  echo "~/.netrc has wandb credentials: yes"
else
  echo "~/.netrc has wandb credentials: NO -> run 'wandb login' on the login node first"
fi

echo "=================== NCCL SMOKE (2 GPUs) ==================="
# sc3-c98 (2026-07-31): NCCL segfaults in ncclNetPluginInit while loading an
# external net plugin. Test BOTH ways so we know whether the workaround is still
# needed: first as-is, then with plugin discovery disabled.
if [ "$(nvidia-smi -L 2>/dev/null | wc -l)" -ge 2 ]; then
  # Must live in a real file: mp.spawn re-imports __main__ in the child, and a
  # heredoc on stdin gives the child a '<stdin>' path it cannot re-open.
  NCCL_TEST="$(mktemp /tmp/nccl_check_XXXXXX.py)"
  cat > "$NCCL_TEST" <<'PY'
import os, torch, torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29555",
                      RANK=str(rank), WORLD_SIZE="2")
    # set_device BEFORE init_process_group: otherwise both ranks initialize NCCL
    # against device 0 and the collective segfaults.
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2,
                            device_id=torch.device(f"cuda:{rank}"))
    t = torch.ones(1024, device=f"cuda:{rank}")
    dist.all_reduce(t)
    if rank == 0:
        print("NCCL all_reduce OK, value:", t[0].item(), "(expect 2.0)")
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(worker, nprocs=2, join=True)
PY
  echo "--- attempt 1: stock settings (plugin discovery ON) ---"
  if NCCL_DEBUG=WARN python "$NCCL_TEST"; then
    echo "RESULT: NCCL works as-is -- the NCCL_NET_PLUGIN workaround is NOT needed"
  else
    echo "RESULT: stock NCCL FAILED (expected here: segfault in ncclNetPluginInit)"
    echo "--- attempt 2: NCCL_NET_PLUGIN=none (skip external net plugin) ---"
    if NCCL_NET_PLUGIN=none NCCL_DEBUG=WARN python "$NCCL_TEST"; then
      echo "RESULT: works with NCCL_NET_PLUGIN=none -- keep that export in the training scripts"
    else
      echo "RESULT: still failing. Next things to try, in order:"
      echo "  NCCL_IB_DISABLE=1        (skip InfiniBand transport)"
      echo "  NCCL_P2P_DISABLE=1       (skip peer-to-peer)"
      echo "  NCCL_DEBUG=INFO          (shows which plugin/transport it loads)"
      echo "  -> if none work, single-GPU vLLM + non-torchrun training is the fallback"
    fi
  fi
  rm -f "$NCCL_TEST"
else
  echo "fewer than 2 GPUs visible; skipping NCCL check"
fi

echo "=================== DONE ==================="
