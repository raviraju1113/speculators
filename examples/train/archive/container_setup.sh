#!/bin/bash
# Test whether the NGC container fixes our NCCL crash, and set up an env inside it.
#
# WHY: on bare metal (sc3-c98, driver 595) NCCL segfaults inside ncclNetPluginInit
# (plugin/net.cc:216) during ncclCommInitRank -- it hits vLLM TP>=2 and any
# torchrun launch. NCCL_NET_PLUGIN=none / NCCL_IB_DISABLE=1 / NCCL_SOCKET_IFNAME=lo
# did not help. Ravi runs vLLM TP=4 on this same node INSIDE the NGC container,
# so the container is the prime suspect for a fix: the host's broken
# libnccl-net.so is simply not visible inside it.
#
# HOW TO USE (sngpu --image only works interactively; it fails in batch):
#
#   sngpu --interactive --time 5:59:59 --cpu 24 --mem 200000 --gpu 2 \
#     --image nvcr.io/nvidia/pytorch:25.12-py3
#   # then, inside the container shell:
#   bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/container_setup.sh
#
# Ravi's docker notes: the image defaults to a 64 MB /dev/shm, which is far too
# small for the DataLoader; he passes --ipc=host --ulimit memlock=-1
# --ulimit stack=67108864. Check step 0 below and add those if sngpu does not.
#
# STRATEGY: try the cheap thing first.
#   Phase 1 -- reuse our existing conda env inside the container (no installs).
#             If NCCL works, the container was the fix and nothing else is needed.
#   Phase 2 -- only if phase 1's env is unusable in here: build a fresh env.

set -uo pipefail

REPO=/import/ml-sc-scratch1/mengmengj/speculators
CONDA_ENV=/import/ml-sc-scratch1/mengmengj/condaenvs/dspark
CONDA_SH=/import/snvm-sc-scratch1/mengmengj/miniconda3/etc/profile.d/conda.sh

echo "############ 0. Where are we? ############"
hostname
if [ -f /.dockerenv ] || grep -qE "docker|containerd" /proc/1/cgroup 2>/dev/null; then
  echo "in a container: YES"
else
  echo "in a container: NO  <-- run this inside the sngpu --image shell"
fi
echo "-- /dev/shm (needs to be GBs, not the 64 MB docker default) --"
df -h /dev/shm | tail -1
echo "-- GPUs --"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
echo "-- host NCCL net plugin visible in here? (absent is what we want) --"
ls -la /usr/lib64/libnccl-net*.so* /usr/lib/x86_64-linux-gnu/libnccl-net*.so* 2>/dev/null \
  || echo "no libnccl-net.so visible -- good sign"

echo
echo "############ 1. Try our existing conda env, no installs ############"
if [ -x "$CONDA_ENV/bin/python" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH" 2>/dev/null && conda activate "$CONDA_ENV" 2>/dev/null
  PY="$CONDA_ENV/bin/python"
  echo "using $PY"
  "$PY" - <<'PY'
import importlib
for mod in ("torch", "vllm", "transformers", "speculators"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:14s} {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        print(f"{mod:14s} FAILED: {type(e).__name__}: {e}")
import torch
print("cuda:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
PY
else
  echo "conda env not reachable; skip to phase 2"
  PY=python
fi

echo
echo "############ 2. THE test: does NCCL work in here? ############"
if [ "$(nvidia-smi -L | wc -l)" -lt 2 ]; then
  echo "need >= 2 GPUs for this test; re-allocate with --gpu 2"
else
  NCCL_TEST=$(mktemp /tmp/nccl_container_XXXXXX.py)
  cat > "$NCCL_TEST" <<'PY'
import os, torch, torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29577",
                      RANK=str(rank), WORLD_SIZE="2")
    torch.cuda.set_device(rank)          # before init: else both ranks use dev 0
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
  if NCCL_DEBUG=WARN "$PY" "$NCCL_TEST"; then
    echo
    echo ">>> RESULT: NCCL WORKS in the container."
    echo "    The host NCCL plugin was the problem. Inside here you can run the"
    echo "    normal multi-GPU layout -- no TP=1 workaround needed:"
    echo "      VLLM_TP=2 TRAIN_GPUS_N=2 bash $REPO/examples/train/dspark_online_gemma4_31b.sh"
  else
    echo
    echo ">>> RESULT: NCCL still fails in the container with our env."
    echo "    That points at the NCCL shipped in our pip torch rather than a host"
    echo "    plugin. Next: phase 3 -- use the container's OWN torch instead."
  fi
  rm -f "$NCCL_TEST"
fi

echo
echo "############ 3. Fallback: build a fresh env on the image's torch ############"
cat <<'EOF'
Only if phase 2 failed. The NGC image ships a matched torch+CUDA+NCCL; the goal
is to add vLLM and speculators WITHOUT replacing that torch.

  python -c "import torch; print(torch.__version__)"   # note the image's version

  # vLLM pins a torch version and will happily replace the image's one, which
  # throws away the reason for using the image. Try without deps first:
  pip install --no-deps vllm==0.26.0
  python -c "import vllm; print(vllm.__version__)"     # if this imports, good
  # if it fails on a missing dep, install those individually rather than letting
  # pip resolve vllm's full tree.

  pip install -e /import/ml-sc-scratch1/mengmengj/speculators/hs_connectors
  pip install -e /import/ml-sc-scratch1/mengmengj/speculators --no-deps
  pip install pydantic pydantic-settings loguru datasets safetensors typer rich

  # then re-run the phase 2 NCCL test with this python.

If vLLM cannot be made to work on the image's torch, the honest fallback is the
bare-metal TP=1 / single-rank layout we already have working, and accepting
lower hidden-state throughput.
EOF

echo
echo "############ SUMMARY: report back ############"
cat <<'EOF'
1. in a container? /dev/shm size?
2. was libnccl-net.so visible?
3. did our conda env import torch/vllm/speculators in here?
4. did the NCCL all_reduce pass?  <-- the answer that matters
EOF
