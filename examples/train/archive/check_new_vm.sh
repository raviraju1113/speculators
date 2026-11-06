#!/bin/bash
# Go/no-go check for running DSpark Gemma-4 training on a NEW machine (e.g. a
# 3x A100 VM). Standalone: no repo, no conda env, no speculators install needed.
#
#   scp examples/train/check_new_vm.sh <vm>:~/    # or run it from the shared FS
#   bash ~/check_new_vm.sh 2>&1 | tee vm_check.txt
#
# Checks, in order of how likely they are to kill the plan:
#   1. GPU memory  -- the verifier is 62.6 GB; A100-40GB changes everything
#   2. driver      -- torch 2.11+cu130 needs a CUDA 13 driver (~580+)
#   3. shared FS   -- are the model/data/env paths even visible from here
#   4. /dev/shm    -- the DataLoader ships hidden states through it
#   5. local disk  -- hidden states want fast local scratch, not NFS
#   6. glibc       -- our conda env was built on RHEL8 (glibc 2.28)
#   7. egress      -- HF downloads and W&B logging

echo "############ HOST ############"
hostname; date; uname -r
echo "distro: $(grep -h PRETTY_NAME /etc/os-release 2>/dev/null | cut -d= -f2- | tr -d '\"')"

echo
echo "############ 1. GPUs (memory is the make-or-break) ############"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  echo
  GPU_COUNT=$(nvidia-smi -L | wc -l)
  MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "GPU count : $GPU_COUNT"
  echo "per-GPU   : ${MEM_MIB} MiB"
  # gemma-4-31B-it weights are ~62.6 GB = ~64100 MiB.
  if [ "${MEM_MIB:-0}" -ge 70000 ]; then
    echo "VERDICT   : 80GB-class -> verifier fits on ONE GPU (TP=1 possible, ~17 GB KV)"
  elif [ "${MEM_MIB:-0}" -ge 38000 ]; then
    echo "VERDICT   : 40GB-class -> verifier does NOT fit on one GPU; needs TP=2 (2x40=80 GB,"
    echo "            ~17 GB total KV headroom). Workable but tight; TP=2 is mandatory,"
    echo "            so with 3 GPUs you get exactly vLLM TP=2 + 1 training rank."
  else
    echo "VERDICT   : under 40 GB -> not viable for a 62.6 GB verifier"
  fi
else
  echo "!! nvidia-smi not found -- no GPU driver on this host?"
fi

echo
echo "############ 2. Driver vs our torch build ############"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
CUDA_RT=$(nvidia-smi 2>/dev/null | grep -o "CUDA Version: [0-9.]*" | head -1)
echo "driver: ${DRV:-unknown}   ${CUDA_RT:-}"
DRV_MAJOR=${DRV%%.*}
if [ -n "$DRV_MAJOR" ] && [ "$DRV_MAJOR" -ge 580 ] 2>/dev/null; then
  echo "VERDICT: OK -- supports CUDA 13, our existing env (torch 2.11.0+cu130) works as-is"
elif [ -n "$DRV_MAJOR" ] && [ "$DRV_MAJOR" -ge 570 ] 2>/dev/null; then
  echo "VERDICT: CUDA 12.8-era -- need to rebuild the env against a cu128 torch + matching vLLM"
elif [ -n "$DRV_MAJOR" ]; then
  echo "VERDICT: TOO OLD (same wall as sc-c96 at 565) -- needs a driver bump"
else
  echo "VERDICT: could not read driver"
fi

echo
echo "############ 3. Shared filesystem visibility ############"
# If these are absent we must copy ~62.6 GB of model + 3.4 GB of prepared data
# (and rebuild the conda env), which is a day of work on its own.
for p in \
  /import/ml-sc-scratch1/mengmengj \
  /import/ml-sc-scratch5/chenw/models/gemma-4-31B-it \
  /import/ml-sc-scratch1/mengmengj/datasets/gemma4_dspark \
  /import/ml-sc-scratch1/mengmengj/condaenvs/dspark \
  /import/ml-sc-scratch1/mengmengj/speculators \
  /import/snvm-sc-scratch1/mengmengj/hf_cache
do
  if [ -r "$p" ]; then echo "OK      $p"; else echo "MISSING $p"; fi
done

echo
echo "############ 4. /dev/shm ############"
df -h /dev/shm 2>/dev/null
SHM_G=$(df -BG --output=size /dev/shm 2>/dev/null | tail -1 | tr -dc '0-9')
if [ "${SHM_G:-0}" -ge 32 ]; then
  echo "VERDICT: fine ( >=32 GB ) -- run bare-metal, no docker --shm-size needed"
else
  echo "VERDICT: SMALL (${SHM_G:-?} GB) -- lower --num-workers, or run in docker with"
  echo "         --shm-size 16G --ipc=host (see Ravi's TRAINING.md)"
fi

echo
echo "############ 5. Disk (for hidden states + checkpoints) ############"
echo "-- filesystems --"; df -h / /tmp /scratch /local 2>/dev/null | grep -v "^Filesystem" | sort -u
echo
echo "Hidden states stream at ~64 KB/token (5 aux layers + last, bf16)."
echo "With --on-generate delete they are transient, but throughput matters:"
echo "10k tok/s implies ~640 MB/s sustained write+read. Point HS_PATH at the"
echo "fastest LOCAL disk above, never at NFS."
echo "-- quick local write speed test (1 GB to /tmp) --"
dd if=/dev/zero of=/tmp/_hs_speed_test bs=1M count=1024 oflag=direct 2>&1 | tail -1 \
  || dd if=/dev/zero of=/tmp/_hs_speed_test bs=1M count=1024 2>&1 | tail -1
rm -f /tmp/_hs_speed_test

echo
echo "############ 6. CPU / RAM / glibc ############"
echo "cores: $(nproc)"
free -g | head -2
echo "glibc: $(ldd --version 2>/dev/null | head -1)"
echo "(our conda env was built on RHEL8 / glibc 2.28; same-or-newer is fine,"
echo " older would require rebuilding the env here)"

echo
echo "############ 7. Egress (HF + W&B) ############"
for url in https://huggingface.co https://api.wandb.ai; do
  code=$(curl -sS -m 15 -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)
  if [ -n "$code" ] && [ "$code" != "000" ]; then
    echo "OK        $url -> $code (any HTTP code = reachable)"
  else
    echo "NO EGRESS $url  -> use WANDB_MODE=offline and pre-staged models"
  fi
done

echo
echo "############ 8. GPU topology (TP performance) ############"
nvidia-smi topo -m 2>/dev/null | head -12
echo "(NV# links = NVLink, good for TP=2. PHB/SYS = PCIe only: TP still works,"
echo " just slower collectives.)"

echo
echo "############ 9. If the shared FS IS visible, test the env directly ############"
ENVP=/import/ml-sc-scratch1/mengmengj/condaenvs/dspark
if [ -x "$ENVP/bin/python" ]; then
  "$ENVP/bin/python" - <<'PY'
import importlib
for mod in ("torch", "transformers", "vllm", "datasets", "hs_connectors"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:14s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{mod:14s} FAILED: {type(e).__name__}: {e}")
try:
    import torch
    print("cuda_available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device 0:", torch.cuda.get_device_name(0))
        print("bf16 supported:", torch.cuda.is_bf16_supported())
except Exception as e:
    print("torch cuda probe FAILED:", e)
PY
else
  echo "env not reachable from here -> a fresh conda env + pip install is needed"
fi

echo
echo "############ SUMMARY: what to report back ############"
cat <<'EOF'
1. GPU model + per-GPU memory (80GB vs 40GB changes the layout)
2. driver version  (>=580 ideal, 570-579 = env rebuild, <570 = blocked)
3. which /import paths were visible
4. /dev/shm size
5. local disk write speed + where the fast disk is mounted
6. egress yes/no
EOF
