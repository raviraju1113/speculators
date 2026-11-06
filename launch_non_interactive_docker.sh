#!/bin/bash
# Usage: bash launch_non_interactive_docker.sh <cmd> [gpu-devices] [net]
#   <cmd>          command to run inside the container, e.g. "bash my_script.sh"
#   [gpu-devices]  value for --gpus device=...  (default: all)
#   [net]          docker --net value           (default: host)

CMD=$1
NV_VISIBLE_DEVICES=${2:-"all"}
DOCKER_BRIDGE=${3:-"host"}

IMAGE=sc-artifacts2.sambanovasystems.com/sw-docker-scratch/speculators:ngc-24.12

sudo -g docker /usr/bin/cuda-docker-run-wrapper --rm \
  --gpus device=$NV_VISIBLE_DEVICES \
  --net=$DOCKER_BRIDGE \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size 16G \
  -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  -w /workspace/speculators \
  "$IMAGE" $CMD

# --- Iterating on code without rebuilding the image? ---
# Mount your live checkout over the baked-in copy by adding this flag above:
#   -v /import/ml-sc-scratch1/ravir/spec_decoding_work/speculators:/workspace/speculators \
# Then `pip install -e .` in the image means edits on the host take effect
# immediately (Python picks them up; no rebuild/push needed).
