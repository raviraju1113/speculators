#!/bin/bash
set -e
echo "host: $(hostname)"; nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
sudo -g docker /usr/bin/cuda-docker-run-wrapper --rm \
  --gpus device=all \
  --net=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size 16G \
  nvcr.io/nvidia/pytorch:25.12-py3 \
  bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/.job_dspark_docker_inner.sh < /dev/null
