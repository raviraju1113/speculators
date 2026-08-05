#!/bin/bash
echo "=== inside container: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
echo "shm: $(df -h /dev/shm | tail -1)"
echo "/import visible: $( [ -d /import/ml-sc-scratch1/mengmengj ] && echo yes || echo NO )"
bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/container_setup.sh
