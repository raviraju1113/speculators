#!/bin/bash
# Submit a 2-minute driver probe to every candidate GPU node, so we can pick one
# whose driver supports our torch build (2.11.0+cu130 needs a CUDA 13 driver,
# i.e. >= 580; a CUDA 12.8 build would need >= 570.26).
#
#   bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/sngpu_driver_survey.sh
#   # then, once the jobs land:
#   grep -H . logs/driver_*.txt
#
# sc-c96 is already known: 565.57.01 / CUDA 12.7 -> too old for our env.

set -uo pipefail
cd /import/ml-sc-scratch1/mengmengj/speculators
mkdir -p logs

# node:gputype pairs. H200 boxes first: newest hardware, so most likely to carry
# a recent driver, and 141 GB/GPU is roomier than A100-80 for a 62.6 GB verifier.
# Surveyed 2026-07-31: sc3-c98 = 595.71.05 (CUDA 13.2) OK; sc3-c97 / sc-c82 /
# sc-c120 / sc-c96 = 565.57.01 (CUDA 12.7) too old; H200 boxes reserved so the
# probes never ran. sc3-c81 is the remaining unknown -- same hardware and naming
# family as sc3-c98, so it may also be current, which would double our capacity.
CANDIDATES=(
  "sc3-c81:a100m80"    # UNKNOWN -- probe this first
  "sc3-c127:h200m141"  # reserved (mix$); probe stays pending, harmless
  "sc3-c128:h200m141"  # reserved (mix$)
)

PROBE=/import/ml-sc-scratch1/mengmengj/speculators/examples/train/_driver_probe.sh

for entry in "${CANDIDATES[@]}"; do
  node="${entry%%:*}"
  gputype="${entry##*:}"
  echo "submitting driver probe -> $node ($gputype)"
  # Single script path only: sngpu forwards this to `sbatch --wrap`, which
  # refuses a command with arguments (e.g. `bash -c '...'`).
  sngpu --jobname "drv_$node" --partition gpuonly --nodelist "$node" \
    --gpu 1 --gputype "$gputype" --cpu 4 --mem 16000 --time 00:02:00 \
    --output "./logs/driver_${node}.txt" \
    -- bash "$PROBE" \
    || echo "  !! submit rejected for $node -- check node state (sinfo -n $node) and gputype"
done

echo
echo "submitted. check with: squeue -u $USER   then: grep -H . logs/driver_*.txt"
