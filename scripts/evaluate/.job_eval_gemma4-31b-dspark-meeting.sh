#!/bin/bash
set -euo pipefail
echo "host: $(hostname)"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "driver: $DRV"
if [ "${DRV%%.*}" -lt 580 ]; then
  echo "FATAL: driver $DRV too old for this vLLM (needs 595.x). Not starting."; exit 1
fi
export PATH=/import/ml-sc-scratch1/mengmengj/condaenvs/dspark/bin:$PATH
export HF_HOME=/import/snvm-sc-scratch1/mengmengj/hf_cache
cd /import/ml-sc-scratch1/mengmengj/speculators/scripts/evaluate/experiments
_VL=$(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')
if grep -q sample_from_anchor "$_VL/transformers_utils/configs/speculators/algos.py"; then
  echo "algos.py patch: APPLIED (checkpoint sample_from_anchor honoured)"
else
  echo "algos.py patch: NOT APPLIED -- upstream hardcode; True checkpoints will read ~1.8"
fi
python run_experiments.py --config gemma4-31b-dspark-meeting.yaml
echo "=== results table ==="
_OUT=$(python -c 'import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' gemma4-31b-dspark-meeting.yaml)
cat "$_OUT/results_table.md" 2>/dev/null || echo "(no results_table.md)"
