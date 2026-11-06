#!/bin/bash
# Minimal per-node driver probe, invoked by sngpu_driver_survey.sh.
# Must be a standalone FILE: sngpu passes the job command to `sbatch --wrap`,
# which rejects `bash -c '<string>'` ("Script arguments not permitted").
echo "host: $(hostname)"
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader | head -2
nvidia-smi | grep -i "CUDA Version" || true
