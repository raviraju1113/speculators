#!/usr/bin/env bash
# Restart ONLY the SOTA trainer with the small dataloader footprint (8 workers x prefetch 2),
# resuming from the newest checkpoint under its SAVE_PATH (mid-epoch fast-skip resume).
# Verifier job on c131 is untouched. Needs sn50-hdi-1 reservation access.
#   bash restart_sota_trainer_small_loader.sh <trainer_jobid_to_cancel>
set -euo pipefail
RUN=glm53_dflash2_sota1_L5_lr2e-4_wsd
RD=/import/ml-sc-scratch1/mengmengj/speculators/output/glm53/$RUN
ls "$RD/checkpoints" | grep -E "^epoch[0-9]+_" || { echo "!! no checkpoint yet under $RD/checkpoints"; exit 1; }
[ -n "${1:-}" ] && { scancel "$1"; echo "cancelled trainer $1; waiting for the node"; sleep 60; }
ROLE=trainer RUN_NAME=$RUN NODE="${NODE:-sc3-cp108}" NUM_WORKERS=8 PREFETCH=2 \
  bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/glm53/submit_layoutA_job.sh
