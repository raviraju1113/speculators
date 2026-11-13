#!/usr/bin/env bash
# Phase-2 contrast group 1 (PLAN §2b/§6): shared verifier on c130 + 3 independent 8-rank trainers,
# small loader footprint. Same data/seed/LR/schedule as the SOTA run; compare at equal step.
#   bash launch_sweep_group1.sh            (needs sn50-hdi-1 access; nodes default cp100-102)
set -euo pipefail
H=/import/ml-sc-scratch1/mengmengj/speculators/examples/train/glm53
VD=/import/ml-sc-scratch1/mengmengj/speculators/output/glm53/glm53_dflash2_sweep_verifier/run
ROLE=verifier RUN_NAME=glm53_dflash2_sweep_verifier NODE="${VNODE:-sc3-c130}" bash $H/submit_layoutA_job.sh
N=(${TNODES:-sc3-cp100 sc3-cp101 sc3-cp102})
ROLE=trainer RUN_NAME=glm53_dflash2_sweep_L4   NODE=${N[0]} VERIFIER_DIR=$VD NUM_WORKERS=8 PREFETCH=2 DRAFT_CONFIG=$H/glm53_dflash2_incoai_shape_L4.json bash $H/submit_layoutA_job.sh
ROLE=trainer RUN_NAME=glm53_dflash2_sweep_L6   NODE=${N[1]} VERIFIER_DIR=$VD NUM_WORKERS=8 PREFETCH=2 DRAFT_CONFIG=$H/glm53_dflash2_incoai_shape_L6.json bash $H/submit_layoutA_job.sh
ROLE=trainer RUN_NAME=glm53_dflash2_sweep_K3   NODE=${N[2]} VERIFIER_DIR=$VD NUM_WORKERS=8 PREFETCH=2 DRAFT_CONFIG=$H/glm53_dflash2_k3_layout.json bash $H/submit_layoutA_job.sh
