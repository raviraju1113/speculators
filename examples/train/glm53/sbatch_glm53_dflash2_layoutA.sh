#!/usr/bin/env bash
# Submit a layout-A run: verifier job on a B300 + trainer job on a B200 (run_glm53_dflash2_layoutA.sh).
#   RUN_NAME=glm53_dflash2_sota_v1 [VERIFIER_NODE=sc3-c131] [TRAINER_NODE=sc3-cp108] \
#     [LR=2e-4 EPOCHS=5 ...] bash examples/train/glm53/sbatch_glm53_dflash2_layoutA.sh
# Resume: rerun with the same RUN_NAME (train.py picks up the newest checkpoint under SAVE_PATH,
# wandb continues the run id); RUN_DIR is derived from RUN_NAME, so the hand-shake files are
# cleared first.
set -euo pipefail
: "${RUN_NAME:?}"
export RUN_NAME ROLE
export REPO="${REPO:-/import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2/code_295f434}"
export RUN_DIR="${RUN_DIR:-/import/ml-sc-scratch1/mengmengj/speculators/output/glm53/$RUN_NAME/run}"
LOGS=/import/ml-sc-scratch1/mengmengj/speculators/data/glm53/logs
VERIFIER_NODE="${VERIFIER_NODE:-sc3-c131}"; TRAINER_NODE="${TRAINER_NODE:-sc3-cp108}"
mkdir -p "$RUN_DIR"; rm -f "$RUN_DIR/ip_verifier" "$RUN_DIR/TRAIN_DONE" "$RUN_DIR/VLLM_DIED"
SCRIPT=/import/ml-sc-scratch1/mengmengj/speculators/examples/train/glm53/run_glm53_dflash2_layoutA.sh
COMMON=(-p gpuonly --reservation="${RESERVATION:-sn50-hdi-1}" -N 1 --exclusive --mem=0 --time="${TIME:-10-00:00:00}" --signal=B:TERM@900 --export=ALL)
V=$(ROLE=verifier sbatch --parsable "${COMMON[@]}" -J "${RUN_NAME}_verifier" -w "$VERIFIER_NODE" --gres=gpu:b300m288:8 --cpus-per-task=128 \
      -o "$LOGS/${RUN_NAME}_verifier_%j.out" "$SCRIPT")
T=$(ROLE=trainer sbatch --parsable "${COMMON[@]}" -J "${RUN_NAME}_trainer" -w "$TRAINER_NODE" --gres=gpu:b200m180:8 --cpus-per-task=128 \
      -o "$LOGS/${RUN_NAME}_trainer_%j.out" "$SCRIPT")
echo "verifier job $V on $VERIFIER_NODE; trainer job $T on $TRAINER_NODE; run dir $RUN_DIR"
