#!/usr/bin/env bash
# Phase 1b: train both arms concurrently on one exclusive 8-GPU node, same hardware.
#   off: vLLM GPU0, train GPUs 1-3     on: vLLM GPU4, train GPUs 5-7
#
#   sbatch -p gpuonly -w sc-c96 --exclusive --gres=gpu:a100m80:8 -c 64 --mem 400G \
#     -t 2-00:00:00 -J onpol_train -o <abl>/logs/train-%j.out \
#     examples/train/onpolicy_ablation/submit_phase1b.sh
#
# Never edit train_arm.sh while this job runs (bash reads scripts by byte offset),
# so it runs a snapshot copy from node-local scratch.
set -uo pipefail
REPO="${REPO:-/import/ml-sc-scratch1/ravir/spec_decoding_work/speculators}"
SNAP="${SLURM_TMPDIR:-/tmp}/train_arm.sh"
cp "$REPO/examples/train/onpolicy_ablation/train_arm.sh" "$SNAP"
export SEED="${SEED:-0}"

ARM=off VLLM_GPU=0 TRAIN_GPUS=1,2,3 PORT=8000 MASTER_PORT=29500 bash "$SNAP" &
P_OFF=$!
sleep 60   # stagger vLLM weight loads
ARM=on VLLM_GPU=4 TRAIN_GPUS=5,6,7 PORT=8100 MASTER_PORT=29600 bash "$SNAP" &
P_ON=$!
trap 'kill $P_OFF $P_ON 2>/dev/null' TERM INT
wait $P_OFF; RC_OFF=$?
wait $P_ON; RC_ON=$?
echo "off exit=$RC_OFF on exit=$RC_ON"
[ $RC_OFF -eq 0 ] && [ $RC_ON -eq 0 ]
