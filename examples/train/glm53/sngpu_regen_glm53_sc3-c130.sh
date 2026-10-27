#!/usr/bin/env bash
# Submit the GLM-5.3 regeneration as a batch job on the 8x B300 node sc3-c130
# (reservation sn50-hdi-0). Wraps run_regen_glm53.sh; every env var it reads
# passes through (REASONING_EFFORT, TEMPERATURE, MAX_TOKENS, LIMIT, ...).
#
#   # 2k-row throughput probe (~1 h incl. weight load), then read tok/s from
#   # the client log and scale the full-run --time from it:
#   LIMIT=2000 TIME=2:00:00 bash examples/train/glm53/sngpu_regen_glm53_sc3-c130.sh
#   # full run (resumes where the probe stopped):
#   TIME=3-00:00:00 bash examples/train/glm53/sngpu_regen_glm53_sc3-c130.sh
#
# The interactive form of the same allocation, for poking at the server:
#   sngpu --nodelist sc3-c130 --reservation sn50-hdi-0 --gputype b300m288 \
#         --gpu 8 --cpu 32 --mem 256000 --time 5:59:59 --interactive
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
NODE="${NODE:-sc3-c130}"
RESERVATION="${RESERVATION:-sn50-hdi-0}"
GPUTYPE="${GPUTYPE:-b300m288}"
TIME="${TIME:-3-00:00:00}"
CPU="${CPU:-32}"
MEM="${MEM:-256000}"
JOBNAME="${JOBNAME:-glm53_regen}"
LOG_DIR="${LOG_DIR:-$REPO/data/glm53/logs}"
mkdir -p "$LOG_DIR"

# sngpu copies the script to a spool dir, so the runner is referenced by its
# absolute path and gets the env through --export (sbatch default is ALL, but
# make the pass-through explicit for the variables the runner reads).
PASS=(REPO MODEL INPUT_JSONL OUTFILE REASONING_EFFORT TEMPERATURE TOP_P MAX_TOKENS
      MAX_MODEL_LEN CONCURRENCY TP KV_CACHE_DTYPE GPU_MEM_UTIL SPEC_MTP LIMIT CONDA_ENV LOG_DIR)
for v in "${PASS[@]}"; do [ -n "${!v:-}" ] && export "$v"; done
export REPO

CMD=(sngpu --nodelist "$NODE" --reservation "$RESERVATION" --gputype "$GPUTYPE"
     --gpu 8 --cpu "$CPU" --mem "$MEM" --time "$TIME" --jobname "$JOBNAME"
     --output "$LOG_DIR/${JOBNAME}_%j.out"
     --filepath "$REPO/examples/train/glm53/run_regen_glm53.sh")
printf '[submit] '; printf '%q ' "${CMD[@]}"; echo
[ -n "${DRY_RUN:-}" ] && exit 0
exec "${CMD[@]}"
