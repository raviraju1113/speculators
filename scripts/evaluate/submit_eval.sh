#!/bin/bash
# Submit an eval config as a BATCH job.
#
# WHY THIS EXISTS
#   Evals kept dying with the interactive session that launched them:
#   2026-08-05 the meeting run was killed by an 8 h --time limit partway through
#   the `baseline` experiment, leaving results/gemma4-31b-dspark-meeting/baseline/
#   with no summary. Batch + generous --time survives that, and survives you
#   closing the laptop.
#
#   Also enforces the driver guard. A run on sc3-c97 (driver 565) died with
#   "driver is too old (found version 12070)" on EVERY server -- 4 model loads
#   wasted before it gave up.
#
# USAGE
#   CONFIG=gemma4-31b-dspark-meeting.yaml bash scripts/evaluate/submit_eval.sh
#   CONFIG=... ONLY=baseline,redhat_k7   bash scripts/evaluate/submit_eval.sh
#   CONFIG=... DRY_RUN=1                 bash scripts/evaluate/submit_eval.sh
#
# Budget ~20-25 min per (experiment x 3 benchmarks) plus ~7 min per server load.
# The 4-experiment meeting config is ~2.5 h; TIME defaults well above that.
set -euo pipefail

REPO="${REPO:-/import/ml-sc-scratch1/mengmengj/speculators}"
CONDA="${CONDA:-/import/ml-sc-scratch1/mengmengj/condaenvs/dspark}"
CONFIG="${CONFIG:?set CONFIG=<file>.yaml (relative to scripts/evaluate/experiments)}"
ONLY="${ONLY:-}"
JOBNAME="${JOBNAME:-eval_$(basename "$CONFIG" .yaml)}"
GPU="${GPU:-1}"
GPUTYPE="${GPUTYPE:-a100m80}"
CPU="${CPU:-16}"
MEM="${MEM:-200000}"
TIME="${TIME:-12:00:00}"
# 565 drivers -- vLLM refuses to start on these.
EXCLUDE="${EXCLUDE:-sc-c96,sc3-c97,sc-c82}"
OUTPUT="${OUTPUT:-$REPO/logs/${JOBNAME}.txt}"

JOBSCRIPT="$REPO/scripts/evaluate/.job_${JOBNAME}.sh"
mkdir -p "$REPO/logs"

{
  printf '#!/bin/bash\nset -euo pipefail\n'
  printf 'echo "host: $(hostname)"\n'
  # Driver guard FIRST -- fail in seconds, not after four 62 GB model loads.
  printf 'DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)\n'
  printf 'echo "driver: $DRV"\n'
  printf 'if [ "${DRV%%%%.*}" -lt 580 ]; then\n'
  printf '  echo "FATAL: driver $DRV too old for this vLLM (needs 595.x). Not starting."; exit 1\n'
  printf 'fi\n'
  printf 'export PATH=%q/bin:$PATH\n' "$CONDA"
  printf 'export HF_HOME=%q\n' "${HF_HOME:-/import/snvm-sc-scratch1/mengmengj/hf_cache}"
  printf 'cd %q/scripts/evaluate/experiments\n' "$REPO"
  # The local algos.py patch is what makes sample_from_anchor=True checkpoints
  # decode correctly. Report its state so the log says which convention was live.
  printf '_VL=$(python -c %s)\n' "'import vllm,os;print(os.path.dirname(vllm.__file__))'"
  printf 'if grep -q sample_from_anchor "$_VL/transformers_utils/configs/speculators/algos.py"; then\n'
  printf '  echo "algos.py patch: APPLIED (checkpoint sample_from_anchor honoured)"\n'
  printf 'else\n'
  printf '  echo "algos.py patch: NOT APPLIED -- upstream hardcode; True checkpoints will read ~1.8"\n'
  printf 'fi\n'
  if [ -n "$ONLY" ]; then
    printf 'python run_experiments.py --config %q --only %q\n' "$CONFIG" "$ONLY"
  else
    printf 'python run_experiments.py --config %q\n' "$CONFIG"
  fi
  printf 'echo "=== results table ==="\n'
  printf '_OUT=$(python -c %s %q)\n' \
    "'import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))[\"output_dir\"])'" "$CONFIG"
  printf 'cat "$_OUT/results_table.md" 2>/dev/null || echo "(no results_table.md)"\n'
} > "$JOBSCRIPT"
chmod +x "$JOBSCRIPT"

SNGPU_ARGS=(--jobname "$JOBNAME" --partition gpuonly --time "$TIME"
            --cpu "$CPU" --mem "$MEM" --gpu "$GPU" --gputype "$GPUTYPE"
            --output "$OUTPUT" --exclude "$EXCLUDE" --filepath "$JOBSCRIPT")

echo "==============================================="
echo " jobname : $JOBNAME   gpus: $GPU   time: $TIME"
echo " config  : $CONFIG${ONLY:+   only: $ONLY}"
echo " log     : $OUTPUT"
echo "==============================================="
cat "$JOBSCRIPT"
echo "--- sngpu ---"; printf 'sngpu'; printf ' %q' "${SNGPU_ARGS[@]}"; echo

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "(DRY_RUN=1 -- script written, not submitted)"; exit 0
fi
sngpu "${SNGPU_ARGS[@]}"
echo "Submitted. tail -f $OUTPUT"
