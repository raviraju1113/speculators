#!/bin/bash
#
# Submit the response-regeneration pipeline as a NON-INTERACTIVE sngpu job.
#
# It writes a small job script, then submits it with `sngpu --filepath`, which
# runs it inside the --image container (via boot_docker.sh). Inside, it
# activates the `speculators` conda env and runs run_all.sh unattended; the vLLM
# server is torn down automatically when regeneration finishes (run_all.sh's
# EXIT trap).
#
# Why --filepath and not --command: `sngpu` forwards its args with an UNQUOTED
# $@, which word-splits a --command string; the helper's `--command` (nargs=+)
# then stops at the first `--flag` inside it (e.g. --model), so wrapped commands
# that contain flags fail with "unrecognized arguments". A job script sidesteps
# this entirely — sngpu only ever sees a single file path.
#
# Usage:
#   # Defaults (gemma-4-31B-it, ultrachat, TP=4, outfile in repo root):
#   bash scripts/response_regeneration/submit_regen.sh
#
#   # Tweak the default run without respelling everything (env knobs):
#   LIMIT=5000 RESUME=1 bash scripts/response_regeneration/submit_regen.sh
#   MAX_MODEL_LEN=262144 bash scripts/response_regeneration/submit_regen.sh
#
#   # Full control — any args are forwarded VERBATIM to run_all.sh, replacing
#   # the default set entirely (you must then include --model yourself):
#   bash scripts/response_regeneration/submit_regen.sh \
#       --model Qwen/Qwen2.5-72B-Instruct --tp-size 4 --dataset magpie \
#       --limit 5000 --outfile /import/.../magpie_qwen.jsonl
#
#   # See the generated job script + sngpu command without submitting:
#   DRY_RUN=1 bash scripts/response_regeneration/submit_regen.sh
#
# Resource / environment knobs (override via env):
#   JOBNAME TIME CPU MEM GPU IMAGE NODELIST GPUTYPE OUTPUT JOBSCRIPT
#   CONDA_SH CONDA_ENV REPO_DIR HF_HOME
#   MODEL DATASET TP_SIZE OUTFILE MAX_MODEL_LEN LIMIT RESUME
#       (the run_* knobs are only used to build the DEFAULT arg set, i.e. when
#        no positional args are passed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

# --- Job resources (mirror the interactive allocation you already use) --------
JOBNAME="${JOBNAME:-regen_gemma}"
TIME="${TIME:-23:59:59}"
CPU="${CPU:-8}"
MEM="${MEM:-100000}"
GPU="${GPU:-4}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}"
NODELIST="${NODELIST:-sc3-c98}"
GPUTYPE="${GPUTYPE:-}"                        # optional; NODELIST already pins the node
OUTPUT="${OUTPUT:-$REPO_DIR/${JOBNAME}.out}"  # job stdout/stderr log
JOBSCRIPT="${JOBSCRIPT:-$SCRIPT_DIR/.job_${JOBNAME}.sh}"  # generated; read at run time

# --- In-container environment -------------------------------------------------
CONDA_SH="${CONDA_SH:-/import/ml-sc-scratch1/ravir/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-speculators}"
HF_HOME="${HF_HOME:-/import/ml-sc-scratch1/ravir/cache}"

# --- run_all.sh arguments -----------------------------------------------------
if [ "$#" -gt 0 ]; then
    # Power-user mode: forward args verbatim (you own the full arg list,
    # including the required --model).
    RUN_ARGS=("$@")
else
    # Default run, tunable via env knobs so the common tweaks (row cap, resume,
    # context length) don't force you to respell the whole arg list.
    MODEL="${MODEL:-google/gemma-4-31B-it}"
    DATASET="${DATASET:-ultrachat}"
    TP_SIZE="${TP_SIZE:-4}"
    OUTFILE="${OUTFILE:-$REPO_DIR/${DATASET}_$(basename "$MODEL").jsonl}"
    RUN_ARGS=(--model "$MODEL" --tp-size "$TP_SIZE" --dataset "$DATASET" --outfile "$OUTFILE")
    [ -n "${MAX_MODEL_LEN:-}" ] && RUN_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
    [ -n "${LIMIT:-}" ]         && RUN_ARGS+=(--limit "$LIMIT")
    [ "${RESUME:-0}" = "1" ]    && RUN_ARGS+=(--resume)
fi

# --- Generate the job script (runs INSIDE the container) ----------------------
# RESPONSE_REGEN_KILL_STALE=1 auto-reaps leftover vLLM workers from a prior run
# on this node instead of aborting (see run_all.sh pre-flight check). %q keeps
# every value correctly quoted regardless of special characters.
{
    printf '#!/bin/bash\n'
    printf 'set -e\n'
    printf 'export RESPONSE_REGEN_KILL_STALE=1\n'
    printf 'export HF_HOME=%q\n' "$HF_HOME"
    printf 'source %q\n' "$CONDA_SH"
    printf 'conda activate %q\n' "$CONDA_ENV"
    printf 'cd %q\n' "$REPO_DIR"
    printf 'bash scripts/response_regeneration/run_all.sh'
    printf ' %q' "${RUN_ARGS[@]}"
    printf '\n'
} > "$JOBSCRIPT"
chmod +x "$JOBSCRIPT"

# --- Assemble sngpu invocation ------------------------------------------------
SNGPU_ARGS=(
    --jobname "$JOBNAME"
    --time "$TIME"
    --cpu "$CPU"
    --mem "$MEM"
    --gpu "$GPU"
    --image "$IMAGE"
    --output "$OUTPUT"
)
[ -n "$NODELIST" ] && SNGPU_ARGS+=(--nodelist "$NODELIST")
[ -n "$GPUTYPE" ]  && SNGPU_ARGS+=(--gputype "$GPUTYPE")
SNGPU_ARGS+=(--filepath "$JOBSCRIPT")

echo "==============================================="
echo "Submitting response-regeneration job"
echo "  jobname   : $JOBNAME"
echo "  node      : ${NODELIST:-<scheduler choice>}   gpus: $GPU   time: $TIME"
echo "  image     : $IMAGE"
echo "  env       : $CONDA_ENV"
echo "  run_all   : ${RUN_ARGS[*]}"
echo "  jobscript : $JOBSCRIPT"
echo "  joblog    : $OUTPUT"
echo "==============================================="
echo "--- generated job script ---"
cat "$JOBSCRIPT"
echo "--- sngpu command ---"
printf 'sngpu'; printf ' %q' "${SNGPU_ARGS[@]}"; printf '\n\n'

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "(DRY_RUN=1 — job script written, not submitting)"
    exit 0
fi

sngpu "${SNGPU_ARGS[@]}"
echo ""
echo "Submitted. Watch progress with:  tail -f $OUTPUT"
