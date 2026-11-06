#!/bin/bash
#
# Submit eagle3 training as a NON-INTERACTIVE sngpu job.
#
# Why the two-layer structure:
#   sngpu has no --shm-size/--ipc knob, so running training directly in its
#   container exhausts /dev/shm as soon as the DataLoader workers pass the large
#   per-sample hidden-state tensors ("unable to allocate shared memory ...
#   Resource temporarily unavailable"). Instead we ask sngpu for BARE-NODE
#   resources (no --image), and the job itself launches the project docker via
#   cuda-docker-run-wrapper with --shm-size 16G --ipc=host -- exactly the
#   settings your interactive runs use. The container runs the training command
#   and exits (--rm, no TTY, stdin from /dev/null), which ends the job cleanly.
#
# Two generated (git-ignored) scripts:
#   .job_<name>.sh        outer: runs on the bare node, docker-runs the inner
#   .job_<name>_inner.sh  inner: runs in the container (conda + torchrun)
#
# Defaults reproduce the on-policy gemma run with the ONE next single-variable
# change: a 2-layer draft (--num-layers 2). Distinct save-path / run-name so the
# baseline is not overwritten. WANDB_API_KEY is read from your shell env and
# baked only into the inner script (git-ignored, redacted from stdout).
#
# Usage:
#   export WANDB_API_KEY=...            # rotate the one pasted in chat
#   bash scripts/submit_train.sh                 # on-policy + 2-layer
#   NUM_LAYERS=1 bash scripts/submit_train.sh    # finish 1-layer on-policy run
#   OFF_POLICY=1 NUM_LAYERS=1 bash scripts/submit_train.sh   # off-policy baseline
#   DRY_RUN=1 bash scripts/submit_train.sh       # preview, don't submit
#   # Power-user: forward a full train.py arg list VERBATIM (torchrun preamble
#   # is still added): bash scripts/submit_train.sh --num-layers 3 ...
#
# Knobs (env): JOBNAME TIME CPU MEM GPU NODELIST GPUTYPE OUTPUT
#   DOCKER_IMAGE NV_VISIBLE_DEVICES DOCKER_NET SHM_SIZE
#   CONDA_SH CONDA_ENV HF_HOME WANDB_PROJECT
#   NPROC CUDA_DEVICES RDZV_ENDPOINT
#   VERIFIER DATA_PATH HIDDEN_STATES_PATH NUM_LAYERS EPOCHS LR SEQ_LEN
#   DRAFT_ARCH NUM_WORKERS OFF_POLICY SAVE_PATH RUN_NAME LOG_DIR

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# --- sngpu (bare node -- NO --image) resources --------------------------------
JOBNAME="${JOBNAME:-train_gemma_onpolicy_2l}"
TIME="${TIME:-23:59:59}"
CPU="${CPU:-8}"
MEM="${MEM:-200000}"
GPU="${GPU:-4}"
NODELIST="${NODELIST:-sc3-c98}"
GPUTYPE="${GPUTYPE:-}"
OUTPUT="${OUTPUT:-$REPO_DIR/${JOBNAME}.out}"
JOBSCRIPT="${JOBSCRIPT:-$SCRIPT_DIR/.job_${JOBNAME}.sh}"
INNERSCRIPT="${INNERSCRIPT:-$SCRIPT_DIR/.job_${JOBNAME}_inner.sh}"

# --- docker container (big /dev/shm, like the interactive runs) ---------------
DOCKER_IMAGE="${DOCKER_IMAGE:-sc-artifacts2.sambanovasystems.com/sw-docker-scratch/speculators:ngc-24.12}"
NV_VISIBLE_DEVICES="${NV_VISIBLE_DEVICES:-all}"
DOCKER_NET="${DOCKER_NET:-host}"
SHM_SIZE="${SHM_SIZE:-16G}"

# --- in-container environment -------------------------------------------------
CONDA_SH="${CONDA_SH:-/import/ml-sc-scratch1/ravir/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-speculators}"
HF_HOME="${HF_HOME:-/import/ml-sc-scratch1/ravir/cache}"
WANDB_PROJECT="${WANDB_PROJECT:-eagle3-speculators}"
WANDB_API_KEY="${WANDB_API_KEY:?export WANDB_API_KEY before submitting (kept out of git)}"

# --- torchrun launcher --------------------------------------------------------
NPROC="${NPROC:-4}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-127.0.0.1:29500}"

# --- train.py arguments (default set = on-policy + 2-layer draft) -------------
VERIFIER="${VERIFIER:-google/gemma-4-31B-it}"
DATA_PATH="${DATA_PATH:-ultrachat_gemma4_31b_50k_seq_len_8k/}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-ultrachat_gemma4_31b_50k_seq_len_8k/hidden_states/}"
NUM_LAYERS="${NUM_LAYERS:-2}"                 # <-- the single experimental variable
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-4}"
SEQ_LEN="${SEQ_LEN:-8192}"
DRAFT_ARCH="${DRAFT_ARCH:-llama}"
NUM_WORKERS="${NUM_WORKERS:-12}"
OFF_POLICY="${OFF_POLICY:-0}"                 # 1 => add --use-off-policy-tokens
SAVE_PATH="${SAVE_PATH:-${DATA_PATH%/}/checkpoints_onpolicy_${NUM_LAYERS}layer}"
RUN_NAME="${RUN_NAME:-eagle3_gemma_onpolicy_${NUM_LAYERS}layer_{time}}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/speculators/output/wandb_logs}"

if [ "$#" -gt 0 ]; then
    TRAIN_ARGS=("$@")   # power-user: full train.py arg list verbatim
else
    TRAIN_ARGS=(
        --verifier-name-or-path "$VERIFIER"
        --data-path "$DATA_PATH"
        --hidden-states-path "$HIDDEN_STATES_PATH"
        --save-path "$SAVE_PATH"
        --epochs "$EPOCHS" --lr "$LR" --total-seq-len "$SEQ_LEN"
        --on-missing skip --draft-arch "$DRAFT_ARCH"
        --num-layers "$NUM_LAYERS" --num-workers "$NUM_WORKERS"
        --logger wandb --run-name "$RUN_NAME"
        --log-dir "$LOG_DIR"
    )
    [ "$OFF_POLICY" = "1" ] && TRAIN_ARGS+=(--use-off-policy-tokens)
fi

# --- Generate the INNER script (runs INSIDE the container) --------------------
{
    printf '#!/bin/bash\n'
    printf 'set -e\n'
    printf 'export HF_HOME=%q\n' "$HF_HOME"
    printf 'export WANDB_PROJECT=%q\n' "$WANDB_PROJECT"
    printf 'export WANDB_API_KEY=%q\n' "$WANDB_API_KEY"
    printf 'source %q\n' "$CONDA_SH"
    printf 'conda activate %q\n' "$CONDA_ENV"
    printf 'cd %q\n' "$REPO_DIR"
    printf 'CUDA_VISIBLE_DEVICES=%q torchrun --nnodes=1 --nproc_per_node=%q' \
        "$CUDA_DEVICES" "$NPROC"
    printf ' --rdzv-backend=c10d --rdzv-endpoint=%q --local-addr=127.0.0.1' \
        "$RDZV_ENDPOINT"
    printf ' scripts/train.py'
    printf ' %q' "${TRAIN_ARGS[@]}"
    printf '\n'
} > "$INNERSCRIPT"
chmod +x "$INNERSCRIPT"

# --- Generate the OUTER job script (runs on the BARE sngpu node) --------------
# cuda-docker-run-wrapper with a command (no -it) + --rm + stdin from /dev/null
# runs the inner script and the container exits when torchrun returns, ending
# the job. --shm-size/--ipc=host give the DataLoader workers real shared memory.
{
    printf '#!/bin/bash\n'
    printf 'set -e\n'
    printf 'sudo -g docker /usr/bin/cuda-docker-run-wrapper --rm \\\n'
    printf '  --gpus device=%q \\\n' "$NV_VISIBLE_DEVICES"
    printf '  --net=%q \\\n' "$DOCKER_NET"
    printf '  --ipc=host \\\n'
    printf '  --ulimit memlock=-1 \\\n'
    printf '  --ulimit stack=67108864 \\\n'
    printf '  --shm-size %q \\\n' "$SHM_SIZE"
    printf '  -w /workspace/speculators \\\n'
    printf '  %q \\\n' "$DOCKER_IMAGE"
    printf '  bash %q < /dev/null\n' "$INNERSCRIPT"
} > "$JOBSCRIPT"
chmod +x "$JOBSCRIPT"

# --- Assemble sngpu invocation (NO --image => bare node) ----------------------
SNGPU_ARGS=(
    --jobname "$JOBNAME"
    --time "$TIME"
    --cpu "$CPU"
    --mem "$MEM"
    --gpu "$GPU"
    --output "$OUTPUT"
)
[ -n "$NODELIST" ] && SNGPU_ARGS+=(--nodelist "$NODELIST")
[ -n "$GPUTYPE" ]  && SNGPU_ARGS+=(--gputype "$GPUTYPE")
SNGPU_ARGS+=(--filepath "$JOBSCRIPT")

echo "==============================================="
echo "Submitting eagle3 training job (bare node -> project docker)"
echo "  jobname   : $JOBNAME"
echo "  node      : ${NODELIST:-<scheduler choice>}   gpus: $GPU   time: $TIME"
echo "  docker    : $DOCKER_IMAGE   shm: $SHM_SIZE   ipc: host"
echo "  env       : $CONDA_ENV"
echo "  num_layers: $NUM_LAYERS   off_policy: $OFF_POLICY   workers: $NUM_WORKERS   epochs: $EPOCHS"
echo "  save_path : $SAVE_PATH"
echo "  run_name  : $RUN_NAME"
echo "  scripts   : $JOBSCRIPT  +  $INNERSCRIPT"
echo "  joblog    : $OUTPUT"
echo "==============================================="
echo "--- outer job script ---"
cat "$JOBSCRIPT"
echo "--- inner job script (WANDB_API_KEY redacted) ---"
sed 's/^export WANDB_API_KEY=.*/export WANDB_API_KEY=<redacted>/' "$INNERSCRIPT"
echo "--- sngpu command ---"
printf 'sngpu'; printf ' %q' "${SNGPU_ARGS[@]}"; printf '\n\n'

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "(DRY_RUN=1 — scripts written, not submitting)"
    exit 0
fi

sngpu "${SNGPU_ARGS[@]}"
echo ""
echo "Submitted. Watch progress with:  tail -f $OUTPUT"
