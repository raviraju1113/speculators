#!/bin/bash
# Draft-OPD round loop (phase 4). One SLURM allocation runs R rounds:
#
#   round r:  serve ckpt[r] (patched vLLM, spec decode, rollout logger)
#             -> rollout P prompts (temperature 0)      [1 GPU]
#             -> assemble + rollouts_to_dataset         [CPU]
#             -> train ONE epoch on that dataset with the online driver
#                (hidden-state vLLM + training ranks), resuming the run dir
#                so optimizer/scheduler/global_step carry across rounds
#                -> checkpoints/<SEED_EPOCH + r + 1> = ckpt[r+1]
#
# Mechanics: the run dir is seeded once with the SFT checkpoint's weights
# (model.safetensors + config.json, no optimizer -> fresh AdamW, warmup in
# round 0) as epoch SEED_EPOCH; round r sets EPOCHS=SEED_EPOCH+r+2 so the
# trainer runs exactly one epoch ("epoch SEED_EPOCH+r+1") on the round's data.
# SCHED_TOTAL is the optimizer-step budget of the WHOLE OPD stage (all rounds),
# so one global schedule spans the rounds (paper: cosine, 5.3K steps).
#
# Pilot (4 A100: rollout on GPU 0; training = 1 HS engine + 1 rank):
#   ROUNDS=2 PROMPTS=data/opd_prompts/pilot_1k.jsonl P=1000 \
#     sbatch -w sc3-c98 --gres=gpu:a100m80:4 examples/train/opd/run_rounds.sh
# Full (8 B200): ROUNDS=8 P=16000 VLLM_DP=4 TRAIN_GPUS_N=4 ACCUM_STEPS=... \
#     sbatch examples/train/opd/run_rounds.sh
#SBATCH -N 1
#SBATCH --partition=gpuonly
#SBATCH --gres=gpu:a100m80:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH -J opd_rounds
#SBATCH --chdir=/import/ml-sc-scratch1/mengmengj/speculators
#SBATCH -o logs/%x-%j.out

set -uo pipefail
export REPO=/import/ml-sc-scratch1/mengmengj/speculators
export MODEL=/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it
export CHAT_TEMPLATE="$MODEL/chat_template.jinja"
EVAL_ENV=/import/snvm-sc-scratch1/mengmengj/condaenvs/dspark-host       # patched vLLM (serving)
TRAIN_ENV=/import/ml-sc-scratch1/mengmengj/condaenvs/dspark             # training
CONDA_SH=/import/snvm-sc-scratch1/mengmengj/miniconda3/etc/profile.d/conda.sh

ROUNDS="${ROUNDS:-2}"
PROMPTS="${PROMPTS:?set PROMPTS (jsonl with turns)}"
P="${P:-1000}"                       # prompts per round (sampled by offset)
K="${K:-8}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
CONC="${CONC:-16}"
SEED_CKPT="${SEED_CKPT:-/import/ml-sc-scratch1/mengmengj/output/gemma4_31b_dspark_nemo782k_conv/checkpoints/25}"
SEED_EPOCH="${SEED_EPOCH:-25}"
VOCAB_DATA="${VOCAB_DATA:-/import/ml-sc-scratch1/mengmengj/datasets/kimi_nemotron_782k}"   # token_freq + d2t/t2d of the seed
RUN_NAME="${RUN_NAME:-opd_rounds_${SLURM_JOB_ID}}"
export OUT_ROOT="${OUT_ROOT:-/import/ml-sc-scratch1/mengmengj/output_opd}"
WORK="$OUT_ROOT/$RUN_NAME"
mkdir -p "$WORK/rounds" "$REPO/logs"

echo "=== job $SLURM_JOB_ID on $(hostname) $(date '+%F %T') ==="
echo "run $RUN_NAME: $ROUNDS rounds x $P prompts, k=$K, seed $SEED_CKPT (epoch $SEED_EPOCH)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
NGPU=$(nvidia-smi -L | wc -l)

# ---- seed the run dir with WEIGHTS ONLY (fresh optimizer) --------------------
CK="$WORK/checkpoints"
if ! ls -d "$CK"/[0-9]* >/dev/null 2>&1; then
  mkdir -p "$CK/$SEED_EPOCH"
  cp "$SEED_CKPT/model.safetensors" "$SEED_CKPT/config.json" "$CK/$SEED_EPOCH/"
  [ -f "$SEED_CKPT/config.py" ] && cp "$SEED_CKPT/config.py" "$CK/$SEED_EPOCH/"
  echo "seeded $CK/$SEED_EPOCH (weights only)"
fi

# shellcheck disable=SC1090
source "$CONDA_SH"

for (( r=0; r<ROUNDS; r++ )); do
  LAST=$(ls -d "$CK"/[0-9]* | xargs -n1 basename | sort -n | tail -1)
  RDIR="$WORK/rounds/r$r"; mkdir -p "$RDIR/engine_log"
  if [ -f "$RDIR/dataset/state.json" ]; then echo "round $r: dataset exists, skipping rollout"; else
  echo "=== round $r: rollout from checkpoints/$LAST $(date '+%T') ==="
  conda activate "$EVAL_ENV"
  PORT=$(( 21000 + (${SLURM_JOB_ID:-0} + r) % 9000 ))
  export DSPARK_ROLLOUT_LOG="$RDIR/engine_log"
  # First GPU of OUR allocation. nvidia-smi on these nodes lists every physical
  # card and SLURM only sets CUDA_VISIBLE_DEVICES, so a literal index would
  # land on another job's GPU (round-0 failure: 1.25 GiB free on cuda:0).
  ROLL_GPU="${CUDA_VISIBLE_DEVICES%%,*}"; ROLL_GPU="${ROLL_GPU:-0}"
  CUDA_VISIBLE_DEVICES="$ROLL_GPU" vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 --max-model-len $(( MAX_TOKENS + 2048 )) --max-num-seqs "$CONC" \
    --speculative-config "{\"model\": \"$CK/$LAST\", \"num_speculative_tokens\": $K, \"method\": \"dspark\"}" \
    --enforce-eager --no-enable-prefix-caching --no-async-scheduling \
    --chat-template "$CHAT_TEMPLATE" > "$RDIR/server.log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; kill -0 $SRV 2>/dev/null || { echo "!! rollout server died"; tail -20 "$RDIR/server.log"; exit 1; }; sleep 15; done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "!! rollout server not healthy" >&2; exit 1; }
  # offset the pool per round so rounds see different prompts when P < pool size
  OFF=$(( (r * P) % $(wc -l < "$PROMPTS") ))
  tail -n +$(( OFF + 1 )) "$PROMPTS" > "$RDIR/prompts.jsonl"; [ "$(wc -l < "$RDIR/prompts.jsonl")" -lt "$P" ] && head -n "$P" "$PROMPTS" >> "$RDIR/prompts.jsonl"
  python scripts/opd/rollout.py --base-url "http://127.0.0.1:$PORT" --model "$MODEL" \
    --prompts "$RDIR/prompts.jsonl" --limit "$P" --concurrency "$CONC" --max-tokens "$MAX_TOKENS" \
    --temperature 0 --tag "r$r" --out "$RDIR" || echo "!! some rollout requests failed (continuing)"
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 5
  unset DSPARK_ROLLOUT_LOG
  python scripts/opd/assemble_rollouts.py --log-dir "$RDIR/engine_log" --responses "$RDIR/responses.jsonl" \
    --out "$RDIR/rollouts.jsonl" --draft-ckpt "$CK/$LAST" --round "$r" | tee "$RDIR/assemble.txt"
  python scripts/opd/rollouts_to_dataset.py --rollouts "$RDIR/rollouts.jsonl" --model "$MODEL" \
    --token-freq "$VOCAB_DATA/token_freq.pt" --output "$RDIR/dataset" | tail -2
  fi

  echo "=== round $r: train one epoch on $RDIR/dataset $(date '+%T') ==="
  conda activate "$TRAIN_ENV"
  export VENV= CONDA_SH CONDA_ENV="$TRAIN_ENV"
  export DATA_PATH="$RDIR/dataset" RUN_NAME
  export EPOCHS=$(( SEED_EPOCH + r + 2 ))
  export MODE=full SAMPLE_FROM_ANCHOR=1 DSPARK_CONV_KERNEL="${DSPARK_CONV_KERNEL:-2}" DSPARK_CONV_GROUP=16
  export ACCUM_STEPS="${ACCUM_STEPS:-1}" SCHED_TOTAL="${SCHED_TOTAL:-5300}" SCHEDULER_TYPE="${SCHEDULER_TYPE:-cosine}"
  export WARMUP_RATIO="${WARMUP_RATIO:-0.05}" LR="${LR:-3e-4}"
  export MAX_ANCHORS="${MAX_ANCHORS:-512}" CHECKPOINT_FREQ=1 SAVE_BEST=0 TRAIN_DATA_RATIO="${TRAIN_DATA_RATIO:-0.98}"
  export ON_GENERATE=delete VLLM_TP=1 VLLM_DP="${VLLM_DP:-1}" TRAIN_GPUS_N="${TRAIN_GPUS_N:-1}"
  export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}" WANDB_MODE="${WANDB_MODE:-offline}" WANDB_PROJECT=gemma4-dspark-opd
  export WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_NAME}" WANDB_RESUME=allow
  rm -rf "/dev/shm/hidden_states_${RUN_NAME}"
  _RUN=$(mktemp "${TMPDIR:-/tmp}/dspark_driver_XXXXXX.sh"); cp "$REPO/examples/train/dspark_online_gemma4_31b.sh" "$_RUN"
  bash "$_RUN"; RC=$?; rm -f "$_RUN"
  [ "$RC" -eq 0 ] || { echo "!! training failed in round $r (rc=$RC)" >&2; exit "$RC"; }
  NEW=$(ls -d "$CK"/[0-9]* | xargs -n1 basename | sort -n | tail -1)
  echo "=== round $r done: checkpoints/$NEW $(date '+%T') ==="
done
echo "=== all $ROUNDS rounds done $(date '+%F %T'); final checkpoint: $CK/$(ls -d "$CK"/[0-9]* | xargs -n1 basename | sort -n | tail -1) ==="
