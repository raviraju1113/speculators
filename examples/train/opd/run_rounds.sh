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
# (k=12 at the default seed needs --max-num-batched-tokens: vLLM 0.26 computes
# a negative scheduled-token budget otherwise -- the k=12 eval hit it first.)
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
MAX_TOKENS="${MAX_TOKENS:-2048}"
CONC="${CONC:-16}"
# Seed = the current best SFT checkpoint (README §0): block-12 warm start of
# conv ep25, served at k=12. Pilots/sweeps before 2026-10-26 used conv ep25
# (block 8): SEED_CKPT=.../gemma4_31b_dspark_nemo782k_conv/checkpoints/25 SEED_EPOCH=25.
SEED_CKPT="${SEED_CKPT:-/import/ml-sc-scratch1/mengmengj/output/gemma4_31b_dspark_nemo782k_conv_blk12/checkpoints/33}"
SEED_EPOCH="${SEED_EPOCH:-33}"
# Block size follows the seed's config.json; rollouts draft K = block size and
# training masks that many slots (a block-8 driver default on a block-12 seed
# would silently train 8 slots). gamma = block size as in the block-12 run.
SEED_BLOCK=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('block_size', 8))" "$SEED_CKPT/config.json")
export BLOCK_SIZE="${BLOCK_SIZE:-$SEED_BLOCK}" DECAY_GAMMA="${DECAY_GAMMA:-$SEED_BLOCK}"
K="${K:-$BLOCK_SIZE}"
VOCAB_DATA="${VOCAB_DATA:-/import/ml-sc-scratch1/mengmengj/datasets/kimi_nemotron_782k}"   # token_freq + d2t/t2d of the seed
RUN_NAME="${RUN_NAME:-opd_rounds_${SLURM_JOB_ID}}"
export OUT_ROOT="${OUT_ROOT:-/import/ml-sc-scratch1/mengmengj/output_opd}"
WORK="$OUT_ROOT/$RUN_NAME"
mkdir -p "$WORK/rounds" "$REPO/logs"

echo "=== job $SLURM_JOB_ID on $(hostname) $(date '+%F %T') ==="
echo "run $RUN_NAME: $ROUNDS rounds x $P prompts, k=$K, block $BLOCK_SIZE gamma $DECAY_GAMMA, seed $SEED_CKPT (epoch $SEED_EPOCH)"
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
  # Data-parallel rollouts: one server per allocated GPU (ROLL_DP of them),
  # the round's prompts split into equal shards. One server on one GPU left the
  # other seven idle for ~2.7 h per 16k-prompt round (pilot 13936853).
  # nvidia-smi lists every physical card and SLURM only sets
  # CUDA_VISIBLE_DEVICES, so GPUs are taken from that variable, never by index.
  IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(( NGPU - 1 )))}"
  ROLL_DP="${ROLL_DP:-${#GPUS[@]}}"; [ "$ROLL_DP" -gt "${#GPUS[@]}" ] && ROLL_DP=${#GPUS[@]}
  # offset the pool per round so rounds see different prompts when P < pool size
  OFF=$(( (r * P) % $(wc -l < "$PROMPTS") ))
  tail -n +$(( OFF + 1 )) "$PROMPTS" | head -n "$P" > "$RDIR/prompts.jsonl"
  [ "$(wc -l < "$RDIR/prompts.jsonl")" -lt "$P" ] && head -n $(( P - $(wc -l < "$RDIR/prompts.jsonl") )) "$PROMPTS" >> "$RDIR/prompts.jsonl"
  split -n "l/$ROLL_DP" -d -a 1 "$RDIR/prompts.jsonl" "$RDIR/prompts.part"
  SRVS=()
  for s in $(seq 0 $(( ROLL_DP - 1 ))); do
    mkdir -p "$RDIR/engine_log/s$s" "$RDIR/s$s"
    DSPARK_ROLLOUT_LOG="$RDIR/engine_log/s$s" CUDA_VISIBLE_DEVICES="${GPUS[$s]}" vllm serve "$MODEL" \
      --host 127.0.0.1 --port $(( PORT + s )) --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.95 --max-model-len $(( MAX_TOKENS + 2048 )) --max-num-seqs "$CONC" \
      --max-num-batched-tokens 16384 \
      --speculative-config "{\"model\": \"$CK/$LAST\", \"num_speculative_tokens\": $K, \"method\": \"dspark\"}" \
      --enforce-eager --no-enable-prefix-caching --no-async-scheduling \
      --chat-template "$CHAT_TEMPLATE" > "$RDIR/server_s$s.log" 2>&1 &
    SRVS+=($!)
  done
  for s in $(seq 0 $(( ROLL_DP - 1 ))); do
    for _ in $(seq 1 120); do
      curl -sf "http://127.0.0.1:$(( PORT + s ))/health" >/dev/null 2>&1 && break
      kill -0 "${SRVS[$s]}" 2>/dev/null || { echo "!! rollout server s$s died"; tail -20 "$RDIR/server_s$s.log"; exit 1; }
      sleep 15
    done
    curl -sf "http://127.0.0.1:$(( PORT + s ))/health" >/dev/null || { echo "!! rollout server s$s not healthy" >&2; exit 1; }
  done
  echo "rollout: $ROLL_DP servers up $(date '+%T')"
  PIDS=()
  for s in $(seq 0 $(( ROLL_DP - 1 ))); do
    python scripts/opd/rollout.py --base-url "http://127.0.0.1:$(( PORT + s ))" --model "$MODEL" \
      --prompts "$RDIR/prompts.part$s" --concurrency "$CONC" --max-tokens "$MAX_TOKENS" \
      --temperature 0 --tag "r${r}s$s" --out "$RDIR/s$s" > "$RDIR/s$s/rollout.log" 2>&1 &
    PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do wait "$pid" || echo "!! some rollout requests failed (continuing)"; done
  for pid in "${SRVS[@]}"; do kill "$pid" 2>/dev/null; done; wait "${SRVS[@]}" 2>/dev/null; sleep 5
  : > "$RDIR/rollouts.jsonl"
  for s in $(seq 0 $(( ROLL_DP - 1 ))); do
    python scripts/opd/assemble_rollouts.py --log-dir "$RDIR/engine_log/s$s" --responses "$RDIR/s$s/responses.jsonl" \
      --out "$RDIR/s$s/rollouts.jsonl" --draft-ckpt "$CK/$LAST" --round "$r" > "$RDIR/s$s/assemble.txt" || { echo "!! assemble failed for shard s$s"; tail -5 "$RDIR/s$s/assemble.txt"; exit 1; }
    cat "$RDIR/s$s/rollouts.jsonl" >> "$RDIR/rollouts.jsonl"
  done
  python - "$RDIR/rollouts.jsonl" "$K" <<'PY' | tee "$RDIR/assemble.txt"
import json, sys
n = steps = acc = full = 0
for line in open(sys.argv[1]):
    r = json.loads(line); n += 1
    for st in r["steps"]:
        steps += 1; acc += st["accepted"]; full += st["accepted"] >= int(sys.argv[2])
k = int(sys.argv[2])
print(f"round rollouts: {n}  draft steps: {steps}  mean accepted/step: {acc/max(steps,1):.3f}  "
      f"accept_rate: {acc/max(steps*k,1):.4f}  full blocks: {full/max(steps,1):.3f}  (k={k})")
PY
  # KEEP_FULL=1 (default): keep fully-accepted blocks. The sweep (PLAN.md,
  # 2026-10-22) showed dropping them is clearly worse: they are the forward-KL
  # anchoring signal; the paper keeps every anchor.
  python scripts/opd/rollouts_to_dataset.py --rollouts "$RDIR/rollouts.jsonl" --model "$MODEL" \
    --token-freq "$VOCAB_DATA/token_freq.pt" --output "$RDIR/dataset" \
    $( [ "${KEEP_FULL:-1}" = "1" ] && echo --keep-full-accept ) | tail -2
  fi

  echo "=== round $r: train one epoch on $RDIR/dataset $(date '+%T') ==="
  conda activate "$TRAIN_ENV"
  export VENV= CONDA_SH CONDA_ENV="$TRAIN_ENV"
  export DATA_PATH="$RDIR/dataset" RUN_NAME
  export EPOCHS=$(( SEED_EPOCH + r + 2 ))
  export MODE=full SAMPLE_FROM_ANCHOR=1 DSPARK_CONV_KERNEL="${DSPARK_CONV_KERNEL:-2}" DSPARK_CONV_GROUP=16
  export ACCUM_STEPS="${ACCUM_STEPS:-1}" SCHED_TOTAL="${SCHED_TOTAL:-5300}" SCHEDULER_TYPE="${SCHEDULER_TYPE:-cosine}"
  # LR: the paper's 3e-4 (24-prompt batches) wrecked our draft at ~10 prompts
  # per step (pilot 12492724); 2e-5 at ACCUM 4 was stable but moved nothing in
  # 91 steps. Default 5e-5 for the next run; scale with the effective batch.
  export WARMUP_RATIO="${WARMUP_RATIO:-0.05}" LR="${LR:-5e-5}"
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
