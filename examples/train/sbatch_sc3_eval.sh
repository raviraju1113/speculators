#!/bin/bash
# sc3-cluster twin of sbatch_eval.sh: standard-benchmark eval (aime/gpqa/lcb)
# for any run_experiments.py yaml in scripts/evaluate/experiments/. 1 B200.
# ~1.5 h per experiment entry (full datasets: aime 30 / gpqa 198 / lcb 1055).
#
#   cd /import/ml-sc-scratch1/mengmengj/speculators
#   CONFIG=gemma4-31b-dspark-conv-ep25.yaml sbatch examples/train/sbatch_sc3_eval.sh
#   ONLY=conv_ep25_k8 CONFIG=... sbatch ...      # subset of experiment names
#
# Differences from the lax01 version: /import paths, conda eval env instead of
# the dspark-host venv, GPU self-test, conv-servability gate (sc3_eval_prelude.sh).
#SBATCH -N 1
#SBATCH --partition=gpuonly
#SBATCH --gres=gpu:b200m180:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
# GPU 2 on these three needs a root reset (2026-10-15); 1-GPU jobs can land on it.
#SBATCH --exclude=sc3-cp78,sc3-cp79,sc3-cp80
#SBATCH -J dspark_eval
#SBATCH --chdir=/import/ml-sc-scratch1/mengmengj/speculators
#SBATCH -o logs/%x-%j.out

# shellcheck disable=SC1091
# REPO may point at a worktree (e.g. speculators-glm): submit with REPO=<path> sbatch --chdir=<path> ...
source "${REPO:-/import/ml-sc-scratch1/mengmengj/speculators}/examples/train/sc3_eval_prelude.sh"

CONFIG="${CONFIG:-gemma4-31b-dspark-conv-ep25.yaml}"
CFG="$REPO/scripts/evaluate/experiments/$CONFIG"
[ -f "$CFG" ] || { echo "!! no such config: $CFG" >&2; exit 2; }

# Gate every draft the config will serve.
for d in $(python -c "import yaml,sys; [print(e['draft']) for e in yaml.safe_load(open(sys.argv[1]))['experiments'] if e.get('draft')]" "$CFG" | sort -u); do
  [ -d "$d" ] || { echo "!! draft checkpoint missing: $d (training not finished?)" >&2; exit 3; }
  check_conv_servable "$d" || exit $?
done

cd "$REPO/scripts/evaluate/experiments"
ONLY_ARGS=(); [ -n "${ONLY:-}" ] && ONLY_ARGS=(--only "$ONLY")
# Unique port per job: SLURM packs several 1-GPU eval jobs onto one node, and
# two servers on 127.0.0.1:<same port> would make one eval score the other's draft.
PORT="${PORT:-$(( 8400 + SLURM_JOB_ID % 1000 ))}"
echo "server port: $PORT"
python run_experiments.py --config "$CONFIG" --port "$PORT" "${ONLY_ARGS[@]+"${ONLY_ARGS[@]}"}"
RC=$?
echo "=== eval done rc=$RC $(date '+%F %T') ==="
exit $RC
