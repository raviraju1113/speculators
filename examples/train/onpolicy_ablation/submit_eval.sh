#!/usr/bin/env bash
# Phase 1c: acceptance eval of both arms + controls, one experiment per GPU in parallel.
# run_experiments.py runs experiments sequentially on one GPU; here each experiment
# gets its own config copy (gpus=i, port 8000+i) and the speedup table is built at
# the end with baseline first. scripts/evaluate/submit_eval.sh is not usable: it
# hard-fails on the 565 driver that every non-reserved IDC node has.
#
#   sbatch -p gpuonly -w sc-c96 --exclusive --gres=gpu:a100m80:8 -c 64 --mem 400G \
#     -t 12:00:00 -J onpol_eval -o <abl>/logs/eval-%j.out \
#     examples/train/onpolicy_ablation/submit_eval.sh
set -euo pipefail
REPO="${REPO:-/import/ml-sc-scratch1/ravir/spec_decoding_work/speculators}"
PY_ENV="${PY_ENV:-/import/ml-sc-scratch1/ravir/miniconda3/envs/dspark_cu12}"
CONFIG="${CONFIG:-$REPO/scripts/evaluate/experiments/llama31-8b-onpolicy-ablation.yaml}"
EXPS="${EXPS:-baseline zlab_dflash off_policy on_policy}"
export HF_HOME="${HF_HOME:-/import/ml-sc-scratch1/ravir/cache}"

source "$(dirname "$PY_ENV")/../etc/profile.d/conda.sh"
conda activate "$PY_ENV"
cd "$REPO/scripts/evaluate/experiments"
OUT=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CONFIG")
mkdir -p "$OUT"
CFG_DIR="${SLURM_TMPDIR:-/tmp}/onpol_eval_cfgs"
mkdir -p "$CFG_DIR"

# fail fast on a missing draft checkpoint rather than after the baseline finishes
python - "$CONFIG" $EXPS <<'PY'
import os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
for e in cfg["experiments"]:
    d = e.get("draft")
    if e["name"] in sys.argv[2:] and d and d.startswith("/") and not os.path.isdir(d):
        sys.exit(f"missing draft for {e['name']}: {d}")
PY

i=0
pids=()
for exp in $EXPS; do
    c="$CFG_DIR/$exp.yaml"
    python - "$CONFIG" "$c" "$i" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
i = int(sys.argv[3])
cfg["gpus"] = str(i)
cfg["server"]["port"] = 8000 + i
yaml.safe_dump(cfg, open(sys.argv[2], "w"), sort_keys=False)
PY
    python run_experiments.py --config "$c" --only "$exp" > "$OUT/run_$exp.log" 2>&1 &
    pids+=($!)
    echo "launched $exp on GPU $i (port $((8000 + i)))"
    i=$((i + 1))
    sleep 20   # stagger weight loads
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done

args=()
for exp in $EXPS; do
    s="$OUT/$exp/mtp_eval_summary.json"
    if [ -f "$s" ]; then args+=("$exp=$s"); else echo "no summary for $exp"; rc=1; fi
done
python "$REPO/scripts/evaluate/mtp_server_eval/compare_speedup.py" "${args[@]}" | tee "$OUT/compare.txt"
exit $rc
