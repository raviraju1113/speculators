#!/usr/bin/env bash
# AgentX concurrency sweep for the Gemma-4-31B full-suite drafts.
# Serves each config (TP=4), then run_agentx.sh at USERS=1/8/16.
#
# Usage (from this directory, in tmux/screen):
#   ./run_gemma4_31b_agentx.sh
#   ./run_gemma4_31b_agentx.sh --dry-run
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

DRY=()
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY=(--dry-run)
fi

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH" PYTHONUNBUFFERED=1

SPECULATOR_PY="${SPECULATOR_PY:-/root/miniconda3/envs/speculator/bin/python}"
VLLM028_PY="${VLLM028_PY:-/nvmedata/chenw/envs/speculator-vllm028/bin/python}"

echo "==> AgentX sweep (0.24: baseline + 3 drafts, then 0.28 DSpark)"
echo "    logs: $HERE/results/gemma4-31b-agentx/run.log"

mkdir -p "$HERE/results/gemma4-31b-agentx"

echo "==> [1/2] vLLM 0.24 (conda speculator)"
export PATH="$(dirname "$SPECULATOR_PY"):$PATH"
"$SPECULATOR_PY" run_experiments.py --config gemma4-31b-agentx.yaml "${DRY[@]}"

echo "==> [2/2] vLLM 0.28 DSpark"
export PATH="$(dirname "$VLLM028_PY"):$PATH"
export FLASHINFER_DISABLE_VERSION_CHECK=1
"$VLLM028_PY" run_experiments.py --config gemma4-31b-agentx-dspark.yaml "${DRY[@]}"

if [[ ${#DRY[@]} -gt 0 ]]; then
  exit 0
fi

echo "==> combined matrix"
"$SPECULATOR_PY" - <<'PY'
from pathlib import Path
root = Path("results/gemma4-31b-agentx")
order = ["baseline", "assistant_k5", "eagle3_k5", "redhat_ft_k5", "dspark_k8"]
labels = {
    "baseline": "baseline",
    "assistant_k5": "Google Assistant (MTP) k=5",
    "eagle3_k5": "Eagle-3 Qwen k=5",
    "redhat_ft_k5": "Eagle-3 Llama k=5",
    "dspark_k8": "DSpark Qwen k=8",
}
parsed = []
header = None
for name in order:
    p = root / name / "matrix.tsv"
    if not p.exists():
        print(f"!! missing {p}")
        continue
    rows = {}
    for line in p.read_text().splitlines():
        cols = line.split("\t")
        if cols[0] == "users":
            header = cols
            continue
        rows[cols[0]] = cols
    parsed.append((labels[name], rows))
if not parsed or header is None:
    raise SystemExit("no AgentX matrices found")
metrics = header[1:]
users = []
seen = set()
for _, rows in parsed:
    for u in rows:
        if u not in seen:
            seen.add(u)
            users.append(u)
out_lines = []
out_lines.append("users\t" + "\t".join(f"{n}:{m}" for n, _ in parsed for m in metrics))
for u in users:
    cells = [u]
    for _, rows in parsed:
        row = rows.get(u, [])
        vals = row[1:] if row else ["NA"] * len(metrics)
        while len(vals) < len(metrics):
            vals.append("NA")
        cells.extend(vals[: len(metrics)])
    out_lines.append("\t".join(cells))
text = "\n".join(out_lines) + "\n"
(root / "comparison.tsv").write_text(text)
print(text)
print(f"wrote {root / 'comparison.tsv'}")
PY
