#!/usr/bin/env bash
# Run a full speculative-decoding evaluation from YAML.
#
#   cd scripts/evaluate/experiments
#   # edit full-eval.yaml (backbone / draft / GPUs)
#   ./run_full_eval.sh --dry-run
#   ./run_full_eval.sh
#   ./run_full_eval.sh --only baseline,draft_k5
#
# Extra args are forwarded to run_experiments.py.
# See ../README.md → "How to run a full evaluation".
set -euo pipefail

cd "$(dirname "$0")"

CONFIG="${FULL_EVAL_CONFIG:-full-eval.yaml}"

if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi

if grep -q 'REPLACE_WITH_' "$CONFIG"; then
  echo "!! Edit $CONFIG: replace REPLACE_WITH_* placeholders for backbone/draft." >&2
  echo "   (Continuing anyway so --dry-run still works.)" >&2
fi

exec python run_experiments.py --config "$CONFIG" "$@"
