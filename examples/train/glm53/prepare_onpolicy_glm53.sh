#!/usr/bin/env bash
# On-policy GLM-5.3 draft data: Mengmeng's GLM-5.3 regeneration of nemo782k
# (reasoning_effort=max, T=1.0, top_p=0.95), tokenized for the Kimi K3 recipe
# (CPU only).
#
#   1. drop rows whose reply was cut by max_tokens (metadata.truncated) and keep
#      only the answer of every NON-final assistant turn (the regeneration ran
#      with clear_thinking=true, so that is what the model saw)
#   2. scripts/prepare_data.py with the GLM tokenizer at SEQ_LENGTH (16384)
#
# The rows are sharegpt {"from", "value"}; each reply is "<reasoning></think>
# <answer>" and the GLM template renders it as "<think>...</think>answer", so
# the loss covers the reasoning as well as the answer.
#
# Loss mask = LAST assistant turn only, with the same end-of-text regex as
# prepare_offpolicy_gemma.sh (GLM emits no end-of-turn token after a reply).
# Not prepare_glm53_data.sh: on the compute nodes (bash 5.2) its ASSISTANT_PATTERN
# default on line 32 is a syntax error ("command substitution: syntax error near
# unexpected token ?!").
#
#   REGEN_JSONL  input regen JSONL (required)
#   OUTPUT       dataset dir (required)
#   SEQ_LENGTH   default 16384
#   WORKERS      tokenizer processes (default 32)
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
MODEL="${MODEL:-/import/ml-sc-nlpcheckpoints-scratch3/mengmengj/glm-5.3-bf16}"
: "${REGEN_JSONL:?regen JSONL}"
: "${OUTPUT:?output dataset dir}"
SEQ_LENGTH="${SEQ_LENGTH:-16384}"
WORKERS="${WORKERS:-32}"
# Single-quoted: bash 5.2 (the sc3 compute nodes) fails to parse this regex as a
# "${VAR:-...}" default even when VAR is set; the login node's bash accepts it.
DEFAULT_ASSISTANT_PATTERN='<\|assistant\|>((?:(?!<\|assistant\|>).)*)$'
ASSISTANT_PATTERN="${ASSISTANT_PATTERN:-$DEFAULT_ASSISTANT_PATTERN}"
PYTHON="${PYTHON:-/import/ml-sc-scratch1/ravir/miniconda3/envs/dspark_sc3_train/bin/python}"
FILTERED="${FILTERED:-${OUTPUT}.clean.jsonl}"

mkdir -p "$(dirname "$FILTERED")"
if [ ! -s "$FILTERED" ]; then
    "$PYTHON" - "$REGEN_JSONL" "$FILTERED" <<'EOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
kept = dropped = bad = 0
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        conv = row.get("conversations") or []
        if row.get("metadata", {}).get("truncated") or not conv or conv[-1].get("from") != "gpt":
            dropped += 1
            continue
        for t in conv[:-1]:
            if t.get("from") == "gpt" and "</think>" in t.get("value", ""):
                t["value"] = t["value"].split("</think>", 1)[1].lstrip()
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        kept += 1
print(f"[prepare] kept {kept}, dropped {dropped} truncated/no-final-reply, {bad} unparsable -> {dst}")
EOF
fi

cd "$REPO"
PYTHONPATH="$REPO/src:${PYTHONPATH:-}" "$PYTHON" scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$FILTERED" \
    --seq-length "$SEQ_LENGTH" \
    --num-preprocessing-workers "$WORKERS" \
    --output "$OUTPUT" \
    --assistant-pattern "$ASSISTANT_PATTERN" \
    "$@"
echo "[prepare] dataset -> $OUTPUT"
ls -la "$OUTPUT" | head
