#!/usr/bin/env bash
# Turn the GLM-5.3 regenerated conversations into a speculators training set:
#   1. drop conversations with a reply cut by --max-tokens (metadata.truncated)
#      and blank the <think>...</think> block of every NON-final assistant turn,
#      which is exactly what the model saw while generating (the regeneration
#      requests use clear_thinking=true, as GLM is served in chat)
#   2. scripts/prepare_data.py with the GLM-5.3 tokenizer/chat template
#      -> arrow shards (input_ids, loss_mask over the assistant span incl. the
#         <think>...</think> block, seq_len) + token_freq.pt
#
# GLM-5.3's template emits no end-of-turn token after an assistant reply (the
# model stops on <|user|>, which the next user turn supplies), so the
# auto-detected assistant regex needs "<|user|>" after the span and never
# matches the LAST reply: loss_mask comes out all-zero. ASSISTANT_PATTERN below
# accepts end-of-text as the terminator too. Keep it unless the template changes.
#
# CPU only; fine on a login node. The first training run derives d2t/t2d from
# token_freq.pt (see README §1: a checkpoint is bound to that mapping).
#
#   REGEN_JSONL  input (default: data/glm53/nemo782k_regen_glm53.jsonl)
#   OUTPUT       dataset dir (default: /import/ml-sc-scratch1/mengmengj/datasets/glm53_nemotron_782k)
#   SEQ_LENGTH   max tokens per row (default 8192, same as the gemma data)
#   KEEP_TRUNCATED=1  keep cut replies instead of dropping them
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
MODEL="${MODEL:-/import/ml-sc-nlpcheckpoints-scratch3/mengmengj/glm-5.3-bf16}"
REGEN_JSONL="${REGEN_JSONL:-$REPO/data/glm53/nemo782k_regen_glm53.jsonl}"
OUTPUT="${OUTPUT:-/import/ml-sc-scratch1/mengmengj/datasets/glm53_nemotron_782k}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
WORKERS="${WORKERS:-32}"
ASSISTANT_PATTERN="${ASSISTANT_PATTERN:-<\\|assistant\\|>((?:(?!<\\|assistant\\|).)*?)(?=<\\|user\\|>|$)}"
PYTHON="${PYTHON:-/import/ml-sc-scratch1/mengmengj/condaenvs/dspark/bin/python}"

FILTERED="${REGEN_JSONL%.jsonl}.clean.jsonl"
if [ "${KEEP_TRUNCATED:-0}" = "1" ]; then
    FILTERED="$REGEN_JSONL"
else
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
        if row.get("metadata", {}).get("truncated"):
            dropped += 1
            continue
        conv = row.get("conversations") or []
        last_gpt = max((i for i, t in enumerate(conv) if t.get("from") == "gpt"), default=-1)
        for i, t in enumerate(conv):
            if t.get("from") == "gpt" and i != last_gpt and "</think>" in t.get("value", ""):
                # mirror clear_thinking=true: earlier turns keep only the answer
                t["value"] = t["value"].split("</think>", 1)[1].lstrip()
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        kept += 1
print(f"[prepare] kept {kept}, dropped {dropped} truncated, {bad} unparsable -> {dst}")
EOF
fi

cd "$REPO"
"$PYTHON" scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$FILTERED" \
    --seq-length "$SEQ_LENGTH" \
    --num-preprocessing-workers "$WORKERS" \
    --output "$OUTPUT" \
    --assistant-pattern "$ASSISTANT_PATTERN" \
    "$@"
echo "[prepare] dataset -> $OUTPUT"
ls -la "$OUTPUT" | head
