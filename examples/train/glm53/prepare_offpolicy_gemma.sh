#!/usr/bin/env bash
# Off-policy GLM-5.3 draft data: the gemma-4-31B-it answers of nemo782k,
# re-tokenized with the GLM-5.3 chat template (CPU only).
#
#   1. build_offpolicy_gemma_jsonl.py: arrow `messages` -> conversations JSONL
#   2. scripts/prepare_data.py with the GLM tokenizer at SEQ_LENGTH (16384, the
#      Kimi K3 recipe's max_seq_length) -> arrow shards + token_freq.pt
#
# Loss mask = LAST assistant turn only (Kimi K3 recipe: last_turn_loss_only).
# GLM's template emits no end-of-turn token after a reply, so the final reply
# runs to end-of-text: the pattern below can only match the last
# <|assistant|> span (an earlier one cannot reach `$` without crossing the
# next <|assistant|>). The span starts with the empty "<think></think>" the
# template renders for a reply without reasoning_content, then the answer.
# The template also injects "<|system|>Reasoning Effort: Max" (the serving
# default) ahead of every conversation.
#
#   SRC_DATASET  gemma arrow dir (default: Mengmeng's kimi_nemotron_782k)
#   WORK         output root (default: /import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2)
#   SEQ_LENGTH   default 16384
#   WORKERS      tokenizer processes (default 32)
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
MODEL="${MODEL:-/import/ml-sc-nlpcheckpoints-scratch3/mengmengj/glm-5.3-bf16}"
SRC_DATASET="${SRC_DATASET:-/import/ml-sc-scratch1/mengmengj/datasets/kimi_nemotron_782k}"
WORK="${WORK:-/import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2}"
JSONL="${JSONL:-$WORK/data/gemma782k_offpolicy.jsonl}"
OUTPUT="${OUTPUT:-$WORK/data/gemma782k_glm53_s16k}"
SEQ_LENGTH="${SEQ_LENGTH:-16384}"
WORKERS="${WORKERS:-32}"
ASSISTANT_PATTERN="${ASSISTANT_PATTERN:-<\\|assistant\\|>((?:(?!<\\|assistant\\|>).)*)\$}"
PYTHON="${PYTHON:-/import/ml-sc-scratch1/ravir/miniconda3/envs/dspark_sc3_train/bin/python}"

mkdir -p "$(dirname "$JSONL")"
if [ ! -s "$JSONL" ]; then
    "$PYTHON" "$REPO/examples/train/glm53/build_offpolicy_gemma_jsonl.py" \
        --dataset "$SRC_DATASET" --output "$JSONL"
fi

cd "$REPO"
PYTHONPATH="$REPO/src:${PYTHONPATH:-}" "$PYTHON" scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$JSONL" \
    --seq-length "$SEQ_LENGTH" \
    --num-preprocessing-workers "$WORKERS" \
    --output "$OUTPUT" \
    --assistant-pattern "$ASSISTANT_PATTERN" \
    "$@"
echo "[prepare] dataset -> $OUTPUT"
ls -la "$OUTPUT" | head
