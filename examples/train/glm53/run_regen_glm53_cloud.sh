#!/usr/bin/env bash
# Regenerate GLM-5.3 responses through SambaCloud (api.sambanova.ai) instead of
# a local vLLM server. Same prompt files, same sampling settings and the same
# resumable client as run_regen_glm53.sh, so the output rows have the identical
# layout ("<reasoning></think><answer>" in the gpt turn, metadata.truncated, ...).
#
# Rows produced here are a different serving stack (RDU numerics, possibly a
# different effort/template) from the bf16 vLLM rows: keep them in their own
# OUTFILE (default suffix _cloud) so they can be compared / excluded / weighted.
#
# Env vars:
#   SN_API_KEY      bearer token (required; read by the client via --api-key-env)
#   ENDPOINT        default https://api.sambanova.ai/v1/chat/completions
#   MODEL           default GLM-5.3 (SambaCloud model id)
#   INPUT_JSONL     prompts file (required)
#   OUTFILE         output JSONL (required)
#   REASONING_EFFORT max | high | medium | low (default max, as the GPU runs)
#   TEMPERATURE / TOP_P / MAX_TOKENS   default 1.0 / 0.95 / 16384 (as the GPU runs)
#   CONCURRENCY     parallel requests (default 32; raise once the rate limit is known)
#   MAX_RETRIES     retries on 429/5xx (default 8, exponential backoff, Retry-After honoured)
#   LIMIT           stop after N rows (probe runs)
#   CONDA_ENV       python env with aiohttp (default dspark)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENDPOINT="${ENDPOINT:-https://api.sambanova.ai/v1/chat/completions}"
MODEL="${MODEL:-GLM-5.3}"
INPUT_JSONL="${INPUT_JSONL:?set INPUT_JSONL}"
OUTFILE="${OUTFILE:?set OUTFILE}"
REASONING_EFFORT="${REASONING_EFFORT:-max}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_RETRIES="${MAX_RETRIES:-8}"
CONDA_ENV="${CONDA_ENV:-dspark}"
: "${SN_API_KEY:?set SN_API_KEY}"

PY="/import/ml-sc-scratch1/mengmengj/condaenvs/$CONDA_ENV/bin/python"
EXTRA_BODY=$(printf '{"temperature": %s, "top_p": %s, "chat_template_kwargs": {"reasoning_effort": "%s", "clear_thinking": true}}' \
  "$TEMPERATURE" "$TOP_P" "$REASONING_EFFORT")

echo "[regen-cloud] endpoint=$ENDPOINT model=$MODEL effort=$REASONING_EFFORT temp=$TEMPERATURE top_p=$TOP_P max_tokens=$MAX_TOKENS concurrency=$CONCURRENCY retries=$MAX_RETRIES"
echo "[regen-cloud] $INPUT_JSONL -> $OUTFILE"

CLIENT=("$PY" "$REPO/scripts/response_regeneration/script_multiendpoint.py"
  --endpoint "$ENDPOINT" --skip-endpoint-validation
  --api-key-env SN_API_KEY --inline-reasoning --max-retries "$MAX_RETRIES"
  --model "$MODEL" --input-jsonl "$INPUT_JSONL" --outfile "$OUTFILE"
  --concurrency "$CONCURRENCY" --max-tokens "$MAX_TOKENS"
  --extra-body "$EXTRA_BODY" --skip-sources "" --resume)
[[ -n "${LIMIT:-}" ]] && CLIENT+=(--limit "$LIMIT")

exec "${CLIENT[@]}"
