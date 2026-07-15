# Server MTP / EAGLE acceptance eval (SGLang & vLLM)

A lightweight, self-contained acceptance-rate + throughput evaluator for a
running speculative-decoding server. Unlike [`../evaluate.py`](../evaluate.py)
(which drives GuideLLM), this sends prompts directly to the server's
OpenAI-compatible streaming API and reads speculative-decoding metrics from the
server's Prometheus endpoint — no GuideLLM dependency, and it supports **both
SGLang and vLLM** backends.

## What it reports (per benchmark)

- **accept_length** — avg tokens committed per target forward pass (`1.0` = no
  speculation benefit).
- **accept_rate** — accepted / draft tokens.
- **decode_tok/s** — output tokens/sec excluding the first token (TTFT); the
  decode-phase speed speculative decoding actually accelerates.
- **e2e_tok/s, ttft** — end-to-end rate and time-to-first-token, for reference.

Benchmarks: **AIME**, **GPQA-Diamond**, **LiveCodeBench** (prepared prompts ship
in [`data/`](./data), one `{benchmark,id,prompt}` per line).

## SGLang vs vLLM — the one real difference

Both evaluators are identical except for **how acceptance is read** from
`/metrics`:

| Backend | Script | Metrics | How |
|---------|--------|---------|-----|
| SGLang | [`run_sglang_eval.py`](./run_sglang_eval.py) | `sglang:spec_accept_length/rate` (windowed gauges, reset each `decode_log_interval`) | polls `/metrics` in a background thread and averages window values |
| vLLM | [`run_vllm_eval.py`](./run_vllm_eval.py) | `vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}` (cumulative counters) | one before/after delta brackets the run |

Both write the same schema (`mtp_eval_summary.json` + `mtp_eval_details.jsonl`),
so [`compare_speedup.py`](./compare_speedup.py) works on either backend's output.

## Usage

```bash
# Launch your SGLang/vLLM server with speculative decoding + metrics enabled,
# then (from this directory):

BACKEND=vllm   BASE_URL=http://127.0.0.1:8000 ./run_eval.sh
BACKEND=sglang BASE_URL=http://127.0.0.1:8080 ./run_eval.sh

# knobs: NUM_SAMPLES, MAX_TOKENS, TEMPERATURE (default greedy=0), BENCHMARKS

# Compare a baseline (spec off) vs a spec-decoding run:
python compare_speedup.py baseline=./results/base/mtp_eval_summary.json \
                          spec=./results/spec/mtp_eval_summary.json
```

Run **greedy** (`temperature=0`) for the canonical acceptance setting.

## Regenerating data

Prepared prompts ship in `data/`. To refresh:

```bash
python prepare_data.py --only aime,livecodebench   # open datasets
python prepare_data.py --only gpqa                 # gated: `hf auth login` first
```

GPQA-Diamond comes from the **gated** `Idavidrein/gpqa` dataset (accept its terms
on Hugging Face before regenerating / redistributing).

## Provenance

Migrated from the MirrorMoEInfer `minimax` (SGLang, MiniMax-M2.7) and
`minimax_m3_vllm` (vLLM, MiniMax-M3) eval trees; the two evaluators were unified
onto shared data + a single `run_eval.sh` backend switch, and internal
infrastructure paths were genericized.
