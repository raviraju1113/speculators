# Server MTP / EAGLE acceptance eval (SGLang & vLLM)

A lightweight, self-contained acceptance-rate + throughput evaluator for a
running speculative-decoding server. Unlike [`../evaluate.py`](../evaluate.py)
(which drives GuideLLM), this sends prompts directly to the server's
OpenAI-compatible streaming API and reads speculative-decoding metrics from the
server's Prometheus endpoint — no GuideLLM dependency, and it supports **both
SGLang and vLLM** backends.

Only dependency beyond the stdlib is `requests` (plus `pandas` /
`huggingface_hub` if you regenerate datasets with `prepare_data.py`).

## What it reports (per benchmark)

- **accept_length** — avg tokens committed per target forward pass (`1.0` = no
  speculation benefit; upper bound is `num_spec_steps + 1`).
- **accept_rate** — accepted / draft tokens, in `[0, 1]`.
- **decode_tok/s** — output tokens/sec excluding the first token (TTFT); the
  decode-phase speed speculative decoding actually accelerates. **This is the
  number to compare against a spec-off baseline.**
- **e2e_tok/s, mean_ttft_s** — end-to-end rate and time-to-first-token, reference.

Benchmarks:

- **aime**, **gpqa** (GPQA-Diamond), **livecodebench** — static prompt sets run
  by `run_eval.sh`; prepared prompts ship in [`data/`](./data), one
  `{benchmark,id,prompt}` per line.
- **gsm8k**, **math500**, **humaneval**, **mbpp**, **mt-bench**, **aime26**,
  **swe-bench-pro**, **swe-rebench**, **aa-lcr** — derived from sibling
  [`../eval_datasets/`](../eval_datasets) turns files (generate AA-LCR /
  aime26 / swe-bench-pro / swe-rebench first; see
  [`../README.md`](../README.md#kimi-k3-dspark-acceptance-suite)).
- **speed-coding**, **speed-multilingual**, **speed-rag**, **speed-qa**,
  **speed-writing**, **speed-low-entropy** — NVIDIA SPEED-Bench slices via
  [`../prepare_speedbench.py`](../prepare_speedbench.py) then
  `prepare_data.py` (`SPEEDBENCH_DIR`).
- **AgentX** — an agentic **trace-replay load test** ([`run_agentx.sh`](./run_agentx.sh)),
  a different mode: it replays real Claude-Code traces at fixed concurrency
  rather than sending prompts from a file. See the [AgentX](#agentx-agentic-trace-replay-load-test)
  section below.

## Backends — the one real difference

Both evaluators are identical except for **how acceptance is read** from
`/metrics`:

| Backend | Script | Metrics | How |
|---------|--------|---------|-----|
| SGLang | [`run_sglang_eval.py`](./run_sglang_eval.py) | `sglang:spec_accept_length/rate` (windowed gauges, reset each `decode_log_interval`) | polls `/metrics` in a background thread and averages window values |
| vLLM | [`run_vllm_eval.py`](./run_vllm_eval.py) | `vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}` (cumulative counters) | one before/after delta brackets the run |

Both write the same schema, so [`compare_speedup.py`](./compare_speedup.py)
works on either backend's output.

## Prerequisites

Launch your server **with speculative decoding and metrics enabled** before
running the eval, e.g.:

- **vLLM:** serve with your `--speculative-config` and Prometheus enabled (the
  `/metrics` endpoint is on by default). Default port here: `8000`.
- **SGLang:** launch with the speculative flags **and** `--enable-metrics` (else
  `sglang:spec_accept_*` won't appear and acceptance is reported `n/a`). Default
  port here: `8080`.

If the spec-decode metrics aren't present the eval still runs and reports
throughput, with `accept_length`/`accept_rate` shown as `n/a` — that is exactly
how you run the **baseline** (spec off).

## Settings

### `run_eval.sh` environment knobs

| Env var | Default | Meaning |
|---------|---------|---------|
| `BACKEND` | `vllm` | `sglang` or `vllm` — selects the evaluator + metric reader |
| `BASE_URL` | `http://127.0.0.1:8000` | server root (the eval appends `/v1/...` and `/metrics`) |
| `BENCHMARKS` | `aime,gpqa,livecodebench` | comma-separated subset; also: `gsm8k,math500,humaneval,mbpp,mt-bench,aime26,swe-bench-pro,swe-rebench,aa-lcr,speed-coding,speed-multilingual,speed-rag,speed-qa,speed-writing,speed-low-entropy` |
| `NUM_SAMPLES` | `20` | prompts per benchmark (`0` = all) |
| `MAX_TOKENS` | `4096` | max generated tokens per request |
| `TEMPERATURE` | `0.0` | `0` = greedy (canonical acceptance setting) |
| `RESULT_DIR` | `./results` | output dir (→ `--output-dir`) |

### Direct CLI args (for knobs the wrapper doesn't expose)

Both evaluators: `--base-url --benchmarks --num-samples --max-tokens
--temperature --output-dir`. **SGLang-only:** `--poll-interval` (seconds between
`/metrics` polls, default `0.25`; keep it below one decode window so windows
aren't skipped). `prepare_data.py`: `--only`, `--lcb-version` (default
`release_v6`). `compare_speedup.py`: `label=path ...`, `--dir`, `--baseline`.

## Recipes by combination

All commands run from this directory (`scripts/evaluate/mtp_server_eval/`).

### A. Quick start

```bash
# vLLM server on :8000
BACKEND=vllm  BASE_URL=http://127.0.0.1:8000 ./run_eval.sh
# SGLang server on :8080
BACKEND=sglang BASE_URL=http://127.0.0.1:8080 ./run_eval.sh
```

### B. Speedup workflow (baseline vs spec-decoding) — the main use

Run the same benchmarks twice against two servers (or the same model with spec
on/off), write to separate dirs, then compare:

```bash
# 1) baseline: server launched WITHOUT speculative decoding
BACKEND=vllm BASE_URL=http://127.0.0.1:8000 RESULT_DIR=./results/base ./run_eval.sh

# 2) spec: server launched WITH the draft/MTP/EAGLE speculator
BACKEND=vllm BASE_URL=http://127.0.0.1:8001 RESULT_DIR=./results/spec ./run_eval.sh

# 3) speedup table (first config = baseline)
python compare_speedup.py \
    base=./results/base/mtp_eval_summary.json \
    spec=./results/spec/mtp_eval_summary.json
```

### C. Benchmark subsets

```bash
BENCHMARKS=aime                       ./run_eval.sh   # single
BENCHMARKS=aime,livecodebench         ./run_eval.sh   # skip gated GPQA
BENCHMARKS=aime,gpqa,livecodebench    ./run_eval.sh   # default three
# extra sets (generate once via prepare_data.py, then):
BENCHMARKS=gsm8k,math500,humaneval,mbpp                    ./run_eval.sh
BENCHMARKS=aime,gpqa,livecodebench,gsm8k,math500,humaneval,mbpp ./run_eval.sh
# Full multi-benchmark suite (prefer YAML — see §H / parent README):
BENCHMARKS=gsm8k,humaneval,mbpp,speed-coding,speed-multilingual,speed-rag,math500,speed-low-entropy,swe-bench-pro,aa-lcr,mt-bench,speed-qa,speed-writing,aime26 \
  NUM_SAMPLES=0 ./run_eval.sh
```

### D. Sampling / length settings

```bash
# more samples, longer generations (reasoning benchmarks)
NUM_SAMPLES=100 MAX_TOKENS=8192 ./run_eval.sh
# whole dataset per benchmark
NUM_SAMPLES=0 ./run_eval.sh
# non-greedy (acceptance drops; greedy=0 is canonical)
TEMPERATURE=0.6 ./run_eval.sh
```

### E. SGLang poll-interval (call the script directly)

`run_eval.sh` doesn't expose `--poll-interval`; invoke the SGLang evaluator when
you need to tune it (e.g. a small `decode_log_interval` needs faster polling):

```bash
python run_sglang_eval.py \
    --base-url http://127.0.0.1:8080 \
    --benchmarks aime,livecodebench \
    --num-samples 50 --max-tokens 8192 --temperature 0 \
    --poll-interval 0.1 \
    --output-dir ./results/sglang_fastpoll
```

### F. Comparing many configs / a sweep

```bash
# explicit labels (first = baseline)
python compare_speedup.py \
    base=./results/base/mtp_eval_summary.json \
    steps3=./results/steps3/mtp_eval_summary.json \
    steps5=./results/steps5/mtp_eval_summary.json

# auto-label by subdir; pick the baseline subdir by name
python compare_speedup.py --dir ./results/sweep --baseline base
```

### G. Regenerating datasets

Prepared prompts for the default three ship in `data/`. To refresh:

```bash
python prepare_data.py --only aime,livecodebench        # open datasets (network)
python prepare_data.py --only gpqa                      # gated: `hf auth login` first
python prepare_data.py --only livecodebench --lcb-version release_v5
# AIME source parquet is environment-specific; set AIME_PARQUET=... to rebuild it,
# otherwise the shipped data/aime.jsonl is used as-is.

# Turns-derived sets (first user turn as the prompt) from ../eval_datasets/:
python prepare_data.py --only gsm8k,math500,humaneval,mbpp,mt-bench,aime26,swe-bench-pro,swe-rebench,aa-lcr
```

Generate `aime26` / `swe-bench-pro` / `swe-rebench` / `aa-lcr` turns files first if missing:

```bash
python ../eval_datasets/convert_eval_datasets_to_jsonl.py MathArena/aime_2026
python ../eval_datasets/convert_eval_datasets_to_jsonl.py ScaleAI/SWE-bench_Pro
python ../eval_datasets/convert_eval_datasets_to_jsonl.py nebius/SWE-rebench
python ../prepare_aa_lcr.py
```

SPEED-Bench slices (set `SPEEDBENCH_DIR` if not `../speedbench_data`):

```bash
# Preferred (full fills): needs HF login for gated upstream sources used by NVIDIA prepare.py
python ../prepare_speedbench.py --data-dir ../speedbench_data \
    --download --configs qualitative,throughput_16k
SPEEDBENCH_DIR=../speedbench_data python prepare_data.py --only \
  speed-coding,speed-multilingual,speed-rag,speed-qa,speed-writing,speed-low-entropy
```

### H. Full multi-benchmark suite (YAML preferred)

For a complete acceptance + throughput sweep, use the YAML runner — see the
parent guide:
[How to run a full evaluation](../README.md#how-to-run-a-full-evaluation).

```bash
cd ../experiments
# edit full-eval.yaml (backbone / draft / GPUs / benchmarks)
./run_full_eval.sh --dry-run
./run_full_eval.sh
```

**Lower-level** (server already running; same benchmark list as `full-eval.yaml`):

```bash
BENCHMARKS=gsm8k,humaneval,mbpp,speed-coding,speed-multilingual,speed-rag,math500,speed-low-entropy,swe-bench-pro,aa-lcr,mt-bench,speed-qa,speed-writing,aime26 \
  NUM_SAMPLES=0 TEMPERATURE=0.0 BASE_URL=http://127.0.0.1:8000 \
  RESULT_DIR=./results/full_eval ./run_eval.sh
```

`full-eval.yaml` `eval:` block:

```yaml
eval:
  backend: vllm
  benchmarks:
    - gsm8k
    - humaneval
    - mbpp
    - speed-coding
    - speed-multilingual
    - speed-rag
    - math500
    - speed-low-entropy
    - swe-bench-pro
    - aa-lcr
    - mt-bench
    - speed-qa
    - speed-writing
    - aime26
  num_samples: 0          # all prompts
  max_tokens: 4096
  temperature: 0.0
```

Notes:

- `aa-lcr` prompts are ~tens of k tokens; raise server `max_model_len` accordingly.
- Without `HF_TOKEN`, SPEED-Bench may only materialise public non-placeholder
  rows (counts below the card). Re-run NVIDIA prepare after `huggingface-cli login`
  for full fills.
- `speed-low-entropy` uses SPEED-Bench `throughput_16k` / `low_entropy` (card
  wording: “low-entropy, 10k input”).

## AgentX (agentic trace-replay load test)

[`run_agentx.sh`](./run_agentx.sh) is the fourth benchmark and a **different mode
of evaluation**: instead of sending prompts from a file, it drives SemiAnalysis's
[InferenceX](https://github.com/SemiAnalysisAI/InferenceX) `trace_replay_tester.py`
to replay real **Claude-Code agentic traces** against your server at a fixed
concurrency. This measures speculative-decoding value under realistic
long-context, multi-user load (where acceptance/throughput behave differently
than at concurrency 1).

Like `run_eval.sh`, it targets a **server you launch yourself** (spec on or off)
— it does not manage the server. It sweeps one axis, **concurrency**
(`USERS_LIST`), and reads acceptance off `/metrics` using the same backend split
(SGLang windowed gauges vs vLLM cumulative counters).

> **Network required:** on first run it clones InferenceX (the trace-replay
> client) and downloads the traces dataset (`semianalysisai/cc-traces-weka-042026`).

### AgentX settings

| Env var | Default | Meaning |
|---------|---------|---------|
| `BACKEND` | `vllm` | `sglang`/`vllm` — acceptance reader |
| `BASE_URL` | `http://127.0.0.1:8000` | server root |
| `USERS_LIST` | `1 8 16` | concurrency levels to sweep |
| `DURATION` | `300` | replay seconds per level (use `1800` for a real run) |
| `TEMPERATURE` | `0` | greedy for comparable acceptance |
| `MAX_CONTEXT` | `128000` | drop traces longer than this |
| `HF_DATASET` | `semianalysisai/cc-traces-weka-042026` | traces dataset |
| `RESULT_DIR` | `./results/agentx` | output dir |
| `AGENTX_DIR` / `AGENTX_BRANCH` / `AGENTX_REPO` | `./.agentx/InferenceX`, `chore/agentx-integration`, SemiAnalysis repo | client checkout |

> **Concurrency feasibility:** each request holds its full context in the KV
> cache, so the server holds only ~`max_total_num_tokens / MAX_CONTEXT` requests
> at once. Beyond that the cache thrashes and throughput collapses for *every*
> config — keep `USERS × MAX_CONTEXT` under the pool (or lower `MAX_CONTEXT` to
> study higher concurrency).

### AgentX recipes

```bash
# concurrency sweep against a vLLM server
BACKEND=vllm BASE_URL=http://127.0.0.1:8000 USERS_LIST="1 8 16" ./run_agentx.sh

# a real run: longer replay, against an SGLang server
BACKEND=sglang BASE_URL=http://127.0.0.1:8080 \
  USERS_LIST="1 8 16 24" DURATION=1800 ./run_agentx.sh

# baseline vs spec: run twice against the two servers, diff the matrices
BASE_URL=http://127.0.0.1:8000 RESULT_DIR=./results/agentx_base ./run_agentx.sh
BASE_URL=http://127.0.0.1:8001 RESULT_DIR=./results/agentx_spec ./run_agentx.sh
```

Each concurrency level writes `results/agentx/users<N>/result.row`; all levels are
collected into `results/agentx/matrix.tsv`
(`users  decode_tok_s  accept_len  accept_rate  out_tok_s`).

## Output files

Each `run_eval.sh` run writes into `--output-dir` / `RESULT_DIR`:

- `mtp_eval_summary.json` — one row per benchmark (the numbers above). Consumed by
  `compare_speedup.py`.
- `mtp_eval_details.jsonl` — one line per request (tokens, ttft, decode time).

`run_agentx.sh` writes `matrix.tsv` plus per-concurrency `users<N>/` dirs (raw
trace-replay output + `result.row`).

## Notes & caveats

- Run **greedy** (`temperature=0`) for canonical acceptance — the target accepts a
  draft token iff it matches argmax.
- SGLang acceptance assumes **concurrency 1** (one request at a time) so the mean
  over windows equals the token-weighted rate; this eval sends requests serially.
- **GPQA-Diamond** comes from the **gated** `Idavidrein/gpqa` dataset — accept its
  terms on Hugging Face before regenerating / redistributing.

## Provenance

Migrated from the MirrorMoEInfer `minimax` (SGLang, MiniMax-M2.7) and
`minimax_m3_vllm` (vLLM, MiniMax-M3) eval trees; the two prompt evaluators were
unified onto shared data + a single `run_eval.sh` backend switch, and the AgentX
trace-replay sweep was decoupled from its internal server-launch/reap harness
into a standalone `run_agentx.sh`. Internal infrastructure paths were genericized.
