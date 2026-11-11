# Evaluating Model Performance

The evaluator uses AIPerf 0.13.0 for dataset loading, requests, concurrency sweeps, native reports, and plotting. Our adapters prepare SPEED-Bench, preserve MRCR histories, and summarize acceptance statistics.

## Install

From the repository root, use a client environment with Python 3.11–3.13:

```bash
pip install -r scripts/evaluate/requirements.txt
```

PNG plotting requires Chrome or Chromium on the machine generating the graphs. If neither is installed, install Chrome with:

```bash
plotly_get_chrome
```

## Serve the model

All examples use [RedHatAI/Qwen3.8-27B-speculator.dspark](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark) with the `Qwen/Qwen3.8-27B` verifier, 7 speculative tokens, and adaptive verification disabled.

```bash
vllm serve Qwen/Qwen3.8-27B \
  --port 9145 \
  --speculative-config '{"model":"RedHatAI/Qwen3.8-27B-speculator.dspark","num_speculative_tokens":7,"method":"dspark","enable_adaptive_verification":false}' \
  --per-request-spec-decode-metrics summary \
  --no-enable-prefix-caching
```

The evaluation commands use `--model Qwen/Qwen3.8-27B` to match the server's model name; the speculator is selected by `--speculative-config`. The experimental per-request response format must match AIPerf's adapter. The wrapper does not validate prefix-cache counters.

## Acceptance

Use `--dataset` to select datasets. When omitted, acceptance defaults to SPEED-Bench qualitative and speedup defaults to HumanEval.

```bash
python scripts/evaluate/evaluate.py acceptance \
  --dataset qualitative \
  --model Qwen/Qwen3.8-27B --url http://localhost:9145 \
  --output-artifact-dir output/qualitative
```

This prepares qualitative once and runs the aggregate split in one native profile. The current source contains 880 conversations across 11 categories and 1,138 user turns. Follow-up turns are generated in the same conversation. For category matrices, use the native sweep below.

Defaults are concurrency 64, timeout 900 seconds, sequential sampling, server-reported token counts, and raw exports. Generation defaults are temperature 1, thinking enabled, and `max_completion_tokens=32768`. Replace that settings object with `--extra-inputs '{"temperature":0,"max_completion_tokens":1024}'`.

Run several datasets sequentially:

```bash
python scripts/evaluate/evaluate.py acceptance \
  --dataset gsm8k math500 qualitative \
  --request-count 8 --concurrency 64 \
  --model Qwen/Qwen3.8-27B --url http://localhost:9145 \
  --output-artifact-dir output/acceptance-smoke
```

The budget applies separately to each dataset. Each dataset keeps native reports under `<output-dir>/<dataset>/`, with combined acceptance CSVs at the root. A native run failure retains its results, subsequent datasets run, and the command returns nonzero.

`gsm8k`, `math500`, `humaneval`, `mbpp`, and `mtbench` select the corresponding native `spec_al_*` public loaders. Native public dataset names also work. Their prompt templates and conversation handling are AIPerf's; MT-Bench includes both turns. Without an explicit budget, these public datasets use AIPerf's default ten requests. AIME26 and Arena-Hard have no equivalent registered presets in this version.

## SPEED-Bench selection

`qualitative` aliases `speed_bench_qualitative`. The wrapper does not expand it into categories. Select individual categories with native names, for example:

```bash
python scripts/evaluate/evaluate.py acceptance \
  --dataset speed_bench_coding speed_bench_math \
  --model Qwen/Qwen3.8-27B --url http://localhost:9145 \
  --output-artifact-dir output/coding-math
```

Each selected qualitative category defaults to all its conversations (currently 80) unless a budget is supplied. Explicitly selecting multiple categories runs them separately, which also supports category-specific speedup measurements.

Generate category reports from those runs:

```bash
aiperf speed-bench-report output/coding-math/speed_bench_* \
  --metric accept_length --format csv \
  --output output/coding-math/acceptance_length_by_category.csv
aiperf speed-bench-report output/coding-math/speed_bench_* \
  --metric accept_rate --format csv \
  --output output/coding-math/acceptance_rate_by_category.csv
```

Throughput selections use native names such as `speed_bench_throughput_8k_high_entropy`. They use native request budgets in acceptance mode and timed sweeps in speedup mode. Select the desired entropy tiers explicitly when separate reports are needed.

## MRCR

```bash
python scripts/evaluate/evaluate.py acceptance \
  --dataset mrcr \
  --dataset-filter needles=8 --dataset-filter max_samples=800 \
  --request-count 800 --concurrency 64 --request-timeout-seconds 3600 \
  --model Qwen/Qwen3.8-27B --url http://localhost:9145 \
  --output-artifact-dir output/mrcr
```

The registered MRCR plugin streams `openai/mrcr` from Hugging Face. Each complete history becomes one request, preserving system messages, user turns, and historical assistant responses. Earlier turns are not regenerated. No client tokenization, truncation, or context filtering occurs.

Select `needles=2`, `4`, or `8` and optionally `revision=<HF-commit>` with `--dataset-filter`. Defaults are eight needles and at most 100 source rows. `max_samples` limits source rows; AIPerf's request budget is separate and can replay them. The normal generation cap and `--extra-inputs` overrides also apply to MRCR. Over-context responses remain errors; this evaluates decoding statistics rather than answer accuracy.

## Acceptance results

- `acceptance.csv`: one row per native run, with pooled acceptance length, acceptance rate, and numeric position rates.
- `acceptance_length_by_category.csv` and `acceptance_rate_by_category.csv`: produced by the native `speed-bench-report` commands above.
- `acceptance_by_prompt_length.csv`: those metrics and request counts grouped by actual prompt-token length. Boundaries start at 8,192 and double through 1,048,576, with overflow and `unknown` bins.

Each profiling record is read once. Warmup and duplicate phase exports are excluded; follow-up turns and repeated conversations count as separate requests. Acceptance counters include successful responses with speculative metrics. Prompt-bin request counts also retain exported failures; missing token lengths appear under `unknown`. Unexported incomplete requests cannot be counted.

Counters are summed before division: acceptance length is `1 + accepted draft tokens / verification steps`; acceptance rate is `accepted draft tokens / proposed draft tokens`. Position k is the fraction of steps accepting at least k draft tokens. Rates are percentages and unavailable values are blank. These pooled values can differ from native means across requests.

Category matrices are generated by AIPerf 0.13.0's `speed-bench-report`, using each category run's `server_metrics_export.json`. Keep server metrics enabled and expose the server's `/metrics` endpoint (collection is enabled by default). Native acceptance rates are fractions from 0 to 1; the `Overall` column is the arithmetic mean of category values. These server-based matrices can differ from the request-based pooled statistics in `acceptance.csv`. Per-position rates for each category remain in that run's row of `acceptance.csv`. There is no custom category aggregation or request-ID lookup.

## Speedup and plotting

```bash
# SPEED-Bench qualitative
python scripts/evaluate/evaluate.py speedup \
  --dataset qualitative \
  --model Qwen/Qwen3.8-27B --url http://localhost:9145 \
  --output-artifact-dir output/speedup-qualitative

# Individual categories
python scripts/evaluate/evaluate.py speedup \
  --dataset speed_bench_coding speed_bench_math \
  --model Qwen/Qwen3.8-27B --url http://localhost:9145 \
  --output-artifact-dir output/speedup-categories

# HumanEval
python scripts/evaluate/evaluate.py speedup \
  --dataset humaneval \
  --model Qwen/Qwen3.8-27B --url http://localhost:9145 \
  --output-artifact-dir output/speedup-humaneval

# Generate plots
aiperf plot output/speedup-qualitative --output output/plots
```

## Server-side evaluators (`mtp_server_eval`, `experiments`)

These evaluators sit alongside the AIPerf evaluator above and are the ones used for this fork's model evaluations (Gemma-4, GLM-5.x, Kimi-K3). They send requests to a running server and read speculative-decoding counters from `/metrics`. The GuideLLM modes (`MODE=throughput`, `MODE=sweep`) also need `pip install guidellm~=0.7.1`.

### Quick Start

#### Easiest: config-driven experiments (recommended)

Describe the run in a YAML — start from the template
[`example.yaml`](https://github.com/vllm-project/speculators/blob/main/scripts/evaluate/experiments/example.yaml),
or a filled-in example
[`gemma4-31b.yaml`](https://github.com/vllm-project/speculators/blob/main/scripts/evaluate/experiments/gemma4-31b.yaml)
(Gemma 4 31B-it backbone + assistant draft) — and let
[`scripts/evaluate/experiments/run_experiments.py`](https://github.com/vllm-project/speculators/blob/main/scripts/evaluate/experiments/run_experiments.py)
do the rest: for each experiment it **launches vLLM (backbone ± draft), waits for
`/health`, runs the eval, stops the server**, then prints a speedup table (first
entry = baseline). No manual server management.

```bash
cd scripts/evaluate/experiments
# Edit example.yaml: set `backbone`, the `draft`(s), and `tensor_parallel_size`/`gpus`.
python run_experiments.py --config example.yaml --dry-run   # preview serve+eval commands, launch nothing
python run_experiments.py --config example.yaml             # baseline vs. each draft, then compare
```

Backbone / drafts / server (`tensor_parallel_size`, `gpus`) / eval settings all
live in the YAML; per-experiment blocks override the defaults. See the
[experiments README](https://github.com/vllm-project/speculators/blob/main/scripts/evaluate/experiments/README.md),
and for a full backbone-specific walkthrough (env, GPU layout, driver
requirements) the Gemma 4 31B-it example:
[`examples/train/gemma4_31b_README.md`](https://github.com/vllm-project/speculators/blob/main/examples/train/gemma4_31b_README.md).

### Measured results — Gemma-4-31B-it (single-GPU, forward-compat node)

Our trained eagle-3 draft (from scratch, on gemma-4-regenerated data, off-policy —
see [Train Eagle-3 for Gemma-4-31B-it](train_eagle3_online_gemma4_31b.md)),
evaluated with the `mtp_server_eval` evaluator via `run_experiments.py`.

**Setup:** single A100 80 GB, `tensor_parallel_size: 1`, `--enforce-eager`,
`max_model_len 8192`; benchmark `gsm8k`, `num_samples 20`, greedy
(`temperature 0.0`), `num_speculative_tokens: 3`. (Single-GPU because NCCL
segfaults under CUDA-13 forward compat — see the training tutorial.) Config:
`scripts/evaluate/experiments/gemma4-31b-regen-1gpu.yaml`.

Head-to-head vs. the two **official** drafts, all at k=3 in one run
(`scripts/evaluate/experiments/gemma4-31b-compare-1gpu.yaml`):

| draft | decode tok/s | accept_len | accept_rate | speedup |
|---|---|---|---|---|
| baseline (backbone only) | 18.0 | — | — | 1.00× |
| **ours** — eagle-3, regen 5k, 3 ep, k=3 | 40.2 | 2.38 | 0.46 | **2.23×** |
| official RedHat eagle-3 (`…-speculator.eagle3`) | 51.2 | 3.10 | 0.70 | 2.84× |
| official Google assistant (`…-it-assistant`, MTP) | 56.8 | 3.69 | 0.90 | 3.16× |

Reading it:
- Every draft roughly **doubles+ decode throughput**; the official drafts lead
  (Google assistant best at 3.16×, 90% acceptance).
- **Ours reaches 2.23×** — about **79%** of the official eagle-3's speedup and
  **71%** of the Google assistant's — from a *quick proof run*: only **5,000**
  regenerated samples, **3 epochs**, and a **reduced draft vocab (32k)**. The
  official drafts are fully trained (and the Google assistant is a larger 4-layer
  MTP head). Expect ours to close much of the gap with more data/epochs and full
  draft vocab. The reduced vocab in particular caps `accept_rate` (the draft
  can't propose out-of-vocab tokens).
- `accept_len` is out of a max of k+1 = 4 (the k speculated tokens + the target's
  always-accepted bonus token).

> **Note on absolute tok/s:** these are single-GPU, `--enforce-eager` numbers, so
> throughput is low in absolute terms; the **speedup ratio** is the meaningful
> figure. On a driver ≥580 you'd use multi-GPU + CUDA graphs for higher absolute
> throughput (ratios stay comparable).

### Measured results — GLM-5.2-FP8, native MTP (8×B200)

GLM-5.2 ships with **native multi-token prediction** built into the checkpoint
(`method=mtp`, `num_speculative_tokens=5`) — no separate draft model. Evaluated
with the `mtp_server_eval` evaluator via `run_experiments.py`.

**Setup:** 8×B200, `tensor_parallel_size: 8`, FP8 weights + FP8 KV cache, CUDA
graphs (not eager), `max_model_len 16384`; benchmarks `aime`, `gpqa`,
`livecodebench`, `num_samples 50`, greedy (`temperature 0.0`),
`num_speculative_tokens: 5`. Config:
`scripts/evaluate/experiments/glm52-eval.yaml`.

Baseline (backbone alone) vs. native MTP, per benchmark:

| benchmark | decode tok/s | e2e tok/s | TTFT (s) | accept_len (of k+1=6) | accept_rate | speedup (e2e) |
|---|---|---|---|---|---|---|
| **AIME** | | | | | | |
| baseline (no MTP) | 520.4 | 104.3 | 23.037 | — | — | 1.00× |
| + native MTP | 1251.7 | 249.3 | 10.089 | 4.54 | 0.708 | **2.39×** |
| **GPQA** | | | | | | |
| baseline (no MTP) | — | 104.2 | 10.073 | — | — | 1.00× |
| + native MTP | — | 182.7 | 4.469 | 3.33 | 0.466 | **1.75×** |
| **LiveCodeBench** | | | | | | |
| baseline (no MTP) | 779.1 | 103.8 | 10.802 | — | — | 1.00× |
| + native MTP | 1695.9 | 211.5 | 5.411 | 3.90 | 0.580 | **2.04×** |

> **Note on GPQA decode tok/s:** the GPQA baseline `decode_tok_s` is anomalously
> high (22 K tok/s) due to a measurement artifact on that run; `e2e tok/s` is
> reliable (104.2 tok/s) and used for the speedup calculation above.

Reading it:
- **Native MTP roughly doubles end-to-end throughput** — up to **2.39× on `aime`**,
  **2.04× on `livecodebench`**, and **1.75× on `gpqa`**.
- **TTFT is also significantly reduced** — AIME TTFT drops from 23s to 10s (+MTP),
  a 2.3× improvement. This is a key latency benefit of speculative decoding that
  is not captured by decode-throughput speedup alone.
- **Acceptance tracks workload predictability**: math (`aime`, 71%) > code
  (`livecodebench`, 58%) > science QA (`gpqa`, 47%). Higher acceptance → longer
  accepted runs (`accept_len = accept_rate × k + 1`) → larger speedup.
- `accept_len` is out of a max of k+1 = 6 (the 5 speculated tokens + the target's
  always-accepted bonus token).

#### KV cache dtype ablation (BF16 vs FP8)

A separate ablation run (`scripts/evaluate/experiments/glm52-kvcache-ablation.yaml`)
tested BF16 KV cache vs. FP8 KV cache on a subset of benchmarks. Results with
native MTP enabled:

| benchmark | KV dtype | decode tok/s | accept_len | accept_rate |
|---|---|---|---|---|
| AIME | FP8 | 1251.7 | 4.54 | 0.708 |
| AIME | BF16 | 246.2 | 4.62 | 0.725 |
| GSM8K | BF16 | 240.3 | 4.40 | 0.679 |
| Math500 | BF16 | 250.2 | 4.66 | 0.733 |

> **Note:** the FP8 and BF16 runs used different server configurations (cold vs.
> warm cache, different sampling parameters), so the raw `decode tok/s` numbers
> are not directly comparable. The accept length/rate numbers are consistent,
> showing that KV cache dtype does not significantly affect MTP acceptance
> quality.

> **Metric provenance:** acceptance comes from vLLM's cumulative Prometheus
> counters (`vllm:spec_decode_*`). The e2e and decode tok/s figures are from
> vLLM's server-side generation logs.

#### Manual: against a server you already run

If you already have a vLLM server up, run GuideLLM through `run_eval.sh`. Full
benchmark pipeline (output-length estimation → performance sweep → CSV):

```bash
cd scripts/evaluate/mtp_server_eval
MODE=sweep BASE_URL=http://localhost:8000 ./run_eval.sh
```

This runs all 9 subsets from `RedHatAI/speculator_benchmarks` and produces `perf_results.csv` under `RESULT_DIR` (default `./results`).

For acceptance rates only (skips the sweep):

```bash
MODE=throughput BASE_URL=http://localhost:8000 ./run_eval.sh
```

See [`examples/evaluate/`](https://github.com/vllm-project/speculators/tree/main/examples/evaluate) for end-to-end examples that launch a vLLM server and run the pipeline.

### Outputs and baseline runs

Both modes support a vLLM server running without speculative decoding. A successful baseline run does not require speculative counters and does not produce `acceptance.csv`. Throughput runs must contain at least one successful request; speculative decoding metrics are optional. Raw GuideLLM results are saved under `artifacts/` in both modes. In `sweep` mode, the run also produces `perf_results.csv` and `max_tokens.json` for comparison with a speculative run.

`perf_results.csv` contains performance metrics for each sweep load point. When speculative counters are available and drafts were generated, each row also carries the sweep's aggregate acceptance metrics (`num_drafts`, `num_accepted_tokens`, `acceptance_length`, etc.) — these values are measured across the entire sweep, not per load point, so they repeat unchanged across rows. The evaluator samples counters before and after the run and cannot attribute acceptance to individual load points. `acceptance.csv` additionally contains one aggregate acceptance row per subset: in `throughput` mode this covers the throughput run; in `sweep` mode it covers the entire sweep, excluding the preceding output-length estimation run.

### Options

`MODE=throughput` and `MODE=sweep` share these env vars (see `run_eval.sh`):

```
  BASE_URL                 server root (script appends /v1)
  DATASET                  HF dataset ID or local dir (default: RedHatAI/speculator_benchmarks)
  SUBSETS                  comma-separated subset names (default: all 9)
  RESULT_DIR               output directory
  MAX_CONCURRENCY          max concurrent requests (default: 128)
  MAX_REQUESTS             max requests per sweep point (default: 200)
  GEN_LEN_RATE             request rate for gen-len estimation (default: 128)
  SWEEP_RATE               number of sweep rate points (default: 10)
  GEN_KWARGS / TEMPERATURE generation kwargs
  SPEEDBENCH_DATA_DIR      required when DATASET=speedbench/...
```

### SPEED-Bench

[NVIDIA SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) provides structured evaluation across qualitative categories (coding, math, reasoning, multilingual, …) and throughput splits with varying input sequence lengths (1 k–32 k tokens).

#### One-time data preparation

SPEED-Bench prompts are fetched from external sources and cannot be redistributed directly. Run the preparation step once to materialise them locally:

```bash
# Fetch and materialise prompts, then split into per-category files (all in one command)
python scripts/evaluate/prepare_speedbench.py \
    --data-dir ./speedbench_data \
    --download

# Or run the two steps separately if you already have the flat files:
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \
    | python3 - --output_dir ./speedbench_data
python scripts/evaluate/prepare_speedbench.py --data-dir ./speedbench_data
```

> **Note:** `prepare_speedbench.py` reads from the URL above to fetch NVIDIA's `prepare.py`. Save a local copy (`--download` does this implicitly) if you anticipate running data preparation again. The materialised files contain data from third-party sources — do not redistribute them.

#### Running evaluations

Pass a `speedbench/<config>` spec via `DATASET` together with `SPEEDBENCH_DATA_DIR`:

```bash
cd scripts/evaluate/mtp_server_eval

# All 11 qualitative categories
MODE=throughput BASE_URL=http://localhost:8000 \
  DATASET=speedbench/qualitative SPEEDBENCH_DATA_DIR=../speedbench_data ./run_eval.sh

# Single category
MODE=throughput BASE_URL=http://localhost:8000 \
  DATASET=speedbench/qualitative/coding SPEEDBENCH_DATA_DIR=../speedbench_data ./run_eval.sh
```

Available configs: `qualitative`, `throughput_1k`, `throughput_2k`, `throughput_8k`, `throughput_32k`.

Results are written to `acceptance.csv` in the output directory with per-category acceptance lengths and per-position acceptance rates, identical in format to the `RedHatAI/speculator_benchmarks` output.

### Sequential acceptance (SGLang or vLLM)

`MODE=acceptance` (the `run_eval.sh` default) sends prompts directly to the
server and reads speculative-decoding metrics off Prometheus. It works against
**both SGLang and vLLM**, only needs `requests`, and reports per-benchmark
**acceptance length/rate** and **decode tok/s**. `compare_speedup.py` prints
speedup vs. a baseline.

```bash
cd scripts/evaluate/mtp_server_eval
BACKEND=vllm   BASE_URL=http://localhost:8000 RESULT_DIR=./results/spec ./run_eval.sh
BACKEND=sglang BASE_URL=http://localhost:8080 RESULT_DIR=./results/spec ./run_eval.sh
python compare_speedup.py base=./results/base/mtp_eval_summary.json \
                          spec=./results/spec/mtp_eval_summary.json
```

Benchmarks (`aime`, `gpqa`, `livecodebench`, …) ship as prompt files; the two
backends differ only in how acceptance is read (SGLang windowed gauges vs. vLLM
cumulative counters). It also includes **AgentX** (`run_agentx.sh`) — an agentic
trace-replay load test for measuring spec-decode value under realistic
multi-user concurrency. See the
[directory README](https://github.com/vllm-project/speculators/blob/main/scripts/evaluate/mtp_server_eval/README.md)
for the full option matrix.
