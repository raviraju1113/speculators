# Evaluating Model Performance

## Prerequisites

```bash
cd scripts/evaluate
pip install -r requirements.txt
```

## Quick Start

### Easiest: config-driven experiments (recommended)

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

## Measured results — Gemma-4-31B-it (single-GPU, forward-compat node)

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

## Measured results — GLM-5.2-FP8, native MTP (8×B200)

GLM-5.2 ships with **native multi-token prediction** built into the checkpoint
(`method=mtp`, `num_speculative_tokens=5`) — no separate draft model. Evaluated
with the `mtp_server_eval` evaluator via `run_experiments.py`.

**Setup:** 8×B200, `tensor_parallel_size: 8`, FP8 weights + FP8 KV cache, CUDA
graphs (not eager), `max_model_len 16384`; benchmarks `aime`, `gpqa`,
`livecodebench`, `num_samples 50`, greedy (`temperature 0.0`),
`num_speculative_tokens: 5`. Config:
`scripts/evaluate/experiments/glm52-eval.yaml`.

Baseline (backbone alone) vs. native MTP, per benchmark:

| benchmark | accept_len (of k+1=6) | accept_rate | decode speedup |
|---|---|---|---|
| baseline (no MTP) | — | — | 1.00× |
| **aime** (math) | **4.54** | **0.708** | **2.38×** |
| **livecodebench** (code) | 3.90 | 0.580 | ~1.5× |
| **gpqa** (science QA) | 3.33 | 0.466 | — |
| **overall** | — | — | **~2.0×** |

Reading it:
- **Native MTP roughly doubles decode throughput** — vLLM's server-side generation
  throughput went from **~104 tok/s (baseline) to ~209 tok/s (MTP)**, ~2.0× overall,
  up to **2.38× on `aime`**, the most predictable workload.
- **Acceptance tracks workload predictability**: math (`aime`, 71%) > code
  (`livecodebench`, 58%) > science QA (`gpqa`, 47%). Higher acceptance → longer
  accepted runs (`accept_len = accept_rate × k + 1`) → larger speedup.
- `accept_len` is out of a max of k+1 = 6 (the 5 speculated tokens + the target's
  always-accepted bonus token).

> **Metric provenance:** acceptance comes from vLLM's cumulative Prometheus
> counters (`vllm:spec_decode_*`) and the speedup from vLLM's own server-side
> generation-throughput logs — both independent of per-request client timing. The
> `gpqa` throughput cell is left blank because its per-request decode timing was
> unreliable in this run; its acceptance (from Prometheus) is unaffected.

### Manual: against a server you already run

If you already have a vLLM server up, run the GuideLLM evaluator directly. Full
benchmark pipeline (output-length estimation → performance sweep → CSV):

```bash
python evaluate.py sweep --target http://localhost:8000/v1
```

This runs all 9 subsets from `RedHatAI/speculator_benchmarks` and produces `perf_results_<timestamp>/perf_results.csv`.

For acceptance rates only (skips the sweep):

```bash
python evaluate.py throughput --target http://localhost:8000/v1
```

See [`examples/evaluate/`](https://github.com/vllm-project/speculators/tree/main/examples/evaluate) for end-to-end examples that launch a vLLM server and run the pipeline.

## Options

Both `throughput` and `sweep` share the same options:

```
  --target URL               vLLM server endpoint (required)
  --dataset DATASET          HF dataset ID or local dir (default: RedHatAI/speculator_benchmarks)
  --subsets LIST             Comma-separated subset names (default: all 9)
  --output-dir DIR           Output directory (default: perf_results_TIMESTAMP)
  --max-concurrency N        Max concurrent requests (default: 128)
  --max-requests N           Max requests per sweep point (default: 200)
  --gen-len-rate N           Request rate for gen-len estimation (default: 128)
  --gen-kwargs JSON          Generation kwargs, e.g. '{"temperature":0.6}'
  --data-column-mapper JSON  Column mapping for guidellm (default: '{"text_column":"prompt"}')
```

## Visualization

```bash
# Compare multiple versions
python plot.py compare \
    --source "No Spec=nospec/perf_results.csv" \
    --source "DFlash=dflash/perf_results.csv" \
    --metric latency --output-dir ./plots

# Pairwise speedup (blue = faster, red = regression)
python plot.py speedup \
    --baseline "No Spec=nospec/perf_results.csv" \
    --target "DFlash=dflash/perf_results.csv" \
    --metric latency --title "Qwen3-8B" --output-dir ./plots
```

Both accept CSVs or raw GuideLLM sweep JSONs. Available metrics: `latency`, `itl`, `ttft`, `output_tps`.

## Alternative: direct server eval (SGLang or vLLM)

`evaluate.py` drives GuideLLM against **vLLM**. For a lighter-weight evaluator
that works against **both SGLang and vLLM** — sending prompts directly to the
server's OpenAI-compatible streaming API and reading speculative-decoding metrics
off Prometheus (only `requests` needed) — use
[`scripts/evaluate/mtp_server_eval/`](https://github.com/vllm-project/speculators/tree/main/scripts/evaluate/mtp_server_eval).
It reports per-benchmark **acceptance length/rate** and **decode tok/s**, and
`compare_speedup.py` prints the speedup vs. a baseline.

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

