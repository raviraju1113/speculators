# Evaluating Model Performance

## Prerequisites

```bash
cd scripts/evaluate
pip install -r requirements.txt
```

## Quick Start

### Easiest: config-driven experiments (recommended)

Describe the run in a YAML and let
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

