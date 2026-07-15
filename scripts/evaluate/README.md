# Evaluation

Tools for evaluating a speculative-decoding **draft model** against its
**backbone** (target / verifier): acceptance rate, throughput, and the end-to-end
speedup over running the backbone alone.

All evaluators run against a **live OpenAI-compatible server** (vLLM or SGLang) —
they do not load models themselves. So every evaluation is: **serve → run →
compare vs. baseline**.

## What gets measured

| Metric | Meaning |
|--------|---------|
| **acceptance_length** | avg tokens committed per backbone forward pass (`1.0` = speculation gave no benefit; higher is better, up to `num_spec_tokens + 1`) |
| **acceptance_rate** | accepted / drafted tokens, in `[0, 1]` |
| **decode tok/s** (a.k.a. `output_tokens_per_second`) | decode-phase output speed — the number speculative decoding actually accelerates |
| **speedup** | decode tok/s with the draft ÷ decode tok/s of the backbone alone |

## The two evaluators

| | [`evaluate.py`](./evaluate.py) (GuideLLM) | [`mtp_server_eval/`](./mtp_server_eval) (direct) |
|---|---|---|
| Engine | vLLM | vLLM **or** SGLang |
| Load driver | GuideLLM (rate/sweep control) | direct streaming requests |
| Deps | `guidellm`, `vllm` (see [requirements.txt](./requirements.txt)) | just `requests` |
| Best for | SLA-style rate sweeps, standardized perf runs | quick acceptance + decode-tok/s per benchmark; SGLang; agentic load |
| Extras | `sweep` mode, [`plot.py`](./plot.py) | `compare_speedup.py`, AgentX trace-replay |

Use **either** for single-stream throughput; they report the same core metrics.

## Datasets

- [`eval_datasets/`](./eval_datasets) — 12 static benchmarks as `{turns:[...]}`
  JSONL (gsm8k, math500, aime24/25, humaneval, mbpp, lbpp, livecodebench,
  mt-bench, alpaca, arena-hard-v2, swe-bench) + a converter to regenerate them.
  `evaluate.py` needs a `prompt` column, so run
  [`eval_datasets/to_guidellm.py`](./eval_datasets/to_guidellm.py) once to derive
  GuideLLM-ready files (or use the hosted `RedHatAI/speculator_benchmarks`).
- `mtp_server_eval/data/` — aime / gpqa-diamond / livecodebench prompts for the
  direct evaluator, plus **AgentX** (agentic Claude-Code trace replay).

## Throughput evaluation walkthrough (backbone + draft)

Speculative decoding is lossless, so the goal is **speedup at equal quality** —
always compare draft-on against the backbone-alone baseline, same GPU/config.

### 1. Serve

```bash
# spec-on: a speculators-format draft references its verifier, so serving it
# attaches the backbone
vllm serve <draft-speculator-checkpoint> --port 8001
#   or attach a draft to the backbone explicitly:
# vllm serve <backbone> --port 8001 \
#     --speculative-config '{"model":"<draft>","num_speculative_tokens":5}'

# baseline: backbone alone (the speedup denominator)
vllm serve <backbone> --port 8000
```

### 2a. Run — GuideLLM (`evaluate.py`)

```bash
python eval_datasets/to_guidellm.py            # once: derive prompt-column files
python evaluate.py --target http://localhost:8001/v1 \
    --dataset eval_datasets/guidellm --output-dir ./results/spec \
    throughput --subsets "gsm8k,humaneval,livecodebench" --max-requests 200
# repeat against :8000 -> ./results/base ; use `sweep` for a full rate curve
```

### 2b. Run — direct (`mtp_server_eval`)

```bash
cd mtp_server_eval
BACKEND=vllm BASE_URL=http://localhost:8001 RESULT_DIR=./results/spec ./run_eval.sh
BACKEND=vllm BASE_URL=http://localhost:8000 RESULT_DIR=./results/base ./run_eval.sh
python compare_speedup.py base=./results/base/mtp_eval_summary.json \
                          spec=./results/spec/mtp_eval_summary.json
```

`compare_speedup.py` prints decode tok/s + accept length/rate + the speedup ratio
per benchmark.

### 3. Realistic multi-user load (optional)

Spec-decode gains shrink as concurrency fills the KV cache. To measure that, use
AgentX (agentic trace replay) at several concurrency levels:

```bash
cd mtp_server_eval
BASE_URL=http://localhost:8001 RESULT_DIR=./results/agentx_spec USERS_LIST="1 8 16" ./run_agentx.sh
BASE_URL=http://localhost:8000 RESULT_DIR=./results/agentx_base USERS_LIST="1 8 16" ./run_agentx.sh
```

## Config-driven experiments (recommended for a multi-GPU box)

Rather than launching servers and running evals by hand, describe the whole
experiment in a YAML and let [`experiments/run_experiments.py`](./experiments/run_experiments.py)
drive it: for each entry it launches vLLM (backbone, optionally with a draft via
`--speculative-config`), waits for `/health`, runs the eval, stops the server,
then prints the speedup table (first entry = baseline). Ideal for an 8xA100 box —
set `server.tensor_parallel_size: 8` and `gpus: "0,...,7"` once and sweep configs
unattended.

```yaml
# experiments/example.yaml (excerpt)
backbone: meta-llama/Llama-3.1-70B-Instruct
gpus: "0,1,2,3,4,5,6,7"
server: { tensor_parallel_size: 8, gpu_memory_utilization: 0.9, max_model_len: 8192 }
eval:   { backend: vllm, benchmarks: [aime, gpqa, livecodebench], num_samples: 50 }
experiments:
  - name: baseline                     # backbone alone (speedup denominator)
  - name: eagle3_k5
    draft: RedHatAI/Llama-3.1-70B-Instruct-speculator.eagle3
    num_speculative_tokens: 5
  - name: eagle3_k3
    draft: RedHatAI/Llama-3.1-70B-Instruct-speculator.eagle3
    num_speculative_tokens: 3
    eval: { num_samples: 100 }         # per-experiment overrides
```

```bash
cd experiments
python run_experiments.py --config example.yaml            # run everything
python run_experiments.py --config example.yaml --dry-run  # print serve+eval commands only
python run_experiments.py --config example.yaml --only baseline,eagle3_k5
```

`--dry-run` prints the exact `vllm serve` + eval commands without launching
anything (handy to review on a machine without GPUs). Backbone/draft/server/eval
settings all live in the YAML; per-experiment blocks override the defaults.

## Tips for a fair comparison

- Same GPU/config, dataset/subsets, and request budget for spec-on and baseline.
- **Greedy (`temperature=0`)** — the canonical acceptance setting.
- Compare **decode tok/s**, not end-to-end (spec decoding doesn't speed up
  prefill / time-to-first-token).
- `acceptance_length ≈ 1` → the draft isn't helping on that workload.

## Layout

```
evaluate.py            GuideLLM-based acceptance/throughput/sweep eval
perf_utils.py          metric parsing + GuideLLM invocation helpers
plot.py                plots from sweep output
requirements.txt       guidellm + vllm + viz deps
eval_datasets/         12 benchmark prompt sets + converter + GuideLLM bridge
mtp_server_eval/       direct sglang/vllm eval + compare_speedup + AgentX
experiments/           YAML config-driven runner (serve → eval → compare)
```

See [`mtp_server_eval/README.md`](./mtp_server_eval/README.md) for detailed
per-setting recipes and [`eval_datasets/README.md`](./eval_datasets/README.md)
for the datasets.
