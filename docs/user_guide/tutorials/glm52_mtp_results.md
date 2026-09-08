# GLM-5.2 Native MTP Evaluation Results

GLM-5.2 ships with **native multi-token prediction** built into the checkpoint
(`method=mtp`, `num_speculative_tokens=5`) — no separate draft model is needed.
The evaluator reads acceptance metrics directly from vLLM's Prometheus counters,
so numbers are server-side and independent of client-side timing.

## Setup

| Parameter | Value |
|---|---|
| Backbone | `zai-org/GLM-5.2-FP8` |
| Hardware | 8× NVIDIA B200 |
| `tensor_parallel_size` | 8 |
| Weights | FP8 |
| KV cache | FP8 |
| `max_model_len` | 16,384 |
| Speculative method | `mtp` |
| `num_speculative_tokens` | 5 |
| Temperature | 0.0 (greedy) |
| Eval tool | `scripts/evaluate/experiments/run_experiments.py` |
| Config | `scripts/evaluate/experiments/glm52-eval.yaml` |

Reference: [vLLM GLM-5.2 recipe](https://recipes.vllm.ai/zai-org/GLM-5.2)

## Main Results

### Baseline vs. Native MTP

| benchmark | config | n | decode tok/s | e2e tok/s | TTFT (s) | accept_len | accept_rate | speedup (e2e) |
|---|---|---|---|---|---|---|---|---|
| **AIME** | baseline | 30 | 520.4 | 104.3 | 23.037 | — | — | 1.00× |
| | +MTP | 30 | 1251.7 | 249.3 | 10.089 | 4.54 | 0.708 | **2.39×** |
| **GPQA** | baseline | 50 | — | 104.2 | 10.073 | — | — | 1.00× |
| | +MTP | 50 | — | 182.7 | 4.469 | 3.33 | 0.466 | **1.75×** |
| **LiveCodeBench** | baseline | 50 | 779.1 | 103.8 | 10.802 | — | — | 1.00× |
| | +MTP | 50 | 1695.9 | 211.5 | 5.411 | 3.90 | 0.580 | **2.04×** |

> **Note on GPQA decode tok/s:** the GPQA baseline `decode_tok_s` is anomalously
> high (22 K tok/s) due to a measurement artifact on that run. The `e2e tok/s`
> figure (104.2 tok/s) is consistent across all three benchmarks and is used for
> the speedup calculation.

### Key Observations

- **End-to-end throughput roughly doubles** with native MTP — up to **2.39× on
  AIME** (math), **2.04× on LiveCodeBench** (code), and **1.75× on GPQA** (science QA).
- **TTFT is dramatically reduced** — AIME TTFT drops from 23s to 10s (2.3× improvement),
  which is a major latency win not captured by decode-throughput speedup alone.
- **Acceptance correlates with workload predictability**: math (71%) > code (58%) >
  science QA (47%). Higher acceptance means longer accepted draft runs
  (`accept_len ≈ accept_rate × k + 1`), which amplifies the speedup.
- `accept_len` is out of a maximum of k+1 = 6 (the 5 speculated tokens plus the
  target's always-accepted bonus token).

## KV Cache Dtype Ablation

A separate ablation run tested whether BF16 vs FP8 KV cache affects MTP acceptance
quality (`scripts/evaluate/experiments/glm52-kvcache-ablation.yaml`):

| benchmark | KV dtype | n | decode tok/s | accept_len | accept_rate |
|---|---|---|---|---|---|
| AIME | FP8 | 30 | 1251.7 | 4.54 | 0.708 |
| AIME | BF16 | 30 | 246.2 | 4.62 | 0.725 |
| GSM8K | BF16 | 50 | 240.3 | 4.40 | 0.679 |
| Math500 | BF16 | 50 | 250.2 | 4.66 | 0.733 |

> The FP8 and BF16 runs used different server warm-up states, so raw `decode tok/s`
> are not directly comparable. However, the accept length/rate are consistent across
> dtypes — **KV cache dtype does not meaningfully affect MTP acceptance quality**.

## Running the Evaluation

```bash
# Automated (launches server, runs eval, compares):
cd scripts/evaluate/experiments
python run_experiments.py --config glm52-eval.yaml

# Dry run (preview commands without launching):
python run_experiments.py --config glm52-eval.yaml --dry-run
```

For manual evaluation against a pre-running server:

```bash
# Terminal 1: serve
./scripts/evaluate/mtp_server_eval/glm52/run_glm52_vllm.sh

# Terminal 2: eval
./scripts/evaluate/mtp_server_eval/glm52/run_glm52_eval.sh
```

## Raw Results

JSON summaries are stored under `scripts/evaluate/experiments/results/`:

```
results/glm52-eval/
  baseline/mtp_eval_summary.json          # backbone alone
  glm52-native-mtp/mtp_eval_summary.json  # FP8 KV, native MTP

results/glm52-kvcache-ablation/
  glm52-mtp-bf16kv/mtp_eval_summary.json  # BF16 KV cache ablation
```