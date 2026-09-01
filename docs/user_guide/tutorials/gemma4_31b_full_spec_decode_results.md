# Gemma-4-31B Full-Suite Speculative Decoding Results

Head-to-head acceptance + throughput eval of `gemma-4-31b-it` with four drafts
on the same 25-benchmark suite (50 samples per bench; AIME / AIME26 = 30;
speed-multilingual = 47). Skipped (no JSONL on this box): `aa-lcr`,
`speed-low-entropy`.

## Setup

| Parameter | Value |
|---|---|
| Backbone | `/nvmedata/hf_checkpoints/gemma-4-31b-it` (bf16) |
| Hardware | 4× NVIDIA A100 80GB, TP=4 |
| `max_model_len` | 8,192 |
| `max_tokens` | 4,096 |
| Temperature | 0.0 (greedy) |
| Eval | `scripts/evaluate/experiments/run_experiments.py` (`mode: acceptance`) |

| config | draft | k | vLLM | YAML |
|---|---|---:|---|---|
| baseline (no draft) | — | — | 0.24.0+cu129 | `gemma4-31b-full.yaml` |
| Google Assistant (MTP) | `/nvmedata/hf_checkpoints/gemma-4-31B-it-assistant` | 5 | 0.24.0+cu129 | `gemma4-31b-full.yaml` |
| Eagle-3 Qwen (Ravi) | `/nvmedata/hf_checkpoints/gemma4_draft_model_900k_eagle3_kimi_mtp_stem_code_math` | 5 | 0.24.0+cu129 | `gemma4-31b-full-eagle3.yaml` |
| Eagle-3 Llama (John) | `/nvmedata/hf_checkpoints/gemma4_redhat_draft_ft_2026-07-26_06-06-06` | 5 | 0.24.0+cu129 | `gemma4-31b-full-redhat-ft.yaml` |
| DSpark SWA (Mengmeng) | `/nvmedata/hf_checkpoints/gemma4_31b_dspark_nemo782k_scratch` | 8 | 0.28.0+cu129 | `gemma4-31b-full-dspark-nemo782k.yaml` |

DSpark SWA (Mengmeng) was trained with `block_size=8` and `sample_from_anchor=True`.
vLLM 0.24 cannot serve DSpark (`method=dspark` landed in 0.25; 0.28 is required so
`sample_from_anchor` is honored). k=3 / k=5 are not valid for this checkpoint.

## Metrics

- **decode tok/s** — output tokens/sec excluding TTFT. This is the throughput
  number used for speedup.
- **e2e tok/s** — end-to-end completion tokens / wall time (includes prefill).
- **ttft (s)** — mean time to first token.
- **comp. tokens** — total completion tokens over the bench (sanity-check that
  configs generated similar-length outputs).
- **accept_len** — tokens committed per target forward (includes the bonus
  token). Baseline is n/a. Upper bound is k+1 for Google Assistant (MTP) / Eagle-3 and 8 for
  this DSpark SWA (Mengmeng) run.
- **accept_rate** — accepted_tokens / draft_tokens (vLLM cumulative counters,
  delta over the bench). Baseline is n/a.
- **speedup** — decode tok/s ÷ baseline decode tok/s.
- **wall time** — sum of per-request `e2e_s` from `mtp_eval_details.jsonl`
  (sequential request time only; excludes server load / CUDA-graph capture).

> DSpark SWA (Mengmeng) tok/s and speedup mix vLLM 0.28 with a 0.24 baseline.
> Treat those as approximate. **accept_len / accept_rate are the fair draft-quality comparison.**

## Throughput (decode tok/s vs baseline)

| benchmark | n | baseline | Google Assistant (MTP) | Eagle-3 Qwen (Ravi) | Eagle-3 Llama (John) | DSpark SWA (Mengmeng) | × MTP | × Ravi | × John | × Mengmeng |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aime | 30 | 54.3 | 168.8 | 121.9 | 138.2 | 211.6 | **3.11×** | **2.24×** | **2.55×** | **3.90×** |
| gpqa | 50 | 54.6 | 155.4 | 98.5 | 109.6 | 157.9 | **2.85×** | **1.80×** | **2.01×** | **2.89×** |
| livecodebench | 50 | 54.2 | 152.0 | 102.4 | 119.5 | 184.5 | **2.80×** | **1.89×** | **2.20×** | **3.40×** |
| gsm8k | 50 | 55.3 | 199.2 | 143.6 | 172.0 | 261.1 | **3.60×** | **2.60×** | **3.11×** | **4.72×** |
| humaneval | 50 | 55.1 | 197.1 | 130.1 | 163.9 | 242.3 | **3.58×** | **2.36×** | **2.97×** | **4.40×** |
| mbpp | 50 | 55.1 | 171.7 | 114.7 | 143.6 | 206.3 | **3.12×** | **2.08×** | **2.61×** | **3.74×** |
| math500 | 50 | 54.9 | 185.4 | 137.8 | 160.4 | 252.4 | **3.38×** | **2.51×** | **2.92×** | **4.60×** |
| mt-bench | 50 | 55.1 | 128.5 | 82.7 | 105.7 | 133.8 | **2.33×** | **1.50×** | **1.92×** | **2.43×** |
| aime26 | 30 | 54.0 | 158.7 | 118.7 | 132.5 | 213.8 | **2.94×** | **2.20×** | **2.45×** | **3.96×** |
| bfcl | 50 | 55.2 | 200.5 | 109.4 | 133.3 | 198.9 | **3.63×** | **1.98×** | **2.41×** | **3.60×** |
| swe-bench-pro | 50 | 54.1 | 147.0 | 79.1 | 97.0 | 137.4 | **2.72×** | **1.46×** | **1.79×** | **2.54×** |
| speed-coding | 50 | 55.0 | 174.7 | 110.2 | 137.3 | 201.9 | **3.18×** | **2.00×** | **2.50×** | **3.67×** |
| speed-multilingual | 47 | 55.2 | 169.9 | 73.3 | 80.5 | 90.7 | **3.08×** | **1.33×** | **1.46×** | **1.64×** |
| speed-rag | 50 | 54.8 | 150.6 | 89.7 | 105.7 | 138.7 | **2.75×** | **1.64×** | **1.93×** | **2.53×** |
| speed-qa | 50 | 55.4 | 124.8 | 77.1 | 96.5 | 114.5 | **2.25×** | **1.39×** | **1.74×** | **2.07×** |
| speed-writing | 50 | 53.9 | 103.1 | 65.8 | 82.4 | 101.5 | **1.91×** | **1.22×** | **1.53×** | **1.88×** |
| HumanEval | 50 | 55.2 | 186.6 | 123.7 | 155.0 | 233.4 | **3.38×** | **2.24×** | **2.81×** | **4.23×** |
| math_reasoning | 50 | 55.4 | 201.7 | 143.2 | 175.1 | 262.6 | **3.64×** | **2.58×** | **3.16×** | **4.74×** |
| qa | 50 | 55.4 | 124.8 | 77.1 | 96.5 | 114.5 | **2.25×** | **1.39×** | **1.74×** | **2.07×** |
| question | 50 | 55.1 | 128.3 | 82.9 | 105.6 | 133.6 | **2.33×** | **1.50×** | **1.92×** | **2.42×** |
| rag | 50 | 54.7 | 141.1 | 83.7 | 100.7 | 129.2 | **2.58×** | **1.53×** | **1.84×** | **2.36×** |
| summarization | 50 | 54.5 | 111.2 | 63.2 | 78.0 | 96.8 | **2.04×** | **1.16×** | **1.43×** | **1.78×** |
| tool_call | 50 | 55.2 | 148.5 | 82.2 | 108.3 | 135.7 | **2.69×** | **1.49×** | **1.96×** | **2.46×** |
| translation | 50 | 55.4 | 160.7 | 87.7 | 111.4 | 142.7 | **2.90×** | **1.58×** | **2.01×** | **2.58×** |
| writing | 50 | 55.1 | 128.4 | 82.5 | 105.5 | 133.4 | **2.33×** | **1.50×** | **1.91×** | **2.42×** |

## Acceptance (accept_len and accept_rate)

Baseline has no speculation, so it is omitted here.

| benchmark | Google Assistant (MTP) AL | Eagle-3 Qwen AL | Eagle-3 Llama AL | DSpark SWA AL | Google Assistant (MTP) AR | Eagle-3 Qwen AR | Eagle-3 Llama AR | DSpark SWA AR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aime | 4.848 | 3.620 | 3.759 | 5.952 | 0.7696 | 0.5241 | 0.5518 | 0.6190 |
| gpqa | 4.479 | 2.910 | 2.942 | 4.350 | 0.6958 | 0.3819 | 0.3884 | 0.4188 |
| livecodebench | 4.539 | 3.074 | 3.365 | 5.198 | 0.7077 | 0.4148 | 0.4730 | 0.5247 |
| gsm8k | 5.068 | 3.745 | 4.074 | 6.426 | 0.8135 | 0.5490 | 0.6148 | 0.6783 |
| humaneval | 5.183 | 3.508 | 4.001 | 6.137 | 0.8366 | 0.5015 | 0.6002 | 0.6421 |
| mbpp | 4.524 | 3.089 | 3.529 | 5.239 | 0.7049 | 0.4178 | 0.5058 | 0.5299 |
| math500 | 5.037 | 3.818 | 4.084 | 6.590 | 0.8075 | 0.5635 | 0.6168 | 0.6988 |
| mt-bench | 3.412 | 2.241 | 2.619 | 3.423 | 0.4823 | 0.2482 | 0.3238 | 0.3028 |
| aime26 | 4.769 | 3.613 | 3.699 | 6.102 | 0.7538 | 0.5225 | 0.5399 | 0.6377 |
| bfcl | 5.587 | 3.134 | 3.343 | 5.141 | 0.9174 | 0.4268 | 0.4686 | 0.5177 |
| swe-bench-pro | 4.466 | 2.433 | 2.755 | 3.958 | 0.6932 | 0.2866 | 0.3509 | 0.3697 |
| speed-coding | 4.733 | 3.031 | 3.452 | 5.241 | 0.7466 | 0.4062 | 0.4905 | 0.5301 |
| speed-multilingual | 4.429 | 1.957 | 1.951 | 2.278 | 0.6859 | 0.1914 | 0.1901 | 0.1597 |
| speed-rag | 4.387 | 2.614 | 2.811 | 3.776 | 0.6773 | 0.3227 | 0.3622 | 0.3470 |
| speed-qa | 3.157 | 1.991 | 2.272 | 2.803 | 0.4314 | 0.1982 | 0.2543 | 0.2253 |
| speed-writing | 3.118 | 2.013 | 2.329 | 2.920 | 0.4235 | 0.2025 | 0.2658 | 0.2400 |
| HumanEval | 4.902 | 3.312 | 3.784 | 5.901 | 0.7804 | 0.4624 | 0.5567 | 0.6127 |
| math_reasoning | 5.066 | 3.675 | 4.071 | 6.360 | 0.8131 | 0.5350 | 0.6143 | 0.6700 |
| qa | 3.157 | 1.991 | 2.272 | 2.803 | 0.4314 | 0.1982 | 0.2543 | 0.2253 |
| question | 3.408 | 2.240 | 2.617 | 3.416 | 0.4816 | 0.2480 | 0.3234 | 0.3020 |
| rag | 4.154 | 2.462 | 2.704 | 3.543 | 0.6308 | 0.2924 | 0.3407 | 0.3178 |
| summarization | 3.290 | 1.881 | 2.128 | 2.697 | 0.4579 | 0.1762 | 0.2256 | 0.2121 |
| tool_call | 4.012 | 2.281 | 2.722 | 3.492 | 0.6024 | 0.2562 | 0.3444 | 0.3115 |
| translation | 4.023 | 2.243 | 2.590 | 3.463 | 0.6045 | 0.2487 | 0.3181 | 0.3079 |
| writing | 3.408 | 2.237 | 2.616 | 3.412 | 0.4817 | 0.2474 | 0.3231 | 0.3016 |

## End-to-end throughput and TTFT

| benchmark | base e2e | Google Assistant (MTP) e2e | Eagle-3 Qwen e2e | Eagle-3 Llama e2e | DSpark SWA e2e | base TTFT | Google Assistant (MTP) TTFT | Eagle-3 Qwen TTFT | Eagle-3 Llama TTFT | DSpark SWA TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aime | 54.2 | 166.8 | 119.9 | 136.9 | 208.9 | 0.078 | 0.096 | 0.196 | 0.096 | 0.096 |
| gpqa | 54.2 | 152.0 | 97.1 | 107.8 | 154.6 | 0.104 | 0.129 | 0.133 | 0.129 | 0.119 |
| livecodebench | 53.7 | 147.3 | 100.1 | 116.8 | 177.5 | 0.144 | 0.167 | 0.173 | 0.167 | 0.156 |
| gsm8k | 54.8 | 190.2 | 138.7 | 165.4 | 246.4 | 0.064 | 0.077 | 0.081 | 0.076 | 0.073 |
| humaneval | 54.5 | 185.3 | 124.9 | 155.8 | 226.2 | 0.079 | 0.093 | 0.099 | 0.093 | 0.086 |
| mbpp | 54.9 | 169.1 | 113.5 | 141.8 | 202.7 | 0.052 | 0.066 | 0.071 | 0.066 | 0.063 |
| math500 | 54.7 | 181.4 | 135.5 | 157.4 | 245.5 | 0.070 | 0.085 | 0.090 | 0.085 | 0.078 |
| mt-bench | 54.8 | 126.3 | 81.7 | 104.3 | 131.6 | 0.060 | 0.073 | 0.078 | 0.073 | 0.069 |
| aime26 | 53.9 | 157.5 | 118.0 | 131.6 | 211.8 | 0.078 | 0.095 | 0.097 | 0.094 | 0.087 |
| bfcl | 49.2 | 121.2 | 80.2 | 93.0 | 121.6 | 0.109 | 0.141 | 0.148 | 0.143 | 0.138 |
| swe-bench-pro | 53.3 | 140.0 | 76.9 | 94.0 | 131.6 | 0.213 | 0.234 | 0.244 | 0.238 | 0.222 |
| speed-coding | 54.6 | 168.8 | 107.7 | 133.6 | 194.4 | 0.085 | 0.100 | 0.105 | 0.100 | 0.093 |
| speed-multilingual | 54.8 | 164.5 | 72.3 | 79.3 | 89.4 | 0.068 | 0.082 | 0.087 | 0.082 | 0.076 |
| speed-rag | 48.0 | 104.6 | 70.5 | 79.9 | 99.6 | 0.184 | 0.204 | 0.213 | 0.207 | 0.193 |
| speed-qa | 54.9 | 121.2 | 75.7 | 94.4 | 111.6 | 0.051 | 0.063 | 0.067 | 0.062 | 0.059 |
| speed-writing | 53.6 | 101.6 | 65.2 | 81.4 | 100.2 | 0.176 | 0.196 | 0.204 | 0.199 | 0.186 |
| HumanEval | 54.7 | 178.0 | 119.8 | 149.1 | 220.9 | 0.070 | 0.086 | 0.091 | 0.086 | 0.080 |
| math_reasoning | 54.7 | 187.2 | 135.4 | 164.1 | 239.8 | 0.059 | 0.071 | 0.075 | 0.071 | 0.065 |
| qa | 54.9 | 121.1 | 75.7 | 94.4 | 111.6 | 0.051 | 0.062 | 0.067 | 0.062 | 0.059 |
| question | 54.8 | 126.2 | 82.0 | 104.2 | 131.4 | 0.058 | 0.072 | 0.077 | 0.072 | 0.069 |
| rag | 49.9 | 106.8 | 69.1 | 81.1 | 99.0 | 0.126 | 0.151 | 0.170 | 0.159 | 0.155 |
| summarization | 50.9 | 95.1 | 57.6 | 69.8 | 85.1 | 0.199 | 0.219 | 0.228 | 0.222 | 0.207 |
| tool_call | 51.0 | 116.5 | 71.4 | 90.2 | 109.4 | 0.170 | 0.192 | 0.200 | 0.193 | 0.182 |
| translation | 54.9 | 153.7 | 85.6 | 108.2 | 137.7 | 0.054 | 0.066 | 0.071 | 0.066 | 0.062 |
| writing | 54.9 | 126.3 | 81.7 | 104.2 | 131.3 | 0.048 | 0.070 | 0.074 | 0.069 | 0.067 |

## Wall time per bench

Sum of per-request `e2e_s` (prefill + decode). Sequential eval only — not
including vLLM startup.

| benchmark | n | baseline | Google Assistant (MTP) | Eagle-3 Qwen (Ravi) | Eagle-3 Llama (John) | DSpark SWA (Mengmeng) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aime | 30 | 13.0m | 3.8m | 5.9m | 5.1m | 3.6m |
| gpqa | 50 | 11.7m | 4.6m | 7.3m | 6.0m | 4.5m |
| livecodebench | 50 | 11.5m | 4.3m | 6.1m | 5.8m | 3.3m |
| gsm8k | 50 | 4.6m | 1.3m | 1.8m | 1.5m | 1.0m |
| humaneval | 50 | 4.3m | 1.2m | 1.9m | 1.5m | 1.0m |
| mbpp | 50 | 10.2m | 3.3m | 4.9m | 3.9m | 2.7m |
| math500 | 50 | 10.1m | 3.0m | 4.0m | 3.6m | 2.2m |
| mt-bench | 50 | 7.4m | 3.2m | 5.0m | 3.9m | 3.1m |
| aime26 | 30 | 16.1m | 5.8m | 7.7m | 6.4m | 4.3m |
| bfcl | 50 | 42s | 17s | 26s | 22s | 17s |
| swe-bench-pro | 50 | 10.5m | 4.0m | 7.1m | 6.0m | 4.2m |
| speed-coding | 50 | 7.0m | 2.3m | 3.5m | 2.9m | 2.0m |
| speed-multilingual | 47 | 5.7m | 1.9m | 4.4m | 3.9m | 3.6m |
| speed-rag | 50 | 1.1m | 32s | 47s | 40s | 33s |
| speed-qa | 50 | 3.4m | 1.5m | 2.5m | 2.0m | 1.7m |
| speed-writing | 50 | 20.8m | 10.8m | 17.1m | 13.6m | 11.2m |
| HumanEval | 50 | 4.7m | 1.5m | 2.2m | 1.7m | 1.2m |
| math_reasoning | 50 | 2.6m | 46s | 1.0m | 52s | 35s |
| qa | 50 | 3.4m | 1.5m | 2.5m | 2.0m | 1.7m |
| question | 50 | 7.4m | 3.2m | 5.0m | 3.9m | 3.1m |
| rag | 50 | 1.0m | 30s | 45s | 38s | 32s |
| summarization | 50 | 2.3m | 1.2m | 2.0m | 1.7m | 1.4m |
| tool_call | 50 | 1.7m | 43s | 1.2m | 55s | 45s |
| translation | 50 | 3.2m | 1.1m | 2.1m | 1.7m | 1.3m |
| writing | 50 | 7.5m | 3.2m | 5.0m | 4.0m | 3.1m |
| **total** | | **2.86h** | **1.09h** | **1.70h** | **1.41h** | **1.05h** |

## Full per-benchmark table

Every metric on one row per (benchmark, config), including baseline.

| benchmark | config | n | decode tok/s | e2e tok/s | ttft (s) | comp. tokens | accept_len | accept_rate | speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aime | baseline (no draft) | 30 | 54.3 | 54.2 | 0.078 | 42422 | — | — | 1.00× |
|  | Google Assistant (MTP) | 30 | 168.8 | 166.8 | 0.096 | 37851 | 4.848 | 0.7696 | **3.11×** |
|  | Eagle-3 Qwen (Ravi) | 30 | 121.9 | 119.9 | 0.196 | 42190 | 3.620 | 0.5241 | **2.24×** |
|  | Eagle-3 Llama (John) | 30 | 138.2 | 136.9 | 0.096 | 41859 | 3.759 | 0.5518 | **2.55×** |
|  | DSpark SWA (Mengmeng) | 30 | 211.6 | 208.9 | 0.096 | 45502 | 5.952 | 0.6190 | **3.90×** |
| gpqa | baseline (no draft) | 50 | 54.6 | 54.2 | 0.104 | 38159 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 155.4 | 152.0 | 0.129 | 41798 | 4.479 | 0.6958 | **2.85×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 98.5 | 97.1 | 0.133 | 42692 | 2.910 | 0.3819 | **1.80×** |
|  | Eagle-3 Llama (John) | 50 | 109.6 | 107.8 | 0.129 | 38871 | 2.942 | 0.3884 | **2.01×** |
|  | DSpark SWA (Mengmeng) | 50 | 157.9 | 154.6 | 0.119 | 41824 | 4.350 | 0.4188 | **2.89×** |
| livecodebench | baseline (no draft) | 50 | 54.2 | 53.7 | 0.144 | 36918 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 152.0 | 147.3 | 0.167 | 38301 | 4.539 | 0.7077 | **2.80×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 102.4 | 100.1 | 0.173 | 36654 | 3.074 | 0.4148 | **1.89×** |
|  | Eagle-3 Llama (John) | 50 | 119.5 | 116.8 | 0.167 | 40730 | 3.365 | 0.4730 | **2.20×** |
|  | DSpark SWA (Mengmeng) | 50 | 184.5 | 177.5 | 0.156 | 35346 | 5.198 | 0.5247 | **3.40×** |
| gsm8k | baseline (no draft) | 50 | 55.3 | 54.8 | 0.064 | 15166 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 199.2 | 190.2 | 0.077 | 15030 | 5.068 | 0.8135 | **3.60×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 143.6 | 138.7 | 0.081 | 15101 | 3.745 | 0.5490 | **2.60×** |
|  | Eagle-3 Llama (John) | 50 | 172.0 | 165.4 | 0.076 | 15079 | 4.074 | 0.6148 | **3.11×** |
|  | DSpark SWA (Mengmeng) | 50 | 261.1 | 246.4 | 0.073 | 15126 | 6.426 | 0.6783 | **4.72×** |
| humaneval | baseline (no draft) | 50 | 55.1 | 54.5 | 0.079 | 13913 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 197.1 | 185.3 | 0.093 | 13577 | 5.183 | 0.8366 | **3.58×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 130.1 | 124.9 | 0.099 | 14225 | 3.508 | 0.5015 | **2.36×** |
|  | Eagle-3 Llama (John) | 50 | 163.9 | 155.8 | 0.093 | 13709 | 4.001 | 0.6002 | **2.97×** |
|  | DSpark SWA (Mengmeng) | 50 | 242.3 | 226.2 | 0.086 | 13894 | 6.137 | 0.6421 | **4.40×** |
| mbpp | baseline (no draft) | 50 | 55.1 | 54.9 | 0.052 | 33490 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 171.7 | 169.1 | 0.066 | 33221 | 4.524 | 0.7049 | **3.12×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 114.7 | 113.5 | 0.071 | 33336 | 3.089 | 0.4178 | **2.08×** |
|  | Eagle-3 Llama (John) | 50 | 143.6 | 141.8 | 0.066 | 33114 | 3.529 | 0.5058 | **2.61×** |
|  | DSpark SWA (Mengmeng) | 50 | 206.3 | 202.7 | 0.063 | 32942 | 5.239 | 0.5299 | **3.74×** |
| math500 | baseline (no draft) | 50 | 54.9 | 54.7 | 0.070 | 33011 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 185.4 | 181.4 | 0.085 | 32922 | 5.037 | 0.8075 | **3.38×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 137.8 | 135.5 | 0.090 | 32852 | 3.818 | 0.5635 | **2.51×** |
|  | Eagle-3 Llama (John) | 50 | 160.4 | 157.4 | 0.085 | 33580 | 4.084 | 0.6168 | **2.92×** |
|  | DSpark SWA (Mengmeng) | 50 | 252.4 | 245.5 | 0.078 | 33047 | 6.590 | 0.6988 | **4.60×** |
| mt-bench | baseline (no draft) | 50 | 55.1 | 54.8 | 0.060 | 24239 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 128.5 | 126.3 | 0.073 | 24276 | 3.412 | 0.4823 | **2.33×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 82.7 | 81.7 | 0.078 | 24410 | 2.241 | 0.2482 | **1.50×** |
|  | Eagle-3 Llama (John) | 50 | 105.7 | 104.3 | 0.073 | 24608 | 2.619 | 0.3238 | **1.92×** |
|  | DSpark SWA (Mengmeng) | 50 | 133.8 | 131.6 | 0.069 | 24556 | 3.423 | 0.3028 | **2.43×** |
| aime26 | baseline (no draft) | 30 | 54.0 | 53.9 | 0.078 | 52100 | — | — | 1.00× |
|  | Google Assistant (MTP) | 30 | 158.7 | 157.5 | 0.095 | 55142 | 4.769 | 0.7538 | **2.94×** |
|  | Eagle-3 Qwen (Ravi) | 30 | 118.7 | 118.0 | 0.097 | 54233 | 3.613 | 0.5225 | **2.20×** |
|  | Eagle-3 Llama (John) | 30 | 132.5 | 131.6 | 0.094 | 50612 | 3.699 | 0.5399 | **2.45×** |
|  | DSpark SWA (Mengmeng) | 30 | 213.8 | 211.8 | 0.087 | 54529 | 6.102 | 0.6377 | **3.96×** |
| bfcl | baseline (no draft) | 50 | 55.2 | 49.2 | 0.109 | 2086 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 200.5 | 121.2 | 0.141 | 2086 | 5.587 | 0.9174 | **3.63×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 109.4 | 80.2 | 0.148 | 2086 | 3.134 | 0.4268 | **1.98×** |
|  | Eagle-3 Llama (John) | 50 | 133.3 | 93.0 | 0.143 | 2086 | 3.343 | 0.4686 | **2.41×** |
|  | DSpark SWA (Mengmeng) | 50 | 198.9 | 121.6 | 0.138 | 2086 | 5.141 | 0.5177 | **3.60×** |
| swe-bench-pro | baseline (no draft) | 50 | 54.1 | 53.3 | 0.213 | 33430 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 147.0 | 140.0 | 0.234 | 33295 | 4.466 | 0.6932 | **2.72×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 79.1 | 76.9 | 0.244 | 32540 | 2.433 | 0.2866 | **1.46×** |
|  | Eagle-3 Llama (John) | 50 | 97.0 | 94.0 | 0.238 | 33833 | 2.755 | 0.3509 | **1.79×** |
|  | DSpark SWA (Mengmeng) | 50 | 137.4 | 131.6 | 0.222 | 33502 | 3.958 | 0.3697 | **2.54×** |
| speed-coding | baseline (no draft) | 50 | 55.0 | 54.6 | 0.085 | 22795 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 174.7 | 168.8 | 0.100 | 23699 | 4.733 | 0.7466 | **3.18×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 110.2 | 107.7 | 0.105 | 22922 | 3.031 | 0.4062 | **2.00×** |
|  | Eagle-3 Llama (John) | 50 | 137.3 | 133.6 | 0.100 | 23043 | 3.452 | 0.4905 | **2.50×** |
|  | DSpark SWA (Mengmeng) | 50 | 201.9 | 194.4 | 0.093 | 23064 | 5.241 | 0.5301 | **3.67×** |
| speed-multilingual | baseline (no draft) | 47 | 55.2 | 54.8 | 0.068 | 18886 | — | — | 1.00× |
|  | Google Assistant (MTP) | 47 | 169.9 | 164.5 | 0.082 | 18523 | 4.429 | 0.6859 | **3.08×** |
|  | Eagle-3 Qwen (Ravi) | 47 | 73.3 | 72.3 | 0.087 | 19011 | 1.957 | 0.1914 | **1.33×** |
|  | Eagle-3 Llama (John) | 47 | 80.5 | 79.3 | 0.082 | 18714 | 1.951 | 0.1901 | **1.46×** |
|  | DSpark SWA (Mengmeng) | 47 | 90.7 | 89.4 | 0.076 | 19052 | 2.278 | 0.1597 | **1.64×** |
| speed-rag | baseline (no draft) | 50 | 54.8 | 48.0 | 0.184 | 3222 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 150.6 | 104.6 | 0.204 | 3370 | 4.387 | 0.6773 | **2.75×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 89.7 | 70.5 | 0.213 | 3325 | 2.614 | 0.3227 | **1.64×** |
|  | Eagle-3 Llama (John) | 50 | 105.7 | 79.9 | 0.207 | 3229 | 2.811 | 0.3622 | **1.93×** |
|  | DSpark SWA (Mengmeng) | 50 | 138.7 | 99.6 | 0.193 | 3286 | 3.776 | 0.3470 | **2.53×** |
| speed-qa | baseline (no draft) | 50 | 55.4 | 54.9 | 0.051 | 11308 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 124.8 | 121.2 | 0.063 | 11197 | 3.157 | 0.4314 | **2.25×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 77.1 | 75.7 | 0.067 | 11259 | 1.991 | 0.1982 | **1.39×** |
|  | Eagle-3 Llama (John) | 50 | 96.5 | 94.4 | 0.062 | 11467 | 2.272 | 0.2543 | **1.74×** |
|  | DSpark SWA (Mengmeng) | 50 | 114.5 | 111.6 | 0.059 | 11361 | 2.803 | 0.2253 | **2.07×** |
| speed-writing | baseline (no draft) | 50 | 53.9 | 53.6 | 0.176 | 66840 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 103.1 | 101.6 | 0.196 | 65919 | 3.118 | 0.4235 | **1.91×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 65.8 | 65.2 | 0.204 | 66827 | 2.013 | 0.2025 | **1.22×** |
|  | Eagle-3 Llama (John) | 50 | 82.4 | 81.4 | 0.199 | 66435 | 2.329 | 0.2658 | **1.53×** |
|  | DSpark SWA (Mengmeng) | 50 | 101.5 | 100.2 | 0.186 | 67444 | 2.920 | 0.2400 | **1.88×** |
| HumanEval | baseline (no draft) | 50 | 55.2 | 54.7 | 0.070 | 15439 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 186.6 | 178.0 | 0.086 | 15665 | 4.902 | 0.7804 | **3.38×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 123.7 | 119.8 | 0.091 | 15659 | 3.312 | 0.4624 | **2.24×** |
|  | Eagle-3 Llama (John) | 50 | 155.0 | 149.1 | 0.086 | 15586 | 3.784 | 0.5567 | **2.81×** |
|  | DSpark SWA (Mengmeng) | 50 | 233.4 | 220.9 | 0.080 | 15523 | 5.901 | 0.6127 | **4.23×** |
| math_reasoning | baseline (no draft) | 50 | 55.4 | 54.7 | 0.059 | 8507 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 201.7 | 187.2 | 0.071 | 8535 | 5.066 | 0.8131 | **3.64×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 143.2 | 135.4 | 0.075 | 8494 | 3.675 | 0.5350 | **2.58×** |
|  | Eagle-3 Llama (John) | 50 | 175.1 | 164.1 | 0.071 | 8506 | 4.071 | 0.6143 | **3.16×** |
|  | DSpark SWA (Mengmeng) | 50 | 262.6 | 239.8 | 0.065 | 8477 | 6.360 | 0.6700 | **4.74×** |
| qa | baseline (no draft) | 50 | 55.4 | 54.9 | 0.051 | 11308 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 124.8 | 121.1 | 0.062 | 11197 | 3.157 | 0.4314 | **2.25×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 77.1 | 75.7 | 0.067 | 11259 | 1.991 | 0.1982 | **1.39×** |
|  | Eagle-3 Llama (John) | 50 | 96.5 | 94.4 | 0.062 | 11467 | 2.272 | 0.2543 | **1.74×** |
|  | DSpark SWA (Mengmeng) | 50 | 114.5 | 111.6 | 0.059 | 11361 | 2.803 | 0.2253 | **2.07×** |
| question | baseline (no draft) | 50 | 55.1 | 54.8 | 0.058 | 24329 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 128.3 | 126.2 | 0.072 | 24281 | 3.408 | 0.4816 | **2.33×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 82.9 | 82.0 | 0.077 | 24420 | 2.240 | 0.2480 | **1.50×** |
|  | Eagle-3 Llama (John) | 50 | 105.6 | 104.2 | 0.072 | 24556 | 2.617 | 0.3234 | **1.92×** |
|  | DSpark SWA (Mengmeng) | 50 | 133.6 | 131.4 | 0.069 | 24521 | 3.416 | 0.3020 | **2.42×** |
| rag | baseline (no draft) | 50 | 54.7 | 49.9 | 0.126 | 3085 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 141.1 | 106.8 | 0.151 | 3158 | 4.154 | 0.6308 | **2.58×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 83.7 | 69.1 | 0.170 | 3143 | 2.462 | 0.2924 | **1.53×** |
|  | Eagle-3 Llama (John) | 50 | 100.7 | 81.1 | 0.159 | 3116 | 2.704 | 0.3407 | **1.84×** |
|  | DSpark SWA (Mengmeng) | 50 | 129.2 | 99.0 | 0.155 | 3125 | 3.543 | 0.3178 | **2.36×** |
| summarization | baseline (no draft) | 50 | 54.5 | 50.9 | 0.199 | 6923 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 111.2 | 95.1 | 0.219 | 6875 | 3.290 | 0.4579 | **2.04×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 63.2 | 57.6 | 0.228 | 6835 | 1.881 | 0.1762 | **1.16×** |
|  | Eagle-3 Llama (John) | 50 | 78.0 | 69.8 | 0.222 | 6941 | 2.128 | 0.2256 | **1.43×** |
|  | DSpark SWA (Mengmeng) | 50 | 96.8 | 85.1 | 0.207 | 6902 | 2.697 | 0.2121 | **1.78×** |
| tool_call | baseline (no draft) | 50 | 55.2 | 51.0 | 0.170 | 5075 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 148.5 | 116.5 | 0.192 | 5012 | 4.012 | 0.6024 | **2.69×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 82.2 | 71.4 | 0.200 | 5089 | 2.281 | 0.2562 | **1.49×** |
|  | Eagle-3 Llama (John) | 50 | 108.3 | 90.2 | 0.193 | 4958 | 2.722 | 0.3444 | **1.96×** |
|  | DSpark SWA (Mengmeng) | 50 | 135.7 | 109.4 | 0.182 | 4921 | 3.492 | 0.3115 | **2.46×** |
| translation | baseline (no draft) | 50 | 55.4 | 54.9 | 0.054 | 10607 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 160.7 | 153.7 | 0.066 | 10492 | 4.023 | 0.6045 | **2.90×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 87.7 | 85.6 | 0.071 | 10667 | 2.243 | 0.2487 | **1.58×** |
|  | Eagle-3 Llama (John) | 50 | 111.4 | 108.2 | 0.066 | 10735 | 2.590 | 0.3181 | **2.01×** |
|  | DSpark SWA (Mengmeng) | 50 | 142.7 | 137.7 | 0.062 | 10582 | 3.463 | 0.3079 | **2.58×** |
| writing | baseline (no draft) | 50 | 55.1 | 54.9 | 0.048 | 24711 | — | — | 1.00× |
|  | Google Assistant (MTP) | 50 | 128.4 | 126.3 | 0.070 | 24280 | 3.408 | 0.4817 | **2.33×** |
|  | Eagle-3 Qwen (Ravi) | 50 | 82.5 | 81.7 | 0.074 | 24455 | 2.237 | 0.2474 | **1.50×** |
|  | Eagle-3 Llama (John) | 50 | 105.5 | 104.2 | 0.069 | 24705 | 2.616 | 0.3231 | **1.91×** |
|  | DSpark SWA (Mengmeng) | 50 | 133.4 | 131.3 | 0.067 | 24497 | 3.412 | 0.3016 | **2.42×** |

## Observations

DSpark SWA (Mengmeng) wins math/code (gsm8k AL 6.43, math500 6.59) and beats
both Eagle-3 drafts on most benches. Google Assistant (MTP) still wins chat,
multilingual, and BFCL (AR 92% vs 52%). Eagle-3 Llama (John) beats Eagle-3
Qwen (Ravi) 25/25 on decode tok/s.

## Raw outputs

| config | summary |
|---|---|
| baseline | `scripts/evaluate/experiments/results/gemma4-31b-full/baseline/` |
| Google Assistant (MTP) | `scripts/evaluate/experiments/results/gemma4-31b-full/assistant_k5/` |
| Eagle-3 Qwen (Ravi) | `scripts/evaluate/experiments/results/gemma4-31b-full-eagle3/eagle3_k5/` |
| Eagle-3 Llama (John) | `scripts/evaluate/experiments/results/gemma4-31b-full-redhat-ft/redhat_ft_k5/` |
| DSpark SWA (Mengmeng) | `scripts/evaluate/experiments/results/gemma4-31b-full-dspark-nemo782k/dspark_nemo782k_k8/` |

## Re-running

```bash
cd scripts/evaluate/experiments

# Google Assistant (MTP) + baseline (vLLM 0.24, conda env speculator)
python run_experiments.py --config gemma4-31b-full.yaml

# Eagle-3 Qwen (Ravi) / Eagle-3 Llama (John) (same env; skip baseline)
python run_experiments.py --config gemma4-31b-full-eagle3.yaml
python run_experiments.py --config gemma4-31b-full-redhat-ft.yaml

# DSpark SWA (Mengmeng) (vLLM 0.28.0+cu129; skip baseline)
source /nvmedata/chenw/envs/speculator-vllm028/bin/activate
export CUDA_HOME=/usr/local/cuda-12.9 PATH="$CUDA_HOME/bin:$PATH"
export PYTHONUNBUFFERED=1 FLASHINFER_DISABLE_VERSION_CHECK=1
python run_experiments.py --config gemma4-31b-full-dspark-nemo782k.yaml
```
