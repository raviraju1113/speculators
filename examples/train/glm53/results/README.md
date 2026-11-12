# GLM-5.3 DFlash2 draft — eval results (2026-11-03 .. 11-06)

> **Start with [`../SUMMARY.md`](../SUMMARY.md)**: the whole GLM-5.3 DFlash2 story (training
> lineage, the incoai gap analysis, conclusions, next steps). This file covers the Nov 3–6 eval
> suites in detail.

Draft acceptance for our GLM-5.3 DFlash2 drafts vs the released incoai GLM-5.3-DFlash2 card
draft, measured with Mengmeng's evaluation harness (`scripts/evaluate/experiments/run_experiments.py`
+ `mtp_server_eval/run_vllm_eval.py` on `mengmengj/glm-dflash2`).

## Setup

- **Server:** official FP8 GLM-5.3 (`zai-org/GLM-5.3`), TP8 on ONE 8xB200 node, vLLM 0.33,
  `--block-size 64 --enable-expert-parallel`. bf16 GLM-5.3 (1.51 TB) does not fit one B200 node,
  and DFlash2 cannot run with pipeline parallelism (its aux hidden states stay on their PP stage),
  so FP8 is the only single-node option. vLLM 0.28's DFlash2 path is broken (accept_len ~1.4).
- **Sampling:** the data-gen/serving regime — T 1.0, top-p 0.95, `reasoning_effort=max`,
  `clear_thinking=true`, max_tokens 4096, seed 42. Greedy rows (3-benchmark set only) use T 0.
- **Drafts:** ours are speculators DFlash2 checkpoints (5 layers, head_dim 64, block 7,
  `sample_from_anchor=false`, served at k=6) converted with `../export_dflash2_vllm.py`;
  incoai is 6 layers, head_dim 128, block 8, served at k=7.
- **Speedup** = draft decode tok/s / baseline decode tok/s on the same benchmark and node.
  `accept_len` is vLLM's mean acceptance length (accepted drafts + 1 per verify step).
- **Configs:** `../eval_glm53_full22_fp8_v033.yaml` (22-benchmark suite = Mengmeng's
  `glm53-full-v033-b300.yaml` list minus six SPEED-Bench categories with no data on sc3),
  `../eval_glm53_dflash2_fp8_v033.yaml` and `../eval_glm53_dflash2_ft_fp8_v033.yaml` (3-benchmark set).

## Drafts evaluated

| name | what | data |
|---|---|---|
| `ours_ep9_k6` | Kimi-K3-recipe DFlash2, epoch 9 of 10 (`glm53_dflash2_gemma782k_kimirecipe_v2/checkpoints/8`) | off-policy: gemma-4-31B answers to nemo782k (782k rows) |
| `ftA_*` | continued FT of `ours_ep9` (LR 2e-5 cosine, 3 epochs) — **clean** | on-policy GLM-5.3 regen, shard A only (`nemo782k_regen_glm53_snap20pct_shardA_20261104`, 141,818 rows) |
| `ftA_ep2best_k6` | `checkpoint_best` of that run (end of epoch 2; `glm53_dflash2_onpolicyA_ft3ep_lr2e-5/checkpoints/1`) | same |
| `ft_*` (3-bench suite) | same recipe on the OLD snapshot — **contaminated, superseded** | `snap20pct_20261031`: 56% shard-B rows generated on 2-node PP=2 pairs |
| `incoai_k7` | released incoai GLM-5.3-DFlash2 | — |

### Why the first on-policy FT (`ft_*`) is contaminated

Shard-B rows of the old `snap20pct_20261031` snapshot were generated on two-node PP=2 B200 pairs
and degenerate: truncated at the cap 37.4% vs 8.6% for shard A, no `</think>` 34.9% vs 7.9%,
median 4,733 vs 1,491 completion tokens, highly repetitive tails (zlib ratio < 0.15) 26.3% vs
0.1% — replies loop "...with Python:</think>## Setting Up..." until max_tokens. Shard-A rows are
byte-identical between the old and the new snapshot. ~57k shard-B rows survived the truncation
filter (46% of that FT's training rows). Treat `ft_*` numbers as superseded by `ftA_*`.

## Takeaways

1. **On-policy data helps, modestly:** clean on-policy FT lifts mean accept_len 2.83 → ~3.0
   (+6%; mean speedup 1.90x → 2.01x) and improves 20/22 benchmarks, most on code and RAG
   (HumanEval +12%, humaneval / swe-bench-pro +9%, speed-rag +8%).
2. **It saturates within ~half an epoch** of the 141k-row snapshot; epochs 1–3 are within noise.
3. **Still 23% short of incoai** (3.89, 2.53x). The gap opens at position 0 (0.68 vs 0.80 on the
   3-benchmark set), and is largest on speed-multilingual (ours/incoai 0.61) and tool_call (0.67).
4. **Noise floor:** n=50 per benchmark; single benchmarks move ±5–10% between checkpoints
   (gsm8k 4.24 → 4.75 → 4.53 → 4.24; tool_call 2.55 → 2.85 → 2.58). Compare means, not cells.
5. **Next levers:** more on-policy rows (Mengmeng's single-node shard-B regen), function-calling
   and multilingual prompts, and an incoai-shaped draft (6 layers / head_dim 128 / block 8).

Regenerate the tables below: `python examples/train/glm53/results/make_tables.py`
(per-prompt records: `<suite>/<draft>/mtp_eval_details.jsonl`).

## Tables

### glm53-full22-fp8-v033

| draft | mean accept_len | mean speedup |
|---|---:|---:|
| ours_ep9_k6 | 2.83 | 1.90x |
| ftA_ep0.5_k6 | 2.98 | 2.00x |
| ftA_ep1_k6 | 3.01 | 2.02x |
| ftA_ep2best_k6 | 2.99 | 2.01x |
| incoai_k7 | 3.89 | 2.53x |
| baseline (no spec) | -- | 1.00x (124 tok/s) |

Accept length per benchmark (decode speedup vs baseline):

| benchmark | n | ours_ep9_k6 | ftA_ep0.5_k6 | ftA_ep1_k6 | ftA_ep2best_k6 | incoai_k7 | baseline tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 50 | 4.24 (2.82x) | 4.75 (3.16x) | 4.53 (3.00x) | 4.24 (2.80x) | 5.42 (3.47x) | 127 |
| humaneval | 50 | 3.36 (2.24x) | 3.56 (2.38x) | 3.62 (2.42x) | 3.65 (2.44x) | 4.67 (3.02x) | 124 |
| mbpp | 50 | 2.94 (1.96x) | 3.06 (2.04x) | 3.10 (2.08x) | 3.15 (2.10x) | 4.11 (2.66x) | 124 |
| speed-coding | 50 | 2.83 (1.89x) | 3.07 (2.05x) | 3.01 (2.01x) | 3.04 (2.03x) | 4.04 (2.61x) | 124 |
| speed-multilingual | 50 | 2.44 (1.64x) | 2.55 (1.72x) | 2.60 (1.75x) | 2.58 (1.74x) | 4.21 (2.75x) | 124 |
| speed-rag | 50 | 2.83 (1.91x) | 2.95 (1.99x) | 3.12 (2.11x) | 3.07 (2.07x) | 3.84 (2.51x) | 124 |
| math500 | 50 | 3.86 (2.56x) | 4.07 (2.70x) | 4.11 (2.73x) | 4.06 (2.70x) | 4.82 (3.10x) | 124 |
| swe-bench-pro | 50 | 2.49 (1.67x) | 2.62 (1.76x) | 2.67 (1.79x) | 2.71 (1.81x) | 3.60 (2.34x) | 123 |
| aa-lcr | 50 | 2.93 (1.99x) | 3.02 (2.05x) | 3.10 (2.11x) | 3.04 (2.07x) | 3.93 (2.59x) | 119 |
| mt-bench | 50 | 2.23 (1.50x) | 2.34 (1.57x) | 2.35 (1.58x) | 2.35 (1.58x) | 2.96 (1.94x) | 124 |
| speed-qa | 50 | 2.26 (1.53x) | 2.37 (1.60x) | 2.39 (1.61x) | 2.39 (1.61x) | 3.06 (2.00x) | 125 |
| speed-writing | 50 | 2.06 (1.40x) | 2.13 (1.44x) | 2.13 (1.44x) | 2.15 (1.45x) | 2.71 (1.78x) | 123 |
| aime26 | 30 | 3.28 (2.19x) | 3.46 (2.30x) | 3.47 (2.31x) | 3.50 (2.33x) | 4.28 (2.76x) | 124 |
| HumanEval | 50 | 3.21 (2.14x) | 3.48 (2.32x) | 3.48 (2.32x) | 3.58 (2.39x) | 4.65 (3.00x) | 124 |
| math_reasoning | 50 | 3.89 (2.60x) | 3.97 (2.64x) | 3.97 (2.64x) | 4.05 (2.69x) | 4.99 (3.22x) | 126 |
| qa | 50 | 2.27 (1.53x) | 2.36 (1.59x) | 2.40 (1.62x) | 2.39 (1.61x) | 3.10 (2.03x) | 125 |
| question | 50 | 2.24 (1.51x) | 2.33 (1.57x) | 2.32 (1.56x) | 2.33 (1.56x) | 2.96 (1.93x) | 124 |
| rag | 50 | 2.74 (1.85x) | 2.90 (1.95x) | 2.90 (1.96x) | 2.90 (1.96x) | 3.68 (2.41x) | 124 |
| summarization | 50 | 2.75 (1.86x) | 2.90 (1.96x) | 2.95 (1.99x) | 2.93 (1.97x) | 3.85 (2.52x) | 124 |
| tool_call | 50 | 2.58 (1.74x) | 2.55 (1.72x) | 2.85 (1.91x) | 2.58 (1.73x) | 3.84 (2.49x) | 124 |
| translation | 50 | 2.65 (1.78x) | 2.82 (1.90x) | 2.80 (1.89x) | 2.85 (1.92x) | 3.97 (2.58x) | 126 |
| writing | 50 | 2.25 (1.51x) | 2.33 (1.57x) | 2.33 (1.57x) | 2.33 (1.57x) | 2.96 (1.93x) | 124 |

### glm53-dflash2-kimirecipe-ep9-fp8-v033

| draft | mean accept_len | mean speedup |
|---|---:|---:|
| ours_ep9_k6 | 2.77 | 1.85x |
| incoai_k7 | 3.69 | 2.38x |
| ours_ep9_k6_greedy | 3.08 | 2.07x |
| incoai_k7_greedy | 4.31 | 2.80x |
| baseline (no spec) | -- | 1.00x (124 tok/s) |

Accept length per benchmark (decode speedup vs baseline):

| benchmark | n | ours_ep9_k6 | incoai_k7 | ours_ep9_k6_greedy | incoai_k7_greedy | baseline tok/s |
|---|---:|---:|---:|---:|---:|---:|
| aime | 30 | 3.23 (2.15x) | 4.43 (2.84x) | 3.62 (2.42x) | 4.90 (3.17x) | 124 |
| gpqa | 50 | 2.12 (1.43x) | 2.75 (1.79x) | 2.54 (1.72x) | 3.69 (2.43x) | 124 |
| livecodebench | 50 | 2.95 (1.96x) | 3.90 (2.51x) | 3.09 (2.07x) | 4.33 (2.81x) | 124 |

### glm53-dflash2-onpolicy-ft-fp8-v033

**Contaminated first on-policy FT (`ft_*`, old snapshot with PP=2 shard-B rows) — superseded by
`ftA_*` above; kept for the record.** Same 3-benchmark set as the table above.

| draft | mean accept_len |
|---|---:|
| ft_ep0.5_k6 | 2.94 |
| ft_ep0.5_k6_greedy | 3.30 |
| ft_ep1_k6 | 2.97 |
| ft_ep1_k6_greedy | 3.32 |
| ft_ep2_k6 | 2.99 |
| ft_ep2_k6_greedy | 3.33 |
| ft_ep3_k6 | 2.95 |
| ft_ep3_k6_greedy | 3.32 |

Accept length per benchmark:

| benchmark | n | ft_ep0.5_k6 | ft_ep0.5_k6_greedy | ft_ep1_k6 | ft_ep1_k6_greedy | ft_ep2_k6 | ft_ep2_k6_greedy | ft_ep3_k6 | ft_ep3_k6_greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aime | 30 | 3.49 | 3.82 | 3.54 | 3.86 | 3.58 | 3.88 | 3.51 | 3.85 |
| gpqa | 50 | 2.28 | 2.84 | 2.26 | 2.84 | 2.29 | 2.85 | 2.26 | 2.88 |
| livecodebench | 50 | 3.05 | 3.25 | 3.10 | 3.27 | 3.10 | 3.25 | 3.08 | 3.22 |

