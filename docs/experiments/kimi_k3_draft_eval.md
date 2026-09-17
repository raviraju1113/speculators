# Kimi-K3 speculator evaluation: EAGLE3 vs DSpark (2026-09-14 / 2026-09-15)

Offline evaluation of two speculative-decoding drafts for Kimi K3 — the
TorchSpec-trained EAGLE3 draft and the published `RadixArk/Kimi-K3-DSpark`
draft — performed offline because the serving-side spec-decode path is
broken for both methods (see "Why offline" below).

**Contents**
- [Draft under test (EAGLE3)](#draft-under-test)
- [Setup (measurement conditions)](#setup-measurement-conditions)
- [Metric definitions](#metric-definitions)
- [Why offline](#why-offline)
- [Capture convention (largest pitfall found)](#capture-convention-largest-pitfall-found)
- [Result 1 — general chat (EAGLE3)](#result-1--general-chat-torchspecs-eval-set-teacher-forced)
- [Result 2 — context-length sweep (EAGLE3)](#result-2--context-length-sweep-aa-lcr-on-policy)
- [Result 3 — benchmark slice + throughput (EAGLE3)](#result-3--benchmark-slice--throughput)
- [RadixArk/Kimi-K3-DSpark — draft under test + serving notes](#radixarkkimi-k3-dspark-hf--second-draft-evaluated-2026-09-15)
- [Reproduction check against the model card](#reproduction-check-against-the-model-card)
- [Full 25-benchmark sweep (DSpark)](#full-25-benchmark-sweep)
- [Per-slot accuracy + confidence calibration (DSpark)](#per-slot-accuracy--confidence-calibration-dspark)
- [Normalized view: accept_length / accept_rate](#normalized-view-accept_length--accept_rate)
- [Bottom line](#bottom-line)
- [Open follow-ups](#open-follow-ups)
- [Artifacts](#artifacts)

## Draft under test

- `Eagle3DeepseekV2ForCausalLM` (`model_type: deepseek_v3`): single DeepSeek-V3
  MLA decoder layer; hidden 7168; q_lora 1536 / kv_lora 512; head dims
  nope/rope/v = 128/64/128; full 163 840 vocab (no t2d/d2t pruning);
  `rope_theta` 10 000; aux hidden-state layers `[48, 68, 88]`
  (TorchSpec 0-based convention); TTT length 4.
- Trained on-policy on K3-regenerated code/math data
  (`train_regen_code_math_merged_converted.jsonl`).
- Assets: `/import/ml-sc-scratch5/chenw/models/kimi-k3-draft-torchspec/`
  (native distcp `iter_0039388/`, HF exports `hf/` and `hf-vllm/`, TorchSpec
  source, training data).
- **Training-time metadata** (`iter_0039388/meta.json`): `eval/avg_acc` 0.594,
  `eval/simulated_acc_len` 1.436, `eval/avg_loss` 2.167.

## Setup (measurement conditions)

| item | value |
|---|---|
| Target model | Kimi-K3-patched (~1.5 TB mxfp4 MoE, 93 layers, 896 experts, KDA+MLA hybrid) |
| Hardware | 1 node, 8× NVIDIA B300 SXM6 (275 GB), tensor-parallel 8 |
| Serving stack | vLLM 0.28.0 (internal build with `vllm/models/kimi_k3/`), torch 2.13.0+cu130, conda env `/import/ml-sc-scratch6/chenw/conda_env/kimi_k3` |
| Hidden-state capture | `extract_hidden_states` method + `ExampleHiddenStatesConnector` (file backend), `prefix_only` mode, vLLM layer ids `[49, 69, 89]` + final (93) — see "capture convention" |
| Draft eval harness | TorchSpec's own `Eagle3Model` TTT wrapper + `Eagle3DeepseekV2ForCausalLM`, weights loaded from the native distcp checkpoint (0 missing / 0 unexpected keys), bf16, 1 GPU |
| Draft attention backend | `flex_attention` for sequences > ~2k (verified vs `sdpa`: avg_acc 0.4924 vs 0.4921 on the chat set; sdpa's TTT cached path is O(S²) memory) |
| Generation (on-policy sets) | greedy (temperature 0), target-only server |
| Scripts | `scripts/evaluate/kimi_k3_offline_eval/{prep_and_collect,sweep_collect,run_ttt_eval}.py` |

## Metric definitions

All metrics follow TorchSpec's training-time eval exactly
(`torchspec/training/eagle3_trainer.py`), so numbers are directly comparable
to the checkpoint metadata.

- **Teacher-forced TTT step accuracy** `acc_i` (i = 0..3): at TTT step *i*,
  the fraction of loss-masked positions where the draft's argmax token equals
  the **target model's argmax** (computed as
  `argmax(final_norm(last_hidden) · lm_headᵀ)` from the captured pre-norm
  final hidden states). Positions are teacher-forced: the draft always
  receives ground-truth previous tokens and, recursively, its own previous
  step's hidden output.
- **Average accuracy** `avg_acc = mean_i(acc_i)`.
- **Simulated acceptance length**
  `sim_acc_len = Σ_{k=0..3} Π_{i≤k} acc_i` — the expected number of
  *accepted draft tokens* per verify step under the independence
  approximation. Note this excludes the bonus token; serving-effective tokens
  per target forward ≈ `1 + sim_acc_len`.
- **Position loss** `vloss_i`: cross-entropy of the draft distribution vs the
  target soft distribution at step *i* (TorchSpec `LazyTarget`).
- **Aggregation**: per-sample batch means, averaged uniformly over samples
  (matches TorchSpec's `_aggregate_eval_metrics`).
- **decode_tok/s**: single-stream decode throughput reported by
  `scripts/evaluate/mtp_server_eval/run_vllm_eval.py` (completion tokens ÷
  decode wall time, per request, streaming; TTFT excluded).
- **Loss masks**: chat set masked by TorchSpec's `KimiK3Parser` (assistant
  think+response channels); on-policy sets masked over the generated span
  only.

## Why offline

vLLM 0.28 (this internal build) corrupts the *target's own* output stream when
eagle3 speculative decoding is enabled: token-loop degeneration after ~10–12
tokens (raw + chat, greedy + sampled), acceptance ~0. Target-only serving is
fully healthy, and `--enforce-eager` does not help, so this is spec-decode
state corruption (onset matches `attn_res_block_size = 12`; prime suspects are
the AttnRes bank commits / KDA recurrent-state handling of rejected
speculative tokens). Serving-side acceptance/throughput with spec decode is
therefore unmeasurable until fixed. Details: TRAINING.md §7.

## Capture convention (largest pitfall found)

The draft's aux features depend critically on *which tensor* the target server
captures. Diagnosed with a 9-layer window capture (47–49, 67–69, 87–89) and
per-combo replay on TorchSpec's 64-conversation chat eval set:

| capture mode | vLLM layer ids | avg_acc | sim_acc_len | acc_0 |
|---|---|---|---|---|
| prefix_only | 48, 68, 88 (as-given) | 0.031 | 0.075 | 0.073 |
| prefix_only | 47, 67, 87 (−1) | 0.103 | 0.119 | 0.105 |
| attn_res_stream | 48, 68, 88 (as-given) | 0.073 | 0.161 | 0.148 |
| attn_res_stream | 47, 67, 87 (−1) | 0.154 | 0.239 | 0.200 |
| attn_res_stream | 49, 69, 89 (+1) | 0.366 | 0.486 | 0.294 |
| **prefix_only** | **49, 69, 89 (+1)** | **0.492** | **0.936** | **0.480** |

The correct convention — confirmed against TorchSpec's own engine code
(`torchspec/inference/engine/vllm_engine.py` shifts config ids +1 for vLLM):

- **Mode `prefix_only`**: leave `VLLM_KIMI_K3_AUX_ATTN_RES_STREAM` unset/0.
- **vLLM ids = TorchSpec ids + 1**: TorchSpec `[48, 68, 88]` →
  `launch_vllm.py --target-layer-ids 49 69 89` (final layer 93 auto-appended
  for `verifier_last_hidden_states`, pre-final-norm).

The wrong combination silently costs up to **16×** in measured accuracy.
Sanity check isolating aux states as the variable: the captured final hidden
reproduces target next-token behavior at 0.675 argmax-vs-ground-truth
agreement on masked positions (normal for teacher-forced chat).

## Result 1 — general chat (TorchSpec's eval set, teacher-forced)

64 conversations (`eval_conversations.jsonl`, the set used for the training
metadata), correct capture convention:

| metric | measured | training metadata |
|---|---|---|
| avg_acc | **0.4921** | 0.594 |
| sim_acc_len | **0.9364** | 1.436 |
| acc_0 / acc_1 / acc_2 / acc_3 | 0.4800 / 0.5594 / 0.4843 / 0.4446 | — |
| vloss_0 / 1 / 2 / 3 | 2.908 / 2.712 / 3.148 / 3.405 | avg_loss 2.167 (position-weighted) |

The residual gap vs metadata is attributed to cross-build/hardware numerics
(their capture: TP8×PP2 on B200s on their vLLM build; ours: TP8 on B300 with
different autotuned mxfp4/attention kernels) and their eval running inside the
training loop (batching/padding, 8-rank all-reduce). Directionally verified:
the draft is real but modest.

## Result 2 — context-length sweep (AA-LCR, on-policy)

Paired design: same 20 questions per bin, documents truncated to the target
prompt length (`prepare_aa_lcr_sweep.py`, K3 tokenizer, ±32 tokens). Target
greedily generates ≤256 tokens per prompt; acceptance measured on the
generated span only.

| prompt tokens | n | avg_acc | sim_acc_len | acc_0 / acc_1 / acc_2 / acc_3 | vloss_0..3 |
|---|---|---|---|---|---|
| 1 024 | 20 | 0.5020 | **1.135** | 0.573 / 0.619 / 0.417 / 0.398 | 2.22 / 1.92 / 2.65 / 2.75 |
| 2 048 | 20 | 0.4546 | 0.803 | 0.437 / 0.515 / 0.441 / 0.426 | 3.84 / 3.48 / 3.77 / 3.84 |
| 4 096 | 20 | 0.4851 | 1.063 | 0.550 / 0.593 / 0.413 / 0.384 | 2.33 / 1.94 / 2.67 / 2.79 |
| 8 192 | 20 | 0.3778 | 0.553 | 0.327 / 0.457 / 0.377 / 0.350 | 4.15 / 3.61 / 4.02 / 4.12 |
| 16 384 | 20 | 0.0806 | **0.156** | 0.136 / 0.140 / 0.023 / 0.023 | 4.87 / 4.60 / 6.54 / 7.09 |

Roughly flat through 4k (bin-to-bin variance from 20 paired questions), clear
degradation at 8k, **collapse at 16k**. Likely causes, both actionable for the
next draft: (1) draft `rope_theta = 10 000` — a short-context rope base (the
target's own MLA layers are NoPE, so this was purely a draft-side choice);
(2) training data dominated by short sequences.

## Result 3 — benchmark slice + throughput

On-policy, 15 samples/benchmark, ≤512-token greedy generations, prompts from
`scripts/evaluate/mtp_server_eval/data/`. Baseline decode throughput measured
separately on a plain target-only server (no extraction overhead;
`run_vllm_eval.py`, 8 prompts × ≤512 tokens, temperature 0, single-stream):

| benchmark | avg_acc | sim_acc_len | acc_0 / 1 / 2 / 3 | baseline decode | projected w/ spec* |
|---|---|---|---|---|---|
| aime | 0.5563 | **1.212** | 0.565 / 0.642 / 0.522 / 0.495 | 108.0 tok/s | ~200–210 tok/s (≤2.21×) |
| livecodebench | 0.5539 | **1.102** | 0.507 / 0.647 / 0.530 / 0.531 | 108.0 tok/s | ~190–200 tok/s (≤2.10×) |
| gpqa | 0.3914 | 0.682 | 0.408 / 0.464 / 0.328 / 0.366 | 108.0 tok/s | ~145–155 tok/s (≤1.68×) |
| general chat (Result 1) | 0.4921 | 0.936 | — | — | ~165–175 tok/s (≤1.94×) |

Baseline details: decode 108.0 tok/s on all three benchmarks (decode-bound at
these context lengths); e2e 97.2–105.4 tok/s; mean TTFT 0.088–0.538 s.
Raw JSONs: `/import/ml-sc-scratch5/chenw/models/kimi-k3-data/throughput_baseline/`.

\* **Projection, not a measurement** (spec-decode serving blocked by the vLLM
bug): ceiling = baseline × (1 + sim_acc_len), discounted ~10% for draft
forward + verification overhead (1-layer MLA draft + 163k-vocab lm_head vs the
1T MoE target). Single-stream, short-context; batching and long contexts
change the picture.

The domain pattern matches the training data: aime/livecodebench (in-domain
code/math) sit ~1.8× higher in sim_acc_len than gpqa (out-of-distribution
science QA).

## RadixArk/Kimi-K3-DSpark (HF) — second draft evaluated (2026-09-15)

DSpark draft from https://huggingface.co/RadixArk/Kimi-K3-DSpark: 5-layer
Qwen3-style decoder (64 heads × 64 head_dim, GQA 16 kv-heads, intermediate
14 336), block_size 7 (K=7 draft tokens; the model card's "1 current + 7
  draft tokens"), Markov head (rank 256, vanilla)
+ confidence head, mask_token_id 163 824, **yarn rope** (factor 16, original
64k), full 163 840 vocab, dflash aux layer ids `[7, 23, 51, 67, 83]`
(0-based → vLLM capture ids `[8, 24, 52, 68, 84]` + final 93).

Serving notes:

- vLLM misroutes a generic `"DSparkDraftModel"` architecture into its
  weights-in-target DeepSeek-V4 path and force-inherits the target's mxfp4
  quantization (crash). Declaring `"Qwen3DSparkModel"` in config.json fixes
  loading. `num_speculative_tokens: 6` is required.
- **The spec-decode target-corruption bug is method-independent**: DSpark
  serving shows the same failure signature as eagle3 (correct first ~10–12
  tokens then token loops; acceptance 7/666 ≈ 1%). The bug is in K3's
  spec-decode state handling (KDA/AttnRes vs rejected tokens), not in any
  draft or method.

Offline eval (`run_dspark_eval.py`): weights load into this repo's
`DSparkDraftModel` with zero core mismatches (it is a speculators-family
export); captured hidden states replay through the repo's training forward.
Metrics differ from the EAGLE3 TTT eval: `accept_len` here is the *analytical*
expected accepted draft tokens per block — cumulative product over the 7 draft
slots of `accept_rate = 1 − TV(draft, target)` (probabilistic rejection
sampling), vs the argmax-agreement product over 4 TTT steps for EAGLE3.
Directionally comparable, not identical. `position_k_acc` is top-1 argmax
agreement at block slot k (slot 0 = anchor-adjacent, comparable to EAGLE3
acc_0).

| set | accept_len (of 7) | accept_rate | pos-0 acc | pos-0..6 acc |
|---|---|---|---|---|
| general chat (64 convs, teacher-forced) | **2.48** | 0.445 | 0.658 | 0.66/0.54/0.45/0.41/0.38/0.36/0.33 |
| aime (on-policy) | **2.27** | 0.346 | 0.503 | 0.50/0.46/0.43/0.40/0.38/0.36/0.33 |
| livecodebench (on-policy) | **2.13** | 0.335 | 0.578 | 0.58/0.49/0.41/0.38/0.32/0.29/0.29 |
| gpqa (on-policy) | 1.85 | 0.262 | 0.457 | 0.46/0.39/0.32/0.28/0.26/0.22/0.20 |

Head-to-head vs the TorchSpec EAGLE3 draft (expected accepted draft
tokens/verify step; EAGLE3 = sim_acc_len over 4 steps):

| set | EAGLE3 draft | DSpark draft | DSpark projected decode* |
|---|---|---|---|
| general chat | 0.94 | **2.48** | ~330–360 tok/s (≤3.48×) |
| aime | 1.21 | **2.27** | ~310–340 tok/s (≤3.27×) |
| livecodebench | 1.10 | **2.13** | ~295–325 tok/s (≤3.13×) |
| gpqa | 0.68 | **1.85** | ~270–295 tok/s (≤2.85×) |

\* Same projection method as before (baseline 108 tok/s × (1 + accept_len),
~5–10% overhead discount — DSpark drafts a whole block in ONE draft forward,
so overhead is lower than sequential EAGLE3 drafting). Not a measurement.

Also better positioned for long context on paper (yarn rope, 64k original) —
the aa-lcr length sweep for this draft is the natural follow-up.

### Reproduction check against the model card

The card reports `acc_len` measured live in SGLang serving (real rejection
sampling, K=7). Our numbers are offline/analytical (distribution-overlap
acceptance, vLLM `extract_hidden_states`) — a different measurement path, so
this is a sanity check on load correctness, not an apples-to-apples repro.
25 on-policy samples/set, same harness as above.

| dataset | our accept_len (of 7) | our AL (1+accept_len) | card acc_len | ratio |
|---|---|---|---|---|
| MT-Bench | 2.555 | **3.56** | 3.9342 | 0.90 |
| GSM8K | 2.982 | **3.98** | 5.4176 | 0.73 |

MT-Bench lands within 10% of the card; GSM8K undershoots more (still same
direction — GSM8K > MT-Bench > AIME26 in both, and our GSM8K AL of ~4 already
beats every set in the head-to-head table above). Structural load checks also
passed: `fc.weight` is exactly 5×7168 (5 aux layers, matches the card), and all
62 tensors loaded into the repo's `DSparkDraftModel` with zero missing/
unexpected keys. **Conclusion: the draft is loaded correctly** — the gap to
the card is attributable to the offline-analytical vs. live-SGLang-serving
measurement gap (and dataset-specific prompt sampling), not a loading bug.
The earlier "not that good" read was comparing against benchmarks (aime,
gpqa, livecodebench, general chat) that are harder than the card's easiest
sets (GSM8K, HumanEval, MBPP) where the card's own numbers also peak highest.

### Full 25-benchmark sweep

Every set in `scripts/evaluate/mtp_server_eval/data/` plus 2 aa-lcr length
bins, evaluated with `run_dspark_eval.py`. On-policy sets (all except
`chat64`) use greedy target generations; `gsm8k`/`mtbench` used
`--max-gen-tokens 512`, the other 20 new sets used `--max-gen-tokens 256`
(kept short to make the batch tractable) — a methodological difference from
the earlier 512-token aime/gpqa/livecodebench/chat64 runs, noted for anyone
comparing across sets. `n` is samples/questions, capped at 15–25 per set for
turnaround time. `speed-low-entropy` is omitted: every prompt in it exceeds
the 15,360-token server context and was skipped (long repetitive code
boilerplate). `qa` and `speed-qa` share the same 15 underlying questions
(verified via diff/md5) — identical numbers are expected, not a bug.

AL = 1 + accept_len (bonus token included, the vLLM/SGLang serving
convention); AR = measured `accept_rate` (mean per-position acceptance over
the 7 draft slots (K=7), not a linear back-derivation from AL); `full_acc` =
top-1 argmax agreement pooled over all 7 slots (stricter than AR — see the
per-slot section below for why); `conf_err` = confidence head's mean
absolute calibration error against realized acceptance (lower = better);
`proj. decode*` = projected single-stream decode throughput, **not
measured** (spec-decode serving is blocked by the vLLM bug — see "Why
offline"): `108 tok/s baseline × AL × 0.90` (a flat 10% discount for draft
forward + verification overhead; the real number would vary by domain since
overhead is roughly constant per step while AL varies, so the discount ratio
isn't actually flat — treat this column as directional, not precise).
Sorted by AL descending.

| dataset | n | AR | AL | full_acc | conf_err | proj. decode* |
|---|---|---|---|---|---|---|
| **bfcl** | 15 | 0.570 | **4.58** | 0.625 | 0.230 | **445 tok/s** |
| **math500** | 15 | 0.551 | **4.47** | 0.611 | 0.240 | 434 tok/s |
| aa-lcr-4k | 15 | 0.496 | 4.26 | 0.549 | 0.207 | 414 tok/s |
| aa-lcr-1k | 15 | 0.486 | 4.20 | 0.539 | 0.217 | 408 tok/s |
| tool_call | 15 | 0.491 | 4.19 | 0.550 | 0.223 | 407 tok/s |
| mbpp | 15 | 0.502 | 4.18 | 0.568 | 0.232 | 406 tok/s |
| aime26 | 15 | 0.469 | 4.02 | 0.520 | 0.229 | 391 tok/s |
| gsm8k | 25 | 0.475 | 3.98 | 0.544 | 0.249 | 387 tok/s |
| translation | 15 | 0.420 | 3.73 | 0.474 | 0.249 | 363 tok/s |
| speed-multilingual | 15 | 0.410 | 3.63 | 0.479 | 0.238 | 353 tok/s |
| writing | 15 | 0.415 | 3.58 | 0.474 | 0.237 | 348 tok/s |
| qa / speed-qa | 15 | 0.422 | 3.57 | 0.481 | 0.244 | 347 tok/s |
| mt-bench | 25 | 0.398 | 3.55 | 0.467 | 0.237 | 345 tok/s |
| chat64 (teacher-forced) | 64 | 0.445 | 3.48 | 0.448 | 0.199 | 338 tok/s |
| speed-coding | 15 | 0.389 | 3.48 | 0.449 | 0.226 | 338 tok/s |
| swe-rebench | 15 | 0.389 | 3.45 | 0.464 | 0.222 | 335 tok/s |
| speed-writing | 15 | 0.372 | 3.39 | 0.440 | 0.219 | 330 tok/s |
| rag | 15 | 0.345 | 3.29 | 0.403 | 0.213 | 320 tok/s |
| aime | 15 | 0.346 | 3.27 | 0.408 | 0.243 | 318 tok/s |
| speed-rag | 15 | 0.325 | 3.16 | 0.369 | 0.211 | 307 tok/s |
| livecodebench | 15 | 0.335 | 3.13 | 0.394 | 0.222 | 304 tok/s |
| summarization | 15 | 0.327 | 3.00 | 0.397 | 0.208 | 292 tok/s |
| swe-bench-pro | 15 | 0.301 | 2.93 | 0.358 | 0.203 | 285 tok/s |
| **gpqa** | 15 | 0.263 | **2.85** | 0.304 | 0.199 | 277 tok/s |
| speed-low-entropy | 0/15 | — | — | — | — | — *(excluded: all prompts exceed 15,360-tok context)* |

Every domain projects at **2.6–4.1× the measured 108 tok/s baseline**, even
the weakest (gpqa, 2.85×) — no domain falls near parity the way the EAGLE3
draft's 16k-context bin did.

Range: **AL 2.85–4.58** across 24 evaluated domains (median ≈ 3.5). Highest on
structured/in-distribution tasks (bfcl function-calling, math500, tool_call,
mbpp — all draft-training-adjacent code/math) and the two short/medium aa-lcr
bins; lowest on gpqa (out-of-domain science QA) and swe-bench-pro (long,
unusual-format software-engineering prompts). No domain drops below AL 2.85 —
unlike the EAGLE3 draft, which fell to 1.68 on gpqa and collapsed to 1.16 by
16k context.

### Per-slot accuracy (DSpark)

`pos-k` is `full_acc` (see the master table above for the pooled value)
broken out per draft slot: slot 0 = immediately after the anchor, slot 6 =
the last/deepest draft position.

| dataset | pos-0 | pos-1 | pos-2 | pos-3 | pos-4 | pos-5 | pos-6 |
|---|---|---|---|---|---|---|---|
| bfcl | 0.76 | 0.73 | 0.67 | 0.59 | 0.57 | 0.54 | 0.51 |
| math500 | 0.72 | 0.70 | 0.65 | 0.62 | 0.57 | 0.51 | 0.51 |
| aa-lcr-4k | 0.71 | 0.66 | 0.60 | 0.55 | 0.48 | 0.44 | 0.40 |
| mbpp | 0.68 | 0.67 | 0.58 | 0.57 | 0.53 | 0.51 | 0.45 |
| gsm8k | 0.67 | 0.67 | 0.58 | 0.52 | 0.51 | 0.46 | 0.41 |
| aa-lcr-1k | 0.69 | 0.65 | 0.60 | 0.53 | 0.48 | 0.43 | 0.39 |
| tool_call | 0.72 | 0.65 | 0.60 | 0.52 | 0.49 | 0.45 | 0.42 |
| aime26 | 0.64 | 0.61 | 0.55 | 0.49 | 0.50 | 0.45 | 0.40 |
| translation | 0.64 | 0.58 | 0.52 | 0.47 | 0.42 | 0.37 | 0.32 |
| speed-multilingual | 0.62 | 0.56 | 0.49 | 0.45 | 0.43 | 0.43 | 0.38 |
| qa / speed-qa | 0.60 | 0.55 | 0.52 | 0.45 | 0.43 | 0.42 | 0.40 |
| writing | 0.60 | 0.55 | 0.50 | 0.45 | 0.43 | 0.40 | 0.39 |
| chat64 (teacher-forced) | 0.66 | 0.54 | 0.45 | 0.41 | 0.38 | 0.36 | 0.33 |
| mt-bench | 0.58 | 0.51 | 0.49 | 0.45 | 0.43 | 0.42 | 0.39 |
| swe-rebench | 0.58 | 0.55 | 0.51 | 0.43 | 0.43 | 0.40 | 0.36 |
| speed-coding | 0.55 | 0.55 | 0.46 | 0.42 | 0.42 | 0.39 | 0.35 |
| speed-writing | 0.57 | 0.54 | 0.44 | 0.40 | 0.38 | 0.38 | 0.36 |
| aime | 0.50 | 0.46 | 0.43 | 0.40 | 0.38 | 0.36 | 0.33 |
| rag | 0.51 | 0.49 | 0.43 | 0.40 | 0.36 | 0.33 | 0.31 |
| summarization | 0.53 | 0.50 | 0.44 | 0.35 | 0.33 | 0.30 | 0.33 |
| livecodebench | 0.58 | 0.49 | 0.41 | 0.38 | 0.32 | 0.29 | 0.29 |
| speed-rag | 0.52 | 0.47 | 0.37 | 0.35 | 0.31 | 0.29 | 0.26 |
| swe-bench-pro | 0.48 | 0.44 | 0.38 | 0.34 | 0.30 | 0.28 | 0.28 |
| gpqa | 0.46 | 0.39 | 0.32 | 0.28 | 0.26 | 0.22 | 0.20 |

Two observations beyond the master table's AR/AL/full_acc/conf_err columns:

- **Decay shape is consistent across domains**: pos-0 is always highest,
  decaying roughly monotonically to pos-6, but the *slope* varies —
  bfcl/math500 decay gently (0.76→0.51, a 33% relative drop) while
  gpqa/swe-bench-pro decay sharply (0.46→0.20, a 57% drop). This means the
  Markov head's benefit (predicting later tokens conditioned on earlier draft
  tokens, not just the anchor) is domain-dependent — it holds up longer on
  structured/short-answer domains than on open-ended or unfamiliar ones.
  `summarization` is the one set that doesn't decay monotonically (0.33 at
  pos-3 dips below pos-6's 0.33 — noise at n=15, not a real signal).
- **`conf_err` sits in a narrow band (0.20–0.25) across every domain**,
  including the worst-performing one (gpqa, 0.199 — actually the *best*
  calibrated). This is a genuinely good sign for deployment: the confidence
  head's calibration doesn't degrade on hard/OOD inputs the way raw
  acceptance does, so adaptive-verification budget sizing (`--enable-adaptive-verification`
  in vLLM's dspark path) should remain reliable even where the draft itself
  is weak — worth verifying once serving is unblocked.

## Normalized view: accept_length / accept_rate

Both drafts restated in vLLM's serving-counter conventions —
`accept_length = 1 + accepted_tokens/step` (includes the bonus token; maps
directly to the speedup ceiling vs the 108 tok/s baseline) and
`accept_rate = accepted_tokens / drafted_tokens`. Derived from the measured
per-step/per-slot numbers above (EAGLE3 drafts 4 tokens/step, DSpark 7).

| set | EAGLE3 rate | EAGLE3 len | DSpark rate | DSpark len |
|---|---|---|---|---|
| general chat | 0.234 | 1.94 | 0.445 | **3.48** |
| aime | 0.303 | 2.21 | 0.346 | **3.27** |
| livecodebench | 0.275 | 2.10 | 0.335 | **3.13** |
| gpqa | 0.170 | 1.68 | 0.263 | **2.85** |
| aa-lcr 1k / 4k | 0.284 / 0.266 | 2.13 / 2.06 | 0.486 / 0.496 | 4.20 / 4.26 |
| aa-lcr 8k | 0.138 | 1.55 | — | — |
| aa-lcr 16k | 0.039 | 1.16 | — | — |

Footnote: EAGLE3 rates derive from greedy argmax agreement (TTT), DSpark's
from the analytical distribution-overlap acceptance — both estimate
rejection-sampling acceptance but are not bit-identical definitions. DSpark's
AR here is the measured per-position acceptance rate (`accept_rate` field),
not (AL-1)/K — those aren't the same quantity since acceptance decays across
the 7 draft slots. The DSpark aa-lcr 8k/16k bins have not been run yet.

## Bottom line

**EAGLE3 draft (TorchSpec, iter 39388):**
- Verified draft quality (correct convention, TorchSpec's own harness):
  **avg_acc 0.492, sim_acc_len 0.94** on general chat; **1.10–1.21** on
  in-domain code/math benchmarks; **0.68** on out-of-domain science QA.
  In serving terms ≈ 1.7–2.2 effective tokens per target forward (incl. the
  bonus token) where the draft works.
- **Acceptance collapses at 16k context** (sim_acc_len 0.16) — do not use this
  draft for long-context serving; the next draft should fix `rope_theta` and
  train on long sequences.
- Hidden-state capture convention for this draft/method: `prefix_only` + vLLM
  ids `[49, 69, 89]`. The wrong choice silently costs up to 16×.

**DSpark draft (RadixArk/Kimi-K3-DSpark):**
- Clearly stronger and more robust: **AL 2.85–4.58 across 24 domains**
  (median ≈3.5) — roughly **1.5–2.7× the EAGLE3 draft's AL** on every
  comparable set, and no domain collapse (worst case, gpqa, still beats
  EAGLE3's best case).
- Reproduction check against the model card's live-SGLang numbers landed
  within 10–27% (MT-Bench, GSM8K) on a different (offline/analytical)
  measurement path — confirms the checkpoint loads and runs correctly.
- Hidden-state capture convention for this draft/method: `prefix_only` +
  vLLM ids `[8, 24, 52, 68, 84]` (dflash `target_layer_ids [7,23,51,67,83]`
  +1). Note the convention is method-specific — do not reuse EAGLE3's ids.
- Long-context (aa-lcr 8k/16k) not yet measured for this draft — the natural
  next step given its yarn rope design, especially since that's exactly where
  the EAGLE3 draft failed.

**Both drafts:**
- Serving-side spec decode (and therefore measured speedup) remains blocked
  by the vLLM target-corruption bug — confirmed **method-independent**
  (eagle3 and dspark corrupt the target identically). All AL/AR numbers above
  are offline/analytical, not live-served.

## Open follow-ups

Roughly in priority order:

1. **Escalate the vLLM spec-decode target-corruption bug.** Confirmed
   method-independent (eagle3 and dspark both fail identically); this is the
   single blocker on every serving-side number in this doc (real throughput,
   real acceptance under batching, concurrent requests). Evidence is in "Why
   offline" and the DSpark serving notes above.
2. **DSpark aa-lcr 8k/16k.** The one benchmark axis not yet measured for the
   stronger draft, and the most interesting given its yarn rope design —
   directly testable against the EAGLE3 draft's collapse at the same lengths.
3. **RULER V2 1M** (from the model card) — needs either the vLLM fix or
   SGLang, since 1M-token prompts are impractical to capture via
   `extract_hidden_states` (86 GB+ of hidden states per sample at 6 layers).
4. **DSpark's `--enable-adaptive-verification` path** — the confidence-head
   calibration numbers above suggest it should work well; untestable until
   serving is unblocked.
5. If a from-scratch MTP/DSpark draft is trained (tasks already staged, see
   TRAINING.md §7): fix `rope_theta` (use a large base or match the target's
   NoPE design) and include long sequences in training data, per the EAGLE3
   long-context failure analysis in Result 2.
6. Widen the 25-benchmark sweep's `n` (currently 15–25/set) and standardize
   `--max-gen-tokens` (currently 256 vs 512 depending on when the set was
   collected) if these numbers need to support a specific go/no-go decision
   rather than a directional read.

## Artifacts

- Harness: `scripts/evaluate/kimi_k3_offline_eval/{prep_and_collect,sweep_collect,run_ttt_eval,run_dspark_eval}.py`
- EAGLE3 collected hidden states + metric JSONs:
  `/import/ml-sc-scratch5/chenw/models/kimi-k3-data/{eval_hs_*,sweep_hs_*,bench_hs_*}/`
- DSpark collected hidden states + metric JSONs:
  `/import/ml-sc-scratch5/chenw/models/kimi-k3-data/dspark_hs_*/` (25 sets;
  each dir has `dspark_eval_results.json`)
- DSpark checkpoint (local copy, arch patched to `Qwen3DSparkModel` for vLLM
  loading): `/import/ml-sc-scratch5/chenw/models/Kimi-K3-DSpark/`
- Throughput: `/import/ml-sc-scratch5/chenw/models/kimi-k3-data/throughput_baseline/`
- Sweep prompt bins: `/import/ml-sc-scratch5/chenw/models/kimi-k3-data/aa_lcr_sweep_kimi/`
