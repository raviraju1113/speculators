# Ablation: on-policy vs off-policy completions for DSpark drafts

**Question.** Does a DSpark draft trained on the target's *own* completions
(on-policy) accept materially more tokens than one trained on *another model's*
completions (off-policy), when everything else is held fixed?

**Why it matters.** Every new target model currently triggers a full response
regeneration (an 8-GPU, multi-hour job for gemma-4-31B). If off-policy data is
within noise of on-policy, we can reuse one curated corpus across targets and skip
regeneration entirely.

> Terminology. "On/off-policy" here is about **where the training text came
> from**. It is *not* `--use-off-policy-tokens` (EAGLE-3's teacher-forced TTT
> chaining), which does not apply to DSpark.

## Design

| | |
|---|---|
| Target | `meta-llama/Llama-3.1-8B-Instruct` (32 layers, hidden 4096, vocab 128256) — local copy `/import/ml-sc-scratch1/mengmengj/models/llama3.1-8b-instruct` |
| Why not Qwen3-4B | Qwen3 is a hybrid thinking model: on-policy regen would emit `<think>` traces (or empty think tags in no-think mode) that the off-policy text never has, adding a format confound on top of the policy one. Llama-3.1-8B is also the canonical spec-decoding target with published DFlash drafts to sanity-check the harness against. |
| Prompt pool | `kimi_nemotron_782k` (kimi-mtp + nemotron stem/code/math), **single-turn rows only**, seed-0 shuffle → 100k train + 1k held-out prompts. Multi-turn rows (11%) excluded: their later user turns were written against the original replies, so regenerating replies makes them incoherent. 403 rows truncated at the source's 8192 cap dropped. (`export_prompts.py`, stats in `split_stats.json`) |
| Prompt-set matching | After regen, any prompt whose ON reply errored or hit the 6144-token cap is dropped from **both** arms (`finalize_arms.py`), so both arms train on the identical prompt set. Smoke test: 7/64 replies capped — Llama-3.1-8B at T=0.6 loops on hard math (digit spirals). Keeping them would make ~40% of ON tokens degenerate loops; dropping them only from ON would change the prompt set. |
| Draft | DSpark, block 8, **5 layers**, aux layers `[1, 8, 15, 22, 29]` (matches `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`), mask token 128002 `<\|reserved_special_token_0\|>` (the auto-fallback picks `<\|eot_id\|>`), draft vocab 32k, Markov rank 256 vanilla, confidence head w/ Markov, loss `{"ce":0.1,"tv":0.9}`, **`sample_from_anchor=False`** (vLLM's serving convention — see the `dspark-train-serve-parity` skill) |

### Arms

| Arm | Training text | Hard labels (CE, 10%) | Soft targets (TV, 90%) |
|---|---|---|---|
| **OFF** | original corpus responses (Kimi-family, not Llama) | off-policy tokens | Llama's distribution on off-policy prefixes |
| **ON** | Llama-3.1-8B-Instruct regenerations of the *same* 100k prompts (its `generation_config`: T=0.6, top-p 0.9, max 6144 new tokens) | on-policy tokens | Llama's distribution on on-policy prefixes |

Everything else is identical: prompts, sample count, seed, data order, hyperparameters,
epochs, hardware, eval harness. The only variable is the response text.

Why this might come out equal: 90% of DSpark's loss is TV against the **verifier's
own** next-token distribution, computed live by vLLM on whatever prefix is fed. So
even the OFF arm is distilled toward Llama's predictions; what stays off-policy is
(a) the *prefix distribution* the draft is trained on (exposure mismatch at serve
time) and (b) the 10% hard-label CE term. The optional 2×2 below separates those.

### Controlled-for, and what is deliberately left in

- **Response length differs between arms**, so steps/epoch differ (packed 8192-token
  batches). Primary comparison is at **equal epochs over equal samples** — that is
  the real decision ("reuse this corpus vs regenerate it"). Quarter-epoch checkpoints
  give a secondary **matched-training-tokens** comparison.
- **Draft vocab (top-32k from `token_freq.pt`) is built per arm from its own data.**
  That is faithful to the no-regen scenario (you only have off-policy text to build
  it from). Draft-vocab coverage of the target's eval outputs is reported for both
  arms so any vocab-driven share of a gap is visible.
- Training-time val metrics are **per-arm diagnostics only** — each arm's val split
  comes from its own text, so they are not comparable across arms. vLLM acceptance on
  a shared benchmark suite is the result.

### Evaluation (the result)

vLLM 0.26 serve, Llama-3.1-8B-Instruct TP=1 on 1×A100, `method: dspark`,
`num_speculative_tokens: 7` (native for `sample_from_anchor=False`, block 8),
`scripts/evaluate/experiments/run_experiments.py`.

- Benchmarks: `mt-bench` (80, chat), `alpaca` (300, chat), `gsm8k` (300, math),
  `math500` (300, math), `humaneval` (164, code), `livecodebench` (200, code),
  **`heldout_eval`** (500 held-out corpus prompts, in-distribution).
- Temperature **0.0** primary (canonical acceptance); **0.6/top-p 0.9** secondary on
  mt-bench + gsm8k + heldout (the sampling regime the ON data was generated under).
- Metrics: `accept_len` (primary), per-position acceptance from vLLM's
  `SpecDecoding metrics` log line, decode tok/s and speedup vs a same-config
  baseline (experiment #0).
- Harness controls: `baseline` (no draft) first; `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`
  as a published known-good draft — if it scores sanely and ours don't, the problem
  is our checkpoint, not the harness.

### Decision rule (fixed before looking at results)

Let Δ = AL(ON) − AL(OFF), macro-averaged over the 7 benchmarks at T=0, at the final
checkpoint.

- **Equivalent** ("no need to regenerate") if Δ ≤ 3% of AL(OFF) **and** no single
  domain (chat / math / code / in-dist) has Δ > 6%.
- **On-policy wins** if Δ > 3% and exceeds 2× the seed-to-seed spread.
- Phase 2 (seed replicates) is triggered when |Δ| falls within [1.5%, 6%] — the
  zone where seed noise could flip the call.

## Phases and compute (A100/H100 only — B200s and H200s reserved for the K3 demo)

| Phase | What | Where | Est. |
|---|---|---|---|
| 0 | Env + smoke test (regen 64 prompts; online DSpark train 50 steps; export; serve) | sc-c120 / sc-c96 | < 1 h |
| 1a | ON-arm regeneration: 100k prompts, vLLM DP=8 on sc-c96 | sc-c96 (8×A100) | ~1–2 h (≈55M output tokens) |
| 1b | Train OFF and ON concurrently, 4 GPUs each (vLLM hidden-state server ×1 + 3 train ranks, online `--on-generate delete`) | sc-c96 | sized from Phase-0 throughput |
| 1c | Eval both + baseline + z-lab control | 1–2 A100s | ~4–6 h |
| 2 | (triggered, done) seed 1 for both arms | sc-c96 | 4h30m |
| 3 | (optional) 2×2: add CE-only (`{"ce":1.0}`) for both arms — does the TV distillation term explain any OFF≈ON result? | sc-c96 | = 1b |

## Results (2026-10-22): on-policy wins, by ~10% acceptance length

**Verdict under the pre-registered rule: ON wins.** Over the 7 benchmarks, ON beats OFF
on accept_len by **+9.6%** (seed 0 +9.1%, seed 1 +10.3%). That is above the 3% bar, and
above 2× the seed-to-seed spread (arm-level spread is 2.6% for OFF and 4.1% for ON).
Every domain is above 6%. No benchmark has OFF ahead in both seeds. So reusing
another model's completions does **not** come free. For Llama-3.1-8B, regenerating the
data with the target buys ~10% acceptance length.

| benchmark | z-lab DFlash k=9 | OFF s0 | OFF s1 | ON s0 | ON s1 | Δ s0 | Δ s1 | Δ mean |
|---|---|---|---|---|---|---|---|---|
| mt-bench | 3.98 | 2.47 | 2.64 | 2.65 | 2.63 | +7.5% | −0.4% | +3.4% |
| writing | 3.83 | 2.29 | 2.59 | 2.62 | 2.83 | +14.6% | +9.2% | +11.7% |
| qa | 4.45 | 2.61 | 2.62 | 2.80 | 3.02 | +7.2% | +15.3% | +11.3% |
| gsm8k | 4.43 | 3.35 | 3.64 | 3.60 | 3.57 | +7.2% | −2.0% | +2.4% |
| math500 | 4.97 | 4.33 | 4.55 | 4.67 | 5.12 | +7.9% | +12.4% | +10.2% |
| humaneval | 4.88 | 3.63 | 3.15 | 3.60 | 3.73 | −0.8% | +18.2% | +8.0% |
| livecodebench | 4.48 | 3.51 | 3.57 | 4.22 | 4.26 | +20.2% | +19.4% | +19.8% |
| kn782k_heldout | 4.48 | 3.61 | 3.61 | 4.19 | 4.03 | +16.1% | +11.6% | +13.8% |

The values are accept_len, and Δ = AL(ON)/AL(OFF) − 1. By domain (mean over seeds): chat +8.8%,
math +6.3%, code +13.9%, in-dist +13.8%. Macro over all 8 benchmarks: +10.1%.

Setup:
- Data: 94,032 prompts in both arms. 5,935 (5.9%) were dropped because the ON reply hit
  the 6144 cap, and 33 because the prompt exceeded 2048 tokens.
- ON replies are ~12% shorter, so over 3 epochs ON took ~6.43k steps vs OFF's ~7.55k.
  This confound biases against ON, so the gap is if anything understated.
- Each pair of arms trained in ~4h30m, concurrently on sc-c96.
- Eval: T=0, max 2048 new tokens, A100 TP1, batch 1. Each benchmark has 200 samples,
  except mt-bench, writing and qa (80) and humaneval (164).

### Eval reproducibility (read before re-running)

- **The first seed-0 eval (job 12527538) did not reproduce, so it is excluded.** It is
  kept in `results/llama31-8b-onpolicy-ablation-t0/{off,on}_policy`.
  - Two later evals of the same, unchanged checkpoints (`-t0-rerun`, `-t0-rerun2`)
    agree with each other exactly: 1204/1204 samples have the same completion length,
    and accept_len matches to the decimal.
  - The seed-1 checkpoints were also bit-reproducible across two evals.
  - The bad run gave +3.4% instead of +9.1%, and it flipped chat to −2.9%.
  - Ruled out as causes: the weights (untouched), the torch.compile cache (seed 1 also
    compiled fresh and reproduced), and the GPU/port mapping.
  - The bad run was the only one whose drafts compiled while the baseline and z-lab
    servers were starting on neighbouring GPUs. That is the remaining suspect, but it
    is unproven.
  - **Rule: evaluate any new checkpoint twice and confirm bit-identical
    `completion_tokens` before trusting a number.**
- At T=0 a draft's continuations still differ from the no-draft baseline: ~37% of
  samples have the same length, z-lab included. That is expected spec-decode numerics,
  and it is deterministic run to run.
- The z-lab and baseline numbers come from the same job as the excluded run. They are
  a sanity control only.

Other caveats:
- Both arms reach 72–81% of z-lab's accept_len (z-lab used a larger training budget
  and block 10). The comparison is relative only.
- Training is near-deterministic. Final val accept_len is OFF 3.683 / 3.667 and
  ON 4.345 / 4.346 across seeds. Seed-to-seed eval differences of up to ±15% per
  benchmark are therefore mostly from the small per-benchmark n. Trust macro and
  domain numbers over single benchmarks.
- Only epoch-end checkpoints were kept, so no matched-training-tokens comparison was
  possible.
- Not yet tested: whether this holds for a different target (e.g. Qwen3-4B), and how
  much of the gap the TV term closes. Phase 3 (CE-only 2×2) would answer the latter.

## Files

- `export_prompts.py` — carve splits from the 782k corpus
- `run_regen.sh` — Phase 1a, on-policy regeneration
- `finalize_arms.py` — Phase 1a', drop capped/failed prompts from both arms
- `smoke_test.sh` — Phase 0
- `train_arm.sh` — Phase 1b, one arm (`ARM=off|on`)
- `scripts/evaluate/experiments/llama31-8b-onpolicy-ablation.yaml` — Phase 1c
- `RESULTS.md` — numbers, once they exist
