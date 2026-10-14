# Gemma-4-31B-it speculative-decoding experiments

Results for the **dense** `gemma-4-31b-it` backbone. The MoE sibling
(`gemma-4-26B-A4B-it`) has its own doc:
[gemma4_26b_moe_results.md](gemma4_26b_moe_results.md).

All evals: single-stream (batch=1), greedy (`temperature=0`), via
`scripts/evaluate/mtp_server_eval/run_vllm_eval.py`. Metrics:

- **accept_len** — avg tokens committed per target forward pass (max = `k+1`).
- **accept_rate** — accepted / drafted tokens.
- **decode tok/s** — decode-phase output speed (what spec-decoding accelerates).
- **e2e tok/s / ttft** — end-to-end rate / time-to-first-token (reference).
- **speedup** — decode tok/s ÷ the target-alone baseline.

---

## 1. MTP assistant, k sweep (4xA100, tp=4)

Target: `gemma-4-31b-it`, draft: `gemma-4-31B-it-assistant`. Baseline = backbone alone.

| benchmark | config | accept_len | accept_rate | decode tok/s | speedup |
|---|---|--:|--:|--:|--:|
| aime | baseline | — | — | 54.4 | 1.00× |
| | assistant k=3 | 3.549 | 85.0% | 133.3 | 2.45× |
| | assistant k=5 | 4.788 | 75.8% | 165.3 | **3.04×** |
| gpqa | assistant k=5 | 4.465 | 69.3% | 155.6 | 2.84× |
| gsm8k | assistant k=5 | 5.074 | 81.5% | 199.3 | **3.58×** |
| humaneval | assistant k=5 | 5.155 | 83.1% | 195.4 | 3.53× |
| livecodebench | assistant k=5 | 4.523 | 70.5% | 151.1 | 2.78× |
| math500 | assistant k=5 | 5.045 | 80.9% | 185.7 | 3.36× |
| mbpp | assistant k=5 | 4.510 | 70.2% | 170.7 | 3.08× |

(baseline ≈ 54–56 tok/s across benchmarks; k=3 rows omitted for brevity — see
`scripts/evaluate/experiments/results/gemma4-31b/results_table.md`.)

**Takeaways:** k=5 > k=3 everywhere (longer accepted runs beat higher per-token
accept rate); **2.8–3.6× speedup**, best on short-output math/code (gsm8k 3.58×).

---

## 2. AgentX / InferenceX concurrency sweep, DSpark drafts (4xA100, tp=4)

**This is the only non-batch=1 section in this doc** -- it is a concurrency
sweep, so the preamble's "single-stream" note does not apply here.

Harness: `aiperf --scenario inferencex-agentx-mvp` against the
`cc-traces-weka-062126` corpus, via
`scripts/evaluate/experiments/gemma4-31b-agentx-dspark-v2.yaml`.
Raw: `scripts/evaluate/experiments/results/gemma4-31b-agentx-v2/`.

Arms: baseline (backbone alone); **ours** =
`gemma4_31b_dspark_nemo782k_scratch`, DSpark trained from scratch, served at its
native k=8 (`sample_from_anchor=True`); **RedHat** =
the published `gemma-4-31B-it-speculator.dspark`, native k=7. Both are 5 layers
/ SWA 2048 / `draft_vocab=32000` / `markov_rank=256`, so neither carries the
full-262k-vocab Omega penalty that section 15 of the 26B doc diagnoses.

All 18 cells are `submission_valid=true` (`duration: 1024` >= the scenario's
900s floor; `max_context: 65536` admits the corpus, whose shortest trace is
44,048 tokens -- the superseded v1 config's 32768 cap admitted nothing).

`decode tok/s` here is **per-user** decode speed (it tracks `1000/ITL`);
`out tok/s` is **aggregate** server throughput. They diverge sharply under load.

| users | accept_len (ours) | accept_len (RedHat) | decode tok/s base / ours / RH | out tok/s base / ours / RH | TTFT s base / ours / RH |
|--:|--:|--:|--:|--:|--:|
| 1 | **3.464** | 3.145 | 38.7 / **53.2** / 48.4 | 26.8 / **35.6** / 31.2 | 1.3 / 1.3 / 1.3 |
| 8 | **2.936** | 2.810 | 33.2 / **36.6** / 35.0 | 36.3 / 35.9 / 34.5 | 1.5 / 1.6 / 1.3 |
| 16 | **2.996** | 2.886 | **13.0** / 7.0 / 6.9 | **52.2** / 30.2 / 30.1 | **7.6** / 19.4 / 20.3 |
| 32 | **2.945** | 2.882 | 1.2 / 3.6 / 3.6 | 11.5 / 21.4 / 20.7 | 79 / 156 / 151 |
| 64 | **2.993** | 2.694 | 1.2 / 3.6 / 3.2 | 8.5 / 15.8 / 16.7 | 277 / 326 / 331 |
| 128 | 2.816 | **2.891** | 1.2 / 3.4 / **3.6** | 9.8 / 16.0 / **18.3** | 413 / 451 / 451 |

### Ours beats the published RedHat draft on acceptance

Our from-scratch draft wins on `accept_len` at **5 of 6** levels, by +0.06 to
+0.32 (it loses only at 128 users, by 0.075). At one user, 3.464 vs 3.145.
This is the cleanest claim in the table and the one worth keeping.

### But the throughput win is single-user only

At 1 user, per-user decode is 1.375x baseline for ours vs 1.251x for RedHat. By
8 users that margin is nearly gone (1.10x / 1.05x).

### At 16 users both drafts LOSE to the baseline

Per-user decode falls to 0.54x baseline (7.0 vs 13.0) and aggregate throughput
to 0.58x (30.2 vs 52.2). Higher acceptance does not rescue it: `accept_len` is
essentially unchanged from 8 to 16 users (2.936 -> 2.996), so the draft is
drafting just as well and still losing. This is the same Omega mechanism as
sections 15/19 of the 26B doc -- once the GPU is compute-bound, the draft's
extra per-step work competes with the batch it is meant to accelerate, and
acceptance stops being the binding term. **Do not quote a 31B AgentX speedup
without naming the concurrency.**

### At 32+ users the sweep stops measuring speculation

Baseline TTFT goes 79s -> 277s -> 413s and baseline ITL pins near 800ms. The
run is queue-bound: requests spend nearly all their wall-clock waiting for
admission, not decoding. The apparent "3x per-user decode" at these levels
(3.6 vs 1.2) is an artifact of the *baseline's* queueing, not a draft win, and
the aggregate 1.9x at 32 users comes with TTFT twice as bad (156s vs 79s).
These three rows are valid submissions but are not usable speedup evidence.

**Caveat on sample size:** the fixed 1024s window completes only 51-162
requests per cell, so single cells are coarse; the AL ordering is consistent
across levels, which is why it is the claim this section leads with.

---

## Related 31B material elsewhere in the repo

- Eval configs: `scripts/evaluate/experiments/gemma4-31b*.yaml` (full suite,
  AgentX concurrency sweeps, BFCL, DSpark, EAGLE3, RedHat fine-tune).
- Raw results: `scripts/evaluate/experiments/results/gemma4-31b*/`.
- Training recipe: `examples/train/eagle3_online_gemma4_31b.sh` and the
  tutorial `docs/user_guide/tutorials/train_eagle3_online_gemma4_31b.md`.
- BFCL results doc: `docs/user_guide/tutorials/gemma4_31b_assistant_bfcl_results.md`.
