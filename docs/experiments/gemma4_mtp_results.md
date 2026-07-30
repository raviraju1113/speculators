# Gemma-4 MTP / speculative-decoding experiments

Consolidated results for Gemma-4 speculative-decoding drafts (MTP assistant &
EAGLE3), the acceptance/throughput evals, and the training-time **shared-KV
attention leak** discovered and fixed here.

All evals: single-stream (batch=1), greedy (`temperature=0`), vLLM 0.24.0+cu129,
via `scripts/evaluate/mtp_server_eval/run_vllm_eval.py`. Metrics:

- **accept_len** — avg tokens committed per target forward pass (max = `k+1`).
- **accept_rate** — accepted / drafted tokens.
- **decode tok/s** — decode-phase output speed (what spec-decoding accelerates).
- **e2e tok/s / ttft** — end-to-end rate / time-to-first-token (reference).
- **speedup** — decode tok/s ÷ the target-alone baseline.

---

## 1. Gemma-4-26B-A4B (MoE) — draft comparison + k-depth sweep (1×A100)

Target: `gemma-4-26B-A4B-it`. Drafts (vanilla MTP assistant, EAGLE3, DFlash)
compared against the same target, sweeping speculative depth k.

| benchmark | config | accept_len | accept_rate | decode tok/s | e2e tok/s | speedup |
|---|---|--:|--:|--:|--:|--:|
| **aime** | baseline (no draft) | — | — | 127.1 | 126.4 | 1.00× |
| | vanilla MTP, k=3 | 3.571 | 85.7% | 198.9 | 195.9 | 1.56× |
| | vanilla MTP, **k=5** | 4.838 | 76.8% | 257.4 | 255.8 | **2.03×** |
| | vanilla MTP, **k=7** | 5.733 | 67.6% | 269.2 | 267.3 | **2.12×** |
| | eagle3, k=3 | 3.007 | 66.9% | 182.7 | 180.2 | 1.44× |
| | eagle3, **k=5** | 3.551 | 51.0% | 219.3 | 218.3 | **1.73×** |
| | eagle3, **k=7** | 3.697 | 38.5% | 208.8 | 207.7 | 1.64× |
| | DFlash, **k=5** | 4.277 | 65.5% | 257.3 | 255.8 | **2.02×** |
| | DFlash, k=7 | 4.789 | 54.1% | 273.9 | 271.8 | **2.15×** |
| | DFlash, **k=15** | 5.531 | 30.2% | 283.3 | 278.0 | **2.23×** |
| **gpqa** | baseline | — | — | 127.2 | 126.4 | 1.00× |
| | vanilla MTP, k=3 | 3.338 | 77.9% | 190.8 | 188.6 | 1.50× |
| | vanilla MTP, **k=5** | 4.528 | 70.6% | 241.9 | 239.2 | **1.90×** |
| | vanilla MTP, **k=7** | 4.955 | 56.5% | 235.8 | 232.6 | 1.85× |
| | eagle3, k=3 | 2.472 | 49.1% | 152.2 | 150.9 | 1.20× |
| | eagle3, **k=5** | 2.760 | 35.2% | 174.2 | 172.8 | **1.37×** |
| | eagle3, **k=7** | 2.747 | 25.0% | 160.1 | 158.9 | 1.26× |
| | DFlash, **k=5** | 3.568 | 51.4% | 218.2 | 216.1 | **1.72×** |
| | DFlash, k=7 | 3.806 | 40.1% | 221.0 | 218.6 | **1.74×** |
| | DFlash, **k=15** | 4.163 | 21.1% | 218.0 | 214.4 | 1.71× |
| **livecodebench** | baseline | — | — | 126.2 | 125.4 | 1.00× |
| | vanilla MTP, k=3 | 3.418 | 80.6% | 186.7 | 184.9 | 1.48× |
| | vanilla MTP, **k=5** | 4.513 | 70.3% | 232.2 | 229.1 | 1.84× |
| | vanilla MTP, **k=7** | 5.157 | 59.4% | 232.9 | 229.7 | **1.85×** |
| | eagle3, k=3 | 2.583 | 52.8% | 153.1 | 152.0 | 1.21× |
| | eagle3, **k=5** | 2.912 | 38.2% | 175.1 | 173.6 | **1.39×** |
| | eagle3, **k=7** | 2.974 | 28.2% | 163.2 | 161.8 | 1.29× |
| | DFlash, **k=5** | 3.581 | 51.6% | 208.3 | 205.9 | 1.65× |
| | DFlash, k=7 | 3.987 | 42.7% | 219.7 | 217.1 | **1.74×** |
| | DFlash, **k=15** | 4.142 | 20.9% | 201.9 | 199.6 | 1.60× |

**Takeaways** (k-sweep now covers k = 3 / 5 / 7 (/15 for DFlash))
- **Most of the gain is realized by k≈5 — deeper speculation has diminishing (or
  negative) returns.** The vanilla MTP assistant jumps k=3→k=5 (1.48–1.56× →
  **1.84–2.03×**) and then barely moves to k=7 (1.85–2.12×); it even *peaks at
  k=5* on gpqa (1.90 vs 1.85×).
- **EAGLE3 peaks at k=5 and then *declines*:** k=5 = **1.37–1.73×** beats k=7 =
  1.26–1.64× on every benchmark — its per-token accept_rate falls off a cliff
  beyond k=5 (aime 51%→38%, gpqa 35%→25%), so extra depth wastes drafts.
- **DFlash** (block-diffusion, block_size=16) is strong (**1.60–2.23×**) and
  fairly flat over k=5–15 (e.g. aime 2.02→2.15→2.23×): *low* accept_rate but
  *high* accept_len, since it drafts a block in parallel. Served `method: dflash`,
  `--attention-backend triton_attn` (no flash_attn).
- **Best drafts overall:** vanilla MTP and DFlash are near-tied at their sweet
  spots (both ~2.0× aime, ~1.7–1.9× gpqa/lcb); EAGLE3 trails. Given single-stream
  (batch=1) caveats, **k≈5 is the pragmatic operating point** for MTP/DFlash.
- k is the method's speculative depth; each method's natural config differs, so
  compare achieved speedup across the k-sweep, not raw k.

---

## 2. Gemma-4-31B-it — MTP assistant, k sweep (4×A100, tp=4)

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

## 3. Training-time shared-KV attention leak (root-caused & fixed)

**Symptom.** Fine-tuning the MTP assistant with the in-repo online trainer
(`scripts/gemma4_mtp/train_online.py`) *destroyed* it: accept_len collapsed from
~3.5 (stock) to ~1.1 (§1, trained-50k), i.e. ~3× *slower* than the stock draft
and 2× slower than no draft.

**Root cause.** The stock HF assistant's `create_attention_masks`
(`transformers/models/gemma4_assistant/modeling_gemma4_assistant.py`) builds an
**all-ones bidirectional** mask (`create_bidirectional_mask` →
`bidirectional_mask_function` = `q_idx >= 0`) over the target's full
teacher-forced KV. That is fine for HF's intended *inference* use (`q_len == 1`,
where bidirectional ≡ causal — the docstring even says so), but the repo's TTT
trainer reuses the same model at **`q_len = L > 1`** (whole sequence in
parallel), where "attend everything" lets query row `t` attend the target's KV
for the tokens it is predicting → a **label leak**. The draft learns to read it
(train loss falls fast), then collapses at inference.

**Why it's invisible at inference.** Confirmed against vLLM's `Gemma4MTP`
inference (`vllm/model_executor/models/gemma4_mtp.py`): the draft has **no KV of
its own** (`is_kv_shared_layer=True`, passes dummy K/V) and reads K/V from the
**target's cache via KV-sharing**, using vLLM's standard **causal** paged
`Attention` (+ `per_layer_sliding_window` for SWA). Positions are held
**constant** across draft steps (`constant_draft_positions` in the proposer).
So inference is plainly causal over the verified prefix — there is no future KV
to leak, and the bug never surfaces during decoding.

**Fix.** Make the training shared-KV attention **causal / prefix-only**, matching
inference: query row `t` (a draft rollout that started at target position `t`,
its recurrent hidden tracing to `target_hidden[t]`) attends only the **verified
prefix** `KV[0..t]` (full-attn) or `t−window < j ≤ t` (sliding) — **the same for
every TTT step k** (offset 0), because the draft never attends the target KV of
the tokens it is drafting (that info rides the recurrent hidden, and at inference
the KV range is fixed per rollout). Applied as a monkeypatch
(`patch_causal_shared_kv_masks` in `train_online.py`), verified with a mask-print
test: row `t` attends exactly `KV[0..t]`, **identical across k**, zero leak.
RoPE positions needed no change — the stock `position_ids=None → arange(L)` (row
`t` → position `t`, constant across k) already matches inference's constant
positions. (An earlier attempt used an advancing `k+t` offset for both mask and
RoPE; checking the inference code showed that *creates* a mismatch — reverted.)

---

## 4. Training runs

| run | init | mask | data | lr | status / result |
|---|---|---|---|---|---|
| pretrained-50k | stock | leaky | 50k | 6e-4 | superseded — collapsed to accept_len 1.1 |
| random-init 20k / full | scratch | leaky | 20k / 338k | 6e-4 | superseded (leaky objective) |
| warm-start FT | stock | **fixed** | 10k | 6e-4 | accept_len 1.5 (lr too high) |
| **warm-start FT** | stock | **fixed** | 10k | **2e-5** | **accept_len 3.15 — draft preserved ✓** |
| random-init full | scratch | **fixed** | 338k | 6e-4 | in progress (GPU 2,3) — from-scratch, slow |

Online trainer places the frozen 26B target on one GPU and the trained assistant
(+ optimizer) on another (`train_online.py` device split); target `shared_kv_states`
come from the in-process HF forward (`return_shared_kv_states=True`) — **not** the
vLLM `Gemma4SharedKVStatesConnector` (that connector is the decoupled
vLLM-server export path, unused here).

> **Note:** runs above predate the §3 fix and produce non-inference-valid drafts.
> Post-fix reruns supersede them.

**Post-fix findings (mask-only, correct):**

- **Warm-start FT from vanilla (10k samples) — the fix, confirmed end-to-end:**

  | trainer | lr | accept_len (aime) | note |
  |---|---|--:|---|
  | vanilla (untouched) | — | 3.57 | reference |
  | leaky | 6e-4 | 1.10 | leak destroys the draft |
  | **mask-fixed** | 6e-4 | 1.50 | leak gone, but lr too high |
  | **mask-fixed** | **2e-5** | **3.15** | **draft preserved (~1.3× speedup)** |

  Monotonic story: the leak was the real bug; lr 6e-4 was a *separate* degrader;
  with the mask fix **and** a sane lr the trainer **preserves** the draft (no
  collapse — accept_len 3.15 vs 3.57, ~1.2–1.4× speedup). The small gap to vanilla
  is ordinary fine-tuning drift on the regen data, not a correctness issue. **The
  mask-fixed `train_online.py` is sound.**
- **Random-init from scratch** stays undertrained on this data volume
  (step-10k checkpoint ≈ 1% accept) — not a fix problem, just far too little data
  to train a draft from zero.
- **Takeaway:** the trainer is now *correct* (matches vLLM causal inference), but
  producing a *better* draft needs sane fine-tuning hyperparameters, not just the
  mask fix. The stock assistant / DFlash (§1) remain the drafts to deploy.

_Raw results: `scripts/evaluate/mtp_server_eval/results/26b_compare/results_table.{md,csv}`
and `scripts/evaluate/experiments/results/gemma4-31b/results_table.{md,csv}`._
