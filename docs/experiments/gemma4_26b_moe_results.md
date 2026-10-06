# Gemma-4-26B-A4B (MoE) speculative-decoding experiments

Consolidated results for **`gemma-4-26B-A4B-it` (MoE)** speculative-decoding
drafts (MTP assistant, EAGLE3, DFlash, DSpark, P-EAGLE), the acceptance/throughput
evals, and two training-time bugs found & fixed here: the **shared-KV attention
leak** (§2) and a **hidden-state off-by-one** (§4), plus the feature-distillation
quality push, the six-way 25-benchmark profile (§5), and the 400k DSpark
scale-up (§6), its results (§7), and the continuation run that shows more
epochs are exhausted (§8), and a production-style workload where the
stock assistant wins decisively (§9), and a concurrency sweep where the
speedup disappears under load (§10), and a draft-vocabulary test that
refuted its own hypothesis (§11), and the warm-start from the stock DFlash
backbone that finally fixed it (§12), plus what published drafts train on
(§13).

The dense 31B sibling has its own doc:
[gemma4_31b_results.md](gemma4_31b_results.md).

All evals: single-stream (batch=1), greedy (`temperature=0`), vLLM 0.24.0+cu129,
via `scripts/evaluate/mtp_server_eval/run_vllm_eval.py`. Metrics:

- **accept_len** — avg tokens committed per target forward pass (max = `k+1`).
- **accept_rate** — accepted / drafted tokens.
- **decode tok/s** — decode-phase output speed (what spec-decoding accelerates).
- **e2e tok/s / ttft** — end-to-end rate / time-to-first-token (reference).
- **speedup** — decode tok/s ÷ the target-alone baseline.

---

## 1. Draft comparison + k-depth sweep (1×A100)

Target: `gemma-4-26B-A4B-it`. Drafts (vanilla MTP assistant, EAGLE3, DFlash)
compared against the same target, sweeping speculative depth k.

> **Scope / currency.** This section is the **k-depth sweep over *stock*
> (published) drafts on 3 benchmarks, measured on vLLM 0.24.0** (baseline
> ~127 tok/s). It is kept for the k-sweep, which later sections do not repeat.
> For the **current** picture — the drafts trained here, at their native k,
> over 25-26 benchmarks on vLLM 0.28 — see §5 (six-way comparison) and §7
> (DSpark 400k, the present best at **1.89x** mean / **2.31x** aime, which
> supersedes the "best drafts overall" note below). Do not compare speedups
> across engine versions; compare within a section.

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
| | featdistill (ours), k=3 | 2.536 | 51.2% | 141.7 | 141.3 | 1.12× |
| | featdistill (ours), **k=5** | 2.918 | 38.4% | 156.4 | 155.8 | 1.23× |
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
| | featdistill (ours), k=3 | 1.963 | 32.1% | 111.9 | 111.4 | 0.88× |
| | featdistill (ours), **k=5** | 2.203 | 24.1% | 119.9 | 119.3 | 0.94× |
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
| | featdistill (ours), k=3 | 2.188 | 39.6% | 119.4 | 118.6 | 0.95× |
| | featdistill (ours), **k=5** | 2.439 | 28.8% | 126.0 | 125.0 | 1.00× |

`featdistill (ours)` = our from-scratch feature-distilled draft
(`assistant_featdistill/step3200`, §4); its own matched baseline (aime 126.9, gpqa 127.5,
lcb 125.5 tok/s) matches the row above, so speedups are directly comparable.

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
- **Our from-scratch feature-distilled draft (`featdistill`, §4)** is now in the table
  for comparison — markedly weaker than the stock drafts: aime **1.12× (k3) / 1.23× (k5)**
  vs vanilla MTP's 1.56–2.03×, and *net-negative* on gpqa/lcb at k=3 (0.88–0.95×, because
  accept < the ~2.0 break-even), reaching only break-even by k=5. Deeper k still helps it
  (aime accept 2.54→2.92). See §4 for the full multi-domain picture and why (data coverage).

---

## 2. Training-time shared-KV attention leak (root-caused & fixed)

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

## 3. Training runs

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

Representative repro command for the post-fix trainer run (with soft-CE enabled):

```bash
/root/miniconda3/envs/speculator/bin/python -u scripts/gemma4_mtp/train_online.py \
  --target /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it \
  --assistant /nvmedata/chenw/speculators/output/gemma4_26b_mtp_rinit_shiftfix_4gpu/checkpoints/step10000 \
  --data /nvmedata/data/kimi-regen-gemma4-26b-moe/train_regen.jsonl \
  --output /nvmedata/chenw/speculators/output/gemma4_26b_mtp_assistant_featdistill/checkpoints \
  --epochs 10 --batch-size 1 --grad-accum 128 --lr 2e-4 --max-length 1024 \
  --ttt-steps 7 --max-samples 0 --bf16 \
  --soft-ce-weight 0.5 --hard-ce-weight 0.1 --feature-l1-weight 0.9 \
  --num-workers 4 --log-every 5 --save-every 200
```

**Multi-draft online training** (YAML `drafts:` list, sequential/isolated —
config-controlled N): see `examples/train/gemma4_26b_mtp_online_multi.yaml`.

```bash
CONFIG=examples/train/gemma4_26b_mtp_online_multi.yaml \
  bash examples/train/gemma4_26b_mtp_online.sh
```

Each draft under `drafts:` is trained alone (own target signals, loss I/O,
optimizer, `<output_dir>/<name>/` checkpoints). Drafts are not mixed in one step.
> **Note:** runs above predate the §2 fix and produce non-inference-valid drafts.
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
- **Random-init from scratch** first looked "undertrained" (~1% accept) — but that
  was a **second, separate bug** (§4), not data volume: a hidden-state off-by-one
  (train fed `h_t`, vLLM feeds `h_{t-1}`). After that fix, from-scratch reaches
  accept **~2.2**, and feature distillation lifts it to **~2.5** (§4).
- **Takeaway:** the trainer is now *correct* on both counts (mask fix **and** the
  hidden shift, §4). A *deployable general* draft is now a **training-data-coverage**
  problem (§4), not a correctness one. The stock assistant / DFlash (§1) remain the
  drafts to deploy today.

_Raw results: `scripts/evaluate/mtp_server_eval/results/26b_compare/results_table.{md,csv}`._

---

## 4. Second bug + quality push: hidden-state off-by-one → feature distillation

Full write-up: `gemma4_mtp_vllm_hidden_shift_bug.md`.

**Bug (distinct from §2's mask leak).** Even with the mask fix, *from-scratch* drafts
collapsed to **accept_len ~1.07 in vLLM** while scoring ~0.94 next-token agreement in HF.
Cause: a **hidden-state off-by-one** — the trainer fed the draft `hidden[t]`
(`build_target_signals`), but vLLM (EAGLE/MTP convention) feeds `hidden[t-1]` + `embed(x_t)`.
One-position shift → **1.07 → ~2.2**. Folded into `training_step.py` (`_shift_right`; the
`patch_hidden_shift` monkeypatch is now a no-op).

**Feature distillation.** Pure logit-distillation then caps accept ~2.2 — the recurrent
`backbone_hidden` never learns to match the target (`feat_l1` ≈ random). Adding an
EAGLE/DSpark smooth-L1 feature loss (`_feature_l1`, `--feature-l1-weight`) lifts **2.2 → 2.5**.
Adding soft-CE on top *hurt* (erodes the feature) — **feature-only (0.1 hard-CE + 0.9 feat)
is the best recipe**; watch `feat_l1` rising as a "getting worse" proxy.

**Checkpoint comparison** (26B target, k=3, ~100 prompts; accept_len / accept_rate):

| checkpoint | aime | livecodebench | gpqa |
|---|--:|--:|--:|
| `assistant_featdistill/step3200` (feature-distill, **best**) | **2.51 / 0.50** | **2.18 / 0.39** | **2.01 / 0.34** |
| `assistant_featdistill_soft03/step1400` (soft0.3 + feat) | 2.40 | 2.04 | 1.88 |
| `rinit_shiftfix_4gpu/step10000` (pure soft-CE) | 2.22 / 0.41 | 1.88 / 0.29 | 1.75 / 0.25 |

**Full 15-benchmark eval** (soft03/step1400 vs 26B baseline, k=3, ~100 prompts): **mean
1.04×, 8/15 positive** — a **math/code specialist**: gsm8k 1.41×, math500 1.38×, humaneval
1.28×, aime 1.20×, mbpp 1.14×; but **net-slower** on gpqa 0.96×, mt-bench 0.87×,
swe-bench-pro 0.84×, speed-qa 0.79×, **speed-writing 0.68×**. Speedup tracks accept vs the
**~2.0 break-even** at k=3. This is a **training-data-coverage** limit (kimi-regen =
math/reasoning only), not a method bug — a general draft needs diverse data; the stock
assistant / DFlash (§1) remain the deploy-today drafts. (NB: full eval ran on the weaker
soft03 ckpt; `step3200` clears break-even on lcb/gpqa and would be net-positive on more.)

_Results: `scripts/evaluate/experiments/results/full-eval-soft03-step1400/`;
`scripts/evaluate/mtp_server_eval/results/eval3_*`._

---

## 5. Six-way draft comparison — 25-benchmark profile

All five supported draft types trained **simultaneously on one 4×A100 node**
(shared frozen-target hidden-states server + shared feature cache; see
[`examples/train/gemma4_26b_penta_draft_online.sh`](../../examples/train/gemma4_26b_penta_draft_online.sh)
and `docs/TRAINING.md` → "Simultaneous multi-draft training"). Identical budget:
**30k samples × 2 epochs** from the merged regen pool (686k rows: 26B-MoE
kimi-regen + 31B kimi-regen + 31B tool regen, deduped/shuffled), seq 4096,
draft vocab 32k. MTP-ft = fine-tune of the official assistant at lr 5e-5 (the
rinit lr 6e-4 *degrades* it much harder: aime accept 4.83 → 2.98 — §4 lesson
re-confirmed). The official **vanilla assistant** is the reference column.

Eval: vLLM 0.28, 1×A100 tp=1, greedy, 30 prompts/benchmark, same-config
baseline (~126 tok/s). P-EAGLE serves via `speculative_config: {method:
eagle3}` — vLLM 0.28's method auto-detection has no peagle branch and
otherwise routes it to the hidden-state-less `draft_model` proposer, which
crashes at warmup (upstream bug; the eagle3 path picks up `pard_token`).

> **Note:** `qa`/`speed-qa` and `question`/`writing` are duplicate prompt
> sets (see the suite caveat in §7), so those rows are each counted twice in
> the mean. De-duplicated means are +0.02 to +0.04 and preserve the ordering.

### Per-benchmark decode speedup / accept_len (rows sorted easiest→hardest; bold = best per row)

| benchmark | DSpark k=8 | DFlash k=7 | MTP-ft k=5 | EAGLE3 k=5 | P-EAGLE k=4 | vanilla asst k=5 |
|---|---|---|---|---|---|---|
| math_reasoning | **2.51 / 4.67** | 2.44 / 4.56 | 1.87 / 4.01 | 2.11 / 3.86 | 1.08 / 2.02 | 2.37 / 5.11 |
| gsm8k | **2.53 / 4.76** | 2.44 / 4.61 | 1.79 / 3.88 | 2.07 / 3.82 | 1.07 / 2.01 | 2.32 / 5.03 |
| math500 | 2.22 / 4.23 | **2.24 / 4.29** | 1.94 / 4.38 | 1.85 / 3.54 | 1.01 / 2.00 | 2.24 / 5.02 |
| humaneval | 2.01 / 3.83 | 2.06 / 3.93 | 1.88 / 4.14 | 1.78 / 3.34 | 1.02 / 1.99 | **2.26 / 4.99** |
| bfcl | 1.65 / 3.18 | 1.78 / 3.47 | 2.26 / 5.32 | 1.37 / 2.62 | 0.95 / 1.88 | **2.39 / 5.75** |
| aime26 | 1.90 / 3.83 | 1.87 / 3.83 | 1.88 / 4.35 | 1.68 / 3.35 | 0.95 / 1.98 | **2.10 / 4.90** |
| HumanEval | 1.92 / 3.69 | 1.90 / 3.65 | 1.67 / 3.69 | 1.69 / 3.19 | 1.01 / 1.94 | **2.20 / 4.78** |
| aime | 1.84 / 3.67 | 1.86 / 3.75 | 1.82 / 4.18 | 1.60 / 3.18 | 0.95 / 1.97 | **2.10 / 4.86** |
| mbpp | 1.83 / 3.53 | 1.81 / 3.49 | 1.56 / 3.43 | 1.67 / 3.15 | 0.98 / 1.90 | **2.03 / 4.46** |
| livecodebench | 1.60 / 3.28 | 1.66 / 3.40 | 1.60 / 3.77 | 1.46 / 2.93 | 0.90 / 1.91 | **1.92 / 4.52** |
| speed-coding | 1.43 / 2.93 | 1.52 / 3.12 | 1.53 / 3.57 | 1.32 / 2.64 | 0.91 / 1.90 | **1.93 / 4.46** |
| gpqa | 1.46 / 2.89 | 1.46 / 2.94 | 1.56 / 3.60 | 1.33 / 2.63 | 0.88 / 1.82 | **1.89 / 4.36** |
| tool_call | 1.29 / 2.44 | 1.35 / 2.54 | 1.36 / 3.01 | 1.23 / 2.28 | 0.88 / 1.70 | **1.78 / 3.93** |
| translation | 1.31 / 2.44 | 1.32 / 2.43 | 1.23 / 2.64 | 1.25 / 2.29 | 0.93 / 1.70 | **1.83 / 3.89** |
| swe-bench-pro | 1.26 / 2.55 | 1.32 / 2.69 | 1.36 / 3.17 | 1.16 / 2.33 | 0.84 / 1.77 | **1.79 / 4.18** |
| speed-rag | 1.21 / 2.50 | 1.23 / 2.56 | 1.26 / 3.00 | 1.16 / 2.35 | 0.84 / 1.74 | **1.77 / 4.16** |
| rag | 1.19 / 2.42 | 1.18 / 2.42 | 1.14 / 2.69 | 1.15 / 2.28 | 0.82 / 1.68 | **1.68 / 3.89** |
| writing | 1.28 / 2.44 | 1.26 / 2.41 | 1.04 / 2.28 | 1.22 / 2.28 | 0.87 / 1.68 | **1.47 / 3.18** |
| question | 1.28 / 2.44 | 1.26 / 2.40 | 1.04 / 2.28 | 1.22 / 2.28 | 0.87 / 1.68 | **1.47 / 3.17** |
| speed-multilingual | 0.97 / 1.76 | 0.97 / 1.76 | 1.60 / 3.44 | 0.93 / 1.71 | 0.76 / 1.44 | **1.89 / 4.03** |
| mt-bench | 1.28 / 2.45 | 1.26 / 2.41 | 1.04 / 2.28 | 1.22 / 2.28 | 0.86 / 1.68 | **1.44 / 3.16** |
| speed-qa | 1.14 / 2.09 | 1.14 / 2.10 | 0.98 / 2.08 | 1.13 / 2.04 | 0.88 / 1.61 | **1.42 / 2.98** |
| qa | 1.14 / 2.09 | 1.14 / 2.10 | 0.98 / 2.08 | 1.13 / 2.04 | 0.88 / 1.61 | **1.42 / 2.98** |
| speed-writing | 1.04 / 2.15 | 1.04 / 2.16 | 0.91 / 2.14 | 1.01 / 2.04 | 0.77 / 1.62 | **1.27 / 2.96** |
| summarization | 0.97 / 1.98 | 0.99 / 2.02 | 0.96 / 2.27 | 0.93 / 1.86 | 0.76 / 1.58 | **1.37 / 3.20** |
| **MEAN** | **1.53×** | **1.54×** | **1.45×** | **1.39×** | **0.91×** | **1.85×** |

### Takeaways

- **The vanilla Google assistant wins overall (mean 1.85×) and is the
  deploy-today draft.** None of the same-budget trained drafts beat it in the
  mean; the from-scratch drafts approach or edge past it **only in math**
  (DSpark gsm8k 2.53× / math_reasoning 2.51× vs vanilla 2.32× / 2.37×).
- **Fine-tuning the assistant was net-harmful on ALL 25 benchmarks**
  (mean 1.85× → 1.45×), even at the gentle lr 5e-5. Its apparent bfcl /
  multilingual "strengths" vs the from-scratch drafts are inherited from
  vanilla (bfcl 2.39×, multilingual 1.89×), merely degraded less than other
  domains. Fine-tune the assistant only with genuinely in-domain data for a
  narrow deployment — never as a general upgrade.
- **DSpark ≈ DFlash (1.53/1.54× mean)** lead the from-scratch field and
  dominate structured domains; EAGLE3 trails (1.39×); **P-EAGLE is below
  break-even at this budget (0.91× mean)** — accept 1.4–2.0 matches its
  training simulation, so it is served correctly and simply needs a bigger
  budget or recipe work.
- **The multilingual/summarization hole of the pruned-vocab drafts (<1.0×) is
  the 32k draft vocab, not data volume** — the full-vocab assistant is immune.
  Fix = larger `--draft-vocab-size`, not more samples.
- Chat/QA-style outputs (short, high-entropy) are the intrinsic hard case:
  even vanilla only reaches 1.3–1.5× there.
- DSpark caveat: this checkpoint trains `sample_from_anchor=True` (native
  k=8); it needs a vLLM whose speculators loader reads that field (0.28+
  here). Later runs use `--no-sample-from-anchor` for stock-vLLM portability
  (k=7).

_Configs: `scripts/evaluate/experiments/gemma4-26b-penta-eval.yaml`,
`gemma4-26b-pre400k-eval.yaml`. Raw results + generated tables:
`scripts/evaluate/experiments/results/gemma4-26b-{penta-eval,pre400k-eval}/`
(`results_table.md` / `.csv`)._

---

## 6. Scaling DSpark to 400k samples — setup

Follow-up to §5: give the best from-scratch draft (DSpark) ~13x the data and let
validation decide the epoch count. Started 2026-09-16.

| Parameter | Value |
|---|---|
| Data | 400k of the merged regen pool (686k rows), seq 4096 |
| Layout | 3-GPU DDP trainer (GPUs 1-3) + shared hidden-states server (GPU 0) |
| Features | **streaming** (`--on-generate delete`) — no feature cache, flat disk |
| Epochs | up to 3, `--early-stop-patience 1` (stop 1 epoch after val loss plateaus) |
| Checkpoints | `--checkpoint-freq 0.25` (quarter-epoch) |
| Convention | **`--no-sample-from-anchor`** — portable on stock vLLM, native k=7 |
| Other | lr 3e-4, 3 layers, block 8, markov_rank 256, confidence head, ce 0.1/tv 0.9 |

### Epoch 0 validation (40,240 steps, ~1 day)

| metric | 400k epoch 0 (k=7 slots) | 30k x 2ep, §5 (k=8 slots) |
|---|---|---|
| **accept_len** | **2.937** | 2.889 |
| full_acc | 0.520 | 0.496 |
| first slot acc | 0.696 | 0.688 |
| last slot acc | 0.406 | 0.386 |
| loss_epoch | 0.585 | 0.542 |

**Read this comparison carefully:** the two runs use different block conventions,
so the columns are not strictly like-for-like. The 400k run trains
`sample_from_anchor=False` (1+N bonus-anchor fill, **7** predicted slots,
reported as `position_1..7`); the §5 run trained `True` (**8** slots,
`position_0..7`). The 400k draft therefore reaches a *higher* accept_len from
*fewer* draft slots after *one* epoch — a real improvement — while `loss_epoch`
is not comparable across conventions (different loss I/O and Markov
conditioning). The serving numbers are what will settle it.

### Status / next

- **Completed — final results and the 26-benchmark eval are in §7.**
- Operational note: three separate session-boundary `SIGTERM`s killed trainers
  during this work (the parent shell's process group is reaped). The run is now
  launched via
  [`examples/train/_resume_400k_detached.sh`](../../examples/train/_resume_400k_detached.sh),
  which `setsid`s the server and trainer into their own sessions and records
  PIDs under `<output>/pids/`. Sub-epoch checkpoints capped the largest loss at
  ~1.6% of an epoch.

_Config/logs: `output/gemma4_26b_dspark_400k/` (`logs/`, `dspark/checkpoints/`,
`data_prep/`)._

---

## 7. DSpark at 400k samples — final results

The §6 run finished: **3 epochs** (the cap; validation was still improving, so
`--early-stop-patience 1` never fired), `checkpoint_best` = epoch 2.

| epoch | val loss | val accept_len | full_acc |
|---|---|---|---|
| 0 | 0.585 | 2.937 | 0.520 |
| 1 | 0.555 | 3.130 | 0.551 |
| 2 (best) | **0.495** | **3.520** | **0.606** |

Served at its native **k=7** (trained `--no-sample-from-anchor`, so it is
portable to stock vLLM) against a **freshly measured same-day baseline**
(drift vs the §5 baseline was +1.9%, so the older columns remain comparable).
26 benchmarks — the 25 of §5 plus `heldout_chat`, which existed as a data file
but was unregistered in both eval runners until now.

> **Suite caveat — two duplicate prompt sets (found 2026-09-20 during a numbers
> audit).** Of the 26 benchmark files, only **24 are distinct**: `qa` and
> `speed-qa` contain the same 80 prompts, as do `question` and `writing`
> (verified by hashing the prompt sets). Those pairs therefore carry double
> weight in any mean below. The effect is small and uniform — dropping one of
> each pair raises every draft's mean by +0.02 to +0.04 and changes no
> ordering — but the de-duplicated figures are the honest ones:
>
> | draft | mean as tabulated | mean over distinct sets |
> |---|---|---|
> | DSpark 400k (§7) | 1.89× (n=26) | **1.93×** (n=24) |
> | vanilla assistant | 1.85× (n=25) | **1.89×** (n=23) |
> | DFlash 30k | 1.54× | 1.57× |
> | DSpark 30k | 1.53× | 1.56× |
> | MTP-ft 30k | 1.45× | 1.49× |
> | EAGLE3 30k | 1.39× | 1.41× |
> | P-EAGLE 30k | 0.91× | 0.91× |
>
> On the 23 **distinct, shared** benchmarks the headline still holds:
> DSpark 400k **1.933×** vs vanilla **1.890×**, winning 12 of 23 (the
> "13 of 25" split elsewhere counts the duplicated pairs).

### Speedup / accept_len (AL) / accept_rate (AR) — bold = beats the vanilla assistant

**AL** = accepted tokens committed per target forward (max `k+1`). **AR** =
`(AL-1)/k`, so it is *normalized by draft depth* and therefore **not comparable
across columns with different k** — a draft with more slots is penalized for the
deeper, harder ones. Compare **AL** (and speedup) across drafts; read AR only
within a column, as "how much of this draft's own budget landed".

| benchmark | **DSpark 400k** (k=7)<br>speedup / AL / AR | DSpark 30k (k=8)<br>speedup / AL / AR | vanilla assistant (k=5)<br>speedup / AL / AR |
|---|---|---|---|
| gsm8k | **2.99 / 5.56 / 0.652** | 2.53 / 4.76 / 0.471 | 2.32 / 5.03 / 0.805 |
| math_reasoning | **2.94 / 5.39 / 0.628** | 2.51 / 4.67 / 0.459 | 2.37 / 5.11 / 0.821 |
| math500 | **2.71 / 5.11 / 0.587** | 2.22 / 4.23 / 0.404 | 2.24 / 5.02 / 0.805 |
| humaneval | **2.63 / 4.95 / 0.564** | 2.01 / 3.83 / 0.354 | 2.26 / 4.99 / 0.797 |
| HumanEval | **2.47 / 4.65 / 0.521** | 1.92 / 3.69 / 0.336 | 2.20 / 4.78 / 0.756 |
| aime26 | **2.33 / 4.65 / 0.521** | 1.90 / 3.83 / 0.354 | 2.10 / 4.90 / 0.780 |
| aime | **2.31 / 4.55 / 0.507** | 1.84 / 3.67 / 0.334 | 2.10 / 4.86 / 0.772 |
| bfcl | 2.31 / 4.40 / 0.486 | 1.65 / 3.18 / 0.272 | 2.39 / 5.75 / 0.951 |
| mbpp | **2.23 / 4.31 / 0.472** | 1.83 / 3.53 / 0.316 | 2.03 / 4.46 / 0.692 |
| livecodebench | **2.03 / 4.10 / 0.443** | 1.60 / 3.28 / 0.285 | 1.92 / 4.52 / 0.705 |
| heldout_chat | 1.95 / 3.69 / 0.385 | — | — |
| speed-coding | **1.93 / 3.91 / 0.415** | 1.43 / 2.93 / 0.241 | 1.93 / 4.46 / 0.692 |
| gpqa | 1.76 / 3.48 / 0.355 | 1.46 / 2.89 / 0.236 | 1.89 / 4.36 / 0.671 |
| translation | 1.66 / 3.03 / 0.289 | 1.31 / 2.44 / 0.180 | 1.83 / 3.89 / 0.579 |
| swe-bench-pro | 1.65 / 3.30 / 0.329 | 1.26 / 2.55 / 0.194 | 1.79 / 4.18 / 0.635 |
| tool_call | 1.59 / 2.95 / 0.279 | 1.29 / 2.44 / 0.180 | 1.78 / 3.93 / 0.586 |
| speed-rag | 1.57 / 3.19 / 0.314 | 1.21 / 2.50 / 0.188 | 1.77 / 4.16 / 0.632 |
| question | **1.52 / 2.83 / 0.262** | 1.28 / 2.44 / 0.181 | 1.47 / 3.17 / 0.435 |
| writing | **1.52 / 2.83 / 0.261** | 1.28 / 2.44 / 0.181 | 1.47 / 3.18 / 0.435 |
| mt-bench | **1.52 / 2.83 / 0.262** | 1.28 / 2.45 / 0.181 | 1.44 / 3.16 / 0.432 |
| rag | 1.48 / 2.98 / 0.283 | 1.19 / 2.42 / 0.177 | 1.68 / 3.89 / 0.578 |
| qa | 1.34 / 2.44 / 0.206 | 1.14 / 2.09 / 0.136 | 1.42 / 2.98 / 0.395 |
| speed-qa | 1.34 / 2.44 / 0.206 | 1.14 / 2.09 / 0.136 | 1.42 / 2.98 / 0.395 |
| speed-writing | 1.25 / 2.55 / 0.222 | 1.04 / 2.15 / 0.143 | 1.27 / 2.96 / 0.392 |
| summarization | 1.18 / 2.40 / 0.199 | 0.97 / 1.98 / 0.122 | 1.37 / 3.20 / 0.440 |
| speed-multilingual | 1.06 / 1.92 / 0.131 | 0.97 / 1.76 / 0.095 | 1.89 / 4.03 / 0.606 |
| **MEAN (25 shared)** | **1.89 / 3.63 / 0.376** | 1.53 / 2.97 / 0.246 | 1.85 / 4.16 / 0.632 |

### Per-slot acceptance (server-side, pooled over the whole 26-benchmark run)

Cumulative acceptance at each draft slot, from vLLM's own `SpecDecoding`
metrics (weighted by drafted tokens); `1 + sum` reproduces the measured AL.

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | => AL |
|---|---|---|---|---|---|---|---|---|---|
| **DSpark 400k** (k=7) | 0.752 | 0.561 | 0.420 | 0.323 | 0.251 | 0.193 | 0.146 | — | 3.65 |
| DSpark 30k (k=8) | 0.689 | 0.466 | 0.312 | 0.211 | 0.141 | 0.092 | 0.059 | 0.036 | 3.00 |
| vanilla assistant (k=5) | 0.917 | 0.830 | 0.750 | 0.674 | 0.598 | — | — | — | 4.77 |

- Scaling to 400k mostly bought **depth**: +9% at slot 1 over the 30k draft but
  **+147% at slot 7** — blocks stay alive longer, which is where AL comes from.
- **Vanilla wins every slot yet loses on speedup, and that is the whole story.**
  Its per-slot decay is ~0.90x/slot vs DSpark's ~0.75x, giving a much higher AL
  (4.77 vs 3.65) and AR (0.63 vs 0.38) — but it is a *true autoregressive*
  draft, so those 5 tokens cost **5 sequential forwards**, while DSpark drafts
  its whole 7-slot block in **one parallel forward**. Per-slot AR measures draft
  *quality*; speedup is quality / cost.
- **Train/serve parity is clean**: served slot-1 (0.752) matches training
  position-1 (0.769), and served AL (3.65) exceeds the training-reported 3.52.
  The failure mode to watch for is serving at ~half of training, which signals a
  `sample_from_anchor` convention mismatch (see
  `.claude/skills/dspark-train-serve-parity`).

### Takeaways

- **Scaling the data worked: 1.53× → 1.89× mean** (AL 2.97 → 3.63, AR 0.246 →
  0.376), and the 400k DSpark now **edges past the vanilla Google assistant
  (1.85×)** — the first draft we have trained that does. Note it wins on
  *speedup* while losing on AL (3.63 vs 4.16) and AR (0.376 vs 0.632): its
  block drafts in one forward pass, vanilla's needs five.
- **The win is uneven: 13 of 25 benchmarks beat vanilla, 12 do not.** It wins
  decisively where drafting is easy — gsm8k **2.99×** (+0.67 over vanilla),
  math_reasoning 2.94×, math500 2.71×, humaneval 2.63× — and loses on
  retrieval/chat-ish and multilingual work.
- **The §5 draft-vocab diagnosis is confirmed.** More data barely moved
  multilingual (0.97× → 1.06×, still 0.83 behind vanilla's 1.89×) while math
  jumped ~0.5. The 32k pruned draft vocab, not data volume, is the binding
  constraint there — the next experiment worth running is a larger
  `--draft-vocab-size`, not more samples.
- **Deployment read:** for a math/code-heavy workload, this checkpoint is now
  the best option; for general multilingual traffic, the vanilla assistant is
  still safer. A vocab fix would likely settle it outright.
- Training was still improving at the epoch cap (epoch 1→2 gained *more* than
  0→1), so a 4th/5th epoch is also unexplored headroom.

_Results: `scripts/evaluate/experiments/results/gemma4-26b-post400k-eval/`
(merged from the four parallel `post400k-*` shards + the `heldout_chat` legs).
Checkpoint: `output/gemma4_26b_dspark_400k/dspark/checkpoints/checkpoint_best`._

---

## 8. Does more training help? Continuation run (4 more epochs) — no

§7 left the 400k run still improving at its 3-epoch cap, so this tests whether
more epochs pay. **They do not, in any practical sense.**

Not a plain resume: the original run used the default **linear** LR decay and
ended at lr ≈ 6.5e-07, so resuming into the same `--save-path` would restore
that dead schedule and train at ~zero LR. Instead the weights were loaded with
`--from-pretrained` into a fresh save-path and the schedule restarted at a
lower peak (**1e-4**, vs the original 3e-4) with **cosine** decay, 4 epochs,
`--early-stop-patience 1`. (`--from-pretrained` carries the architecture, so
model-definition flags like `--num-layers` must not be passed with it.)

### Validation trajectory

| epoch | val loss | val AL | note |
|---|---|---|---|
| — | 0.4950 | 3.520 | the original checkpoint (the bar) |
| 0 | 0.5172 | 3.386 | LR restart perturbs it — *worse* than the start |
| 1 | 0.4979 | 3.523 | recovered, ~tie |
| 2 | 0.4899 | 3.582 | clears the bar |
| **3 (best)** | **0.4897** | **3.587** | plateaued (LR ≈ 0) |

Early stopping never fired: each epoch improved on the previous. Note it fires
against *this run's* best, not the previous run's — worth watching when
continuing into a fresh save-path.

### Served result (26 benchmarks, fresh matched baseline)

| benchmark | continuation (7 ep total) | original (3 ep) |
|---|---|---|
| gsm8k | 2.97 / 5.52 | 2.99 / 5.56 |
| math_reasoning | **2.95 / 5.42** | 2.94 / 5.39 |
| math500 | **2.75 / 5.17** | 2.71 / 5.11 |
| humaneval | **2.66 / 4.99** | 2.63 / 4.95 |
| HumanEval | **2.48 / 4.67** | 2.47 / 4.65 |
| aime26 | 2.33 / 4.65 | 2.33 / 4.65 |
| bfcl | 2.29 / 4.35 | 2.31 / 4.40 |
| aime | 2.28 / 4.54 | 2.31 / 4.55 |
| mbpp | **2.28 / 4.34** | 2.23 / 4.31 |
| livecodebench | **2.04 / 4.12** | 2.03 / 4.10 |
| heldout_chat | **1.98 / 3.75** | 1.95 / 3.69 |
| speed-coding | 1.90 / 3.87 | 1.93 / 3.91 |
| gpqa | **1.80 / 3.52** | 1.76 / 3.48 |
| translation | **1.66 / 3.05** | 1.66 / 3.03 |
| swe-bench-pro | **1.65 / 3.32** | 1.65 / 3.30 |
| tool_call | **1.60 / 2.99** | 1.59 / 2.95 |
| speed-rag | 1.56 / 3.19 | 1.57 / 3.19 |
| mt-bench | **1.54 / 2.88** | 1.52 / 2.83 |
| question | **1.54 / 2.88** | 1.52 / 2.83 |
| rag | **1.49 / 3.00** | 1.48 / 2.98 |
| qa | **1.34 / 2.45** | 1.34 / 2.44 |
| speed-writing | **1.26 / 2.57** | 1.25 / 2.55 |
| summarization | 1.18 / 2.39 | 1.18 / 2.40 |
| speed-multilingual | 1.06 / 1.91 | 1.06 / 1.92 |
| **MEAN (24 distinct)** | **1.942× / 3.731** | 1.934× / 3.716 |

Continuation wins 16 of 24 benchmarks — consistent, but the margin is
**+0.4%**.

### Takeaways

- **4 extra epochs (~4 days on 4 GPUs) bought +0.4% served speedup**
  (1.934× → 1.942×). The 400k data is **saturated**; more passes are not
  where the remaining headroom is.
- **Validation AL overstates deployment gain**: training AL rose +1.9%
  (3.520 → 3.587) but served speedup moved only +0.4%. Judge
  drafts by served speedup, not val AL.
- The LR-restart dip (epoch 0 *worse* than the starting checkpoint) is the
  expected cost of re-warming; budget at least ~2 epochs before a continuation
  shows any gain, or it will look like a failure at epoch 0.
- Standing versus the reference: **1.940× vs the vanilla assistant's
  1.890×** on the 23 distinct shared benchmarks — still ahead, still
  by a small margin, still lost on multilingual. The open lever remains a
  larger `--draft-vocab-size`, plus the ~286k samples never trained on.

_Checkpoint: `output/gemma4_26b_dspark_400k_cont/dspark/checkpoints/checkpoint_best`
(epoch 3). Results: `scripts/evaluate/experiments/results/gemma4-26b-cont400k-eval/`. Launcher:
[`examples/train/_continue_400k_detached.sh`](../../examples/train/_continue_400k_detached.sh)._

---

## 9. Production-style workload: `sc1_delta` — the assistant wins decisively

The 26-benchmark suite (§5-§8) is public benchmarks with short prompts. This
section evaluates the same drafts on **`/nvmedata/data/sc1_delta_v2.jsonl`**
(MAI Profile V3 delta interest extraction): long, heavily structured
production prompts — **mean 3879 prompt tokens** (max ~16k) producing
**~1196 output tokens**. 100 prompts, **single A100, tp=1**, greedy,
vLLM 0.28. Registered as the `sc1_delta` benchmark in both eval runners.

| config | decode tok/s | AL | AR | QPS | output tok/s | latency (s) | TTFT (s) |
|---|---|---|---|---|---|---|---|
| baseline | 118.2 | — | — | 0.096 | 114.3 | 10.47 | 0.36 |
| DSpark 400k+cont (k=7) | 135.2 | 2.48 | 0.211 | 0.109 | 130.4 | 9.20 | 0.33 |
| **vanilla assistant (k=5)** | **230.7** | **5.06** | **0.812** | **0.177** | **214.4** | **5.64** | 0.40 |

| speedup vs baseline | decode | QPS | output throughput |
|---|---|---|---|
| DSpark 400k+cont | 1.14x | 1.14x | 1.14x |
| **vanilla assistant** | **1.95x** | **1.86x** | **1.88x** |

### Takeaways

- **This reverses the suite result.** On the 26 public benchmarks the two were
  near-tied (DSpark 1.94x vs vanilla 1.89x, §8). Here the assistant delivers
  **1.88x** output throughput against DSpark's
  **1.14x** — about **64% more throughput**.
- **The acceptance numbers show why:** vanilla reaches **AL 5.06 of a
  possible 6 at 81% acceptance**, while DSpark manages
  **AL 2.48 of 8 at 21%**. Not a serving bug — the same
  DSpark checkpoint serves at AL ~3.7 across the public suite with this exact
  setup. It is a **domain mismatch**: sc1_delta is long structured extraction
  with a fixed system prompt, unlike the kimi-regen math/chat training mix, and
  the 32k pruned draft vocab likely compounds it on structured tokens. The
  assistant carries the full 262k vocab and Google's broad training.
- **QPS gains trail decode gains** (1.86x vs
  1.95x) because prefill of ~3879-token prompts does not
  accelerate — speculation only speeds the decode phase. On prompt-heavy
  traffic, judge drafts by QPS / output throughput, not decode tok/s.
- **Deployment read: for traffic like sc1_delta, ship the vanilla assistant.**
  A DSpark draft would need in-domain training data (and a larger draft vocab)
  before competing here. The §7 lesson generalizes: our from-scratch drafts win
  where their training distribution matches, and lose where it does not.
- Batch=1 figures. The concurrency picture is a separate question (AgentX).

_Data: `/nvmedata/data/sc1_delta_v2.jsonl` -> `mtp_server_eval/data/sc1_delta.jsonl`
(100 of 1000 prompts sampled). Results: `scripts/evaluate/experiments/results/sc1-{base,dspark,vanilla}/`._

---

## 10. AgentX: concurrency sweep — the speedup does not survive load

Every other section is **batch=1**, where the GPU idles between tokens and
speculation is nearly free. AgentX replays agentic traces
(`semianalysisai/cc-traces-weka-062126`, no trace under 44k tokens) at rising
concurrency, which is the regime real serving runs in.

Setup: 26B-A4B backbone, **tp=4** on 4xA100, 65,536 context, **1024s per
concurrency level**, greedy. Config:
[`gemma4-26b-agentx-cont.yaml`](../../scripts/evaluate/experiments/gemma4-26b-agentx-cont.yaml).
All rows are stamped `submission_valid=true`.

> **Validity gotcha:** AgentX requires **>=900s per level**. Below that
> `run_agentx.sh` silently passes `--unsafe-override` and stamps every row
> `submission_valid=false` ("plumbing validation only, never for reported
> numbers"). Our first attempt used `duration: 600`, copied from the existing
> `gemma4-31b-agentx*.yaml` configs — **which still carry 600**, so any 31B
> AgentX numbers taken from `results/gemma4-31b-agentx/` are sub-threshold too
> (their matrices predate the `valid` column entirely).

| users | baseline out tok/s | DSpark out tok/s | ratio | baseline decode | DSpark decode | ratio | DSpark AL |
|---|---|---|---|---|---|---|---|
| 1 | 48.2 | 47.7 | **0.99x** | 91.2 | 99.2 | 1.09x | 3.17 |
| 8 | 54.9 | 56.5 | **1.03x** | 87.3 | 83.8 | 0.96x | 2.95 |
| 16 | 80.4 | 88.1 | **1.10x** | 78.4 | 78.2 | 1.00x | 3.16 |
| 32 | 148.2 | 139.9 | **0.94x** | 57.4 | 51.4 | 0.90x | 2.80 |
| 64 | 196.8 | 202.9 | **1.03x** | 14.8 | 18.0 | 1.22x | 2.88 |
| 128 | 64.8 | 82.4 | **1.27x** | 3.7 | 11.6 | 3.14x | 3.01 |

### Takeaways

- **Speculative decoding buys essentially nothing here at any concurrency.**
  Output throughput sits at parity (0.94x-1.27x, no trend), versus **1.94x**
  for the same checkpoint on the batch=1 public suite (§8).
- **Even at users=1 it is only 1.09x decode**, far below the suite's 1.94x.
  These traces carry 44k+ token contexts, so attention over a huge KV cache
  dominates each step and drafting saves proportionally far less. Long context
  erodes speculative gains *independently* of batching.
- **Acceptance is not the problem.** AL holds at **2.8-3.2 across every
  concurrency level** — the draft predicts just as well at 128 users as at 1.
  The benefit disappears for purely economic reasons: in
  `speedup = AL / (1 + omega)`, contention inflates `omega` because the draft's
  extra compute now competes for saturated SMs. Same AL, no speed.
- **Capacity finding independent of drafts:** aggregate throughput peaks at
  **64 users** (~197-203 tok/s) and *collapses* at 128 (~65-82 tok/s) for both
  configs — the serving system thrashes past 64 concurrent 64k-context
  requests. That is a larger deployment lever than any draft choice.
- **Practical read:** justify a draft on the regime you actually serve. A
  1.9x batch=1 number does not transfer to a loaded, long-context agentic
  server.

_Results: `scripts/evaluate/experiments/results/gemma4-26b-agentx-cont/`.
The vanilla-assistant sweep (same settings) is a separate run._

---

## 11. Draft-vocabulary size: the hypothesis did NOT hold

§7/§9 pinned DSpark's weakness on the **32k pruned draft vocab** (12.2% of the
target's 262,144 ids; measured token coverage 82.8% on sc1_delta, 54.3% on
multilingual). Every unreachable token is a guaranteed rejection, so doubling
the vocab should have lifted exactly those domains. **It did not.**

| benchmark | DSpark 32k (400k samples) | DSpark 64k (100k samples) | delta |
|---|---|---|---|
| speed-multilingual | 1.06x | 1.03x | -0.03 |
| translation | 1.66x | 1.60x | -0.06 |
| summarization | 1.18x | 1.15x | -0.03 |
| gsm8k | 2.97x | 2.96x | -0.01 |
| humaneval | 2.66x | 2.49x | -0.17 |
| bfcl | 2.29x | 2.40x | +0.12 |
| **suite mean (24 benchmarks)** | **1.942x** | **1.875x** | -0.067 |

sc1_delta, against its own baseline:

| draft | speedup | AL |
|---|---|---|
| DSpark 32k (400k samples) | 1.14x | 2.48 |
| DSpark 64k (100k samples) | 1.21x | 2.75 |
| **stock DFlash (full vocab)** | **2.14x** | **5.03** |

### Reading

- **The decisive row is multilingual.** Doubling the vocab raised its coverage
  54.3% -> 63.1%, and speedup moved 1.06x -> **1.03x** — nothing. That is the
  one domain the coverage story required to improve.
- sc1_delta gained a little (+0.07, AL 2.48 -> 2.75) but remains far from
  DFlash's 2.14x / AL 5.03.
- **Confound:** the 64k run trained on 100k samples vs the 32k run's 400k.
  That 4x data disadvantage cost only ~3.5% of suite mean, while the vocab
  doubling bought ~nothing where it was predicted to. The confound weakens the
  test but does not rescue the hypothesis.
- With §9's architecture explanation already refuted by stock DFlash (same
  parallel block drafting, 2.14x on sc1), **the remaining explanation for
  DFlash's lead is training-data breadth** — consistent with the per-slot decay
  evidence (DFlash 0.84/slot vs ours 0.52 on sc1), which is a sequence-modeling
  signal rather than a coverage one.

### Future work (parked)

Reduced draft vocab is **not** exonerated — it was simply not the dominant
term here, and a larger vocab carries costs we hit directly:

- **Training cost is real.** Full 262k vocab inflates the draft 1,085M ->
  ~1.8B params (lm_head 90M -> 738M, markov_w2 8M -> 67M) plus optimizer
  state; the full-vocab run OOMed at 69.8 GiB resident with a 12 GiB loss
  allocation (`max_anchors x 8 positions x 262144 x 4B`). It needs
  max_anchors <= ~512, a shorter sequence, or the DFlash approach of
  **borrowing the verifier's lm_head instead of training its own**
  (stock DFlash ships *no* lm_head or embed_tokens at all — 430M params total
  vs our 1,085M, 82% of which is vocabulary tables).
- **The silent-failure risk is the real concern.** The pruned vocab is derived
  from *training-data* token frequency, so a checkpoint silently underperforms
  on any traffic whose token distribution differs — and standard benchmarks do
  not reveal it. Ours looked fully competitive on the 26-benchmark suite
  (1.94x, §8) while collapsing to 1.14x on production traffic (§9). Any future
  pruned-vocab checkpoint should be gated on a production-traffic slice, and
  coverage measured on the *deployment* distribution rather than assumed.
- Untested: re-deriving the 32k mapping from domain-matched token frequencies
  (free, no cost change), and a matched 64k-at-400k run to remove the data
  confound above.

_Results: `scripts/evaluate/experiments/results/gemma4-26b-vocab64k/`. Checkpoint:
`output/gemma4_26b_dspark_vocab65536/dspark/checkpoints/checkpoint_best`._

---

## 12. Warm-starting DSpark from the stock DFlash backbone — the fix

§9-§11 falsified both candidate explanations for why our from-scratch DSpark
collapsed on production traffic: **architecture** (stock DFlash is the same
parallel block draft and reaches 2.14x on sc1_delta) and **vocabulary**
(doubling 32k -> 64k moved multilingual 1.06x -> 1.03x, §11). What remained was
**training breadth**. Rather than try to reproduce that with our kimi-regen
mix, this run *inherits* it: DSpark is by its own docstring "DFlash backbone
plus a Markov logit-bias head and a confidence head", so the backbone is
initialized from the published DFlash checkpoint and training only has to fit
the two new heads.

Build: convert the vLLM-native stock DFlash to speculators format
(`convert_model(..., algorithm='dflash')`), relabel as `dspark` with
`markov_rank=256` / confidence head enabled, then train with
`--from-pretrained`. The Markov and confidence heads load as freshly
initialized; everything else transfers. Launcher:
[`examples/train/_dspark_from_dflash_detached.sh`](../../examples/train/_dspark_from_dflash_detached.sh).

Inherited from DFlash: full **262k** vocab (no d2t/t2d), **6** aux layers
`[2,7,12,18,23,28]`, `block_size 16`. 2 epochs, lr 1e-4 cosine, max_anchors 512.

### Training: better than from-scratch ever reached, in 2 epochs not 3

| run | val loss | val AL |
|---|---|---|
| **from DFlash, epoch 1 (best)** | **0.2788** | **4.480** |
| from DFlash, epoch 0 | 0.2850 | 4.352 |
| from scratch 400k, best of 3 epochs | 0.4954 | 3.520 |

### Served: first draft over 2.0x on the suite

| benchmark | **DSpark←DFlash** | DSpark scratch | stock DFlash | vanilla asst |
|---|---|---|---|---|
| gsm8k | **3.02x** | 2.97x | 2.62x | 2.32x |
| math_reasoning | 2.90x | **2.95x** | 2.58x | 2.37x |
| bfcl | 2.88x | 2.29x | **2.92x** | 2.39x |
| math500 | 2.63x | **2.75x** | 2.65x | 2.24x |
| humaneval | 2.62x | **2.66x** | 2.60x | 2.26x |
| HumanEval | **2.49x** | 2.48x | 2.38x | 2.20x |
| aime | 2.41x | 2.28x | **2.42x** | 2.10x |
| aime26 | 2.39x | 2.33x | **2.42x** | 2.10x |
| mbpp | 2.28x | **2.28x** | 2.11x | 2.03x |
| livecodebench | **2.10x** | 2.04x | 1.98x | 1.92x |
| gpqa | **2.01x** | 1.80x | 1.96x | 1.89x |
| speed-coding | **1.98x** | 1.90x | 1.89x | 1.93x |
| swe-bench-pro | **1.84x** | 1.65x | 1.77x | 1.79x |
| translation | 1.81x | 1.66x | 1.67x | **1.83x** |
| speed-rag | **1.80x** | 1.56x | 1.72x | 1.77x |
| tool_call | 1.70x | 1.60x | 1.59x | **1.78x** |
| rag | **1.68x** | 1.49x | 1.60x | 1.68x |
| mt-bench | 1.47x | **1.54x** | 1.28x | 1.44x |
| speed-multilingual | 1.43x | 1.06x | 1.57x | **1.89x** |
| question | 1.42x | **1.54x** | 1.28x | 1.47x |
| qa | 1.32x | 1.34x | 1.21x | **1.42x** |
| summarization | 1.25x | 1.18x | 1.15x | **1.37x** |
| speed-writing | 1.21x | 1.26x | 1.08x | **1.27x** |
| **MEAN (23)** | **2.029x** | 1.940x | 1.933x | 1.890x |

End-to-end throughput tracks decode speedup closely (prefill is a small share
of these short-prompt benchmarks):

| draft | decode | e2e throughput | decode tok/s | e2e tok/s |
|---|---|---|---|---|
| **DSpark<-DFlash** | **2.029x** | **1.939x** | 263.6 | 245.8 |
| DSpark scratch | 1.940x | 1.874x | 252.2 | 237.8 |
| stock DFlash | 1.933x | 1.853x | 247.3 | 231.1 |
| vanilla assistant | 1.890x | 1.820x | 241.8 | 226.9 |
| baseline | 1.000x | 1.000x | 130.1 | 126.5 |

### sc1_delta: 1.14x -> 1.91x

| draft | speedup | AL |
|---|---|---|
| stock DFlash | 2.14x | 5.03 |
| vanilla assistant | 1.95x | 5.06 |
| **DSpark<-DFlash** | **1.91x** | **4.84** |
| DSpark from scratch | 1.14x | 2.48 |

Per-slot acceptance on the suite shows where the gain comes from — the decay
per slot goes from ~0.52 to ~0.86:

| draft | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| **DSpark<-DFlash** | 0.889 | 0.772 | 0.685 | 0.608 | 0.512 | 0.446 | 0.356 |
| DSpark from scratch | 0.752 | 0.561 | 0.420 | 0.323 | 0.251 | 0.193 | 0.146 |

### Takeaways

- **Best suite result to date: 2.029x**, and it beats the DFlash
  backbone it started from (1.933x) — so DSpark's Markov and
  confidence heads *do* add value, but only on top of a well-trained backbone.
- **sc1_delta recovers from 1.14x to 1.91x (+67%)**, closing nearly all the gap
  to DFlash/assistant. 400k samples, 4 extra epochs and a doubled vocab could
  not do this; inheriting a broadly-trained backbone did.
- **Cheaper, too:** 2 epochs vs the from-scratch path's 3 + 4-epoch
  continuation.
- **Serving caveat / upstream bug:** the checkpoint advertises
  `speculative_tokens: 15` (block_size 16), but vLLM's DSpark runtime dies at
  k=15 with a CUDA `index out of bounds` during graph capture
  (`_sample_logits` -> `map_draft_to_target`). It serves cleanly at **k=7**,
  which is what is measured here. Worth reporting upstream; also means the
  checkpoint's own native depth is untested.

_Results: `scripts/evaluate/experiments/results/gemma4-26b-from-dflash/`. Init checkpoint:
`output/dspark_from_dflash_init`. Trained:
`output/gemma4_26b_dspark_from_dflash/dspark/checkpoints/checkpoint_best`._

---

## 13. What other published drafts train on (external registry)

From the shared **"Advanced Speculative Decoding"** registry
(`docs.google.com/spreadsheets/d/1WSNlOwJgGbBPhCZn47dCPmvOGeORPu8jZ4eZXec_rzA`,
tab `Speculative_Decoding_Architectures`, read 2026-10-03). Useful because
§12 concluded that **training breadth**, not architecture or vocabulary, is
what separated our from-scratch DSpark from the published drafts — so it is
worth knowing what the published ones actually train on.

| draft model | backbone | training datasets |
|---|---|---|
| `Inferact/Kimi-K3-DSpark` | Kimi-K3 | **kimi-mtp, OpenCodeInstruct, Nemotron, aya** (regenerated) |
| `Inferact/MiniMax-M3-EAGLE3` | MiniMax-M3 | mix2 (SWE-bench, OpenCodeInstruct, kimi-mtp) |
| `gemma4_draft_model_900k_eagle3_kimi_mtp_stem_code_math` | Gemma-4-31B | kimi-mtp + 3 nemotron splits (coding, stem, math) |
| `lightseekorg/kimi-k3-dspark` | Kimi-K3 | kimi-mtp (regenerated) |
| `lightseekorg/kimi-k2.5-eagle3-mla` | Kimi-K2.5 | open-perfectblend (regenerated) |
| `RedHatAI/*-speculator.{eagle3,dflash,dspark}` (6 models) | Gemma-4-31B, Gemma-4-26B-A4B, GLM-5.2, GPT-OSS-120b, Kimi-K3 | **Magpie + UltraChat** (regenerated) |
| `RedHatAI/Qwen3-32B-speculator.eagle3` | Qwen3-32B | ShareGPT + UltraChat |

Rows whose dataset cell reads "HuggingFace" are column-shifted in the source
sheet — the dataset was not recorded, not that it is unknown-by-design.

### Why this matters here

- **The breadth recipes match the §12 conclusion.** The most ambitious entry
  (Inferact's Kimi-K3 DSpark) covers chat/math + **code** + reasoning +
  **multilingual** — exactly the axes where our kimi-regen-only draft was
  weakest (multilingual 1.06x, §11; sc1_delta 1.14x, §9). `prepare_aya.py`
  and `prepare_opencodeinstruct.py` are already in this repo
  (`scripts/response_regeneration/`), so that mix is reproducible here.
- **The common published baseline is simpler than ours**: six RedHat
  speculators all use just **Magpie + UltraChat (regenerated)**. Breadth of
  *domain coverage* appears to matter more than raw sample count — our 400k
  kimi-regen run lost to drafts trained on a narrower-sounding but
  better-balanced mix.
- **CORRECTION (2026-10-04): no published DSpark exists for our backbone.**
  This section previously claimed `RedHatAI/gemma-4-26B-A4B-it-speculator.dspark`
  existed and was "the proper method-vs-method reference point". It does not
  exist. A direct Hub query returns no such repo, and `docs/index.md` agrees:
  the only published speculator for Gemma-4-26B-A4B is **eagle3**. The earlier
  sections this entry "corrected" were right. The nearest published DSparks are
  `RedHatAI/gemma-4-31B-it-speculator.dspark` (dense 31B) and
  `RedHatAI/Qwen3.6-35B-A3B-speculator.dspark` (a MoE backbone) — useful as
  recipe references, but neither shares our backbone, so neither can join the
  six-way table in §5.

The registry also carries a `Gemma4-dspark_training_eval` tab with another
team's DSpark runs on this family (15-24 epochs, lr 6e-4, sweeping
`sample_from_anchor` True/False at k=3/7/8; AL 5.89 at k=8 after 24 epochs,
AL 4.226 at k=7 `False` after 9). They hit the same `sample_from_anchor`
convention question documented in `.claude/skills/dspark-train-serve-parity`.


---

## 14. Registry training corpora — fetched locally (2026-10-04)

§12 concluded that **training breadth** is what separated our from-scratch
DSpark from the published drafts, and §13 recorded what those drafts train on.
This section records actually pulling that mix onto this machine.

### What the published DSpark configs specify

Read directly from the Hub (`config.json`), not from the registry sheet:

| speculator | backbone | layers | SWA | block | `draft_vocab_size` | `sample_from_anchor` |
|---|---|---|---|---|---|---|
| `RedHatAI/gemma-4-31B-it-speculator.dspark` | dense 31B | 5 | 2048 | 8 | **32000** | False |
| `RedHatAI/Qwen3.6-35B-A3B-speculator.dspark` | MoE | 5 | 2048 | 8 | **32000** | True |
| `RedHatAI/GLM-5.3-speculator.dspark` | GLM-5.3 | — | — | 8 | 154880 | True |
| **ours** (`gemma4_26b_dspark_from_dflash`) | 26B-A4B MoE | 5 | 2048 | 8 | **262144** | False |

**Every published DSpark prunes the draft vocabulary; ours does not.** §11
rejected reduced vocab on acceptance grounds alone — but acceptance is only
half of `speedup = AL / (1 + Ω)`, and §15 shows the Ω side is where the full
vocab costs us. §11's conclusion was drawn on incomplete evidence.

### Corpora fetched

Into `/nvmedata/data/registry_datasets/`. Repo IDs resolved against the Hub
rather than inferred from the registry's informal names — `aya` in particular
has moved org (`CohereLabs/`, not `CohereForAI/`, which 404s).

| local dir | repo | covers |
|---|---|---|
| `aya_dataset` | `CohereLabs/aya_dataset` | multilingual |
| `OpenCodeInstruct` | `nvidia/OpenCodeInstruct` | code |
| `nemotron_post_training` | `nvidia/Llama-Nemotron-Post-Training-Dataset` | coding / stem / math |
| `open-perfectblend` | `mlabonne/open-perfectblend` | general blend |
| `magpie_pro_300k` | `Magpie-Align/Magpie-Pro-300K-Filtered` | the common RedHat baseline |
| `ultrachat_200k` | `HuggingFaceH4/ultrachat_200k` | the other half of that baseline |
| `sharegpt_vicuna` | `anon8231489123/ShareGPT_Vicuna_unfiltered` | chat |
| `swe_bench_verified` | `princeton-nlp/SWE-bench_Verified` | part of the `mix2` recipe |

`kimi-mtp` is not re-fetched — already local as `/nvmedata/data/kimi-regen-*`.

Fetched 2026-10-04, 136 GB total, no failures:

| local dir | on disk |
|---|---|
| `nemotron_post_training` | **122 GB** |
| `OpenCodeInstruct` | 6.4 GB |
| `sharegpt_vicuna` | 4.1 GB |
| `ultrachat_200k` | 1.6 GB |
| `open-perfectblend` | 1.4 GB |
| `magpie_pro_300k` | 538 MB |
| `aya_dataset` | 134 MB |
| `swe_bench_verified` | 2.1 MB |

### Nemotron needs subsampling — do not take it whole

`nvidia/Llama-Nemotron-Post-Training-Dataset` is 122 GB because its SFT
config is enormous. The registry recipe
(`gemma4_draft_model_900k_eagle3_kimi_mtp_stem_code_math`) uses the coding,
stem and math splits — which together are **~32.9 M samples**:

| SFT split | on disk | samples |
|---|---|---|
| `math` | 71 GB | 22,066,397 |
| `code` | 45 GB | 10,108,883 |
| `science` (the "stem" split) | 5.7 GB | 708,920 |
| `chat` | 244 MB | 39,792 |
| `safety` | 56 MB | — |

For scale, our largest run to date (§6–7) used **400 k** samples. Regenerating
responses for 32.9 M through a 26 B target is not feasible on this machine.

**Sample the mix balanced, not proportional.** Proportional sampling would make
the corpus ~67% math — precisely the narrowness §12 blamed for our from-scratch
DSpark losing to better-balanced published drafts. A ~400–500 k mix drawing
comparably from math / code / science / aya (multilingual) plus the
Magpie+UltraChat baseline covers the axes where we measured weakness
(multilingual 1.06×, §11; `sc1_delta` 1.14×, §9).

### These are not yet training data

They are **raw corpora**. A draft must be trained on responses produced by *its
own target model*; training on the original human or third-model responses
teaches it to predict the wrong distribution. Every registry entry is marked
"(regenerated)" for this reason. The remaining pipeline is:

1. convert to conversation JSONL — `scripts/response_regeneration/prepare_aya.py`,
   `prepare_opencodeinstruct.py` (note: both default to
   `/import/ml-sc-scratch5/...` paths from the SambaNova scratch cluster that do
   not exist on this box; pass `--input-dir` explicitly)
2. **regenerate responses through gemma-4-26B-A4B** — the dominant GPU cost
3. generate hidden states / train

### Source-access note

The registry sheet could not be re-read on 2026-10-04: the Drive connector
returns its only visible tab (`Speculative Decoding Architectures Registry`) as
empty and CSV export fails with an internal error, although the file is 4.3 MB
and was modified 2026-10-03 16:14 UTC. The dataset list above is therefore the
2026-10-03 snapshot already captured in §13, not a fresh read.


---

## 15. Why DSparkFlash lost under load: draft overhead, not draft quality (2026-10-04)

§9 and §12 left DSparkFlash behind stock DFlash on `sc1_delta` throughput
despite winning the 25-benchmark suite. The assumption was that our draft was
simply worse on production traffic. **It is not — it is the better drafter, and
it was losing on cost.**

### Measurement protocol (read this before trusting any throughput number here)

Earlier `sc1_delta` throughput figures in this doc were measured with
`--enable-prefix-caching` **on**, over a fixed prompt set. Each run warms the
cache for the next, so throughput depends on run history. Measured directly:
three identical back-to-back runs on one unchanged server gave

    524.3  ->  608.4  ->  869.8 tok/s

a **66% drift** with no configuration change. Every number in this section was
therefore taken with `--no-enable-prefix-caching`, one discarded warmup per
cell, and 2 measured repeats; spreads are ≤1.4% unless stated. Prefix caching
remains correct for production — it is just unusable for A/B comparison.

**Consequence for this document:** absolute `sc1_delta` throughput figures in
§9/§10/§12 and any difference under ~10% are not reliable. Relative orderings
within a single sweep are, since those arms shared a run history.

### The diagnosis

vLLM's spec-decode counters separate the three candidate causes. With
`speedup = AL / (1 + Ω)`, measured against a baseline run through the identical
path (c1 108.5, c128 707.2 tok/s):

| | k_eff | AR | AL | speedup | Ω |
|---|---|---|---|---|---|
| DFlash c1 | 7.000 | 0.5346 | 4.742 | 1.852× | 1.560 |
| DFlash c128 | 7.000 | 0.5414 | 4.790 | 1.567× | 2.057 |
| DSparkFlash c1 | 7.000 | 0.5568 | **4.898** | 1.830× | 1.676 |
| DSparkFlash c128 | 7.000 | 0.5581 | **4.907** | 1.397× | **2.513** |

- `k_eff = 7.000` everywhere: the confidence head never shortens a draft. It
  cannot — `gemma4_dspark.py` **skips `confidence_head` at weight load**, so the
  head we train is absent at serve time entirely.
- **AL is flat across a 128× batch change** and DSparkFlash beats DFlash at
  *every* draft position p0–p6. Drafting quality is not the problem.
- Ω is: DSparkFlash starts 7% more expensive and grows 50% vs DFlash's 32%.

### Root cause

`draft_vocab_size = 262144` (full vocab; every published DSpark uses 32000,
see §14). The DSpark Markov head is applied **sequentially, once per draft
position**, and its `markov_w2` is `262144 × 256` bf16 = **134 MB** — far past
L2, so it re-streams from HBM 7× per draft, ~0.94 GB of traffic. At batch 1
that is ~9% on top of the shared draft+lm_head traffic (measured Ω gap: +7.4%);
at batch 128 the compute term switches on and 7 sequential skinny GEMMs
(M=128, K=256, N=262144) stop hiding latency (measured gap: +22%).

### The fix: `dspark_draft_topk`

vLLM already ships a gathered path (`_sample_sequential_topk` in
`v1/worker/gpu/spec_decode/dspark/speculator.py`) that does `topk` once for all
positions and corrects only those rows. It is off by default
(`dspark_draft_topk=None`) and absent from every published config — upstream
does not need it at 32k vocab.

| c128 | tok/s | AL | Ω |
|---|---|---|---|
| DSparkFlash full vocab | 987.8 | 4.907 | 2.513 |
| **DSparkFlash + `dspark_draft_topk=64`** | **1102.3** | **4.912** | **2.151** |
| DSparkFlash + `dspark_draft_topk=16` | 1097.8 | 4.891 | 2.151 |
| stock DFlash | 1108.0 | 4.790 | 2.057 |

**+11.6% for a serve-time flag on the existing checkpoint**, with AL unchanged
(4.912 vs 4.907) and no retraining. The c128 deficit vs DFlash goes from −10.8%
to −0.5%. At batch 1 it wins outright (203.8 vs 200.9).

`k` is insensitive between 16 and 64 (Ω identical at 2.151) — meaning the Ω
benefit **saturates below k=16**. That matters given the suite regression below:
a *larger* k should retain nearly all the Ω gain while giving back the lost
acceptance. k=256 / k=1024 are the experiments that should decide the setting;
k=64 is not yet established as the right choice.

### Suite validation: truncation is a TRADE, not free throughput

The sc1_delta result above (AL 4.912 vs 4.907) does **not** generalise. Running
the full 28-benchmark suite with both arms in one session — k=64 and full vocab,
same flags, same seed, servers started together — gives:

| | mean | median |
|---|---|---|
| ΔAL (k=64 − full vocab) | **−0.18** | −0.08 |
| Δdecode tok/s | +1.33% | +3.55% |

So top-64 truncation **does** cost acceptance on average. It wins throughput on
most benchmarks and loses on some (`livecodebench` ΔAL −1.02, −11.4% tok/s).
Whether it nets positive is workload-dependent — on `sc1_delta` it clearly wins,
on code benchmarks it does not.

#### The AL noise floor on this suite is ±0.31 — most single rows mean nothing

The suite contains duplicate prompt sets under different names, which gives a
free internal noise estimate. In the full-vocab arm, `mt-bench`, `question` and
`writing` each produced **identical total output (13,189 completion tokens** —
same prompts, same generations), yet reported:

| benchmark | AL | AR |
|---|---|---|
| mt-bench | 3.686 | 0.3837 |
| question | 3.375 | 0.3393 |
| writing | 3.544 | 0.3634 |

A **0.31 AL spread on byte-identical output**, because the draft's proposals vary
run to run through nondeterministic GPU reductions — accepted/drafted counts move
even when greedy output does not. (`qa`/`speed-qa` are likewise duplicates:
5,218 tokens, AL 2.787 in both.)

Consequences:
- **Per-benchmark ΔAL below ~0.3 is not interpretable.** The headline
  "regressions" on `mt-bench` (−0.725) and `writing` (−0.583) are ~2σ *and* are
  duplicates of each other, so they are neither significant nor independent.
- **The mean survives**: ~24 distinct benchmarks, σ≈0.31 → SEM≈0.06, so
  ΔAL = −0.18 ± 0.06 is real at ~3σ.
- `livecodebench` (−1.02) is the only row large enough to stand alone.
- Use `--num-samples` well above 20, or duplicate-aware averaging, for any
  future per-benchmark claim.

#### Do not compare suite runs across sessions

Comparing the k=64 arm against the *previous session's* full-vocab suite gave
`sc1_delta` ΔAL = −0.495, against +0.005 from the controlled measurement — same
model, same flag, same benchmark. Server flags (prefix caching), `max_model_len`
and seed all differ between sessions and move AL and throughput by more than the
effects under study. **Only compare arms started together in one run.**

### What remains

Ω bottoms out at 2.151 vs DFlash's 2.057 because `compute_draft_logits` still
runs its lm_head over all 262144 entries — `topk` fixes only the Markov head.
Retraining with `draft_vocab_size=32000` (the published recipe, §14) should
close it: at DFlash's Ω our AL would give 4.912/3.057 = **1.607×**, ≈1136 tok/s,
**beating DFlash outright**. That is the strongest argument yet for revisiting
§11's rejection of a reduced draft vocabulary.

### Negative result: MoE kernel auto-tuning was not the win

vLLM ships no tuned fused-MoE config for `E=128, N=704`, and baseline decode
sat at ~45–54% of an estimated A100 roofline, so a 12-hour `benchmark_moe.py`
tuning run was done on the theory that untuned kernels explained the gap. They
do not. The tuned config loads correctly
(`fused_moe.py:1148 Using configuration from …E=128,N=704,…A100_80GB_PCIe.json`)
and delivers **0% at batch 1** (110.2 untuned vs 110.1 tuned) and nothing
outside noise at c128. The roofline gap is elsewhere — per-step overheads that
a GEMM tile config does not touch. Tuned JSON retained at
`fused_moe/configs/E=128,N=704,device_name=NVIDIA_A100_80GB_PCIe.json`, but it
is unproven and lives only in `site-packages`, so a venv rebuild silently
discards it.


---

## 16. AgentX with `dspark_draft_topk` — speculation loses on agentic load (2026-10-05)

Concurrency sweep on 4xA100 TP=4, 64k context, 1024s per level
(`submission_valid=true` on every cell). Config:
`scratchpad/agentx-topk.yaml`, results `results/agentx-topk{,-rerun}/`.

### Baseline: the server collapses past 64 concurrent users

| users | decode tok/s | wall-clock out tok/s |
|---|---|---|
| 1 | 88.4 | 48.2 |
| 8 | 86.0 | 54.5 |
| 16 | 77.8 | 80.5 |
| 32 | 56.4 | 148.8 |
| 64 | 13.4 | **187.8** (peak) |
| 128 | 3.5 | **59.5** (collapse) |

Aggregate throughput **peaks at 64 users then falls 3x**, and per-request decode
degrades 25x (88.4 -> 3.5). This is overload, not saturation: at 64+ concurrent
44k-token requests the KV cache cannot hold the working set, so the server
spends its time on preemption and prefill recompute. **For this workload,
admission control capping concurrency at 32-64 is worth more than any draft-model
choice.** This is independent of speculative decoding.

Reproducibility: a second run a day later gave 89.2 / 86.9 / 78.8 / 58.1 for
users 1-32 — within 3%. Unlike the `sc1_delta` harness (section 15), AgentX has
no prefix-caching confound; the scenario deliberately cache-busts the first-turn
prefix, so its numbers *are* comparable across runs.

### Speculation is a net loss below 64 users

| users | baseline | DSparkFlash full vocab | speedup | AL |
|---|---|---|---|---|
| 1 | 88.4 | 83.4 | **0.94x** | 3.034 |
| 8 | 86.0 | 69.1 | **0.80x** | 2.697 |
| 16 | 77.8 | 61.8 | **0.79x** | 2.774 |
| 32 | 56.4 | 41.0 | **0.73x** | 2.556 |
| 64 | 13.4 | 14.6 | 1.09x | 2.601 |
| 128 | 3.5 | 9.8 | 2.80x | 2.664 |

The 2.80x at 128 users is **not a win worth having** — it is one overloaded
configuration beating another, both an order of magnitude below the server's
own peak. The real result is 0.73-0.94x everywhere the server is healthy.

**Why acceptance is so low here (AL ~2.6-3.0 vs ~4.9 on `sc1_delta`):** the
AgentX trace corpus carries **no prompt text** — only per-request token counts
and KV block hashes — so aiperf synthesizes prompts reproducing each trace's
length and prefix-sharing structure. Synthetic token streams are far less
predictable than real text. **AgentX measures the serving regime, not draft
quality**; its AL is not comparable to the real-text benchmarks, and the
slowdown here is not evidence that the draft is bad.

With AL ~2.6 and Omega ~2.4, the draft costs more than its acceptance repays.

### `dspark_draft_topk=64` reverses this completely

Re-run 2026-10-05 with an in-run baseline (which reproduced the previous day's
curve within 3%, so the full-vocab arm above is comparable):

| users | baseline | full vocab | **k=64** | full-vocab speedup | **k=64 speedup** |
|---|---|---|---|---|---|
| 1 | 89.2 | 83.4 | **110.2** | 0.94x | **1.24x** |
| 8 | 86.9 | 69.1 | **98.4** | 0.80x | **1.13x** |
| 16 | 78.8 | 61.8 | **89.0** | 0.79x | **1.13x** |
| 32 | 58.1 | 41.0 | **61.4** | 0.73x | **1.06x** |
| 64 | 14.3 | 14.6 | **18.5** | 1.09x | **1.29x** |
| 128 | 3.7 | 9.8 | **13.1** | 2.80x | **3.54x** |

**+27% to +50% over full vocab at every level**, and above 1.0x everywhere the
server is healthy — speculation goes from a net loss to a net win.

**The gain came from acceptance, not Omega** — the opposite of section 15:

| | AL @1 | AL @32 | Omega @1 |
|---|---|---|---|
| full vocab | 3.034 | 2.556 | ~2.22 |
| k=64 | **4.015** | **3.738** | ~2.25 |

Omega is essentially unchanged; AL rose by **+1.0**. On the 28-benchmark suite
the same flag *cost* 0.18 AL (section 15). Both are measured; the direction
depends on the workload.

**Conjecture, not finding:** on synthetic low-predictability token streams the
full-vocab draft may spread probability over a long noisy tail, and top-64
truncation concentrates mass on candidates the target might actually pick —
truncation as denoising. On real text, where the draft is already calibrated,
the same truncation removes mass that was genuinely needed. This fits both
observations but is untested. The practical reading is simpler and safer:
**`dspark_draft_topk` is workload-dependent and must be measured per workload,
not assumed.**

### Gotcha: a failed arm reports success

The first sweep's `dsparkflash_topk64_k7` arm produced **no cells at all** while
`run_experiments.py` still exited **rc=0**. The server had died at startup:

    pydantic_core.ValidationError: 1 validation error for SpeculativeConfig
      Value error, dspark_draft_topk is only supported by DSpark

vLLM validates `dspark_draft_topk` against `method == "dspark"` when
`SpeculativeConfig` is constructed — **before** it auto-detects the method from
the checkpoint. The YAML runner only sets `model` and `num_speculative_tokens`,
so `method` was `None`. Passing `--speculative-config` by hand (as section 15
did) sets `method` explicitly, which is why this never appeared there.

Two things to carry forward:

1. **Always set `method: dspark` in `speculative_config`** when using
   `dspark_draft_topk` through the YAML runner.
2. **`run_experiments.py` rc=0 does not mean every arm ran.** Check for
   `!! server exited early` and count the cells before trusting a sweep.
   Worth fixing upstream — a silent missing arm is the kind of failure that
   quietly invalidates a comparison.


---

## 17. Open experiments (2026-10-06)

### Settled, no further work needed

- **Draft depth: keep k=7.** The checkpoint is `block_size=16`
  (`sample_from_anchor=False`, native k=15), inherited from the DFlash warm
  start. **k=15 does not crash** — it serves cleanly, produces correct output
  and drafts exactly 15 tokens; whatever caused the earlier CUDA
  index-out-of-bounds does not reproduce. But on a matched prompt set it buys
  only **+5% AL (3.026 -> 3.181) for 2.14x the drafted tokens**, and the tail is
  nearly worthless (p7-p14 run 0.053 down to 0.005). Since
  `speedup = AL/(1+Omega)` and Omega scales with drafted tokens, k=15 is a net
  loss. Serving at k=7 is near-optimal, not a workaround.
  Per-position acceptance is **identical at p0-p1** between k=7 and k=15, so the
  16-wide-trained block loses nothing measurable when served 8-wide — the
  train/serve mismatch concern was unfounded.
- **`block_size`: stay at 16**, matching DFlash. Upstream's own note
  (`schema.py:741`) is that the block_size=16 recipe "consistently outperformed"
  8 on DFlash and that DSpark keeps 8 only because "that combination was never
  tested there". Do not retrain to 8.
- **MoE kernel auto-tuning: closed, negative** (section 15).

**These are four independent experiments.** None blocks another, they test
different terms, and each needs its own in-run control. Do not combine them in
a single run: an earlier plan bundled `block_size=8` with
`draft_vocab_size=32000` into one retrain, which would have made both
uninterpretable. Combine only after each is measured alone.

| # | tests | needs GPUs | blocked on |
|---|---|---|---|
| E1 | DSA context selection (draft **attention** cost) | 4 (1 server + 3 arms) | long-context data |
| E2 | draft vocabulary (draft **lm_head** cost) | 4 (1 server + 3 arms) | nothing |
| E3 | training **breadth** (acceptance, not cost) | 4 + regeneration pass | regeneration |
| E4 | re-measure the suite under the clean protocol | 2-4 serving only | nothing |

E2 and E4 can start today. E1 is blocked on data. E3 is blocked on a
regeneration pass that itself needs the GPUs.

### E1 — Test DSA context selection where it can actually pay

The DSA top-k context selection (`src/speculators/models/dspark/topk.py`,
commit 227c3a6) is aimed at the draft's attention cost over long prefixes.
That is the right target: section 16 shows speculation is a **net loss at
0.73-0.94x** under agentic load, where prefixes are 44k+ and the draft's
attention — not its vocabulary — dominates Omega.

**The existing A/B cannot settle it.** `f10ab67` found top-k 128 costs 1.2%
accept_len at 32% density and concluded the draft's useful context is local.
But that ran on `data_prep` whose lengths are:

    400,000 rows   median=641  mean=1210  p90=3759  max=4096 (capped)

At a 641-token median a local window covers most of the useful prefix by
construction, so **locality was favoured by the data, not necessarily by the
draft**. A three-arm ablation at matched density (dense / 128-ranked+32-local /
1-ranked+159-local) was built and then abandoned for this reason: its likely
B ~= C outcome would have been uninformative.

**Blocker: we have no long-context training data.** Approximate token lengths
of every local source:

| source | median | >8k | >16k |
|---|---|---|---|
| `merged_all_regen.jsonl` | 492 | 0.3% | **0%** |
| `sc1_delta_v2.jsonl` | 2,854 | 15.4% | **0%** |
| `train_regen.jsonl` | 380 | 0% | **0%** |

So E1 is really two steps:

1. **Source long-context training data** (16k-32k+ documents), re-prep at a
   matching `--total-seq-len`. Note hidden-state generation cost scales with
   context, so this is substantially more expensive per sample than the 4k runs.
2. **Then** run the three-arm ablation. The question it answers —
   *does indexer ranking beat plain locality when the prefix is long enough that
   a window must miss relevant content?* — is the one that decides whether the
   indexer, its KL term and its warm-up phase earn their place.

### E2 — Decide draft vocabulary on the lm_head, not the recipe

`dspark_draft_topk` and `draft_vocab_size` are the **same lever** pulled at
different times — a dynamic per-position vocabulary restriction vs a static
global one:

| | `draft_vocab_size=32000` | `dspark_draft_topk=64` |
|---|---|---|
| chosen | once, at training time | per draft position, at runtime |
| shrinks `markov_w2` | yes | yes (via gather) |
| shrinks `lm_head` | **yes** | **no** |
| can a token become unreachable | yes, permanently | no |

This explains why Omega bottoms out at 2.151 with top-k and will not go lower:
`compute_draft_logits` still runs the full 262144-row lm_head. **The case for a
32k retrain is specifically that it shrinks the lm_head, which top-k
structurally cannot** — not "the published recipe does it". Against that,
section 11 measured static pruning hurting multilingual, and the dynamic variant
excludes nothing currently probable, so static 32k should be expected to cost
*at least* the -0.18 AL top-k costs.

### E3 — Breadth-data retrain

The 8 registry corpora are on disk (section 14, 136 GB). Remaining work is
convert -> **regenerate responses through gemma-4-26B** -> train. Regeneration
dominates the cost. Nemotron must be subsampled **balanced, not proportional**
(32.9M samples, 67% math if taken proportionally — the exact narrowness
section 12 blamed for our from-scratch draft losing).

`scratchpad/prepare_registry_mix.py` converts all 8 to the pipeline's schema
and is validated end-to-end; it stride-samples the large JSONL shards so the
41GB math shard costs ~1 minute, and keeps only prompts (responses are
regenerated anyway, which is what makes 122GB of Nemotron tractable).

### E4 — Re-measure the suite under the clean protocol

The 2.03x suite figure (section 12) predates the measurement fixes in
section 15 and was taken with prefix caching on. It is not comparable to
anything measured since and should not be quoted until re-run with
`--no-enable-prefix-caching`, warmup, and repeats.
