# Gemma-4-26B-A4B (MoE) speculative-decoding experiments

Consolidated results for **`gemma-4-26B-A4B-it` (MoE)** speculative-decoding
drafts (MTP assistant, EAGLE3, DFlash, DSpark, P-EAGLE), the acceptance/throughput
evals, and two training-time bugs found & fixed here: the **shared-KV attention
leak** (§2) and a **hidden-state off-by-one** (§4), plus the feature-distillation
quality push, the six-way 25-benchmark profile (§5), and the 400k DSpark
scale-up (§6), its results (§7), and the continuation run that shows more
epochs are exhausted (§8).

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
