# Kimi-K3 speculator evaluation: EAGLE3 vs DSpark

Evaluation of two speculative-decoding drafts for Kimi K3 — the
TorchSpec-trained EAGLE3 draft and the published `RadixArk/Kimi-K3-DSpark`
draft — plus two more architecturally distinct DSpark checkpoints
(`Inferact/Kimi-K3-DSpark` and `lightseekorg/kimi-k3-dspark`, both native
MLA) evaluated live for cross-checkpoint comparison. **Four real issues
were found and fixed along the way (two in vLLM, one in EAGLE3's
checkpoint export, one missing feature in vLLM's DSpark model support)**
— root causes, evidence, and the original (now-superseded) offline
methodology are in the [Appendix](#appendix). The body below reflects the
current, corrected state: what each draft is, how it's served,
standardized metric definitions, the up-to-date real-vs-real comparison,
conclusions, the block-width ablation, and open follow-ups.

**Contents**
- [Draft under test (EAGLE3)](#draft-under-test)
- [RadixArk/Kimi-K3-DSpark — draft under test + serving notes](#radixarkkimi-k3-dspark-hf--second-draft-evaluated-2026-09-15)
- [Metric definitions](#metric-definitions)
- [Normalized view: Accept Length (AL) / Accept Rate (AR)](#normalized-view-accept-length-al--accept-rate-ar)
- [Bottom line](#bottom-line)
- [Ablation: drafting fewer tokens than trained](#ablation-drafting-fewer-tokens-than-trained-block_size-7--3)
- [Open follow-ups](#open-follow-ups)
- [Artifacts](#artifacts)
- [Appendix: bug history, superseded methodology and results](#appendix)
  - [Update (2026-09-18/19/20): four issues found, all fixed](#update-2026-09-181920-four-issues-found-all-fixed)
  - [Setup (measurement conditions)](#setup-measurement-conditions)
  - [Why offline](#why-offline)
  - [Capture convention (largest pitfall found)](#capture-convention-largest-pitfall-found)
  - [Result 1 — general chat (EAGLE3)](#result-1--general-chat-torchspecs-eval-set-teacher-forced)
  - [Result 2 — context-length sweep (EAGLE3)](#result-2--context-length-sweep-aa-lcr-on-policy)
  - [Result 3 — benchmark slice + throughput (EAGLE3)](#result-3--benchmark-slice--throughput)
  - [Reproduction check against the model card (DSpark)](#reproduction-check-against-the-model-card)
  - [Full 25-benchmark sweep (DSpark)](#full-25-benchmark-sweep)
  - [Per-slot accuracy (DSpark)](#per-slot-accuracy-dspark)

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
  loading. `num_speculative_tokens: 7` is required (matches `block_size=7`
  — K=7 draft tokens, not 6; see model card's "1 current + 7 draft tokens").
- **Historical, resolved 2026-09-18** — the spec-decode target-corruption
  bug described here (correct first ~10–12 tokens then token loops;
  acceptance 7/666 ≈ 1%) was confirmed method-independent (DSpark and
  eagle3 both showed it) and is now fixed in vLLM 0.29.0 — see the Update
  section for root cause, evidence, and real post-fix measurements.

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

**On-policy numbers originally shown here (aime, livecodebench, gpqa) were
measured on generation data later found corrupted by Bug 1 — removed rather
than kept as known-wrong.** See the Update section's "Corrected 24-set
DSpark sweep" for current values (aime AL 3.16, livecodebench AL 4.46, gpqa
AL 3.01) and "Full 24-set live-serving sweep" for the real measured
counterparts. One value from this original table is still valid and kept:
**general chat (64 convs, teacher-forced)**, unaffected by Bug 1 since it
uses pre-existing conversations, not on-policy generation — `accept_len`
(= real AL, no `+1` needed) **2.48**, accept_rate 0.445, pos-0..6 acc
0.66/0.54/0.45/0.41/0.38/0.36/0.33.

Also better positioned for long context on paper (yarn rope, 64k original) —
the aa-lcr length sweep for this draft is the natural follow-up.

## Metric definitions

This doc standardizes on two industry-standard speculative-decoding
metrics, **Accept Rate (AR)** and **Accept Length (AL)**, used everywhere
below. Both drafts compute them via genuinely different underlying
formulas (that's a real methodological difference, not a naming choice) —
each draft's specific computation is given below, alongside the
codebase-native field/variable names for traceability back to the actual
code and raw output files.

**Accept Rate (AR)** — the fraction of a drafted token's probability mass
the target model agrees with; equivalently, the expected fraction of
proposed draft tokens a correct rejection-sampling procedure would accept.
- *DSpark's computation* (codebase field: `accept_rate`): analytical
  distributional overlap, `AR = 1 - TV(draft_dist, target_dist)` — see the
  TV-distance discussion above for the exact formula. This is an exact,
  no-sampling-noise calculation from the full probability distributions.
- *EAGLE3's computation* (codebase field: `acc_i`, aggregated as `avg_acc`
  or `mean_i(acc_i)`): argmax-agreement — does the draft's single
  highest-probability token exactly match the target's single
  highest-probability token, at teacher-forced TTT step *i* (draft always
  receives ground-truth previous tokens). A discrete hit/miss, not a
  distributional-overlap calculation.
- *Real live-serving AR* (both drafts, where measured): `accepted_tokens /
  drafted_tokens`, read directly from vLLM's own
  `spec_decode_num_{accepted,draft}_tokens_total` Prometheus counters —
  actual counted accept/reject events, no formula involved.

**Accept Length (AL)** — the expected number of tokens emitted per verify
step, including the always-emitted bonus/anchor token (`AL = 1 +
<expected accepted draft tokens>`).
- *DSpark's computation* (codebase field: `accept_len`): cumulative
  product of per-position AR across the draft block, `AL =
  Σ_k Π_{i≤k} AR_i + 1`. The `+1` is already included in the raw
  `accept_len` field — do not add it again (this exact double-counting
  was a real bug caught and fixed earlier in this investigation).
- *EAGLE3's computation* (codebase field: `sim_acc_len`, TorchSpec's own
  name for this quantity — "sim" flags that it assumes independence across
  positions, an approximation, unlike DSpark's exact TV-based cumulative
  product): `sim_acc_len = Σ_{k=0..3} Π_{i≤k} acc_i`, which **excludes**
  the bonus token — so here, `AL = 1 + sim_acc_len` (the `+1` convention
  differs by which codebase field you start from; always check whether a
  given raw field already includes the bonus token before adding `+1`).
- *Real live-serving AL* (both drafts, where measured): `1 +
  accepted_tokens_total / drafts_total` (verify-step count), from vLLM's
  own counters — an actual counted average, not a formula.

**Other metrics used:**
- **Top-1 Accuracy** (codebase field: `full_acc`, DSpark only) — top-1
  argmax agreement pooled over all draft slots. Stricter than AR: AR gives
  partial credit for distributional overlap even when the single top
  guesses disagree; Top-1 Accuracy is a hard yes/no on whether the top
  guesses match exactly.
- **Calibration Error** (codebase field: `conf_err`/`confidence_abs_error`,
  DSpark only) — mean absolute difference between the confidence head's
  predicted accept probability and the realized AR; lower is better.
- **Cross-Entropy Loss** (codebase field: `vloss_i`, EAGLE3 only) — training
  loss of the draft's distribution vs. the target's soft distribution at
  TTT step *i*; a training-loss diagnostic, not a speculative-decoding
  quality metric per se.
- **Decode throughput (tok/s)**: real, measured single-stream decode speed
  — completion tokens ÷ decode wall time, per request. Both offline
  baseline measurements (`run_vllm_eval.py`, plain serving, no draft) and
  live speculative-decoding measurements (real requests against a live
  vLLM server with a draft attached, this doc's later "real" tables) use
  this same definition; only whether a draft is attached differs.
- **Aggregation**: per-sample batch means, averaged uniformly over samples
  for offline/analytical numbers (matches TorchSpec's
  `_aggregate_eval_metrics`); real live-serving numbers are pooled
  (summed counts, not averaged per-sample) since they come from
  cumulative Prometheus counters.
- **Loss masks**: chat set masked by TorchSpec's `KimiK3Parser` (assistant
  think+response channels); on-policy sets masked over the generated span
  only.

## Normalized view: Accept Length (AL) / Accept Rate (AR)

Both drafts restated in the doc's standardized **AL**/**AR** terms (see
[Metric definitions](#metric-definitions); vLLM's own serving counters use
the field names `accept_length`/`accept_rate`, same quantities) —
`AL = 1 + accepted_tokens/step` (includes the bonus token; maps directly
to the speedup ceiling vs the 108 tok/s baseline) and `AR = accepted_tokens
/ drafted_tokens`. Derived from the measured per-step/per-slot numbers
above (EAGLE3 drafts 4 tokens/step, DSpark 7).

**Updated 2026-09-19 — this table now uses real live-serving numbers for
both drafts** (vLLM 0.29.0, post-Bug-3-fix for EAGLE3), not the offline
analytical/TTT estimates the 2026-09-18 version of this table used. This is
a meaningful revision, not just a refresh: EAGLE3's real live AR/AL run
noticeably *higher* than its own offline TTT numbers (see the "Full 24-set
live-serving sweep — EAGLE3" section for why) — e.g. `aime` AR went from
0.176 (offline) to 0.296 (real). Two rows have no live equivalent and are
still offline-only, clearly marked below.

| set | EAGLE3 AR (live, K=4) | EAGLE3 AL (live, K=4) | DSpark AR (live, K=7) | DSpark AL (live, K=7) |
|---|---:|---:|---:|---:|
| aime | 0.296 | 2.19 | 0.233 | 2.63 |
| livecodebench | 0.448 | 2.79 | 0.365 | 3.55 |
| gpqa | 0.319 | 2.28 | 0.242 | 2.70 |
| aa-lcr-1k | 0.325 | 2.30 | 0.246 | 2.72 |
| aa-lcr-4k | 0.350 | 2.40 | 0.282 | 2.97 |

**Offline-only, no live equivalent measured for either draft:**

| set | EAGLE3 AR (offline, K=4) | EAGLE3 AL (offline, K=4) | DSpark AR (offline, K=7) | DSpark AL (offline, K=7) |
|---|---:|---:|---:|---:|
| general chat (chat64) | 0.234 | 1.94 | 0.445 | 2.48 |
| aa-lcr 8k (EAGLE3's own context sweep, `Result 2`, different methodology) | 0.138 | 1.55 | — | — |
| aa-lcr 16k (same) | 0.039 | 1.16 | — | — |

`chat64` uses 64 pre-existing conversations, not part of the 24-set
on-policy live sweep for either draft — no live number exists for it yet.
The aa-lcr 8k/16k rows come from EAGLE3's own paired-bin context-length
sweep (`prepare_aa_lcr_sweep.py`), distinct from the 24-set sweep's own
`aa-lcr-1k`/`aa-lcr-4k` entries (which do now have live numbers, in the
table above) — 8k/16k have not been rerun live for either draft.

Footnote: EAGLE3's offline AR derives from greedy argmax agreement (TTT),
DSpark's offline AR from the analytical distribution-overlap acceptance —
both estimate rejection-sampling acceptance but are not bit-identical
definitions (see [Metric definitions](#metric-definitions)). The live
numbers above are directly comparable between the two drafts, since both
are read the same way from vLLM's own `spec_decode_num_*_total` counters.

## Bottom line

**EAGLE3 draft (TorchSpec, iter 39388):**
- **Real, live-serving confirmation (vLLM 0.29.0, corrected config, the
  authoritative numbers): AL 2.06–3.92 across 24 domains, mean AL 2.79,
  202 tok/s mean throughput** — beats the 108 tok/s no-draft baseline on
  every single set (1.4–2.5×). See "Full 24-set live-serving sweep —
  EAGLE3" in the Update section. Getting here required finding and fixing
  Bug 3 (checkpoint export shipped wrong aux-layer ids for live serving) —
  the first live attempt was *slower than no draft at all*.
- Offline TTT-harness quality (correct capture convention): **AL 1.94** (AR
  0.234; native: avg_acc 0.492, sim_acc_len 0.94) on general chat; **AL
  2.03–2.42** on in-domain code/math benchmarks (gsm8k, mbpp, math500,
  livecodebench); AL **1.67** on out-of-domain science QA (gpqa). Real
  live-serving AL runs consistently *higher* than these offline numbers
  (ratio 1.20–1.67, see the live sweep section for why) — treat the offline
  numbers as a *relative* domain ranking, not an absolute ceiling. See
  [Normalized view](#normalized-view-accept-length-al--accept-rate-ar) for
  the full offline table.
- **Acceptance collapses at 16k context** (AL 1.16, AR 0.039; native
  sim_acc_len 0.16, offline TTT harness only, not yet re-verified live) —
  do not use this draft for long-context serving; the next draft should fix
  `rope_theta` and train on long sequences.
- Hidden-state capture convention for this draft/method: `prefix_only` + vLLM
  ids `[49, 69, 89]`. The wrong choice silently costs up to 16× offline, and
  live-serving-fatal in practice (see Bug 3) — sanity check the served
  config's log line (`Using Eagle3 auxiliary layers from config: ...`)
  before trusting any live number.

**DSpark draft (RadixArk/Kimi-K3-DSpark):**
- Clearly stronger and more robust: **AL 2.88–5.97 across 24 domains on
  corrected clean data** (see the Update section) — no domain collapse.
- Reproduction check against the model card's live-SGLang numbers: initially
  looked like a 10-27% shortfall, but that comparison used pre-correction
  data on both bugs (double-counted AL, and corrupted on-policy generation).
  Now confirmed via **live vLLM 0.29.0 serving with real acceptance
  metrics** (68.2% AR, AL 5.77) that the checkpoint's own reported numbers
  were correct all along.
- Hidden-state capture convention for this draft/method: `prefix_only` +
  vLLM ids `[8, 24, 52, 68, 84]` (dflash `target_layer_ids [7,23,51,67,83]`
  +1). Note the convention is method-specific — do not reuse EAGLE3's ids.
- Long-context (aa-lcr 8k/16k) not yet measured for this draft — the natural
  next step given its yarn rope design, especially since that's exactly where
  the EAGLE3 draft failed.

**Inferact/Kimi-K3-DSpark (third checkpoint, MLA-based, vLLM-native):**
- **Real, live-serving, full 24-set sweep (vLLM 0.29.0): AL 2.63–5.55
  across 24 domains, mean AL 3.66, mean throughput 247 tok/s** — beats
  RadixArk's own real numbers on 18 of 24 sets, roughly ties on 1
  (mtbench), trails narrowly (within 3%) on the other 5 (gsm8k, mbpp, qa,
  speed-qa, writing) — mean AL 3.66 vs 3.47, mean tok/s 247 vs 235 overall.
  Notable despite a completely different architecture (native MLA matching
  the target, vs RadixArk's full-attention Qwen3-style design) and
  training pipeline (vLLM-native vs SGLang). Pooled: AR = 33.9%, AL = 3.37.
- This checkpoint isn't supported by this repo's own offline eval harness
  (`run_dspark_eval.py` targets RadixArk's `DSparkDraftModel`
  architecture specifically) — live serving is the only evaluation path
  available for it, so there's no offline/analytical number to
  cross-check against here, unlike EAGLE3 and RadixArk DSpark (lightseekorg's
  checkpoint below has this same limitation, for the same reason).
- Requires `attention_backend: FLASHINFER_MLA` and
  `draft_sample_method: greedy` in `--speculative-config`; aux/target
  layer ids `[2, 23, 47, 71, 89]` (already vLLM-ready in this checkpoint's
  config, unlike EAGLE3's — no Bug-3-style shift issue here, confirmed via
  a clean gsm8k spot-check before the full sweep).

**lightseekorg/kimi-k3-dspark (fourth checkpoint, MLA-based, vLLM-native, TokenSpeed-trained):**
- **Real, live-serving, full 24-set sweep (vLLM 0.29.0 + custom patch):
  AL 2.45–5.32 across 24 domains, mean AL 3.60, mean throughput 244
  tok/s** — beats the 108 tok/s baseline on every set. Cross-checked
  against the checkpoint's own README-reported benchmarks (measured
  independently via TokenSpeed): ratios scattered 0.92-1.16× with no
  systematic bias, a strong signal the evaluation is measuring the
  checkpoint correctly.
- Required a real fix, not just a config change: vLLM's installed
  `K3DSparkModel` is missing a per-tap `fc_norm` step this checkpoint's
  training contract requires (Bug 4) — worked around via a
  `vllm.ModelRegistry` override that doesn't touch the installed
  package, active only when a checkpoint's config declares `fc_norm`
  (verified inert for Inferact's checkpoint, which doesn't).
- This checkpoint's authors' reference serving stack is **TokenSpeed**,
  not vLLM — not installed in this environment; the vLLM path used here
  is a from-scratch community-side fix, not something the checkpoint
  authors validated themselves.

**All four checkpoints:**
- Serving-side spec decode is **no longer blocked** — see the
  [Update section](#update-2026-09-181920-four-issues-found-all-fixed) in the
  Appendix. vLLM 0.28.0 had a real, now-fixed bug (issues #50851, #51039)
  that corrupted the target's output under live speculative decoding,
  method- and checkpoint-independent (eagle3 and dspark, and multiple
  DSpark checkpoints, all failed identically on 0.28.0). vLLM 0.29.0
  (2026-09-09+) produces clean output and real measured speedups across
  all four checkpoints tested (RadixArk, Inferact, lightseekorg, EAGLE3).
  All four now have full 24-set live sweeps (real AR/AL/throughput, see
  the "Full 24-set live-serving sweep" tables in the Appendix, and the
  block=7 vs block=3 real comparisons under the Ablation section above
  for RadixArk).

## Ablation: drafting fewer tokens than trained (block_size 7 → 3)

**Rerun 2026-09-18 on the Bug-1-fixed clean data** (see the
[Update section](#update-2026-09-181920-four-issues-found-all-fixed) in the
Appendix) — the tables and analysis below replace the original
degenerate-data version. The qualitative headline conclusion is unchanged
(AR/full_acc up at block=3, AL down, unanimous across all 24 sets), but two
quantitative findings flipped or shifted materially — flagged inline below.

Question: DSpark was trained with `block_size=7` (K=7 draft tokens/step) —
if it were served drafting only 3 tokens/step instead, does per-token quality
drop, or does it just cap the ceiling?

Implemented as a real architectural change, not post-hoc pooling:
`model.block_size` is a plain runtime int (no weight tensor is shaped by it —
`MarkovHead`/`ConfidenceHead` operate per-token, block_size-agnostic), read
fresh at every forward call by the anchor-selection, same-block attention
mask (`create_anchor_block_mask_mod`, `dflash/attention.py` — draft slots
attend to each other **non-causally** within a block for full-attention
layers), and target-gathering logic. Overriding it to 3 genuinely shrinks the
synthetic block to 3 slots (1 anchor-seeded + 2 real `[MASK]` positions) —
slots 3-6 are never instantiated, not masked out of an existing 7. Reuses the
already-collected hidden states; no new server or collection needed.
`--block-size-override` in `run_dspark_eval.py`. Verified `n` (sample count)
is identical between the block=7 and block=3 run for all 24 sets — same
sample files, same target generations, only `model.block_size` differs.

One confound checked and ruled out: anchor selection (`select_anchors`,
`dflash/utils.py`) excludes only the last `block_size` sequence positions
from eligibility, and since `max_anchors=3072` always exceeds the actual
pool of loss-masked positions in these samples, *every* eligible position
becomes an anchor regardless of block_size (`k = min(max_anchors, pool) =
pool`) — so block=7 and block=3 select the same anchor positions almost
everywhere, differing only by a few extra tail positions block=3 newly makes
eligible. Slot 0 always predicts `anchor+1` regardless of block_size, so the
two runs are, for the vast majority of positions, scoring the *identical*
(anchor, target-token) pairs — the attention-window width is close to the
only real variable.

| dataset | AR@7 | AR@3 | AL@7 | AL@3 | full_acc@7 | full_acc@3 |
|---|---|---|---|---|---|---|
| gsm8k | 0.847 | 0.907 | 5.97 | 3.59 | 0.855 | 0.910 |
| bfcl | 0.723 | 0.837 | 4.97 | 3.34 | 0.734 | 0.843 |
| speed-coding | 0.710 | 0.827 | 4.78 | 3.28 | 0.727 | 0.835 |
| mbpp | 0.714 | 0.824 | 4.65 | 3.25 | 0.729 | 0.828 |
| livecodebench | 0.666 | 0.787 | 4.46 | 3.14 | 0.680 | 0.792 |
| math500 | 0.667 | 0.775 | 4.44 | 3.09 | 0.672 | 0.775 |
| tool_call | 0.645 | 0.778 | 4.19 | 3.09 | 0.664 | 0.788 |
| speed-rag | 0.636 | 0.776 | 4.17 | 3.10 | 0.658 | 0.786 |
| rag | 0.637 | 0.775 | 4.12 | 3.08 | 0.658 | 0.787 |
| translation | 0.640 | 0.758 | 3.96 | 2.99 | 0.667 | 0.776 |
| swe-bench-pro | 0.594 | 0.735 | 3.83 | 2.94 | 0.616 | 0.750 |
| speed-multilingual | 0.568 | 0.699 | 3.75 | 2.84 | 0.591 | 0.716 |
| summarization | 0.571 | 0.729 | 3.65 | 2.92 | 0.598 | 0.743 |
| speed-writing | 0.554 | 0.691 | 3.49 | 2.79 | 0.579 | 0.705 |
| qa | 0.550 | 0.694 | 3.43 | 2.79 | 0.573 | 0.708 |
| speed-qa | 0.550 | 0.694 | 3.43 | 2.79 | 0.573 | 0.708 |
| aa-lcr-4k | 0.515 | 0.659 | 3.38 | 2.70 | 0.523 | 0.659 |
| aa-lcr-1k | 0.496 | 0.634 | 3.22 | 2.61 | 0.504 | 0.635 |
| aime | 0.492 | 0.639 | 3.16 | 2.62 | 0.497 | 0.639 |
| swe-rebench | 0.507 | 0.653 | 3.11 | 2.63 | 0.530 | 0.664 |
| mtbench | 0.500 | 0.632 | 3.07 | 2.58 | 0.523 | 0.645 |
| gpqa | 0.463 | 0.618 | 3.01 | 2.56 | 0.475 | 0.622 |
| writing | 0.485 | 0.612 | 2.89 | 2.49 | 0.506 | 0.622 |
| aime26 | 0.461 | 0.600 | 2.88 | 2.47 | 0.459 | 0.595 |

`chat64` is not in this table — the block=3 rerun used the clean on-policy
24-set data only (`chat64` is teacher-forced on pre-existing conversations,
unaffected by Bug 1, and wasn't part of this rerun batch).

**The direction is still unanimous across all 24 sets**: AR and full_acc both
go *up* at block=3 (only scoring the easier early slots), AL goes *down*
(structurally fewer slots to accumulate acceptance across) — no set
contradicts this pattern, and no set shows a genuine per-token quality
regression. **AL retention (AL@3 ÷ AL@7) now ranges 0.60–0.86, mean 0.77**
across domains — a meaningfully wider range and lower mean than the
degenerate-data version reported (0.80–0.96, mean 0.88). The highest-AL
domains lose the most in relative terms even more sharply than before:
gsm8k (now the single highest-AL domain) retains only ~60%, while
writing/aime26 (lowest-AL) still retain ~86%.

Practical reading: **block=3 is not "3/7 as good"** — the model doesn't
degrade at the token level when asked to draft less, it simply forgoes the
long tail of increasingly-unlikely-to-be-accepted later slots. If serving
overhead per drafted token is non-trivial (verification cost, draft-head
compute), trading K=7 for K=3 gives back ~77% of the accept-length benefit
for ~43% of the draft width on average — though this now varies more by
domain than the degenerate-data version suggested (60%-86%, not a flat
~88%) — a real lever worth considering, distinct from retraining a
shorter-block draft from scratch (which this checkpoint was never asked to
do, so this is strictly a serving-time knob, not a training-time one).

### Real measured throughput, block=7 vs block=3 (vLLM 0.29.0, live serving)

**Replaces a previous projection-based table.** That projection (`108 tok/s
baseline × AL × 0.90`) turned out to be substantially wrong once real
numbers were available — both in overall magnitude (it overestimated real
throughput by roughly 1.3-2×) and in one qualitative conclusion (it always
showed block=3 slower than block=7, driven purely by block=3's lower AL).
Rather than keep a known-inaccurate estimate in the doc, it's removed;
below is the real replacement, measured the same way as the "Full 24-set
live-serving sweep" above (per-request wall-clock, matched generation
length per prompt, RadixArk DSpark on vLLM 0.29.0).

| dataset | real tok/s @7 | real tok/s @3 | ratio (3÷7) |
|---|---:|---:|---:|
| gsm8k | 360.7 | 269.2 | 0.75 |
| speed-coding | 307.4 | 255.0 | 0.83 |
| mbpp | 300.5 | 253.9 | 0.84 |
| math500 | 273.2 | 236.6 | 0.87 |
| speed-rag | 271.2 | 242.8 | 0.90 |
| livecodebench | 267.7 | 239.7 | 0.90 |
| translation | 267.7 | 230.1 | 0.86 |
| rag | 261.9 | 237.3 | 0.91 |
| bfcl | 249.3 | 216.7 | 0.87 |
| summarization | 239.8 | 228.2 | 0.95 |
| tool_call | 234.6 | 199.3 | 0.85 |
| swe-bench-pro | 226.8 | 207.1 | 0.91 |
| speed-multilingual | 225.1 | 209.6 | 0.93 |
| qa | 221.7 | 217.6 | 0.98 |
| speed-qa | 221.7 | 217.6 | 0.98 |
| swe-rebench | 212.2 | 202.7 | 0.96 |
| speed-writing | 206.8 | 206.2 | 1.00 |
| mtbench | 204.7 | 201.7 | 0.99 |
| aime | 195.4 | 203.6 | 1.04 |
| gpqa | 194.8 | 196.8 | 1.01 |
| writing | 189.2 | 194.3 | 1.03 |
| aime26 | 184.2 | 191.6 | 1.04 |
| aa-lcr-1k | 164.8 | 174.9 | 1.06 |
| aa-lcr-4k | 164.5 | 177.2 | 1.08 |

Mean real throughput: 235 tok/s @7, 217 tok/s @3 (ratio 0.92 overall) — but
the per-domain pattern is not uniform. **On the highest-AL, most
in-distribution domains (gsm8k, speed-coding, mbpp), block=7 is clearly
faster** — the draft earns enough real acceptance at the deeper slots to
justify the wider block's extra draft/verify compute. **On the hardest,
most out-of-distribution domains (aime, gpqa, writing, aime26, aa-lcr),
block=3 is as fast or measurably faster** — the draft rarely gets accepted
past slot 2-3 anyway on these domains, so the extra compute spent
drafting/verifying slots 3-6 in a block=7 config is largely wasted, and
shrinking to block=3 recovers that wasted work as real throughput. This is
exactly the kind of effect a flat linear projection (throughput ∝ AL) can't
capture, since it has no way to model per-step compute overhead scaling
with block width.

### Per-position accept_rate (AR), block=7 vs block=3

The pooled AR above (mean over all slots) hides the decay shape. This adds a
per-position AR breakdown (TV-overlap acceptance, not argmax) at both block
sizes — extends the earlier `full_acc` per-slot table with the metric this
whole ablation is actually about. Metric added to `compute_metrics`
(`position_{k}_accept_rate_sum/total`, mirroring the pre-existing per-position
`full_acc` pattern); verified via direct tensor-shape instrumentation that
block=3 only ever computes 3 positions (see chat: `logits.shape=[1,9216,...]`
= 3072 anchors × 3, no position_3-6 keys exist in the output at all — not
masked out of a wider computation, never instantiated).

| dataset | AR@7, pos-0..6 | AR@3, pos-0..2 |
|---|---|---|
| gsm8k | 0.940 / 0.908 / 0.878 / 0.849 / 0.821 / 0.784 / 0.749 | 0.940 / 0.908 / 0.872 |
| bfcl | 0.907 / 0.837 / 0.766 / 0.709 / 0.668 / 0.613 / 0.557 | 0.907 / 0.842 / 0.761 |
| speed-coding | 0.892 / 0.834 / 0.769 / 0.707 / 0.646 / 0.589 / 0.534 | 0.892 / 0.830 / 0.759 |
| mbpp | 0.891 / 0.829 / 0.766 / 0.706 / 0.654 / 0.601 / 0.553 | 0.890 / 0.826 / 0.756 |
| livecodebench | 0.865 / 0.790 / 0.716 / 0.651 / 0.594 / 0.548 / 0.500 | 0.865 / 0.787 / 0.710 |
| math500 | 0.838 / 0.775 / 0.714 / 0.658 / 0.607 / 0.558 / 0.519 | 0.840 / 0.776 / 0.708 |
| tool_call | 0.864 / 0.781 / 0.705 / 0.634 / 0.564 / 0.508 / 0.457 | 0.863 / 0.778 / 0.694 |
| speed-rag | 0.864 / 0.785 / 0.698 / 0.620 / 0.556 / 0.491 / 0.437 | 0.863 / 0.780 / 0.686 |
| rag | 0.862 / 0.782 / 0.697 / 0.623 / 0.556 / 0.496 / 0.445 | 0.860 / 0.778 / 0.685 |
| translation | 0.839 / 0.766 / 0.697 / 0.630 / 0.573 / 0.514 / 0.457 | 0.836 / 0.759 / 0.678 |
| swe-bench-pro | 0.818 / 0.743 / 0.658 / 0.581 / 0.510 / 0.452 / 0.395 | 0.820 / 0.738 / 0.648 |
| speed-multilingual | 0.779 / 0.702 / 0.624 / 0.553 / 0.493 / 0.438 / 0.390 | 0.782 / 0.699 / 0.616 |
| summarization | 0.835 / 0.734 / 0.631 / 0.545 / 0.473 / 0.415 / 0.367 | 0.834 / 0.730 / 0.622 |
| speed-writing | 0.795 / 0.692 / 0.602 / 0.525 / 0.469 / 0.419 / 0.377 | 0.795 / 0.686 / 0.591 |
| qa | 0.809 / 0.694 / 0.598 / 0.517 / 0.457 / 0.408 / 0.370 | 0.807 / 0.689 / 0.587 |
| speed-qa | 0.809 / 0.694 / 0.598 / 0.517 / 0.457 / 0.408 / 0.370 | 0.807 / 0.689 / 0.587 |
| aa-lcr-4k | 0.770 / 0.661 / 0.566 / 0.490 / 0.423 / 0.372 / 0.325 | 0.767 / 0.654 / 0.556 |
| aa-lcr-1k | 0.747 / 0.635 / 0.542 / 0.471 / 0.407 / 0.352 / 0.317 | 0.745 / 0.628 / 0.531 |
| aime | 0.752 / 0.639 / 0.539 / 0.460 / 0.399 / 0.348 / 0.309 | 0.752 / 0.635 / 0.529 |
| swe-rebench | 0.760 / 0.653 / 0.557 / 0.479 / 0.415 / 0.363 / 0.321 | 0.761 / 0.647 / 0.549 |
| mtbench | 0.748 / 0.629 / 0.532 / 0.466 / 0.414 / 0.373 / 0.339 | 0.747 / 0.625 / 0.525 |
| gpqa | 0.747 / 0.620 / 0.507 / 0.422 / 0.360 / 0.314 / 0.274 | 0.743 / 0.614 / 0.496 |
| writing | 0.728 / 0.605 / 0.514 / 0.448 / 0.402 / 0.362 / 0.332 | 0.727 / 0.600 / 0.508 |
| aime26 | 0.717 / 0.596 / 0.501 / 0.425 / 0.368 / 0.326 / 0.290 | 0.717 / 0.591 / 0.492 |

### Isolated pos-0/1/2 comparison: block=7's first 3 slots vs block=3

AR decays monotonically position-to-position in every single row at block=7
(no exceptions) — the earliest slot is always the most reliable, consistent
with the full_acc-based decay finding above. But the question actually being
asked here is narrower and more useful: **holding the slot fixed at 0, 1, or
2, does having 4 more mask tokens further out in the same block (block=7)
change that slot's AR at all, versus not having them (block=3)?** Isolate
just positions 0-2 from the block=7 run (discard 3-6) and diff them directly
against block=3's positions 0-2 — same slots, one variable (whether the rest
of the block exists).

| dataset | pos0@7 | pos0@3 | Δ0 | pos1@7 | pos1@3 | Δ1 | pos2@7 | pos2@3 | Δ2 | avg(pos0-2)@7 | avg(pos0-2)@3 | AL@3 (real) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 0.940 | 0.940 | +0.001 | 0.908 | 0.908 | -0.001 | 0.878 | 0.872 | -0.006 | 0.909 | 0.907 | 3.587 |
| bfcl | 0.907 | 0.907 | +0.001 | 0.837 | 0.842 | +0.005 | 0.766 | 0.761 | -0.005 | 0.837 | 0.837 | 3.337 |
| speed-coding | 0.892 | 0.892 | +0.000 | 0.834 | 0.830 | -0.005 | 0.769 | 0.759 | -0.010 | 0.832 | 0.827 | 3.283 |
| mbpp | 0.891 | 0.890 | -0.001 | 0.829 | 0.826 | -0.004 | 0.766 | 0.756 | -0.010 | 0.829 | 0.824 | 3.252 |
| livecodebench | 0.865 | 0.865 | -0.000 | 0.790 | 0.787 | -0.003 | 0.716 | 0.710 | -0.006 | 0.790 | 0.787 | 3.142 |
| math500 | 0.838 | 0.840 | +0.002 | 0.775 | 0.776 | +0.001 | 0.714 | 0.708 | -0.006 | 0.776 | 0.775 | 3.089 |
| tool_call | 0.864 | 0.863 | -0.002 | 0.781 | 0.778 | -0.003 | 0.705 | 0.694 | -0.010 | 0.784 | 0.778 | 3.087 |
| speed-rag | 0.864 | 0.863 | -0.001 | 0.785 | 0.780 | -0.005 | 0.698 | 0.686 | -0.012 | 0.782 | 0.776 | 3.104 |
| rag | 0.862 | 0.860 | -0.002 | 0.782 | 0.778 | -0.004 | 0.697 | 0.685 | -0.011 | 0.780 | 0.775 | 3.081 |
| translation | 0.839 | 0.836 | -0.003 | 0.766 | 0.759 | -0.007 | 0.697 | 0.678 | -0.018 | 0.767 | 0.758 | 2.988 |
| swe-bench-pro | 0.818 | 0.820 | +0.002 | 0.743 | 0.738 | -0.005 | 0.658 | 0.648 | -0.010 | 0.740 | 0.735 | 2.940 |
| speed-multilingual | 0.779 | 0.782 | +0.003 | 0.702 | 0.699 | -0.003 | 0.624 | 0.616 | -0.009 | 0.702 | 0.699 | 2.841 |
| summarization | 0.835 | 0.834 | -0.001 | 0.734 | 0.730 | -0.004 | 0.631 | 0.622 | -0.009 | 0.733 | 0.729 | 2.923 |
| speed-writing | 0.795 | 0.795 | -0.000 | 0.692 | 0.686 | -0.006 | 0.602 | 0.591 | -0.010 | 0.696 | 0.691 | 2.787 |
| qa | 0.809 | 0.807 | -0.002 | 0.694 | 0.689 | -0.006 | 0.598 | 0.587 | -0.011 | 0.700 | 0.694 | 2.793 |
| speed-qa | 0.809 | 0.807 | -0.002 | 0.694 | 0.689 | -0.006 | 0.598 | 0.587 | -0.011 | 0.700 | 0.694 | 2.793 |
| aa-lcr-4k | 0.770 | 0.767 | -0.003 | 0.661 | 0.654 | -0.006 | 0.566 | 0.556 | -0.010 | 0.666 | 0.659 | 2.700 |
| aa-lcr-1k | 0.747 | 0.745 | -0.002 | 0.635 | 0.628 | -0.007 | 0.542 | 0.531 | -0.011 | 0.641 | 0.634 | 2.611 |
| aime | 0.752 | 0.752 | -0.000 | 0.639 | 0.635 | -0.004 | 0.539 | 0.529 | -0.010 | 0.644 | 0.639 | 2.621 |
| swe-rebench | 0.760 | 0.761 | +0.001 | 0.653 | 0.647 | -0.006 | 0.557 | 0.549 | -0.008 | 0.657 | 0.653 | 2.632 |
| mtbench | 0.748 | 0.747 | -0.001 | 0.629 | 0.625 | -0.004 | 0.532 | 0.525 | -0.007 | 0.636 | 0.632 | 2.575 |
| gpqa | 0.747 | 0.743 | -0.003 | 0.620 | 0.614 | -0.006 | 0.507 | 0.496 | -0.011 | 0.625 | 0.618 | 2.563 |
| writing | 0.728 | 0.727 | -0.001 | 0.605 | 0.600 | -0.005 | 0.514 | 0.508 | -0.006 | 0.616 | 0.612 | 2.490 |
| aime26 | 0.717 | 0.717 | -0.001 | 0.596 | 0.591 | -0.005 | 0.501 | 0.492 | -0.010 | 0.605 | 0.600 | 2.474 |
| **mean Δ** | | | **-0.0007** | | | **-0.0041** | | | **-0.0095** | | | |
| **max \|Δ\|** | | | 0.003 | | | 0.007 | | | 0.018 | | | |

**avg(pos0-2)@7 / avg(pos0-2)@3** = mean of that row's own pos0/1/2 (a real,
exact quantity — no approximation error). **AL@3 (real)** = the exact
`accept_len` from the genuine block=3 model run (same value as the main
ablation table above, not re-derived).

**The independence approximation is now much closer to correct.** The
degenerate-data version of this table reported that deriving AL@3 as
`1 + p0 + p0·p1 + p0·p1·p2` from the pooled per-position means underestimated
the real AL@3 by 32-42% across every set. On clean data that gap shrinks to
**1-6%** — the earlier large gap was itself substantially an artifact of the
corrupted generation data (which likely exaggerated within-block correlation
by making "easy" and "hard" positions more extreme/bimodal than they
genuinely are), not a real property of the draft. Real per-block accept
rates are still slightly positively correlated within a block (hence a small
residual gap, not zero), just far less than the degenerate data suggested.

**The second-order pattern reported previously has flipped and shrunk.** The
degenerate-data version found slot 0 unanimously *higher* at block=3 (mean
+0.011) and slot 2 mostly *lower* (mean -0.0050). On clean data:

- **Slot 0 is now essentially flat** — mean Δ -0.0007, split 7 positive / 17
  negative, max magnitude 0.003 (0.3 percentage points). No meaningful
  effect either direction.
- **Slot 1 is now consistently, slightly negative** — 22/24 sets, mean
  -0.0041 (opposite direction from the old data's slight-positive finding).
- **Slot 2 is now unanimously negative across all 24/24 sets**, mean -0.0095,
  up to -0.018 (translation) — the same direction as before but the finding
  is now unanimous rather than 16/24.

Interpretation: on clean data, having 4 more mask tokens further out in the
same block (block=7) has **no effect on slot 0**, and a small, consistent,
now-unambiguous *negative* effect on slots 1-2 when they're restricted to a
narrower block (i.e., the wider block=7 context very slightly *helps* slots
1-2, opposite of the earlier "shrinking helps the early slot" reading, which
does not survive the data-corruption fix). All effects remain small in
absolute terms (max magnitude 1.8 percentage points across all 72 dataset×slot
cells) — the "near-independent" characterization is, if anything, *more*
strongly supported by clean data than it was before, given the independence
approximation's error also shrank from 32-42% to 1-6%.

### Isolated pos-0/1/2 comparison — real live-serving counterpart (vLLM 0.29.0)

Same question as above (does slot 0/1/2's accept rate change when the rest
of the block is 7 wide vs 3 wide), but measured live: two full 24-set
live-serving sweeps (RadixArk DSpark, `num_speculative_tokens=7` and `=3`),
per-position accept counts read from vLLM's
`spec_decode_num_accepted_tokens_per_pos_total` counter.

| position | mean Δ (block=3 − block=7) | direction |
|---|---:|---|
| pos0 | +0.020 | 23/24 sets higher at block=3 |
| pos1 | +0.024 | 21/24 sets higher at block=3 |
| pos2 | +0.027 | 20/24 sets higher at block=3 |

**All three positions are consistently higher at block=3 here, with the
effect size growing slightly deeper into the block** — a third, distinct
pattern from both offline versions (degenerate data: pos0 up, pos2 down;
clean data: all roughly flat-to-slightly-down).

**Important methodological caveat, stated plainly rather than glossed over:
this comparison is less tightly controlled than the offline version.** The
offline analytical method replays the *exact same* fixed target-generated
text through both block sizes — only the draft's processing of that fixed
text differs, isolating one variable. Live serving cannot do this: block=7
and block=3 are two independent live generations, and real accept/reject
outcomes cascade into genuinely different actual output text between the
two runs. So this real delta reflects a mix of (a) genuine block-width
effects and (b) natural content variation between the two live runs — it is
evidence about what happens in real deployment at each block size, but not
as clean a single-variable ablation as the offline table above. Take the
offline table as the controlled ablation and this table as the real-world
confirmation that block=3 does not lose ground on early-slot accuracy in
practice either, by any measure tried.

## Open follow-ups

Roughly in priority order:

1. ~~**Escalate the vLLM spec-decode target-corruption bug.**~~ **Done
   2026-09-18** — resolved by upgrading to vLLM 0.29.0, see the
   [Update section](#update-2026-09-181920-four-issues-found-all-fixed) in the
   Appendix. ~~Remaining follow-up: run a proper
   multi-prompt, multi-benchmark real-throughput sweep on 0.29.0~~ **Also
   done 2026-09-18/19/20** — full 24-set real live-serving sweeps completed
   for **all four checkpoints**: RadixArk/Kimi-K3-DSpark (block=7 and
   block=3: AR, AL, throughput, per-position accept rate), EAGLE3 (AR, AL,
   throughput), Inferact/Kimi-K3-DSpark (AR, AL, throughput), and
   lightseekorg/kimi-k3-dspark (AR, AL, throughput), all in the Update and
   Ablation sections. Getting EAGLE3's sweep right required finding and
   fixing a third bug (Bug 3: checkpoint export shipped wrong aux-layer
   ids for live serving); Inferact's sweep needed no fix, config was
   correct on the first attempt; lightseekorg's checkpoint needed a fourth
   fix (Bug 4: vLLM's DSpark model support is missing a per-tap
   normalization step some training contracts require) — see the Update
   section for all four. Still open: testing under concurrent-request
   batching — issue #50851's comment thread reports a separate, unresolved
   *batched-verify* throughput regression on recent vLLM main, distinct
   from the correctness bug fixed here; all real numbers in this doc are
   single-stream
   (concurrency 1) — worth checking whether 0.29.0 has it.
2. **DSpark aa-lcr 8k/16k.** The one benchmark axis not yet measured for the
   stronger draft, and the most interesting given its yarn rope design —
   directly testable against the EAGLE3 draft's collapse at the same lengths.
3. **RULER V2 1M** (from the model card) — needs either the vLLM fix or
   SGLang, since 1M-token prompts are impractical to capture via
   `extract_hidden_states` (86 GB+ of hidden states per sample at 6 layers).
4. **DSpark's `--enable-adaptive-verification` path** — the confidence-head
   calibration numbers above suggest it should work well; now testable live
   given serving is unblocked (not yet done).
5. If a from-scratch MTP/DSpark draft is trained (tasks already staged, see
   TRAINING.md §7): fix `rope_theta` (use a large base or match the target's
   NoPE design) and include long sequences in training data, per the EAGLE3
   long-context failure analysis in Result 2.
6. Widen the 25-benchmark sweep's `n` (currently 15–25/set) and standardize
   `--max-gen-tokens` (currently 256 vs 512 depending on when the set was
   collected) if these numbers need to support a specific go/no-go decision
   rather than a directional read.

## Artifacts

**Current (clean data, 2026-09-18+) — use these for anything in the Update
section or the rerun Ablation tables:**
- Clean on-policy hidden states + metric JSONs (24 sets):
  `/import/ml-sc-scratch5/chenw/models/kimi-k3-data-clean/<set>/` — each dir
  has `gen_cache/` (cached clean generations), `sample_*.safetensors`
  (extracted hidden states), `dspark_eval_clean.json` (block=7 offline
  metrics), `dspark_eval_clean_block3.json` (block=3 offline metrics).
- Real live-serving sweep results (JSON):
  `/tmp/live_dspark_eval_results.json` (block=7 aggregate),
  `/tmp/live_dspark_eval_results_block7.json` (independent block=7 rerun,
  confirms 0% drift), `/tmp/live_dspark_eval_pos_block7.json` /
  `_block3.json` (per-position, both block sizes),
  `/tmp/live_eagle3_eval_results_fixed.json` (EAGLE3, post-Bug-3-fix, the
  numbers in "Full 24-set live-serving sweep — EAGLE3"),
  `/tmp/live_inferact_eval_results.json` (Inferact, the numbers in "Full
  24-set live-serving sweep — Inferact/Kimi-K3-DSpark"),
  `/tmp/live_lightseekorg_eval_results.json` (lightseekorg, post-Bug-4-fix,
  the numbers in "Full 24-set live-serving sweep — lightseekorg/kimi-k3-dspark").
  `/tmp/live_eagle3_eval_results_BUGGED_wrong_aux_layers.json` is the
  **first** EAGLE3 attempt, run before Bug 3 was found/fixed — kept only as
  evidence for that bug, not a usable measurement.
- Bug 4 fix (not a checkpoint edit — a standalone vLLM model-registry
  override, installed package untouched):
  `k3dspark_fc_norm_plugin.py` (defines `PatchedK3DSparkForCausalLM`,
  adds per-tap `fc_norm` only when a checkpoint's config declares it) and
  `launch_lightseekorg_with_fc_norm.py` (loads the plugin, sets
  `VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=1`, then serves) — both in this
  session's scratchpad.
- Fixed collection pipeline: `scripts/evaluate/kimi_k3_offline_eval/sweep_collect.py`
  (`--phase generate` against a plain server, `--phase extract` against the
  `extract_hidden_states` server — see Bug 1 in the Update section).
- DSpark checkpoint used for all real/clean numbers (RadixArk, local copy,
  arch patched to `Qwen3DSparkModel` for vLLM loading, MD5-verified
  identical to the untouched HF download):
  `/import/ml-sc-scratch5/chenw/models/Kimi-K3-DSpark/`
- Inferact's checkpoint (vLLM-native, native MLA, full 24-set live sweep
  completed — no config fix needed, unlike EAGLE3):
  `/import/ml-sc-scratch5/chenw/models/Inferact-Kimi-K3-DSpark/`
- EAGLE3's checkpoint export, with the Bug 3 fix applied (config.json's
  `eagle_aux_hidden_state_layer_ids` corrected from `[48,68,88]` to
  `[49,69,89]`; original preserved as `config.json.orig_48_68_88`):
  `/import/ml-sc-scratch5/chenw/models/kimi-k3-draft-torchspec/hf-vllm/`
- lightseekorg's checkpoint (vLLM-native, native MLA, requires the Bug 4
  fc_norm patch to serve correctly — see `k3dspark_fc_norm_plugin.py`
  above; unmodified as downloaded, the fix lives entirely in the plugin):
  `/import/ml-sc-scratch5/chenw/models/lightseekorg-kimi-k3-dspark/`

**Historical (pre-2026-09-18, degenerate/buggy data) — kept for the record,
do not use for current numbers:**
- Harness: `scripts/evaluate/kimi_k3_offline_eval/{prep_and_collect,sweep_collect,run_ttt_eval,run_dspark_eval}.py`
- EAGLE3 collected hidden states + metric JSONs:
  `/import/ml-sc-scratch5/chenw/models/kimi-k3-data/{eval_hs_*,sweep_hs_*,bench_hs_*}/`
- DSpark collected hidden states + metric JSONs (original, Bug-1-affected):
  `/import/ml-sc-scratch5/chenw/models/kimi-k3-data/dspark_hs_*/` (25 sets;
  each dir has `dspark_eval_results.json`)
- Throughput: `/import/ml-sc-scratch5/chenw/models/kimi-k3-data/throughput_baseline/`
- Sweep prompt bins: `/import/ml-sc-scratch5/chenw/models/kimi-k3-data/aa_lcr_sweep_kimi/`

## Appendix

Bug history, superseded offline methodology, and the original (later
corrected) per-draft result tables. Everything above this point reflects
the current, corrected state; everything below predates one or more of
the three bugs found during this evaluation, or documents how they were
found.

## Update (2026-09-18/19/20): four issues found, all fixed

Four separate, real defects were affecting either every number in this
doc or blocking a checkpoint from being evaluated at all: two in vLLM
itself (Bug 1, Bug 2), fixed as of vLLM **0.29.0** (released 2026-09-09);
one in the EAGLE3 checkpoint's export (Bug 3, found 2026-09-19 while
running EAGLE3's own live-serving sweep), fixed by correcting the
checkpoint's config; and one missing feature in vLLM's own DSpark model
support (Bug 4, found 2026-09-20 while evaluating a third DSpark
checkpoint), worked around with a small, non-invasive model-registry
patch — see below. Everything under "Why offline" through "Result 3"
below was measured on vLLM **0.28.0**, where Bugs 1 and 2 were both
present.

### Bug 1 — `extract_hidden_states` mode corrupts generation

The offline-eval pipeline generates the target's on-policy answer, then
re-submits the full sequence to capture hidden states. Both steps used the
same server (launched with `extract_hidden_states` speculative_config).
Direct decoding of the "ground truth" generations later in this doc showed
severe repetition-loop degeneration (e.g. an actual captured gsm8k sample:
*"and the fresh duck eggs, and the fresh duck eggs, ..."* repeated ~40+
times; an aime sample repeating *"A: A: A: ..."* ~190 times) — meaning
**every on-policy set in this doc except `chat64`** (which uses pre-existing
human conversations, not on-policy generation) was scored against corrupted
target text.

Root cause: `extract_hidden_states` is implemented via vLLM's
speculative-decoding subsystem internally, which shares the corruption in
Bug 2 below even though it does no real multi-step drafting.

Fix (`scripts/evaluate/kimi_k3_offline_eval/sweep_collect.py`): split into
two explicit phases — `--phase generate` against a **plain** server (no
speculative_config at all), then `--phase extract` against the
`extract_hidden_states` server, reading the cached clean generations. All 24
on-policy sets were regenerated and re-extracted with this fixed pipeline;
**fixing this alone raised measured DSpark accept length by 1.0×–2.1× across
nearly every set** (median ~1.4×) — see the corrected sweep table below.
Verified via a repetition-detector (character n-gram duplication rate,
calibrated against the known-bad sample at 0.937) that all 380 newly
collected samples are clean (max score 0.46, and manual inspection of the
highest-scoring ones showed legitimate structural repetition — code imports,
tool-call syntax — not corruption).

### Bug 2 — live speculative decoding corrupts Kimi K3's output on vLLM 0.28.0

Separately, and more fundamentally: attaching **any** draft to Kimi K3 via
vLLM 0.28.0's real `--speculative-config` (not just `extract_hidden_states`)
corrupted the target's actual output — greedy decoding degenerated into
repetition loops after ~10-12 tokens (or, with some configs, almost
immediately). Confirmed **method- and checkpoint-independent**:

- `RadixArk/Kimi-K3-DSpark` (SGLang-trained, K=7): broken.
- `Inferact/Kimi-K3-DSpark` (vLLM-native, purpose-built for this exact
  serving path, K=7 **and** K=1): broken, ruling out "wide draft chunk" as
  the cause.
- TorchSpec EAGLE3 draft: broken with the same signature.

This is theoretically significant on its own: speculative decoding is
supposed to be **output-preserving** — a bad draft only slows things down
(more rejections → falls back toward target-only speed), it should never
change *what* the target says. Corrupted output at any draft quality means
the bug is in vLLM's accept/reject/verification implementation for this
model, not in draft quality.

**Evidence** — real, filed vLLM issues/PRs matching this exact failure mode,
all merged after our original vLLM 0.28.0 install and before 0.29.0's
2026-09-09 release:

| | title | merged/filed | link |
|---|---|---|---|
| Issue | DSpark speculative decoding broken on H200 nightly | filed | https://github.com/vllm-project/vllm/issues/50851 |
| Issue | Kimi-K3: all requests degenerate to a repeated token after long-context prefill (NaN logits; packed KDA prefill suspected) | filed 2026-08-27 | https://github.com/vllm-project/vllm/issues/51039 |
| PR | [Bugfix][Spec Decode] Fix NaN handling in rejection sampler `tl.argmax` | 2026-08-06 | https://github.com/vllm-project/vllm/pull/50183 |
| PR | [Model][Spec Decode] Tap the pre-norm AttnRes mixture as the Kimi K3 DFlash aux state | 2026-08-14 | https://github.com/vllm-project/vllm/pull/50487 |
| PR | [Perf] Adaptive budget for spec scheduled token, 55%-65% E2E TTFT improvement | 2026-08-11 | https://github.com/vllm-project/vllm/pull/51725 |
| PR | [Spec decode] Support Kimi-K3 DCP with DSpark | 2026-08-17 | https://github.com/vllm-project/vllm/pull/52188 |
| PR | [Bugfix][Spec Decode] Reapply group geometry for FlashAttention metadata (addresses #50851) | 2026-08-24 | https://github.com/vllm-project/vllm/pull/53336 |

Issue #51039's mechanism (NaN logits from the packed-KDA-prefill code path
handling multi-token chunks) is a strong mechanistic match for what real
speculative decoding does every verify step; #50851's routing bugs (aux
hidden states never correctly reaching the draft) independently explain
near-immediate corruption regardless of draft chunk width.

### Confirmation: vLLM 0.29.0 fixes it, with real measured numbers

Upgraded the eval env's vLLM 0.28.0 → 0.29.0 (`pip install vllm==0.29.0`; low
risk — torch stayed at 2.13.0, transformers unaffected). Relaunched live
`--speculative-config` serving with both previously-broken checkpoints, same
prompts that used to degenerate:

| draft | output | real AR (measured) | real AL (measured) | real throughput |
|---|---|---|---|---|
| `RadixArk/Kimi-K3-DSpark` (K=7) | clean, correct | **68.2%** (167/245 accepted) | **5.77** tok/verify-step | 110.3 tok/s |
| `Inferact/Kimi-K3-DSpark` (K=7) | clean, correct | **54.1%** (632/1169 accepted) | **4.78** tok/verify-step | 100-127 tok/s |

These are **real, empirical numbers read directly from vLLM's own
`spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total` Prometheus
counters** after live requests — not analytical estimates, not projections.
Both rows above are single-prompt-set spot-checks; a full 24-set sweep for
each was run later (RadixArk's is above in "Corrected 24-set sweep" and
its own "Full 24-set live-serving sweep"; Inferact's is in "Full 24-set
live-serving sweep — Inferact/Kimi-K3-DSpark" further down) — treat those
as the current numbers, this table as the original discovery-time
confirmation. Baseline (no speculative decoding) on this hardware is ~52 tok/s per a
matching community report on the same 8×B300 config (vLLM issue #50851
comment thread), so this is a genuine **~2× measured speedup**. RadixArk's
higher real acceptance here is consistent with its model card's own
(previously unreproducible) `acc_len` numbers being correct all along — the
gap was never the checkpoint, it was vLLM 0.28.0.

### Corrected 24-set sweep — EAGLE3 vs RadixArk DSpark vs Inferact DSpark

**Note on naming: three different, unrelated checkpoints are all called
"DSpark" in this doc** — `RadixArk/Kimi-K3-DSpark` (SGLang-trained,
full-attention Qwen3-style architecture), `Inferact/Kimi-K3-DSpark`
(vLLM-native, MLA architecture matching the target), and
`lightseekorg/kimi-k3-dspark` (also native MLA, TokenSpeed-trained — see
Bug 4). This table only covers the first two; lightseekorg's checkpoint
has its own "Full 24-set live-serving sweep — lightseekorg/kimi-k3-dspark"
section further down (not merged into this table to keep it readable).
Every column below is labeled **RadixArk DSpark** or **Inferact DSpark**
explicitly — an unqualified "DSpark" elsewhere in this doc almost always
means RadixArk's checkpoint (it was evaluated first and most
extensively), but always check which one a given number is attached to.

Both evaluated on the exact same 380 clean, Bug-1-fixed generations
(24 on-policy sets), so every column below is directly side-by-side
comparable *per set*, though **not from an identical underlying formula**:
RadixArk DSpark's offline AL/AR are exact analytical distribution-overlap
acceptance (`run_dspark_eval.py`, `AR = 1 - TV(draft, target)`, cumulative
product over 7 draft slots — `accept_len` already includes the bonus
token, no separate `+1` needed); EAGLE3's AL/AR are derived from TTT
argmax-agreement (`run_ttt_eval.py --ttt-length 4`, `AL = 1 +
sim_acc_len`, `AR = sim_acc_len / 4`, K=4). **Inferact DSpark has no
offline/analytical columns** — its architecture isn't supported by this
repo's offline eval harness (`run_dspark_eval.py` targets RadixArk's
specific `DSparkDraftModel` class), so live serving is the only
measurement available for it; its AL/AR/tok/s below are all real,
live-measured numbers, not analytical estimates. See
[Metric definitions](#metric-definitions) for the full formulas — treat
this as a convenient side-by-side view, not a claim these are
bit-identical measurements. Fixing Bug 1 alone raised RadixArk's numbers
1.0×-2.1× versus the original (now-removed) measurements; EAGLE3 moved in
the *opposite* direction on `aime` specifically (see below) — see the
Bug 1 section above for that comparison.

| dataset | EAGLE3 AL (K=4) | EAGLE3 AR (K=4) | EAGLE3 tok/s (live) | RadixArk DSpark AL (K=7) | RadixArk DSpark AR (K=7) | RadixArk DSpark tok/s (live) | Inferact DSpark AL (live, K=7) | Inferact DSpark AR (live, K=7) | Inferact DSpark tok/s (live) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 2.42 | 0.355 | 273.4 | 5.97 | 0.847 | 360.8 | 5.55 | 0.650 | 359.9 |
| bfcl | 2.24 | 0.311 | 210.0 | 4.97 | 0.723 | 249.7 | 4.86 | 0.552 | 255.4 |
| mbpp | 2.33 | 0.333 | 234.8 | 4.65 | 0.714 | 300.4 | 4.26 | 0.465 | 291.6 |
| speed-coding | 2.15 | 0.289 | 248.6 | 4.78 | 0.710 | 307.4 | 4.72 | 0.531 | 325.7 |
| livecodebench | 2.05 | 0.262 | 215.3 | 4.46 | 0.666 | 267.7 | 3.67 | 0.382 | 274.1 |
| math500 | 2.03 | 0.258 | 212.5 | 4.44 | 0.667 | 273.1 | 3.82 | 0.402 | 273.8 |
| tool_call | 2.10 | 0.275 | 201.1 | 4.19 | 0.645 | 234.8 | 3.89 | 0.413 | 251.5 |
| speed-rag | 2.06 | 0.264 | 228.6 | 4.17 | 0.636 | 271.0 | 4.09 | 0.441 | 277.9 |
| rag | 2.12 | 0.280 | 220.3 | 4.12 | 0.637 | 261.8 | 4.08 | 0.440 | 270.6 |
| translation | 2.01 | 0.253 | 244.4 | 3.96 | 0.640 | 267.9 | 4.48 | 0.497 | 303.2 |
| swe-bench-pro | 1.92 | 0.230 | 210.3 | 3.83 | 0.594 | 227.1 | 3.80 | 0.400 | 256.3 |
| speed-multilingual | 1.75 | 0.188 | 218.2 | 3.75 | 0.568 | 225.0 | 4.20 | 0.458 | 299.1 |
| summarization | 1.97 | 0.243 | 217.3 | 3.65 | 0.571 | 239.8 | 3.66 | 0.380 | 244.8 |
| speed-writing | 1.94 | 0.234 | 192.7 | 3.49 | 0.554 | 206.7 | 3.20 | 0.314 | 218.2 |
| qa / speed-qa | 2.03 | 0.258 | 192.0 | 3.43 | 0.550 | 221.8 | 3.14 | 0.305 | 217.2 |
| aa-lcr-4k | 1.74 | 0.186 | 148.2 | 3.38 | 0.515 | 165.0 | 3.17 | 0.310 | 188.3 |
| aa-lcr-1k | 1.73 | 0.183 | 155.2 | 3.22 | 0.496 | 164.9 | 3.08 | 0.297 | 188.6 |
| aime | 1.70 | 0.176 | 169.8 | 3.16 | 0.492 | 195.5 | 2.77 | 0.253 | 200.6 |
| swe-rebench | 1.93 | 0.232 | 179.9 | 3.11 | 0.507 | 212.4 | 3.17 | 0.310 | 229.0 |
| mtbench | 2.00 | 0.250 | 184.8 | 3.07 | 0.500 | 204.6 | 2.82 | 0.260 | 202.4 |
| gpqa | 1.67 | 0.166 | 172.4 | 3.01 | 0.463 | 194.7 | 3.03 | 0.289 | 216.4 |
| writing | 1.97 | 0.242 | 177.5 | 2.89 | 0.485 | 189.2 | 2.63 | 0.233 | 186.6 |
| aime26 | 1.65 | 0.162 | 156.8 | 2.88 | 0.461 | 184.2 | 2.66 | 0.238 | 190.0 |

**Throughput columns are all real, live-measured** (from the "Full 24-set
live-serving sweep" tables further down, vLLM 0.29.0, per-request
wall-clock) — not derived from either DSpark checkpoint's offline AL/AR
(where it exists), which has no throughput signal of its own (it's a
teacher-forced/analytical replay, not real generation). Both DSpark
checkpoints' live throughput is consistently higher than EAGLE3's on
every set, tracking their higher AL (more accepted tokens per verify
step, wider K=7 block vs EAGLE3's K=4); Inferact edges out RadixArk on
throughput on most (but not all) sets — see "Full 24-set live-serving
sweep — Inferact/Kimi-K3-DSpark" for the full domain-by-domain breakdown.

Ranges: EAGLE3 AL 1.65–2.42 (median ≈ 2.0, offline), RadixArk DSpark AL
2.88–5.97 (median ≈ 3.7, offline), Inferact DSpark AL 2.63–5.55 (median ≈
3.7, live-only — not directly comparable to the two offline ranges since
it's a different measurement type). Both DSpark checkpoints lead EAGLE3
on every single set here, though by a narrower margin on the hardest
domains (aime, gpqa, aa-lcr) than on easy/structured ones (gsm8k, bfcl,
mbpp). Row order follows RadixArk DSpark's AL, descending. This table
supersedes the "Full 25-benchmark sweep" table further down (RadixArk
DSpark's original pre-fix numbers), which is kept for historical record
with its data table removed.

<details>
<summary>EAGLE3 native TorchSpec field names (<code>sim_acc_len</code>,
<code>avg_acc</code>, <code>acc_0..3</code>), for traceability back to the
raw eval JSON</summary>

| dataset | sim_acc_len | avg_acc | acc_0 / acc_1 / acc_2 / acc_3 |
|---|---:|---:|---|
| gsm8k | 1.420 | 0.699 | 0.490 / 0.816 / 0.772 / 0.719 |
| bfcl | 1.243 | 0.637 | 0.481 / 0.746 / 0.693 / 0.627 |
| mbpp | 1.333 | 0.641 | 0.519 / 0.750 / 0.676 / 0.620 |
| speed-coding | 1.154 | 0.592 | 0.481 / 0.713 / 0.622 / 0.554 |
| livecodebench | 1.050 | 0.576 | 0.445 / 0.696 / 0.613 / 0.548 |
| math500 | 1.031 | 0.589 | 0.428 / 0.699 / 0.643 / 0.587 |
| tool_call | 1.099 | 0.565 | 0.486 / 0.667 / 0.583 / 0.525 |
| speed-rag | 1.057 | 0.539 | 0.488 / 0.645 / 0.543 / 0.480 |
| rag | 1.120 | 0.552 | 0.504 / 0.669 / 0.557 / 0.478 |
| translation | 1.011 | 0.541 | 0.464 / 0.644 / 0.556 / 0.500 |
| swe-bench-pro | 0.920 | 0.506 | 0.445 / 0.608 / 0.519 / 0.453 |
| speed-multilingual | 0.753 | 0.451 | 0.396 / 0.546 / 0.463 / 0.400 |
| summarization | 0.973 | 0.529 | 0.449 / 0.646 / 0.548 / 0.472 |
| speed-writing | 0.936 | 0.510 | 0.447 / 0.622 / 0.520 / 0.452 |
| qa / speed-qa | 1.031 | 0.531 | 0.484 / 0.634 / 0.534 / 0.471 |
| aa-lcr-4k | 0.744 | 0.421 | 0.415 / 0.513 / 0.405 / 0.352 |
| aa-lcr-1k | 0.730 | 0.415 | 0.413 / 0.497 / 0.402 / 0.347 |
| aime | 0.704 | 0.437 | 0.373 / 0.548 / 0.449 / 0.379 |
| swe-rebench | 0.927 | 0.496 | 0.459 / 0.596 / 0.494 / 0.437 |
| mtbench | 0.999 | 0.519 | 0.476 / 0.624 / 0.521 / 0.455 |
| gpqa | 0.666 | 0.412 | 0.369 / 0.516 / 0.414 / 0.348 |
| writing | 0.970 | 0.507 | 0.473 / 0.608 / 0.505 / 0.441 |
| aime26 | 0.649 | 0.399 | 0.371 / 0.491 / 0.395 / 0.337 |

</details>

**EAGLE3 comparison against its original (Bug-1-affected) numbers, same
exact clean prompts, verified via identical `hs_stack` shape (4 layers: 3
aux + 1 final) so this isn't a setup error**: `livecodebench` 1.102→1.050
(0.95×, essentially unchanged), `gpqa` 0.682→0.666 (0.98×, essentially
unchanged), **`aime` 1.212→0.704 (0.58×, a substantial decrease)** — the
opposite direction from DSpark, where every single set increased after the
Bug 1 fix. Best available (not fully proven) hypothesis for aime
specifically: the original corrupted "ground truth" degenerated into a
highly repetitive loop (per Bug 1's description — an aime sample repeating
"A: A: A: ..." ~190 times); once any model falls into that kind of
degenerate pattern, "predict the same token again" becomes trivially easy
to guess correctly for both target and draft, artificially inflating
apparent agreement. The clean, mathematically real reasoning text is
genuinely harder for this draft to predict — a lower but more honest
number. `livecodebench`/`gpqa` barely moved, suggesting their original
degenerate samples were less extreme or less frequent than aime's.

For real, directly-measured (not converted) throughput and AL/AR on both
drafts, see "Full 24-set live-serving sweep" below for DSpark and further
down for EAGLE3 — the tables above are offline/analytical approximations,
not live measurements. (The EAGLE3 live sweep also documents Bug 3, a
config issue found and fixed while running it — its post-fix real AL runs
consistently *higher* than the offline numbers shown here.)

### Full 24-set live-serving sweep — real measurements, vLLM 0.29.0 (the authoritative numbers)

Per your request to "use the correct infra": replayed all 380 already-cached
clean prompts as real HTTP requests against a live vLLM 0.29.0 server with
`RadixArk/Kimi-K3-DSpark` attached via native `--speculative-config`
(K=7, no `extract_hidden_states`, no offline replay). AR/AL are **real**,
read from vLLM's own `spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total`
Prometheus counters (deltas per set); throughput is **real**, measured
per-request wall-clock (`completion_tokens ÷ elapsed`), matching each
prompt's own clean-data generation length so effort is comparable across
sets. This supersedes both the offline analytical table above and the
"Full 25-benchmark sweep" table further down — it is the only table in this
doc measuring the actual deployed behavior rather than an analytical proxy.

| dataset | n | real AR | real AL | offline analytical AL | ratio | real tok/s |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 25 | 0.662 | 5.64 | 5.97 | 0.94 | 360.8 |
| bfcl | 15 | 0.529 | 4.70 | 4.97 | 0.95 | 249.7 |
| speed-coding | 15 | 0.492 | 4.44 | 4.78 | 0.93 | 307.4 |
| mbpp | 15 | 0.484 | 4.39 | 4.65 | 0.94 | 300.4 |
| rag | 15 | 0.418 | 3.93 | 4.12 | 0.95 | 261.8 |
| speed-rag | 15 | 0.415 | 3.91 | 4.17 | 0.94 | 271.0 |
| translation | 15 | 0.410 | 3.87 | 3.96 | 0.98 | 267.9 |
| tool_call | 15 | 0.389 | 3.72 | 4.19 | 0.89 | 234.8 |
| math500 | 15 | 0.376 | 3.63 | 4.44 | 0.82 | 273.1 |
| summarization | 15 | 0.368 | 3.57 | 3.65 | 0.98 | 239.8 |
| livecodebench | 15 | 0.365 | 3.55 | 4.46 | 0.80 | 267.7 |
| swe-bench-pro | 15 | 0.345 | 3.42 | 3.83 | 0.89 | 227.1 |
| qa | 15 | 0.312 | 3.18 | 3.43 | 0.93 | 221.8 |
| speed-qa | 15 | 0.312 | 3.18 | 3.43 | 0.93 | 221.8 |
| speed-multilingual | 15 | 0.305 | 3.14 | 3.75 | 0.84 | 225.0 |
| speed-writing | 15 | 0.288 | 3.02 | 3.49 | 0.86 | 206.7 |
| swe-rebench | 15 | 0.284 | 2.99 | 3.11 | 0.96 | 212.4 |
| aa-lcr-4k | 15 | 0.282 | 2.97 | 3.38 | 0.88 | 165.0 |
| mtbench | 25 | 0.259 | 2.82 | 3.07 | 0.92 | 204.6 |
| aa-lcr-1k | 15 | 0.246 | 2.72 | 3.22 | 0.85 | 164.9 |
| gpqa | 15 | 0.242 | 2.70 | 3.01 | 0.90 | 194.7 |
| writing | 15 | 0.237 | 2.66 | 2.89 | 0.92 | 189.2 |
| aime | 15 | 0.233 | 2.63 | 3.16 | 0.83 | 195.5 |
| aime26 | 15 | 0.223 | 2.56 | 2.88 | 0.89 | 184.2 |

Mean real AL 3.47, mean real throughput 235 tok/s across sets (165-361
tok/s range, driven mostly by generation length/domain, not draft quality —
gsm8k's short, formulaic answers hit the highest tok/s). Pooled across all
9,072 real draft tokens: **AR = 31.6%, AL = 3.21** (pooling differs from the
simple per-set mean because sets have different sample counts/lengths).

**Real AL is consistently a bit lower than the offline analytical
estimate** — ratio 0.80-0.98 across sets, mean ~0.90, no set exceeding its
analytical estimate. This is an expected, legitimate real-vs-analytical gap,
not a bug: the analytical `accept_rate = 1 - TV(draft, target)` is an
idealized expectation under ideal rejection sampling, while real serving
verification can differ subtly (e.g. this launch didn't pin
`draft_sample_method` to `greedy` the way the earlier Inferact spot-check
did, so draft-side sampling may not exactly match target-side greedy
verification at every position). The direction and magnitude are
domain-independent enough (0.80-0.98, no outliers) that the offline
analytical numbers remain a reasonable *relative* ranking across domains,
just a ~10% overestimate of *absolute* real-world AL. **Update:** the
block=7-vs-block=3 ablation further down now also has a real live-serving
counterpart (both throughput and per-position accept rate) — see the "Real
measured throughput" and "Isolated pos-0/1/2 comparison — real live-serving
counterpart" subsections under the Ablation section.

### Bug 3 — EAGLE3 checkpoint export ships un-shifted aux hidden-state layer ids for live serving

A third real, distinct issue, found while running EAGLE3's own version of
the live-serving sweep above (2026-09-18/19). The first attempt produced
numbers far too bad to be real: **AR collapsed to ~1-3%** across all 24
sets (vs. 16-36% offline) and **throughput (68-83 tok/s) fell *below* the
108 tok/s no-draft baseline** — a working draft should never make serving
slower than not drafting at all under correct rejection sampling.

Root cause, confirmed via the live server's own log line and vLLM's
source, not inferred: `Using Eagle3 auxiliary layers from config: (48, 68,
88)`. This is the exact **wrong, un-shifted** aux-layer id convention this
doc's own "Capture convention" section (above) already diagnosed as costing
up to **16× in accuracy** (avg_acc 0.031 vs. 0.492) — just surfacing in a
different code path. The checkpoint's exported
`kimi-k3-draft-torchspec/hf-vllm/config.json` shipped
`eagle_aux_hidden_state_layer_ids: [48, 68, 88]` (TorchSpec's raw,
pre-shift layer numbering) instead of the vLLM-ready `[49, 69, 89]`. Your
own offline capture tool (`scripts/launch_vllm.py`) writes to this
*exact same* config field with no internal shift (confirmed via its source
comment: *"vLLM reuses this same field to configure aux-layer capture for
every downstream algorithm... it is NOT eagle-specific and must keep this
exact key or vLLM will silently fall back to defaults"*) — so this isn't
two different mechanisms disagreeing, it's the same mechanism, fed a
config value that was never shifted for it. This is purely a **checkpoint
export/config bug**, not a training or weights issue — no retraining
needed, only correcting which 3 layers vLLM reads from.

Fix: edited `eagle_aux_hidden_state_layer_ids` in the checkpoint's
`hf-vllm/config.json` (both the nested `eagle_config.` copy and the
top-level copy vLLM actually reads) from `[48, 68, 88]` to `[49, 69, 89]`.
Original backed up as `config.json.orig_48_68_88` in the same directory.
Confirmed via the log after restart: `Using Eagle3 auxiliary layers from
config: (49, 69, 89)`. A gsm8k spot-check (using the correct
already-tokenized `prompt_ids`, matching the sweep's own query method — an
earlier ad hoc raw-text-prompt check was a false alarm caused by querying
`/v1/completions` with an untemplated string, not a real corruption)
produced clean, correct output (right final answer, proper reasoning
trace) with **per-position acceptance rate 0.91/0.87/0.84/0.82, mean
acceptance length 4.44, 86.1% avg draft acceptance** — see the corrected
sweep below for the full 24-set numbers.

### Full 24-set live-serving sweep — EAGLE3, real measurements, vLLM 0.29.0 (corrected aux-layer config, the authoritative numbers)

Same methodology as DSpark's live sweep above (real HTTP requests replaying
the same 380 cached clean prompts, `--speculative-config` eagle3 K=4, AR/AL
read from vLLM's own `spec_decode_num_*_total` counters, throughput
per-request wall-clock) — run twice: once with the Bug-3 config
(preserved as `/tmp/live_eagle3_eval_results_BUGGED_wrong_aux_layers.json`
for the record, **not used below**), and again after the fix, which is
what's reported here.

| dataset | n | real AR | real AL | offline AL (TTT) | ratio | real tok/s |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 25 | 0.729 | 3.92 | 2.42 | 1.62 | 273.4 |
| bfcl | 15 | 0.622 | 3.49 | 2.24 | 1.56 | 210.0 |
| translation | 15 | 0.587 | 3.35 | 2.01 | 1.67 | 244.4 |
| speed-coding | 15 | 0.582 | 3.33 | 2.15 | 1.55 | 248.6 |
| mbpp | 15 | 0.548 | 3.19 | 2.33 | 1.37 | 234.8 |
| speed-rag | 15 | 0.516 | 3.06 | 2.06 | 1.49 | 228.6 |
| rag | 15 | 0.512 | 3.05 | 2.12 | 1.44 | 220.3 |
| swe-bench-pro | 15 | 0.509 | 3.04 | 1.92 | 1.58 | 210.3 |
| summarization | 15 | 0.501 | 3.00 | 1.97 | 1.52 | 217.3 |
| tool_call | 15 | 0.485 | 2.94 | 2.10 | 1.40 | 201.1 |
| speed-multilingual | 15 | 0.483 | 2.93 | 1.75 | 1.67 | 218.2 |
| livecodebench | 15 | 0.448 | 2.79 | 2.05 | 1.36 | 215.3 |
| math500 | 15 | 0.443 | 2.77 | 2.03 | 1.37 | 212.5 |
| speed-writing | 15 | 0.410 | 2.64 | 1.94 | 1.36 | 192.7 |
| qa | 15 | 0.391 | 2.56 | 2.03 | 1.26 | 192.0 |
| speed-qa | 15 | 0.391 | 2.56 | 2.03 | 1.26 | 192.0 |
| swe-rebench | 15 | 0.360 | 2.44 | 1.93 | 1.26 | 179.9 |
| mtbench | 25 | 0.351 | 2.40 | 2.00 | 1.20 | 184.8 |
| aa-lcr-4k | 15 | 0.350 | 2.40 | 1.74 | 1.38 | 148.2 |
| writing | 15 | 0.340 | 2.36 | 1.97 | 1.20 | 177.5 |
| aa-lcr-1k | 15 | 0.325 | 2.30 | 1.73 | 1.33 | 155.2 |
| gpqa | 15 | 0.319 | 2.28 | 1.67 | 1.36 | 172.4 |
| aime | 15 | 0.296 | 2.19 | 1.70 | 1.29 | 169.8 |
| aime26 | 15 | 0.266 | 2.06 | 1.65 | 1.25 | 156.8 |

Mean real AL 2.79, mean real throughput 202 tok/s (148-273 tok/s range).
Pooled across all draft tokens: **AR = 41.0%, AL = 2.64**. All 24 sets beat
the 108 tok/s no-draft baseline (1.4-2.5×), confirming the draft is a real,
working speedup once the config is correct — a sharp contrast with the
Bug-3-affected first attempt, which was slower than no draft at all.

**Real AL consistently *exceeds* the offline TTT estimate — ratio
1.20-1.67, mean 1.41, no set falling short.** This is the opposite
direction from DSpark (whose real AL ran *below* its offline estimate,
ratio 0.80-0.98) and a much larger gap. Best explanation, grounded in this
doc's own [Metric definitions](#metric-definitions): EAGLE3's offline
`sim_acc_len` is a crude **exact-argmax-match** proxy (a token only "counts"
if it's literally the single most-likely token, computed independently
per TTT step) — TorchSpec's own "sim" prefix flags it as an approximation.
Real live serving, by contrast, verifies with genuine **probabilistic
rejection sampling** (a token is accepted if it's consistent with the
target's full probability distribution, not just its single argmax) — a
strictly more forgiving criterion. DSpark's offline metric is already a
distributional-overlap calculation (`1 - TV(draft, target)`), much closer
in spirit to real rejection sampling, so its real-vs-offline gap is small
and driven by sampling-method details rather than a fundamental proxy
mismatch. This is a plausible, not fully proven, explanation — but it is
directionally consistent with every single set (no exceptions) and
qualitatively matches the general chat live spot-check corroboration above.

### Full 24-set live-serving sweep — Inferact/Kimi-K3-DSpark, real measurements, vLLM 0.29.0

Third checkpoint, same methodology as the other two live sweeps (real HTTP
requests replaying the same 380 cached clean prompts, AR/AL read from
vLLM's own `spec_decode_num_*_total` counters, throughput per-request
wall-clock). Architecturally distinct from RadixArk's DSpark — native MLA
matching the target model, `attention_backend: FLASHINFER_MLA`,
`draft_sample_method: greedy`, K=7, aux/target layer ids `[2, 23, 47, 71,
89]` (already vLLM-ready in this checkpoint's config — confirmed via a
clean gsm8k spot-check, no Bug-3-style shift issue found here). This
checkpoint isn't supported by this repo's offline eval harness, so there's
no offline/analytical column to compare against — live serving is the only
number available for it.

| dataset | n | real AR | real AL | RadixArk real AL | ratio | real tok/s |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 25 | 0.650 | 5.55 | 5.64 | 0.98 | 359.9 |
| bfcl | 15 | 0.552 | 4.86 | 4.70 | 1.03 | 255.4 |
| speed-coding | 15 | 0.531 | 4.72 | 4.44 | 1.06 | 325.7 |
| translation | 15 | 0.497 | 4.48 | 3.87 | 1.16 | 303.2 |
| mbpp | 15 | 0.465 | 4.26 | 4.39 | 0.97 | 291.6 |
| speed-multilingual | 15 | 0.458 | 4.20 | 3.14 | 1.34 | 299.1 |
| speed-rag | 15 | 0.441 | 4.09 | 3.91 | 1.05 | 277.9 |
| rag | 15 | 0.440 | 4.08 | 3.93 | 1.04 | 270.6 |
| tool_call | 15 | 0.413 | 3.89 | 3.72 | 1.05 | 251.5 |
| math500 | 15 | 0.402 | 3.82 | 3.63 | 1.05 | 273.8 |
| swe-bench-pro | 15 | 0.400 | 3.80 | 3.42 | 1.11 | 256.3 |
| livecodebench | 15 | 0.382 | 3.67 | 3.55 | 1.03 | 274.1 |
| summarization | 15 | 0.380 | 3.66 | 3.57 | 1.02 | 244.8 |
| speed-writing | 15 | 0.314 | 3.20 | 3.02 | 1.06 | 218.2 |
| aa-lcr-4k | 15 | 0.310 | 3.17 | 2.97 | 1.07 | 188.3 |
| swe-rebench | 15 | 0.310 | 3.17 | 2.99 | 1.06 | 229.0 |
| qa | 15 | 0.305 | 3.14 | 3.18 | 0.99 | 217.2 |
| speed-qa | 15 | 0.305 | 3.14 | 3.18 | 0.99 | 217.2 |
| aa-lcr-1k | 15 | 0.297 | 3.08 | 2.72 | 1.13 | 188.6 |
| gpqa | 15 | 0.289 | 3.03 | 2.70 | 1.12 | 216.4 |
| mtbench | 25 | 0.260 | 2.82 | 2.82 | 1.00 | 202.4 |
| aime | 15 | 0.253 | 2.77 | 2.63 | 1.05 | 200.6 |
| aime26 | 15 | 0.238 | 2.66 | 2.56 | 1.04 | 190.0 |
| writing | 15 | 0.233 | 2.63 | 2.66 | 0.99 | 186.6 |

Mean real AL 3.66, mean real throughput 247 tok/s (187-360 tok/s range).
Pooled across all draft tokens: **AR = 33.9%, AL = 3.37**. All 24 sets beat
the 108 tok/s no-draft baseline (1.7-3.3×).

**Beats RadixArk's real numbers on 18 of 24 sets, ties on 1 (mtbench),
trails narrowly (within 3%) on the other 5** (gsm8k, mbpp, qa, speed-qa,
writing) — no domain shows a large gap in either direction (ratio range
0.97-1.34). The biggest edges for Inferact are on `speed-multilingual`
(1.34×), `translation` (1.16×), and `aa-lcr-1k`/`gpqa` (~1.12-1.13×);
these happen to be domains where RadixArk itself is comparatively weaker,
suggesting Inferact's native-MLA architecture may generalize a bit better
off the highest-acceptance, most in-distribution domains (gsm8k/mbpp,
where RadixArk keeps a slight edge) — a real, not fully explained pattern,
worth further investigation if choosing between the two checkpoints
matters for a specific workload mix.

### Bug 4 — vLLM's K3DSparkModel is missing per-tap `fc_norm`, required by some DSpark checkpoints

A fourth real, distinct issue, found evaluating a third DSpark checkpoint
(`lightseekorg/kimi-k3-dspark`, 2026-09-19/20). Architecturally similar to
Inferact's checkpoint (same `K3DSparkModel`/native-MLA class), but this
checkpoint's own README states two hard requirements under "Important
serving notes": (1) the AttnRes capture stream (this repo already
supports this via `VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=1`, just needed to be
set explicitly — not a bug, just an easy-to-miss flag), and (2) **"each
collected hidden-state tap must pass through its corresponding `fc_norm`
before the five taps are concatenated and passed to the linear context
projection. Skipping this step is incompatible with the training
contract."**

Verified requirement (2) is a real, structural gap in the installed
vLLM's `K3DSparkModel` (`vllm/models/kimi_k3/nvidia/dspark_mla.py`), not
a config issue:
- The checkpoint's actual weights contain 5 separate normalization
  tensors — `fc_norm.0.weight` through `fc_norm.4.weight` (confirmed via
  direct safetensors inspection).
- vLLM's `combine_hidden_states` only implements
  `context_norm(context_proj(concatenated_taps))` — a single norm
  applied *after* concatenation and projection. There is no code path
  that applies 5 independent per-tap norms *before* concatenation, which
  is what the checkpoint's own README says the training contract
  requires. Confirmed via full source read of the model file — this
  isn't a missed flag, the feature doesn't exist in this vLLM build.
- The checkpoint's authors' reference serving stack is **TokenSpeed**,
  not vLLM — TokenSpeed apparently already implements this correctly
  (`TOKENSPEED_DFLASH_AUX_STREAM=attn_res`, a TokenSpeed-specific
  variable unrelated to vLLM's own env var of a similar name). TokenSpeed
  is not installed in this environment.

Rather than declare this checkpoint unevaluable, patched the gap without
modifying the installed vLLM package (avoids touching shared/other-users'
environment): a standalone script
(`k3dspark_fc_norm_plugin.py`) defines a `PatchedK3DSparkModel`/
`PatchedK3DSparkForCausalLM` subclass that adds the 5 per-tap `RMSNorm`
layers and applies them before concatenation — **only when a checkpoint's
config sets `fc_norm: true`** (Inferact's config doesn't, so its behavior
is provably unaffected — it falls straight through to the original,
unmodified code path) — and registers it via `vllm.ModelRegistry
.register_model()` to override the `K3DSparkModel` architecture name for
this process only. Fully reversible (the installed package is untouched;
confirmed via diff against a pre-edit backup after an earlier blocked
in-place edit attempt).

Verified the fix, not just assumed it: three independent spot-checks
(gsm8k/math, aime/reasoning, mtbench/creative-writing) all produced
clean, coherent, correct output — no repetition — with healthy,
naturally-decaying per-position acceptance rates (33.7%-57.5% avg draft
acceptance depending on domain difficulty). Then, after the full 24-set
sweep, cross-checked against the checkpoint's own self-reported
benchmark numbers in its README (measured independently via TokenSpeed,
not something under this session's control) — see the sweep section
below for the comparison; the close match (no systematic one-directional
gap, ratios scattered 0.92-1.16×) is strong evidence the patch is
correct, not just "doesn't crash."

### Full 24-set live-serving sweep — lightseekorg/kimi-k3-dspark, real measurements, vLLM 0.29.0 (custom fc_norm patch)

Fourth checkpoint (third DSpark variant), same methodology as the other
live sweeps. `attention_backend: FLASHINFER_MLA`, `draft_sample_method:
greedy`, K=7, aux/target layer ids `[7, 31, 47, 63, 87]`,
`VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=1` set, `PatchedK3DSparkForCausalLM`
registered (see Bug 4). No offline/analytical column here either — same
reason as Inferact, this architecture isn't supported by this repo's
offline harness.

| dataset | n | real AR | real AL | real tok/s |
|---|---:|---:|---:|---:|
| gsm8k | 25 | 0.617 | 5.32 | 345.8 |
| translation | 15 | 0.512 | 4.58 | 308.4 |
| bfcl | 15 | 0.501 | 4.51 | 245.2 |
| speed-coding | 15 | 0.491 | 4.44 | 310.8 |
| mbpp | 15 | 0.469 | 4.29 | 292.7 |
| rag | 15 | 0.429 | 4.00 | 263.5 |
| speed-multilingual | 15 | 0.427 | 3.99 | 290.4 |
| tool_call | 15 | 0.417 | 3.92 | 254.2 |
| speed-rag | 15 | 0.404 | 3.83 | 263.9 |
| swe-bench-pro | 15 | 0.390 | 3.73 | 250.4 |
| summarization | 15 | 0.379 | 3.65 | 244.2 |
| math500 | 15 | 0.347 | 3.43 | 257.0 |
| livecodebench | 15 | 0.340 | 3.38 | 253.1 |
| swe-rebench | 15 | 0.303 | 3.12 | 229.7 |
| speed-writing | 15 | 0.303 | 3.12 | 212.9 |
| qa | 15 | 0.299 | 3.09 | 214.8 |
| speed-qa | 15 | 0.299 | 3.09 | 214.9 |
| aa-lcr-4k | 15 | 0.276 | 2.93 | 176.5 |
| aa-lcr-1k | 15 | 0.264 | 2.85 | 175.2 |
| mtbench | 25 | 0.256 | 2.79 | 200.2 |
| gpqa | 15 | 0.240 | 2.68 | 188.9 |
| writing | 15 | 0.232 | 2.62 | 186.2 |
| aime | 15 | 0.211 | 2.47 | 180.8 |
| aime26 | 15 | 0.207 | 2.45 | 175.8 |

Mean real AL 3.60, mean real throughput 244 tok/s. All 24 sets beat the
108 tok/s no-draft baseline.

**Cross-check against the checkpoint's own README benchmark table**
(measured independently via TokenSpeed — a genuine external validation,
not a self-consistency check): the README reports `acc_len` (= AL) for 9
of these domains under similar conditions (greedy, K=7, thinking mode).

| dataset | this measurement (AL) | README (AL) | ratio |
|---|---:|---:|---:|
| gsm8k | 5.32 | 5.464 | 0.97 |
| speed-coding | 4.44 | 4.299 | 1.03 |
| speed-rag | 3.83 | 3.533 | 1.08 |
| speed-multilingual | 3.99 | 3.452 | 1.16 |
| math500 | 3.43 | 3.254 | 1.05 |
| speed-qa | 3.09 | 3.113 | 0.99 |
| mtbench | 2.79 | 3.047 | 0.92 |
| speed-writing | 3.12 | 2.722 | 1.15 |
| aime | 2.47 | 2.572 | 0.96 |

No systematic one-directional bias (ratios scattered both above and
below 1.0, range 0.92-1.16) — the kind of pattern expected from
genuinely-different eval methodology (different exact prompt sets/sample
sizes, different serving stack) on a **correctly implemented** model, not
from a structurally broken normalization step (which would far more
likely produce a large, one-directional gap, as Bug 3's near-total
collapse did). Not a proof the patch is bit-identical to TokenSpeed's
implementation, but strong evidence it's substantively correct rather
than merely "doesn't crash."

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

## Why offline

**Historical — resolved 2026-09-18, see the Update section above.** vLLM
0.28 (this internal build) corrupted the *target's own* output stream when
eagle3 speculative decoding is enabled: token-loop degeneration after ~10–12
tokens (raw + chat, greedy + sampled), acceptance ~0. Target-only serving is
fully healthy, and `--enforce-eager` does not help, so this is spec-decode
state corruption (onset matches `attn_res_block_size = 12`; prime suspects are
the AttnRes bank commits / KDA recurrent-state handling of rejected
speculative tokens). Serving-side acceptance/throughput with spec decode was
therefore unmeasurable at the time this section was written. Confirmed as a
real upstream vLLM bug (issues #50851, #51039) fixed in vLLM 0.29.0 — real
measured throughput now available in the Update section. Details: TRAINING.md §7.

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
| AL (`= 1 + sim_acc_len`) | **1.936** | 2.436 |
| AR (`= sim_acc_len / 4`) | **0.234** | 0.359 |
| avg_acc (native) | 0.4921 | 0.594 |
| sim_acc_len (native) | 0.9364 | 1.436 |
| acc_0 / acc_1 / acc_2 / acc_3 | 0.4800 / 0.5594 / 0.4843 / 0.4446 | — |
| Cross-Entropy Loss (`vloss_0..3`) | 2.908 / 2.712 / 3.148 / 3.405 | avg_loss 2.167 (position-weighted) |

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

| prompt tokens | n | AL | AR | avg_acc (native) | sim_acc_len (native) | acc_0 / acc_1 / acc_2 / acc_3 |
|---|---|---:|---:|---|---|---|
| 1 024 | 20 | **2.135** | 0.284 | 0.5020 | 1.135 | 0.573 / 0.619 / 0.417 / 0.398 |
| 2 048 | 20 | 1.803 | 0.201 | 0.4546 | 0.803 | 0.437 / 0.515 / 0.441 / 0.426 |
| 4 096 | 20 | 2.063 | 0.266 | 0.4851 | 1.063 | 0.550 / 0.593 / 0.413 / 0.384 |
| 8 192 | 20 | 1.553 | 0.138 | 0.3778 | 0.553 | 0.327 / 0.457 / 0.377 / 0.350 |
| 16 384 | 20 | **1.156** | 0.039 | 0.0806 | 0.156 | 0.136 / 0.140 / 0.023 / 0.023 |

(AL = 1 + sim_acc_len, AR = sim_acc_len / 4, same conversion used
throughout this doc. `vloss_0..3` — Cross-Entropy Loss per TTT step —
omitted from this table for space; see the raw eval JSONs in `Artifacts`.)

Roughly flat through 4k (bin-to-bin variance from 20 paired questions), clear
degradation at 8k, **collapse at 16k**. Likely causes, both actionable for the
next draft: (1) draft `rope_theta = 10 000` — a short-context rope base (the
target's own MLA layers are NoPE, so this was purely a draft-side choice);
(2) training data dominated by short sequences.

## Result 3 — benchmark slice + throughput

**On-policy avg_acc/sim_acc_len numbers originally shown here (aime,
livecodebench, gpqa) were measured on generation data later found corrupted
by Bug 1 — removed rather than kept as known-wrong.** See the Update
section's "Corrected 24-set sweep — EAGLE3 vs RadixArk DSpark vs Inferact
DSpark" for current values (aime
sim_acc_len 0.704, livecodebench 1.050, gpqa 0.666 — note aime dropped
substantially from the original 1.212, the opposite direction from DSpark's
corrections; see that section for the likely explanation). `general chat`
(Result 1, chat64) is unaffected by Bug 1 and remains valid as originally
reported: avg_acc 0.4921, sim_acc_len 0.9364.

Baseline decode throughput (unaffected by either bug — plain target-only
serving, no draft involved) measured separately (`run_vllm_eval.py`, 8
prompts × ≤512 tokens, temperature 0, single-stream): **108.0 tok/s** on
aime/livecodebench/gpqa (decode-bound at these context lengths); e2e
97.2–105.4 tok/s; mean TTFT 0.088–0.538 s. Raw JSONs:
`/import/ml-sc-scratch5/chenw/models/kimi-k3-data/throughput_baseline/`.

### Reproduction check against the model card

**Numbers originally shown here were measured on generation data later
found corrupted by Bug 1 — removed rather than kept as known-wrong.** Using
the corrected clean-data values instead: GSM8K AL 5.97 vs. the card's
5.4176 (ratio 1.10 — our corrected measurement now slightly *exceeds* the
card's own number, a large swing from the original 0.73 ratio). The card
reports `acc_len` from live SGLang serving; our clean-data AL is still
offline/analytical, so this remains a load-correctness sanity check, not an
apples-to-apples repro — but it's a much closer match now, and the *real*
live-serving numbers in the Update section (68.2% AR, AL 5.77 on RadixArk's
own checkpoint) are the actual apples-to-apples confirmation that the card's
reported numbers were correct all along. Structural load checks (unaffected
by either bug): `fc.weight` is exactly 5×7168 (5 aux layers, matches the
card), and all 62 tensors loaded into the repo's `DSparkDraftModel` with
zero missing/unexpected keys — the draft was always loaded correctly; the
gap was purely the two measurement bugs, not a loading bug.

### Full 25-benchmark sweep

**Superseded 2026-09-18** — this table was measured on generation data later
found to be corrupted by the `extract_hidden_states` generation bug (see the
[Update section](#update-2026-09-181920-four-issues-found-all-fixed) in the
Appendix). Kept for historical record; use the "Corrected 24-set sweep —
EAGLE3 vs DSpark" table in the Update section for current numbers. The
`AL = 1 + accept_len` convention described below also has the separate
double-counting bug described in the Update section — `accept_len` already
includes the bonus token.

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

**The data table originally here has been removed** (measured on generation
data later found corrupted by Bug 1 — see the Update section — kept as a
known-wrong table was worse than no table). Metric definitions (AL, AR,
full_acc, conf_err) and methodology notes above still apply to the
corrected replacement: see the Update section's "Corrected 24-set sweep —
EAGLE3 vs RadixArk DSpark vs Inferact DSpark" for current AL/AR numbers
(that table dropped its `full_acc` column for cross-draft readability —
see "Per-slot accuracy (DSpark)" further down for the current pooled
`full_acc` value per domain), and "Full 24-set live-serving sweep" for real
measured AR/AL/throughput. Range on clean data:
**AL 2.88–5.97** across 24 domains (median ≈ 3.7) — see the Update section
table for the full per-domain breakdown and ordering, which shifted
somewhat from the original (gsm8k is now the top domain, not bfcl).

### Per-slot accuracy (DSpark)

`pos-k` is `full_acc` broken out per draft slot (slot 0 = immediately
after the anchor, slot 6 = the last/deepest draft position); the rightmost
column is the pooled value across all 7 slots (moved here after the
"Corrected 24-set sweep" table dropped its `full_acc` column for
cross-draft readability).

**Rerun 2026-09-18 on clean data** (original table removed — measured on
generation data later found corrupted by Bug 1). `chat64` not included
(unaffected by Bug 1, but wasn't part of this particular rerun batch).

| dataset | pos-0 | pos-1 | pos-2 | pos-3 | pos-4 | pos-5 | pos-6 | full_acc (pooled) |
|---|---|---|---|---|---|---|---|---:|
| gsm8k | 0.94 | 0.91 | 0.88 | 0.86 | 0.83 | 0.79 | 0.76 | 0.855 |
| bfcl | 0.90 | 0.84 | 0.78 | 0.73 | 0.69 | 0.63 | 0.58 | 0.734 |
| speed-coding | 0.90 | 0.84 | 0.78 | 0.72 | 0.67 | 0.62 | 0.56 | 0.727 |
| mbpp | 0.89 | 0.83 | 0.78 | 0.72 | 0.67 | 0.63 | 0.58 | 0.729 |
| livecodebench | 0.87 | 0.79 | 0.73 | 0.67 | 0.61 | 0.57 | 0.52 | 0.680 |
| rag | 0.87 | 0.79 | 0.71 | 0.65 | 0.58 | 0.53 | 0.48 | 0.658 |
| speed-rag | 0.87 | 0.79 | 0.71 | 0.65 | 0.59 | 0.52 | 0.47 | 0.658 |
| tool_call | 0.86 | 0.79 | 0.72 | 0.65 | 0.59 | 0.54 | 0.49 | 0.664 |
| translation | 0.85 | 0.78 | 0.72 | 0.66 | 0.61 | 0.55 | 0.50 | 0.667 |
| summarization | 0.84 | 0.75 | 0.65 | 0.58 | 0.51 | 0.45 | 0.41 | 0.598 |
| math500 | 0.83 | 0.77 | 0.71 | 0.67 | 0.62 | 0.57 | 0.53 | 0.672 |
| swe-bench-pro | 0.83 | 0.75 | 0.67 | 0.61 | 0.54 | 0.48 | 0.43 | 0.616 |
| qa / speed-qa | 0.82 | 0.71 | 0.62 | 0.54 | 0.49 | 0.44 | 0.40 | 0.573 |
| speed-writing | 0.81 | 0.71 | 0.62 | 0.55 | 0.50 | 0.45 | 0.41 | 0.579 |
| speed-multilingual | 0.80 | 0.72 | 0.64 | 0.58 | 0.52 | 0.47 | 0.42 | 0.591 |
| aa-lcr-4k | 0.76 | 0.66 | 0.57 | 0.50 | 0.43 | 0.39 | 0.34 | 0.523 |
| swe-rebench | 0.77 | 0.67 | 0.58 | 0.50 | 0.45 | 0.40 | 0.35 | 0.530 |
| aime | 0.75 | 0.64 | 0.54 | 0.47 | 0.41 | 0.36 | 0.32 | 0.497 |
| gpqa | 0.75 | 0.63 | 0.52 | 0.44 | 0.38 | 0.33 | 0.29 | 0.475 |
| mtbench | 0.75 | 0.64 | 0.55 | 0.49 | 0.45 | 0.41 | 0.37 | 0.523 |
| aa-lcr-1k | 0.74 | 0.64 | 0.55 | 0.48 | 0.42 | 0.36 | 0.34 | 0.504 |
| writing | 0.73 | 0.62 | 0.53 | 0.47 | 0.43 | 0.39 | 0.36 | 0.506 |
| aime26 | 0.71 | 0.59 | 0.50 | 0.42 | 0.37 | 0.33 | 0.30 | 0.459 |

Two observations beyond AR/AL (in the "Corrected 24-set sweep" table) and
pooled `full_acc` (in the table above) — `conf_err` isn't tabulated
per-domain anywhere in this doc, only the range quoted below:

- **Decay shape is consistent across domains**: pos-0 is always highest,
  decaying *strictly* monotonically to pos-6 in every single one of the 24
  sets (no exceptions on clean data — the one exception reported in the
  original degenerate-data table, `summarization`, turned out to be a
  corruption artifact, not a real signal). The *slope* still varies by
  domain — gsm8k decays gentlest (94.3%→76.4%, a 19% relative drop) while
  gpqa decays steepest (74.6%→29.0%, a 61% drop). This means the Markov
  head's benefit (predicting later tokens conditioned on earlier draft
  tokens, not just the anchor) is domain-dependent — it holds up longer on
  structured/short-answer domains than on open-ended or unfamiliar ones.
- **`conf_err` ranges 0.11–0.24 across domains** (wider than the original
  degenerate-data table's reported 0.20–0.25 band), best-calibrated on
  gsm8k (0.113) and worst on qa (0.241, not gpqa as the original table
  claimed). Still no strong correlation between calibration quality and raw
  acceptance quality — gsm8k is both the best-accepted and best-calibrated
  domain, but qa/gpqa (weaker acceptance domains) don't calibrate uniformly
  worse than each other, so calibration isn't simply "worse when the draft
  is worse." Adaptive-verification budget sizing
  (`--enable-adaptive-verification` in vLLM's dspark path) is now testable
  live given serving is unblocked — worth verifying in practice.

