# DSpark sliding-window attention ablation (RadixArk/Kimi-K3-DSpark)

Can a DSpark draft checkpoint trained with full attention be forced into
sliding-window attention at evaluation/serving time, with **no
retraining**, and how much quality is lost? Motivated by
`RedHatAI/Kimi-K3-speculator.dspark` and `lightseekorg/kimi-k3-dspark`
(both trained natively with sliding-window layers), which raised the
question of whether RadixArk's full-attention checkpoint could be
retrofitted the same way as a serving-time knob.

This doc covers only the ablation itself. For the full DSpark/EAGLE3
architecture background, checkpoint comparisons, and the four bugs found
evaluating the five checkpoints, see
[`kimi_k3_draft_eval.md`](kimi_k3_draft_eval.md).

## Background: what "sliding window" restricts here

RadixArk's draft is a 5-layer Qwen3-style decoder. Each layer's attention
has two independent components, combined via `or_masks` in
`create_anchor_block_mask_mod` (`src/speculators/models/dflash/attention.py`):

- **`base_prefix_mod`** — governs attention from the draft's synthetic
  query blocks to the **context KV** (the target model's captured hidden
  states, 5-tap-concatenated, projected through `fc`/`hidden_norm`, then
  K/V-projected per draft layer). This is the *only* place `sliding_window`
  applies:
  ```python
  in_window = (
      (kv_base_pos >= q_anchor - sliding_window)
      if sliding_window is not None else True
  )
  ```
- **`same_block_mod`** — governs attention among the draft's own block
  tokens (1 real anchor-token embedding + `block_size-1` mask-token
  placeholders per block). Bidirectional or causal depending on a
  separate flag:
  ```python
  non_causal = sliding_window is None or sliding_window_non_causal
  ...
  if not non_causal:
      same = same & (kv_idx <= q_idx + total_seq_len)  # forces causal
  ```

The important, easy-to-miss detail: `sliding_window is not None` on its
own **also** flips `non_causal` to `False` unless `sliding_window_non_causal`
is separately set to `True`. These are two unrelated behaviors (long-range
context restriction vs. within-block causality) that share one boolean
expression in this code. Missing this coupling was the source of a real
confound in this ablation's first run (below).

Per-layer routing (which of the 5 layers use the sliding-window mask vs.
full attention) is controlled by `layer_types` (`"full_attention"` vs.
`"sliding_attention"`, one entry per layer) and `sliding_window` (window
size), both read from `config.transformer_layer_config` in
`dflash/core.py`. This is the exact mechanism `RedHatAI`'s and
`lightseekorg`'s checkpoints use natively — RadixArk's checkpoint simply
has `layer_types: ["full_attention"]*5` and `sliding_window: null`.

## Experimental setup

**Checkpoint**: a config-only variant of RadixArk's checkpoint —
`/import/ml-sc-scratch5/chenw/models/Kimi-K3-DSpark-sliding2048/`. The
`model.safetensors` there is a **symlink** to the original checkpoint
(zero weight changes, zero retraining). Only 3 config fields differ from
the original:

| field | original | modified |
|---|---|---|
| `layer_types` | `["full_attention"]×5` | `["sliding_attention"]×5` |
| `sliding_window` | `null` | `2048` |
| `use_sliding_window` | `false` | `true` |

**Harness**: this repo's own offline eval (`run_dspark_eval.py`), replaying
the exact same cached clean hidden-state data already used throughout
`kimi_k3_draft_eval.md` — no new generation/collection needed.

**Sets tested (quick pass, not the full 24-set sweep)**: `gsm8k` (short
prompts, sliding window should barely matter), `aime` (moderate-length
competition math), `aa-lcr-4k` (long-context, ~4k-token prompts — where a
2048-token window should bite hardest). Chosen to span short/medium/long
context, not run at full scale yet.

## Bugs found and fixed in `run_dspark_eval.py` while setting this up

The eval script's `build_model()` hand-constructs a `transformers.Qwen3Config`
from the checkpoint's raw JSON, explicitly listing which fields to copy
over — it does not pass the whole dict through. Two fields relevant to
this experiment were missing from that list, both **silent no-ops** (no
error, just ignored):

1. **`sliding_window` was never passed to `Qwen3Config(...)`.** HF's
   `Qwen3Config` additionally requires `use_sliding_window=True` to be
   passed as a constructor argument for `sliding_window` to be retained at
   all — otherwise it silently resets to `None` regardless of what the
   raw JSON says. Without both fixes, the entire experiment would have
   silently run as full attention with no error or warning.
2. **`sliding_window_non_causal` was never passed to
   `DSparkDraftModel.from_training_args(...)`**, so it always fell back to
   that function's own internal default (`False`) — which, combined with
   `sliding_window` being non-`None`, silently made same-block attention
   causal instead of bidirectional (see the coupling above). This was not
   caught before the first run (below).

Both fixes are additive, backward-compatible one-liners:
```python
sliding_window=rc.get("sliding_window"),
use_sliding_window=rc.get("use_sliding_window", False),
...
sliding_window_non_causal=rc.get("sliding_window_non_causal", True),
```
Verified the unmodified baseline config (`sliding_window: null`,
`use_sliding_window: false`, no `sliding_window_non_causal` key) produces
identical behavior before and after these edits — `sliding_window=None`,
`sliding_window_non_causal` value doesn't matter here since
`non_causal = (sliding_window is None) or ... = True` regardless. Zero
regression risk for every other set/checkpoint already evaluated with
this script.

**A related, separate observation not yet verified end-to-end**: vLLM's
own live-serving conversion registry (`update_dflash` in
`vllm/transformers_utils/configs/speculators/algos.py`) explicitly
handles this same field with a *default of `True`*:
```python
pre_trained_config["dflash_config"]["causal"] = not config_dict.get(
    "sliding_window_non_causal", True
)
```
But the **DSpark** conversion function (`update_dspark`, the one actually
used by every checkpoint in this doc) does not reference
`sliding_window_non_causal` anywhere at all. Whether this means live
DSpark serving via vLLM has a different (or missing) default for this
behavior compared to this repo's own offline harness is a real, open
question — flagged here as a follow-up, not something this ablation
tests or claims to have resolved. This ablation only concerns the offline
harness; no live-serving path is exercised.

## Results

### Verification: no information leaks outside the window

Before trusting the near-zero degradation reported below, verified
directly — twice — that the window genuinely excludes out-of-window
information, rather than being a silent no-op.

**(1) Direct mask inspection.** Called `create_anchor_block_mask_mod`
directly with a late anchor (position 4000, window 2048) and probed the
boolean predicate at specific key positions:

| kv position | expected | actual |
|---|---|---|
| 1951 (1 before boundary) | excluded | `attends=False` |
| 1952 (exactly at boundary, `anchor - window`) | included | `attends=True` |
| 2000–3999 (inside window) | included | `attends=True` |
| 4000 (the anchor's own position) | excluded (`before_anchor` is strict `<`) | `attends=False` |
| 4001 (after the anchor) | excluded | `attends=False` |

Exact boundary, no off-by-one, correctly excludes both the far past and
the future.

**(2) End-to-end perturbation test — the stronger proof.** Mask inspection
alone only checks the predicate in isolation, not the real compiled
computation. So: loaded a real 4526-token `aa-lcr-4k` sample, ran the
actual model (hooked `_backbone_forward` to capture raw logits while still
calling through `model.forward()`, so the real `@conditional_torch_compile`
+ `flex_attention` path is exercised, not a bypassed uncompiled fallback).
Took the latest anchor (position 4518, window boundary at 2470) and
compared its output logits across three runs on the identical input
otherwise:

| perturbation | region corrupted (replaced with random noise) | max abs logit diff |
|---|---|---:|
| none (baseline) | — | — |
| **outside the window** | positions `[0, 2370)` — before the boundary | **`0.0`** |
| **inside the window** (control) | positions `[2570, 4468)` — within the boundary | `11.125` |

Corrupting information strictly outside the window changes **nothing at
all** in the output — not "small," exactly zero, bit-for-bit identical.
Corrupting the same amount of information inside the window changes the
output substantially, confirming the model is genuinely sensitive to its
inputs in general (ruling out the trivial possibility that the test
itself doesn't detect anything). Together, these two checks confirm the
sliding window is real, mathematically enforced, and has zero leakage —
the near-zero degradation reported below reflects genuine robustness of
this checkpoint, not a broken mask silently doing nothing.

### Second run — clean, isolates the context-window effect only

Ran after the `sliding_window_non_causal=True` fix. Same-block attention
is now bidirectional in both conditions, verified via
`model.sliding_window_non_causal == True` for both the baseline
(`sliding_window=None`, so `non_causal` is `True` regardless) and the
sliding-2048 config (`non_causal = False or True = True`) before running.

| set | metric | baseline | confounded (both changes) | clean (window only) |
|---|---|---:|---:|---:|
| gsm8k | accept_len | 5.9666 | 5.8281 (0.977) | 5.9671 (**1.000**) |
| gsm8k | accept_rate | 0.8470 | 0.8374 (0.989) | 0.8470 (**1.000**) |
| gsm8k | full_acc | 0.8545 | 0.8465 (0.991) | 0.8544 (**1.000**) |
| aime | accept_len | 3.1631 | 3.0195 (0.955) | 3.1629 (**1.000**) |
| aime | accept_rate | 0.4923 | 0.4783 (0.971) | 0.4923 (**1.000**) |
| aime | full_acc | 0.4973 | 0.4848 (0.975) | 0.4971 (**1.000**) |
| aa-lcr-4k | accept_len | 3.3831 | 3.1588 (0.934) | 3.3662 (**0.995**) |
| aa-lcr-4k | accept_rate | 0.5154 | 0.4926 (0.956) | 0.5135 (**0.996**) |
| aa-lcr-4k | full_acc | 0.5229 | 0.5010 (0.958) | 0.5203 (**0.995**) |

**The confound was not a minor detail — it was responsible for nearly all
of the originally-measured degradation.** Once same-block bidirectionality
is correctly preserved:
- `gsm8k` and `aime` show **no measurable quality loss at all** (ratio
  1.000 on every metric) from restricting the context window to 2048
  tokens. Both sets' prompts are short enough that a 2048-token window
  essentially never binds.
- `aa-lcr-4k` — the set specifically chosen because its ~4k-token prompts
  *should* stress a 2048-token window — drops only ~0.4-0.6%, an order of
  magnitude smaller than the confounded run's ~4-7% drop.

**Interpretation**: RadixArk's full-attention-trained DSpark checkpoint
appears remarkably robust to being forced into a 2048-token sliding
window at serving time, with no retraining, *provided the same-block
attention pattern is not also disturbed*. The large effect in the first
run was almost entirely an artifact of accidentally making same-block
attention causal, not genuine sensitivity to the context-window
restriction itself. This is a real, useful, and non-obvious finding — it
suggests the practical cost of retrofitting sliding-window attention onto
this checkpoint (for memory savings in long-context serving, per the
KV-cache discussion above) could be very low, *if* implemented carefully
enough to avoid this exact bidirectionality trap.

### Third run — full 24-set sweep, clean (window only)

Same clean configuration as the second run (`sliding_window=2048`,
`sliding_window_non_causal=True`, same-block attention verified
bidirectional in both conditions), extended from the 3-set spot check to
all 24 offline-harness sets used in
[`kimi_k3_draft_eval.md`](kimi_k3_draft_eval.md).

**Updated 2026-09-24 — sample counts expanded from 15/25 to 50 per set**
(30 for `aime`/`aime26`, which only have 30 raw prompts available) for
more statistical confidence, after a methodological challenge raised
whether the small-n results were noisy. New hidden states were collected
via a full generate+extract pass (plain-server generation, then an
`extract_hidden_states` server capture at the same aux layers), and all
four conditions (baseline + 3 window sizes) were rerun at the new sample
counts. The n=15/25/30 numbers below are entirely superseded — see the
verification note after the tables for how the two compare (answer:
almost no change; the original small-sample read was already accurate).
Sorted by `accept_len` ratio ascending (most-affected first):

| set | n | baseline AL | sliding-2048 AL | AL ratio | baseline AR | sliding-2048 AR | AR ratio | baseline full_acc | sliding-2048 full_acc | acc ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aa-lcr-4k | 50 | 3.2311 | 3.2131 | **0.9944** | 0.4951 | 0.4924 | 0.9945 | 0.5009 | 0.4974 | 0.9931 |
| speed-writing | 50 | 3.3732 | 3.3582 | 0.9956 | 0.5392 | 0.5372 | 0.9962 | 0.5635 | 0.5613 | 0.9961 |
| speed-rag | 50 | 4.1891 | 4.1757 | 0.9968 | 0.6349 | 0.6329 | 0.9969 | 0.6569 | 0.6549 | 0.9968 |
| tool_call | 50 | 4.0018 | 3.9976 | 0.9990 | 0.6197 | 0.6194 | 0.9995 | 0.6368 | 0.6366 | 0.9996 |
| swe-bench-pro | 50 | 3.6307 | 3.6300 | 0.9998 | 0.5609 | 0.5608 | 0.9998 | 0.5887 | 0.5886 | 0.9998 |
| translation | 50 | 3.9162 | 3.9160 | 0.9999 | 0.6285 | 0.6285 | 1.0000 | 0.6554 | 0.6554 | 0.9999 |
| bfcl | 50 | 4.6678 | 4.6676 | 1.0000 | 0.6838 | 0.6837 | 0.9999 | 0.6986 | 0.6984 | 0.9998 |
| mbpp | 50 | 4.7488 | 4.7486 | 1.0000 | 0.7288 | 0.7288 | 1.0000 | 0.7435 | 0.7435 | 1.0000 |
| math500 | 50 | 4.6384 | 4.6383 | 1.0000 | 0.6909 | 0.6909 | 1.0000 | 0.6970 | 0.6969 | 0.9999 |
| aime26 | 30 | 2.8670 | 2.8670 | 1.0000 | 0.4591 | 0.4591 | 1.0000 | 0.4549 | 0.4552 | 1.0006 |
| summarization | 50 | 3.7169 | 3.7168 | 1.0000 | 0.5810 | 0.5810 | 1.0000 | 0.6067 | 0.6067 | 1.0000 |
| mtbench | 50 | 3.7712 | 3.7711 | 1.0000 | 0.5855 | 0.5855 | 1.0000 | 0.6048 | 0.6047 | 0.9998 |
| aime | 30 | 3.1432 | 3.1432 | 1.0000 | 0.4903 | 0.4903 | 1.0000 | 0.4950 | 0.4949 | 0.9998 |
| aa-lcr-1k | 50 | 3.2307 | 3.2307 | 1.0000 | 0.5011 | 0.5011 | 1.0000 | 0.5106 | 0.5106 | 1.0001 |
| speed-qa | 50 | 3.3698 | 3.3698 | 1.0000 | 0.5401 | 0.5401 | 1.0000 | 0.5630 | 0.5630 | 1.0000 |
| qa | 50 | 3.3935 | 3.3935 | 1.0000 | 0.5423 | 0.5423 | 1.0000 | 0.5652 | 0.5651 | 0.9999 |
| speed-coding | 50 | 4.8553 | 4.8554 | 1.0000 | 0.7157 | 0.7157 | 1.0000 | 0.7329 | 0.7330 | 1.0001 |
| gpqa | 50 | 3.0785 | 3.0785 | 1.0000 | 0.4694 | 0.4694 | 1.0000 | 0.4830 | 0.4830 | 0.9999 |
| speed-multilingual | 50 | 3.6412 | 3.6413 | 1.0000 | 0.5542 | 0.5542 | 1.0000 | 0.5804 | 0.5804 | 0.9999 |
| writing | 50 | 3.8937 | 3.8938 | 1.0000 | 0.6023 | 0.6023 | 1.0000 | 0.6203 | 0.6202 | 0.9998 |
| livecodebench | 50 | 4.3651 | 4.3652 | 1.0000 | 0.6537 | 0.6537 | 1.0000 | 0.6689 | 0.6691 | 1.0002 |
| gsm8k | 50 | 5.9900 | 5.9902 | 1.0000 | 0.8502 | 0.8502 | 1.0000 | 0.8584 | 0.8583 | 0.9999 |
| swe-rebench | 50 | 3.2403 | 3.2404 | 1.0000 | 0.5235 | 0.5236 | 1.0000 | 0.5451 | 0.5451 | 1.0000 |
| rag | 50 | 4.1350 | 4.1352 | 1.0001 | 0.6342 | 0.6342 | 1.0000 | 0.6567 | 0.6568 | 1.0001 |

**Pattern confirmed at full scale, and again at n=50.** `accept_len`
ratios span only **0.9944 to 1.0001** — every set is within ~0.6% of its
full-attention baseline, and the large majority round to **1.000**
exactly. `aa-lcr-4k` remains the single most-affected set, now at -0.56%
(was -0.5% at n=15) — the same conclusion as before, essentially
unchanged by the 3x-larger sample.

**Verification against the n=15 read**: mean AL ratio at n=50 is 0.9994 —
identical to the n=15 mean (0.9994) to 4 decimal places. Every individual
set's ratio moved by less than 0.5 percentage points except `speed-rag`
(-0.32pp, still trivial in absolute terms: 1.0000→0.9968). The
"long-context sets degrade more" pattern and its magnitude are both
confirmed, not an artifact of the smaller original sample.

**Overall conclusion**: at 24-set scale (now n=50/30 per set), RadixArk's
full-attention-trained DSpark checkpoint tolerates a 2048-token
sliding-window retrofit with essentially no measurable quality cost,
including on the long-context set built to stress it. This strengthens
the "low practical cost" conclusion from the 3-set check into a general
finding across this checkpoint's full evaluation suite, *conditional on*
correctly preserving same-block bidirectionality (see "Second run" above)
and validated live (see "Live-serving verification" below).

### Fourth run — window=512, offline 24-set sweep: a different sensitivity pattern

Follow-up #3 asked whether a much tighter window changes the picture.
Same clean setup as the runs above, `sliding_window` dropped from 2048 to
**512**. Numbers below are the n=50/30 rerun (see note in "Third run"
above); sorted by `accept_len` ratio ascending:

| set | n | baseline AL | sliding-512 AL | AL ratio | baseline AR | sliding-512 AR | AR ratio | baseline full_acc | sliding-512 full_acc | acc ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| summarization | 50 | 3.7169 | 3.4088 | **0.9171** | 0.5810 | 0.5394 | 0.9284 | 0.6067 | 0.5667 | 0.9341 |
| rag | 50 | 4.1350 | 3.7979 | **0.9185** | 0.6342 | 0.5947 | 0.9378 | 0.6567 | 0.6191 | 0.9427 |
| speed-rag | 50 | 4.1891 | 3.8809 | **0.9264** | 0.6349 | 0.5968 | 0.9400 | 0.6569 | 0.6190 | 0.9423 |
| swe-bench-pro | 50 | 3.6307 | 3.3913 | **0.9341** | 0.5609 | 0.5317 | 0.9478 | 0.5887 | 0.5600 | 0.9513 |
| speed-writing | 50 | 3.3732 | 3.2450 | 0.9620 | 0.5392 | 0.5214 | 0.9670 | 0.5635 | 0.5451 | 0.9674 |
| aa-lcr-1k | 50 | 3.2307 | 3.1150 | 0.9642 | 0.5011 | 0.4846 | 0.9671 | 0.5106 | 0.4925 | 0.9646 |
| speed-multilingual | 50 | 3.6412 | 3.5314 | 0.9698 | 0.5542 | 0.5347 | 0.9650 | 0.5804 | 0.5607 | 0.9661 |
| gpqa | 50 | 3.0785 | 2.9994 | 0.9743 | 0.4694 | 0.4580 | 0.9756 | 0.4830 | 0.4722 | 0.9776 |
| livecodebench | 50 | 4.3651 | 4.2532 | 0.9744 | 0.6537 | 0.6415 | 0.9814 | 0.6689 | 0.6577 | 0.9831 |
| tool_call | 50 | 4.0018 | 3.9285 | 0.9817 | 0.6197 | 0.6123 | 0.9880 | 0.6368 | 0.6303 | 0.9897 |
| aime | 30 | 3.1432 | 3.0991 | 0.9860 | 0.4903 | 0.4839 | 0.9869 | 0.4950 | 0.4885 | 0.9869 |
| mtbench | 50 | 3.7712 | 3.7256 | 0.9879 | 0.5855 | 0.5788 | 0.9885 | 0.6048 | 0.5978 | 0.9884 |
| speed-coding | 50 | 4.8553 | 4.8005 | 0.9887 | 0.7157 | 0.7087 | 0.9903 | 0.7329 | 0.7265 | 0.9912 |
| aa-lcr-4k | 50 | 3.2311 | 3.2086 | 0.9930 | 0.4951 | 0.4923 | 0.9943 | 0.5009 | 0.4964 | 0.9911 |
| swe-rebench | 50 | 3.2403 | 3.2283 | 0.9963 | 0.5235 | 0.5220 | 0.9970 | 0.5451 | 0.5439 | 0.9979 |
| aime26 | 30 | 2.8670 | 2.8595 | 0.9974 | 0.4591 | 0.4581 | 0.9978 | 0.4549 | 0.4537 | 0.9974 |
| bfcl | 50 | 4.6678 | 4.6608 | 0.9985 | 0.6838 | 0.6834 | 0.9994 | 0.6986 | 0.6985 | 1.0000 |
| translation | 50 | 3.9162 | 3.9123 | 0.9990 | 0.6285 | 0.6281 | 0.9993 | 0.6554 | 0.6551 | 0.9995 |
| writing | 50 | 3.8937 | 3.8913 | 0.9994 | 0.6023 | 0.6020 | 0.9995 | 0.6203 | 0.6198 | 0.9991 |
| speed-qa | 50 | 3.3698 | 3.3681 | 0.9995 | 0.5401 | 0.5399 | 0.9996 | 0.5630 | 0.5628 | 0.9996 |
| qa | 50 | 3.3935 | 3.3918 | 0.9995 | 0.5423 | 0.5421 | 0.9997 | 0.5652 | 0.5649 | 0.9996 |
| math500 | 50 | 4.6384 | 4.6361 | 0.9995 | 0.6909 | 0.6906 | 0.9995 | 0.6970 | 0.6967 | 0.9996 |
| mbpp | 50 | 4.7488 | 4.7480 | 0.9998 | 0.7288 | 0.7288 | 0.9999 | 0.7435 | 0.7436 | 1.0001 |
| gsm8k | 50 | 5.9900 | 5.9895 | 0.9999 | 0.8502 | 0.8502 | 0.9999 | 0.8584 | 0.8583 | 0.9999 |

Mean AL ratio 0.9778, mean AR ratio 0.9812, mean acc ratio 0.9820 —
essentially unchanged from the n=15 means (0.9779/0.9813/0.9822).

**The sensitivity pattern is qualitatively different from window=2048, not
just a scaled-up version of it.** At 2048, only `aa-lcr-4k` showed a real
effect and the earlier conclusion was "the effect is specific to raw
context length, not task type." At 512, that conclusion no longer holds:
`summarization`, `rag`, `speed-rag`, and `swe-bench-pro` now show
**7-8% degradation** — a real, task-correlated pattern, not noise
(verified below) — while `aa-lcr-4k`, the set specifically built to
stress long context, barely moves (0.9930).

**Confirmed at n=50, with one real correction to the n=15 read**: the
overall pattern (which sets are hit, roughly how hard) reproduces almost
exactly — `rag` 0.9185 (was 0.9177), `summarization` 0.9171 (was 0.9212),
`swe-bench-pro` 0.9341 (was 0.9344), `speed-rag` 0.9264 (was 0.9396, a
modest -1.3pp move toward *more* degradation). The one set whose n=15
result turned out noisy: `speed-multilingual` — n=15 showed -5.0%
(0.9499), n=50 shows only -3.0% (0.9698), a genuine ~2-percentage-point
correction in the *less-bad-than-thought* direction. Everything else
across all 24 sets moved by under 1.3pp between n=15 and n=50.

**Verified this isn't dilution from anchors the window never restricts**
(same check applied to `aa-lcr-4k` earlier), for every set showing a new
effect at 512:

| set | seq_len range | total anchors | frac. with anchor position ≥ 512 |
|---|---|---:|---:|
| `rag` | 1007–1324 | 5878 | **100.0%** |
| `summarization` | 857–1912 | 7126 | **98.0%** |
| `swe-bench-pro` | 811–1923 | 6833 | **96.3%** |
| `speed-rag` | 541–1311 | 6128 | **91.2%** |
| `speed-multilingual` | 469–1337 | 6583 | 48.6% |
| `aa-lcr-4k` (reference) | 4315–4697 | 6514 | 100.0% |

The window genuinely binds for the vast majority of scored anchors in
`rag`/`summarization`/`swe-bench-pro`/`speed-rag` — their 7-8%
degradation is real, not diluted. `speed-multilingual` is the one mixed
case: only ~49% of its anchors actually have the window binding (many
samples are shorter than 512 tokens entirely), so its degradation is
itself diluted by anchors that see zero restriction — the true effect for
anchors where the window *does* bind is likely closer to double the
pooled -3.0%.

**Working interpretation** (plausible, not proven): this isn't really
about raw document length — it's about whether the specific content a
task needs sits within reach of the window. `aa-lcr-4k`'s documents are
~4500 tokens, but apparently whatever the draft needs to predict its
completion sits close enough to the anchor already (within 512 tokens)
that shrinking further from 2048 to 512 costs almost nothing. `rag`/
`summarization`/`swe-bench-pro` have much shorter documents (700-1900
tokens) but their tasks inherently reference content that can sit
anywhere across the prompt (the retrieved passage, the source document
to summarize, the code diff under review) — a 512-token window cuts away
roughly half to three-quarters of a ~1000-1900 token document, which is
enough to lose task-relevant content that a 2048-token window still
comfortably held. So sensitivity looks like it tracks *task type*
(how spread out the relevant content is) more than *raw context length*
once the window gets tight enough — the opposite of what the 2048-only
result suggested.

### Fifth run — window=1024, offline 24-set sweep: mostly recovers, but not uniformly

Same setup again, `sliding_window=1024` (between the 2048 and 512 runs
above). Numbers below are the n=50/30 rerun; sorted by `accept_len` ratio
ascending:

| set | n | AL ratio | AR ratio | acc ratio |
|---|---:|---:|---:|---:|
| speed-writing | 50 | **0.9788** | 0.9819 | 0.9817 |
| aa-lcr-1k | 50 | 0.9816 | 0.9830 | 0.9826 |
| summarization | 50 | 0.9835 | 0.9871 | 0.9885 |
| swe-bench-pro | 50 | 0.9869 | 0.9900 | 0.9901 |
| speed-rag | 50 | 0.9883 | 0.9903 | 0.9902 |
| aa-lcr-4k | 50 | 0.9913 | 0.9926 | 0.9906 |
| tool_call | 50 | 0.9929 | 0.9954 | 0.9960 |
| livecodebench | 50 | 0.9963 | 0.9974 | 0.9978 |
| gpqa | 50 | 0.9975 | 0.9980 | 0.9982 |
| rag | 50 | 0.9977 | 0.9983 | 0.9990 |
| aime | 30 | 0.9987 | 0.9990 | 0.9991 |
| speed-coding | 50 | 0.9997 | 0.9997 | 1.0000 |
| speed-multilingual | 50 | 0.9998 | 0.9997 | 0.9997 |
| mtbench | 50 | 0.9998 | 0.9999 | 0.9996 |
| swe-rebench | 50 | 0.9999 | 0.9998 | 0.9998 |
| translation | 50 | 0.9999 | 1.0000 | 0.9999 |
| bfcl | 50 | 1.0000 | 0.9999 | 0.9998 |
| mbpp | 50 | 1.0000 | 1.0000 | 1.0000 |
| math500 | 50 | 1.0000 | 1.0000 | 0.9999 |
| aime26 | 30 | 1.0000 | 1.0000 | 1.0006 |
| speed-qa | 50 | 1.0000 | 1.0000 | 1.0000 |
| qa | 50 | 1.0000 | 1.0000 | 0.9999 |
| writing | 50 | 1.0000 | 1.0000 | 0.9998 |
| gsm8k | 50 | 1.0000 | 1.0000 | 0.9999 |

Mean AL ratio 0.9955, mean AR ratio 0.9964, mean acc ratio 0.9965 —
essentially unchanged from the n=15 means (0.9963/0.9972/0.9972).

**Confirmed at n=50**: same worst-affected set (`speed-writing`, now
-2.12% vs -1.81% at n=15 — a small, non-alarming move), and the same
general shape (the `rag`/`summarization`/`swe-bench-pro`/`speed-rag`
cluster sits in the middle of the pack, partially recovered from their
512 result but not fully back to baseline). `aa-lcr-1k` moved the most of
any set between n=15 and n=50 at this window size (0.9857→0.9816, -0.41pp)
but remains a small effect in absolute terms.

**The "cliff" from window=512 mostly closes at 1024, but not uniformly
across the affected cluster, and the identity of the worst-hit set
changes.** Comparing all three window sizes for the sets that mattered at
512 (n=50 throughout):

| set | 2048 | 1024 | 512 |
|---|---:|---:|---:|
| summarization | ~1.000 | 0.9835 | **0.9171** |
| rag | 1.0001 | 0.9977 | **0.9185** |
| speed-rag | 0.9968 | 0.9883 | **0.9264** |
| swe-bench-pro | 0.9998 | 0.9869 | **0.9341** |
| speed-multilingual | 1.0000 | 0.9998 | 0.9698 |
| speed-writing | 0.9956 | **0.9788** | 0.9620 |

`rag`, `swe-bench-pro`, and `speed-rag` recover most of the way back
toward baseline by 1024 (each within ~1.2-2.3%), consistent with their
critical content sitting in a chunk of the document that a 1024-token
window still mostly covers even though 512 cuts into it. `summarization`
only partially recovers (still -1.65% at 1024, was -8.3% at 512) — a
shallower cliff, more spread across the 512-1024-2048 range rather than
concentrated right at the 512-1024 boundary. `speed-writing` is the
outlier: it barely moved at 512 (-3.8%) but is now the single
most-affected set at 1024 (-2.1%) — not a monotonic "tighter window is
worse" story for this one set, though the absolute effect stays small
throughout (never more than ~4%). Taken together, three window sizes at
full n=50 confirm this isn't one clean threshold shared across tasks —
different domains hit their own sensitivity point at different window
sizes, consistent with the "content position" interpretation above rather
than a single universal cliff.

## Artifacts

- Modified checkpoint configs (weights symlinked, not copied):
  `/import/ml-sc-scratch5/chenw/models/Kimi-K3-DSpark-sliding2048/`,
  `/import/ml-sc-scratch5/chenw/models/Kimi-K3-DSpark-sliding1024/`,
  `/import/ml-sc-scratch5/chenw/models/Kimi-K3-DSpark-sliding512/`
- Fix: `scripts/evaluate/kimi_k3_offline_eval/run_dspark_eval.py`
  (`sliding_window`/`use_sliding_window`/`sliding_window_non_causal`
  pass-through)
- First (confounded) run: `/tmp/dspark_eval_sliding2048_{gsm8k,aime,aa-lcr-4k}.json`
- Second (clean, 3-set) run: `/tmp/dspark_eval_sliding2048_bidir_{gsm8k,aime,aa-lcr-4k}.json`
- Third (clean, full 24-set, window=2048) run: `/tmp/dspark_eval_sliding2048_bidir_full_<set>.json`
  for all 24 sets listed above
- Fourth (clean, full 24-set, window=512) run: `/tmp/dspark_eval_sliding512_full_<set>.json`
  for all 24 sets listed above
- Fifth (clean, full 24-set, window=1024) run: `/tmp/dspark_eval_sliding1024_full_<set>.json`
  for all 24 sets listed above
- Baseline (unmodified, already existed): `/import/ml-sc-scratch5/chenw/models/kimi-k3-data-clean/<set>/dspark_eval_clean.json`

## Open follow-ups

1. ~~Fill in the clean (bidirectional) results above once the rerun
   completes.~~ **Done** — see "Second run" above. Result: the confound
   was responsible for nearly all of the originally-measured degradation;
   the true context-window effect alone is ~0% (gsm8k, aime) to ~0.5%
   (aa-lcr-4k).
2. ~~**Full 24-set sweep** — the quick 3-set read is very promising
   (near-zero quality loss), worth confirming at full scale before
   drawing a general conclusion about this checkpoint's
   window-robustness.~~ **Done** — see "Third run" above. Result:
   confirmed at full scale, accept_len ratios range 0.9950-1.0001 across
   all 24 sets, with `aa-lcr-4k` (the long-context set) the single
   most-affected at -0.5%, and no other domain showing a comparable
   effect.
3. ~~Try other window sizes (512, 1024, 4096) to map the sensitivity curve
   — given 2048 already shows almost no effect, a smaller window (512)
   would be more informative for finding where it actually starts to bite,
   especially on long-context sets.~~ **512 and 1024 done** — see "Fourth
   run" and "Fifth run" above. Result: the sensitivity pattern changes
   qualitatively, not just in magnitude — at 512, `rag`/`summarization`/
   `swe-bench-pro`/`speed-rag` show real 6-8% degradation (verified not
   diluted), while `aa-lcr-4k` (the long-context set that was the only one
   affected at 2048) barely moves. At 1024, most of that cluster mostly
   recovers (within ~1.5-3% of baseline), but not uniformly, and the
   identity of the worst-hit set changes (`speed-writing` at 1024,
   `rag` at 512) — three different domains hitting their own sensitivity
   point at different window sizes, not one shared threshold. 4096
   remains untested (lower priority now — 2048 already showed near-zero
   effect, so 4096 would likely show the same or less).
4. ~~Investigate the `update_dspark`/`sliding_window_non_causal` gap in
   vLLM's live-serving conversion path before ever attempting to actually
   *serve* a sliding-window-retrofitted DSpark checkpoint live.~~ **Done,
   root cause found and fixed, fix confirmed live at full 24-set scale** —
   see "Live-serving verification" below. The checkpoint's own
   `dflash_config` lacked a `causal` key, so live serving fell back to
   causal (confounded) same-block attention. Fixed directly in the
   checkpoint config. A 2-prompt spot check showed AL 2.88→3.22 (+12%);
   the full 24-set live sweep afterward showed mean AL ratio 0.9994 vs
   RadixArk's unmodified baseline (essentially 1.000, `aa-lcr-4k` the
   single most-affected set at -1.5%) — live serving now matches the
   clean, near-zero-degradation regime validated offline, at full scale,
   not just a spot check.

## Live-serving verification: causal-attention gap, found and fixed

Stood up a real vLLM 0.29.0 server (`Kimi-K3-patched` target,
`Kimi-K3-DSpark-sliding2048` draft attached via native
`--speculative-config`, TP=8) to check what actually happens when this
ablation checkpoint is served live, rather than replayed through the
offline harness. Loaded and served successfully (one unrelated hiccup:
a `flashinfer-cubin`/`flashinfer` version mismatch in this conda env,
worked around with `FLASHINFER_DISABLE_VERSION_CHECK=1`).

**First run — confounded, root cause found.** Generation was coherent and
speculative decoding was genuinely active (85 drafts × 7 = 595 draft
tokens, 160 accepted, AL≈2.88/AR≈0.269 on two ad-hoc prompts), but this
measured the *wrong* regime. vLLM's live-serving code path is a completely
separate implementation from this repo's offline harness —
`vllm/model_executor/models/qwen3_dflash.py` (reused by `qwen3_dspark.py`)
resolves each layer's causality via:

```python
def _dflash_layer_causal(config, layer_idx):
    is_causal = getattr(config, "is_causal", None)
    if is_causal is not None: return bool(is_causal)
    override = (getattr(config, "dflash_config", None) or {}).get("causal")
    if override is not None: return bool(override)
    layer_types = getattr(config, "layer_types", None)
    return bool(layer_types) and layer_types[layer_idx] == "sliding_attention"
```

Initial hypothesis was that `update_dspark`
(`vllm/transformers_utils/configs/speculators/algos.py`) was the culprit —
it never sets `dflash_config`/`causal`/`is_causal` anywhere, unlike
`update_dflash`, which explicitly reads `sliding_window_non_causal` and
sets `dflash_config["causal"]`. **This hypothesis was wrong** —
`update_dspark`/`SpeculatorsConfig` only runs for checkpoints in the
nested "speculators-native" config format (the one `speculators_model_type`
gates, e.g. RedHatAI's checkpoint). `Kimi-K3-DSpark-sliding2048` has no
`speculators_model_type` field at all — it's the flat, already-resolved
format (`architectures: ["Qwen3DSparkModel"]`), loaded directly with no
conversion layer. So `update_dspark` never runs for it, and patching it
would have had zero effect (caught before spending a second GPU cycle on
the wrong fix). The **actual** root cause: the checkpoint's own raw
`dflash_config` dict (`{mask_token_id, target_layer_ids}`) simply had no
`causal` key, so `_dflash_layer_causal` fell through to the `layer_types`
check — `causal = True` for every layer, since every layer's `layer_types`
entry is `"sliding_attention"`. Same bug as the offline harness's
"First run" above, reached by a different path.

**Fix**: added `"causal": false` directly to
`Kimi-K3-DSpark-sliding2048/config.json`'s `dflash_config` dict (plus,
separately, `"sliding_window_non_causal": true` at the top level, harmless
since nothing on this checkpoint's format reads it, but kept for
consistency with the offline harness's own config reading).

**Second run — fix confirmed live.** Same server setup, same two prompts,
temperature 0 (deterministic). Generated text was byte-identical both
times (as expected — greedy decoding of the target always produces the
same output regardless of draft quality); what changed is how many
verify steps it took to get there:

| | drafts | draft tokens | accepted | AL | AR |
|---|---:|---:|---:|---:|---:|
| First run (causal, buggy) | 85 | 595 | 160 | 2.88 | 0.269 |
| Second run (non-causal, fixed) | 76 | 532 | 169 | **3.22** | **0.318** |

AL up ~12%, AR up ~18%, for the *same total output* — fewer, longer accept
chains per verify step. This was only two ad-hoc prompts (not a rigorous
benchmark sample, and a bigger jump than the offline 24-set ablation's
average clean-vs-confounded gap), but the direction was exactly what the
offline ablation predicted, and it was a clean, internally consistent signal
(same text, fewer steps) — not an artifact.

### Third run — full 24-set live sweep, fix confirmed at scale

Per "evaluate across all the benchmarks": replayed all 380 cached clean
prompts (same `gen_cache/gen_*.json` token-id prompts used everywhere else
in this doc, `max_tokens` matched to each prompt's own clean-generation
length, temperature 0) against a fresh server with the fixed checkpoint,
across all 24 sets. AR/AL read from vLLM's real
`spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total` counters
(delta per set); tok/s is the mean of each request's own
`completion_tokens ÷ elapsed`. Compared against RadixArk's own original
full-attention checkpoint's live numbers (`kimi_k3_draft_eval.md`, "Full
24-set live-serving sweep — real measurements, vLLM 0.29.0"), sorted by
AL ratio ascending:

| set | n | baseline AL | fixed AL | AL ratio | baseline AR | fixed AR | AR ratio | baseline tok/s | fixed tok/s | tok/s ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aa-lcr-4k | 15 | 2.97 | 2.9240 | **0.985** | 0.282 | 0.2749 | 0.975 | 165.0 | 160.8 | 0.975 |
| speed-writing | 15 | 3.02 | 3.0137 | 0.998 | 0.288 | 0.2877 | 0.999 | 206.7 | 207.5 | 1.004 |
| mtbench | 25 | 2.82 | 2.8158 | 0.999 | 0.259 | 0.2594 | 1.002 | 204.6 | 205.1 | 1.002 |
| swe-rebench | 15 | 2.99 | 2.9862 | 0.999 | 0.284 | 0.2837 | 0.999 | 212.4 | 213.3 | 1.004 |
| gpqa | 15 | 2.70 | 2.6968 | 0.999 | 0.242 | 0.2424 | 1.002 | 194.7 | 195.4 | 1.004 |
| speed-multilingual | 15 | 3.14 | 3.1365 | 0.999 | 0.305 | 0.3052 | 1.001 | 225.0 | 225.7 | 1.003 |
| swe-bench-pro | 15 | 3.42 | 3.4164 | 0.999 | 0.345 | 0.3452 | 1.001 | 227.1 | 228.3 | 1.005 |
| speed-rag | 15 | 3.91 | 3.9060 | 0.999 | 0.415 | 0.4151 | 1.000 | 271.0 | 271.1 | 1.000 |
| gsm8k | 25 | 5.64 | 5.6362 | 0.999 | 0.662 | 0.6623 | 1.000 | 360.8 | 359.4 | 0.996 |
| writing | 15 | 2.66 | 2.6587 | 1.000 | 0.237 | 0.2370 | 1.000 | 189.2 | 189.6 | 1.002 |
| translation | 15 | 3.87 | 3.8690 | 1.000 | 0.410 | 0.4099 | 1.000 | 267.9 | 268.9 | 1.004 |
| rag | 15 | 3.93 | 3.9295 | 1.000 | 0.418 | 0.4185 | 1.001 | 261.8 | 263.9 | 1.008 |
| bfcl | 15 | 4.70 | 4.7000 | 1.000 | 0.529 | 0.5286 | 0.999 | 249.7 | 248.7 | 0.996 |
| math500 | 15 | 3.63 | 3.6304 | 1.000 | 0.376 | 0.3758 | 0.999 | 273.1 | 276.4 | 1.012 |
| mbpp | 15 | 4.39 | 4.3914 | 1.000 | 0.484 | 0.4845 | 1.001 | 300.4 | 299.2 | 0.996 |
| speed-coding | 15 | 4.44 | 4.4416 | 1.000 | 0.492 | 0.4917 | 0.999 | 307.4 | 307.9 | 1.002 |
| livecodebench | 15 | 3.55 | 3.5523 | 1.001 | 0.365 | 0.3646 | 0.999 | 267.7 | 268.0 | 1.001 |
| tool_call | 15 | 3.72 | 3.7244 | 1.001 | 0.389 | 0.3892 | 1.001 | 234.8 | 220.9 | 0.941 |
| aime | 15 | 2.63 | 2.6334 | 1.001 | 0.233 | 0.2333 | 1.001 | 195.5 | 196.5 | 1.005 |
| summarization | 15 | 3.57 | 3.5747 | 1.001 | 0.368 | 0.3678 | 0.999 | 239.8 | 241.0 | 1.005 |
| aime26 | 15 | 2.56 | 2.5634 | 1.001 | 0.223 | 0.2233 | 1.001 | 184.2 | 185.0 | 1.004 |
| qa | 15 | 3.18 | 3.1849 | 1.002 | 0.312 | 0.3121 | 1.000 | 221.8 | 222.4 | 1.003 |
| speed-qa | 15 | 3.18 | 3.1849 | 1.002 | 0.312 | 0.3121 | 1.000 | 221.8 | 222.6 | 1.004 |
| aa-lcr-1k | 15 | 2.72 | 2.7242 | 1.002 | 0.246 | 0.2463 | 1.001 | 164.9 | 172.4 | 1.045 |

**Mean AL ratio 0.9994, mean AR ratio 0.9992, mean tok/s ratio 1.0009** —
essentially 1.000 across the board. `aa-lcr-4k` is again the single
most-affected set (-1.5% AL), consistent with it being the long-context
set the window is actually sized around; every other set is within noise
of the unmodified baseline. This is the same near-zero-degradation
pattern the offline 24-set sweep found ("Third run" above), now confirmed
with a real, full-scale, live-serving measurement — not just the two
ad-hoc prompts from the second run, and not just the offline analytical
harness. The `tool_call` tok/s ratio (0.941) is the one outlier worth
flagging as probably noise rather than signal: its AL/AR ratios are both
essentially 1.000, so a slower wall-clock time with an unchanged
accept-rate more likely reflects transient server load (only 15 samples,
short prompts) than any real effect of the sliding-window fix. The 8-GPU
server was shut down after this sweep completed to free the resources.

**Open**: this fix is local to this one checkpoint's `config.json` (the
copy used for this ablation). It does not fix `update_dspark` for the
speculators-native format, and does not help any *other* checkpoint that
ships without a `causal` key in `dflash_config` — that would need a real
default-flip decision (should `layer_types: sliding_attention` really
imply causal by default in `qwen3_dflash.py`, given DSpark's own default
is bidirectional?) upstream in vLLM, out of scope for this ablation.

## AgentX (SemiAnalysis InferenceX) attempt — server crash, no valid results

Attempted the same fixed checkpoint against AgentX
(`aiperf --scenario inferencex-agentx-mvp`, `run_agentx.sh`, concurrency
sweep `1 8 16`, 900s/cell — the scenario minimum). Two setup issues were
found and fixed first (neither is a result, just plumbing):
- `aiperf` wasn't installed; installed into a fresh venv
  (`/import/ml-sc-scratch5/chenw/aiperf-venv`), works fine, no dataset
  issues — the trace corpus (`semianalysis_cc_traces_weka_062126`, 274
  conversations) downloaded and loaded correctly on the first real attempt.
- `run_agentx.sh` failed instantly on all 3 cells with a tokenizer error
  (Kimi K3's tokenizer needs `trust_remote_code`, which the script didn't
  pass). Fixed via a local copy with `--tokenizer-trust-remote-code` added
  — **but the copy was placed in a scratch directory, not next to its
  sibling `agentx_metrics.py` in `mtp_server_eval/`**, which broke the
  script's own `cd "$(dirname "$0")"` relative-path assumption. This
  produced a second round of all-`NA` results (visible in
  `agentx_run2.log`) purely from a `python3: can't open file
  .../agentx_metrics.py` error — not a measurement of anything.

**Investigating that second all-NA result (rather than accepting it) surfaced the real problem**: the `users=1` cell's own aiperf console output showed 83 `ConnectionRefusedError`s and one `500 EngineCore encountered an issue` — the vLLM server itself had crashed partway through. Confirmed in the server log
(`vllm_sliding2048_server_agentx.log`): at 12:07:51, ~650s into the
`users=1` cell's 900s window, `Worker_TP0` hit

```
cutlass.cutlass_dsl.tvm_ffi_provider.CUDADialectError: cudaErrorIllegalAddress (error code: 700)
```

inside `vllm/models/kimi_k3/nvidia/mla.py:_forward_prefill_fused` →
`flash_attn_varlen_func` → the CuTeDSL FA4 MLA prefill kernel — **the
target model's own MLA prefill attention, not anything in the DSpark
draft/sliding-window code this ablation modifies.** The worker died,
`EngineCore encountered a fatal error`, and every subsequent request
(this cell's remainder, then both the `users=8` and `users=16` cells,
which never got past their single warmup request) failed with connection
errors. So: **zero valid AgentX measurements exist from this attempt** —
`users=1`'s reported numbers are contaminated by the mid-run crash (an
unknown mix of real pre-crash requests and post-crash connection
failures), and `users=8`/`users=16` never ran at all. None of this is
written up as a result table, per instruction to keep the numbers in this
doc correct.

**What's known vs. not**: the crash is in the target's MLA prefill
kernel, and AgentX's synthesized prompts run much longer
(up to ~121K tokens, `--max-context-length 128000`) than anything in the
24-set static sweep (which tops out in the low thousands of tokens) —
long-context prefill is the most likely trigger. **Not yet confirmed**:
whether this is specific to something about the sliding-window/causal-fix
checkpoint, or a pre-existing Kimi K3 + CuTeDSL FA4 fragility at long
context that the unmodified checkpoint would hit too — the fastest way to
find out would be running the same AgentX cell against RadixArk's
original checkpoint, not yet done. Server was already dead (no manual
shutdown needed); GPUs confirmed free afterward.

## Appendix: does the sliding window ever touch the block's own attention?

**Short answer: no — confirmed on two independent grounds, both verified,
not just assumed.**

**1. Structurally, by construction — true regardless of any config.**
`create_anchor_block_mask_mod` (`dflash/attention.py`) partitions the KV
index space into two disjoint ranges: base-sequence positions
(`kv_idx < total_seq_len`) and synthetic same-block/mask-token positions
(`kv_idx >= total_seq_len`). The two mask functions are each gated on one
side of that partition:

```python
def base_prefix_mod(_b, _h, q_idx, kv_idx):     # applies sliding_window
    kv_is_base = kv_idx < total_seq_len          # <- gate
    ...
    return kv_is_base & same_doc & before_anchor & in_window

def same_block_mod(_b, _h, q_idx, kv_idx):       # governs the block
    kv_is_block = kv_idx >= total_seq_len         # <- opposite gate
    ...                                            # never references sliding_window
```

`base_prefix_mod` is ANDed with `kv_is_base` as its first term, so for any
`kv_idx` in the block region it is forced `False` no matter what
`in_window`/`before_anchor` compute — the window logic is evaluated (on
an aliased, modulo-wrapped position, purely to avoid an out-of-bounds
index) but its result is always discarded there. `same_block_mod` doesn't
reference `sliding_window` at all. `or_masks(base_prefix_mod,
same_block_mod)` unions two predicates that live on domains that never
overlap. **The window restricts only how far back into the base-sequence
context a query can see; it has no code path into the block's own
attention at all.**

**2. Causality — the part that *was* actually broken, now fixed in both
places this ablation touches.** Same-block attention being non-causal
(bidirectional) is logically independent of the context window, but the
code couples them by default:
`non_causal = sliding_window is None or sliding_window_non_causal` — an
ecosystem-wide convention (matching vLLM's own `update_dflash`, "enable
causal masking in SWA for vllm-project/speculators models") that fits
checkpoints natively trained with causal sliding-window attention, but is
wrong for RadixArk's checkpoint, which was trained with non-causal
same-block attention throughout. Both places this ablation exercises that
coupling now override it explicitly:
- **Offline harness**: `sliding_window_non_causal=True` passed into
  `DSparkDraftModel.from_training_args(...)` — verified via
  `model.sliding_window_non_causal == True` before running.
- **Live serving**: `"causal": false` set directly in the checkpoint's
  own `dflash_config` — verified via a live 24-set sweep matching the
  unmodified baseline.

**Empirical confirmation, not just code-reading**: if same-block attention
were still accidentally causal in either setup, the ~2-7% degradation
pattern from the "confounded" run (see "Bugs found and fixed") would
reappear. Instead, both the offline 24-set sweep (AL ratios 0.9950-1.0001)
and the live 24-set sweep (mean AL ratio 0.9994) show near-zero
degradation — which is only possible if same-block attention is genuinely
staying bidirectional in the windowed condition, matching how the
checkpoint was actually trained. In this ablation as run, the sliding
window affects only the base-sequence context window — never the block's
own attention mechanism, structurally or causally.
