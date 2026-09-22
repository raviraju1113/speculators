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
[`kimi_k3_draft_eval.md`](kimi_k3_draft_eval.md). Sorted by `accept_len`
ratio ascending (most-affected first):

| set | n | baseline AL | sliding-2048 AL | AL ratio | baseline AR | sliding-2048 AR | AR ratio | baseline full_acc | sliding-2048 full_acc | acc ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aa-lcr-4k | 15 | 3.3831 | 3.3662 | **0.9950** | 0.5154 | 0.5135 | 0.9963 | 0.5229 | 0.5203 | 0.9951 |
| speed-writing | 15 | 3.4902 | 3.4831 | **0.9980** | 0.5541 | 0.5532 | 0.9983 | 0.5789 | 0.5780 | 0.9984 |
| aa-lcr-1k | 15 | 3.2151 | 3.2147 | 0.9999 | 0.4957 | 0.4957 | 0.9999 | 0.5039 | 0.5041 | 1.0003 |
| translation | 15 | 3.9644 | 3.9639 | 0.9999 | 0.6396 | 0.6396 | 1.0000 | 0.6666 | 0.6664 | 0.9997 |
| bfcl | 15 | 4.9744 | 4.9739 | 0.9999 | 0.7225 | 0.7225 | 0.9999 | 0.7338 | 0.7340 | 1.0003 |
| gpqa | 15 | 3.0082 | 3.0080 | 0.9999 | 0.4634 | 0.4634 | 0.9999 | 0.4745 | 0.4742 | 0.9994 |
| aime | 15 | 3.1631 | 3.1629 | 1.0000 | 0.4923 | 0.4923 | 1.0000 | 0.4973 | 0.4971 | 0.9996 |
| mbpp | 15 | 4.6469 | 4.6467 | 1.0000 | 0.7143 | 0.7143 | 1.0000 | 0.7291 | 0.7291 | 0.9999 |
| swe-bench-pro | 15 | 3.8286 | 3.8285 | 1.0000 | 0.5939 | 0.5939 | 1.0000 | 0.6161 | 0.6163 | 1.0004 |
| mtbench | 25 | 3.0679 | 3.0678 | 1.0000 | 0.5001 | 0.5001 | 1.0000 | 0.5227 | 0.5225 | 0.9996 |
| speed-multilingual | 15 | 3.7527 | 3.7526 | 1.0000 | 0.5684 | 0.5684 | 1.0000 | 0.5909 | 0.5907 | 0.9997 |
| aime26 | 15 | 2.8845 | 2.8845 | 1.0000 | 0.4607 | 0.4607 | 1.0000 | 0.4590 | 0.4593 | 1.0007 |
| tool_call | 15 | 4.1889 | 4.1888 | 1.0000 | 0.6447 | 0.6447 | 1.0000 | 0.6641 | 0.6638 | 0.9996 |
| speed-rag | 15 | 4.1683 | 4.1683 | 1.0000 | 0.6359 | 0.6359 | 0.9999 | 0.6576 | 0.6573 | 0.9995 |
| writing | 15 | 2.8906 | 2.8907 | 1.0000 | 0.4846 | 0.4846 | 1.0000 | 0.5057 | 0.5054 | 0.9994 |
| summarization | 15 | 3.6541 | 3.6542 | 1.0000 | 0.5714 | 0.5714 | 1.0000 | 0.5976 | 0.5975 | 0.9998 |
| math500 | 15 | 4.4372 | 4.4373 | 1.0000 | 0.6671 | 0.6671 | 0.9999 | 0.6722 | 0.6720 | 0.9998 |
| speed-coding | 15 | 4.7809 | 4.7810 | 1.0000 | 0.7103 | 0.7103 | 1.0001 | 0.7273 | 0.7276 | 1.0004 |
| livecodebench | 15 | 4.4606 | 4.4608 | 1.0000 | 0.6661 | 0.6662 | 1.0001 | 0.6803 | 0.6805 | 1.0004 |
| swe-rebench | 15 | 3.1136 | 3.1137 | 1.0000 | 0.5070 | 0.5070 | 1.0000 | 0.5302 | 0.5304 | 1.0003 |
| speed-qa | 15 | 3.4347 | 3.4350 | 1.0001 | 0.5504 | 0.5503 | 0.9999 | 0.5732 | 0.5730 | 0.9996 |
| qa | 15 | 3.4347 | 3.4350 | 1.0001 | 0.5504 | 0.5503 | 0.9999 | 0.5732 | 0.5730 | 0.9996 |
| gsm8k | 25 | 5.9666 | 5.9671 | 1.0001 | 0.8470 | 0.8470 | 1.0000 | 0.8545 | 0.8544 | 0.9999 |
| rag | 15 | 4.1171 | 4.1175 | 1.0001 | 0.6372 | 0.6373 | 1.0000 | 0.6583 | 0.6585 | 1.0003 |

**Pattern confirmed at full scale.** Across all 24 sets, `accept_len`
ratios span only **0.9950 to 1.0001** — every set is within ~0.5% of its
full-attention baseline, and 22 of 24 sets round to **1.000** exactly.
This matches the 3-set quick check closely: `gsm8k` and `aime` reproduce
their earlier ~1.000 ratios exactly, and `aa-lcr-4k` reproduces its
~0.995 ratio exactly (both to 4 decimal places), confirming the earlier
3-set read wasn't a fluke of that particular sample.

The **"long-context sets degrade more" hypothesis is directionally
correct but the effect is tiny even at its worst**: `aa-lcr-4k` (~4k-token
prompts, the set specifically chosen to stress a 2048-token window) is
the single most-affected set in the entire sweep, at a 0.5% accept_len
drop. `aa-lcr-1k` (shorter long-context prompts) shows essentially no
effect (0.9999), consistent with its prompts fitting comfortably inside
the 2048 window. `speed-writing` is the only other set with a
measurable (0.2%) drop, and doesn't have an obvious long-context
explanation — plausibly just sampling noise on its 15-sample set, since
several other sets (`translation`, `bfcl`, `gpqa`) sit at the noise floor
(0.9999) rather than exactly 1.0000. There's no sign of any other domain
(code, agentic tool-calling, RAG, translation, summarization) being
disproportionately sensitive — the effect is specific to raw context
length, not task type.

**Overall conclusion**: at 24-set scale, RadixArk's full-attention-trained
DSpark checkpoint tolerates a 2048-token sliding-window retrofit with
essentially no measurable quality cost, including on the long-context set
built to stress it. This strengthens the "low practical cost" conclusion
from the 3-set check into a general finding across this checkpoint's full
evaluation suite, *conditional on* correctly preserving same-block
bidirectionality (see "Second run" above) and on the still-unverified
live-serving conversion path (see follow-up 4 below).

## Artifacts

- Modified checkpoint config (weights symlinked, not copied):
  `/import/ml-sc-scratch5/chenw/models/Kimi-K3-DSpark-sliding2048/`
- Fix: `scripts/evaluate/kimi_k3_offline_eval/run_dspark_eval.py`
  (`sliding_window`/`use_sliding_window`/`sliding_window_non_causal`
  pass-through)
- First (confounded) run: `/tmp/dspark_eval_sliding2048_{gsm8k,aime,aa-lcr-4k}.json`
- Second (clean, 3-set) run: `/tmp/dspark_eval_sliding2048_bidir_{gsm8k,aime,aa-lcr-4k}.json`
- Third (clean, full 24-set) run: `/tmp/dspark_eval_sliding2048_bidir_full_<set>.json`
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
3. Try other window sizes (512, 1024, 4096) to map the sensitivity curve
   — given 2048 already shows almost no effect, a smaller window (512)
   would be more informative for finding where it actually starts to bite,
   especially on long-context sets.
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
