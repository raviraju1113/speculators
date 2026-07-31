# Gemma4 MTP: why from-scratch drafts collapsed in vLLM (hidden-state off-by-one)

## TL;DR
Training-from-scratch produced Gemma4-MTP drafts that looked perfect in HF/loss but
collapsed to **accept_length ≈ 1.07** in vLLM (no speedup), regardless of how long they
trained. Root cause: **the trainer fed the draft the target hidden state from the wrong
position** — `hidden[t]` instead of `hidden[t-1]`. vLLM (and the EAGLE/MTP convention,
and Google's vanilla draft) feeds `hidden[t-1]` + `embed(x_t)` → predict `x_{t+1}`.
A one-position shift in `build_target_signals` fixes it: from-scratch vLLM accept went
**1.07 → 2.1** (and rising). It was never a vLLM bug, LR, norms, or numerics.

## Symptom
- `google/gemma-4-26B-A4B-it` target + a **from-scratch-trained** MTP draft:
  vLLM `accept_length ≈ 1.07`, `accept_rate ≈ 0.02` at k=3/5/7 (pure overhead, tok/s ~59 vs ~190).
- The **vanilla** assistant draft works fine: accept_length 3.58 (k=3) … 5.68 (k=7).
- Training loss looked healthy the entire time (`step0_soft_ce ≈ 0.4`).

## How we localized it (elimination)
| Test | Result | Conclusion |
|---|---|---|
| Trained draft via HF training-forward, held-out AIME | 0.94 | not overfitting; generalizes |
| HF q_len=L (train fwd) == q_len=1 (infer fwd) | identical (A==B=1.0) | training forward self-consistent |
| Splice vanilla norms into trained ckpt, serve vLLM | still ~0.03 | not the norms |
| Activation magnitudes / logits (HF) | bounded, no nan, confident | not overflow/precision |
| Perturb fed hidden / KV by 10% noise (HF) | flat (robust) | not numerical input brittleness |
| **Dump vLLM's actual fed hidden, compare to HF** | vLLM `fed[t]` ≈ HF `hidden[t-1]` (cos 0.96), not `hidden[t]` (cos 0.57) | **off-by-one hidden shift** |
| Feed trained draft `h_{t-1}` in HF | **0.058** (= vLLM collapse); `h_t` = 0.942 | confirms the shift is the cause |
| Same for vanilla | `h_{t-1}` = 0.92 (its best), `h_t` = 0.84 | vanilla natively expects `h_{t-1}` |

Vanilla is robust to either convention, so it survived; a from-scratch draft has no prior
and overfit the wrong (`h_t`) alignment, so it collapsed when vLLM fed `h_{t-1}`.

## The fix
`patch_hidden_shift()` in `scripts/gemma4_mtp/train_online.py` (monkeypatch — no edit to
`training_step.py`): wrap `build_target_signals` so the draft's INPUT hidden is shifted
right by one (row t → `h_{t-1}`), while the labels (`target_logits`) stay computed from the
UNSHIFTED hidden.

```python
lh = sig["last_hidden"]                                   # row t = h_t
sig["last_hidden"] = torch.cat([lh[:, :1], lh[:, :-1]], 1) # row t = h_{t-1}
```

Safe because `training_step` uses `last_hidden` only as the step-0 draft input + recurrent
pad (never for supervision). Applied in `main()` next to the mask patch.

## Validation
Same broken recipe (random-init, lr 6e-4, 4-GPU), only the alignment corrected
(`output/gemma4_26b_mtp_rinit_shiftfix`):

| checkpoint | vLLM accept_len (aime, k=3) |
|---|---|
| old broken rinit (any # steps) | 1.07 |
| shift-fixed, 500 steps | **2.175** |
| shift-fixed, ~1500 steps | 2.10 |
| vanilla (reference) | 3.58 |

`soft_ce` fell 11.9 → 1.15 by step 500; the draft is now servable and improves with training.

## Residual (open) — the ~2.1 plateau
The shift fix is the dominant fix, but accept plateaus ~2.1, below vanilla's 3.58. Diagnosis:
a **second, training-side mismatch in the multi-step recurrence** — the trainer is
**teacher-forced** (feeds ground-truth tokens through the TTT steps), while vLLM decoding is
**free-running** (feeds the draft's own predicted tokens). Confirmed from the vLLM dumps:
spec step j consumes the token the draft itself produced at step j-1, at a constant position.
This exposure-bias gap caps steps 2–3 acceptance for a weak draft.

Caveat: our hand-rolled HF spec-decode eval also free-runs yet reads ~2.95 vs vLLM's 2.10, so
part of that gap may be HF-eval optimism; the reliable number is vLLM (2.10). A fresh dump
saving the recurrent `backbone_hidden` would separate "training-recurrence mismatch" from
"HF-eval optimism" definitively.

**To reach ~3.5:** implement free-running / scheduled-sampling in the TTT recurrence
(feed the draft's own sampled tokens during training) — a real `training_step.py` change,
not a monkeypatch — plus more training. Not the original bug; ordinary draft-quality work.

## Recommendations
1. Fold the hidden shift into `build_target_signals` (`src/.../gemma4_mtp/training_step.py`)
   so **every** training path (online + offline cache) is correct, not just the monkeypatched
   online path.
2. Put a **vLLM accept-length eval in the training loop** — loss and HF are blind to this
   class of bug; only a vLLM eval exposed it.
3. (Optional) free-running TTT training to close the 2.1 → 3.5 gap.

## Reproduction
Diagnostic scripts in the session scratchpad: `forward_parity_probe.py`, `aime_overfit_probe.py`,
`localize_forward.py`, `magnitude_probe.py`, `hidden_sensitivity.py`, `kv_sensitivity.py`,
`hf_specdecode.py` (dual convention), `dump_run.sh` + `analyze_dump.py`/`compare_hidden*.py`
(vLLM instrumentation; the vLLM patch was reverted after use), `shift_test.py`, `analyze_recurrence.py`.
Env: vLLM 0.24.0+cu129, torch 2.11.0+cu129, conda `speculator`.
