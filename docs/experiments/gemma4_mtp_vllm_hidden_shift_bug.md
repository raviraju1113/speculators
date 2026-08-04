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
| shift-fixed, 500 steps | 2.175 |
| shift-fixed, full epoch (10k steps, pure logit-distill) | 2.217 |
| **+ feature-distillation (step3200)** | **2.512** |
| vanilla (reference) | 3.58 |

`soft_ce` fell 11.9 → 1.15 by step 500; the draft is servable and improves with training.
The shift fix (1.07 → ~2.2) is now folded directly into `training_step.py` (not a monkeypatch).

## The ~2.2 plateau — cause and fix (feature distillation)
After the shift fix, accept plateaued ~2.2 across a full epoch even as `soft_ce` kept
dropping — so it was a **ceiling, not under-training**. Root cause: the trainer used
**pure logit-distillation (soft-CE only)** — it never trained the draft's *hidden feature*.
Evidence: even the accept-2.2 draft has **`feat_l1 ≈ 1.83` (= random init)** — its
`backbone_hidden` (the recurrent state it feeds forward in TTT) does not resemble the
target's hidden at all. Since the multi-token tail depends entirely on that recurrent
hidden, steps 2–3 accept poorly → accept caps ~2.2.

NB: this is *not* a teacher-forcing / free-running bug. The trainer is standard **EAGLE-3
TTT** — recurrent hidden (the draft's own) + teacher-forced tokens — which is correct
(acceptance requires draft==target, so the accepted-path token == ground-truth token; only
the *hidden* has no ground-truth twin, hence it is recurred). An earlier "teacher-forcing"
hypothesis was walked back.

**Fix — EAGLE/DSpark feature distillation:** add a **smooth-L1 loss between the draft's
`backbone_hidden` and the target's UNSHIFTED hidden** at the aligned position `t+k`, plus
DSpark's recipe (loss `0.1 hard-CE + 0.9 feature-L1`, global batch 512, TTT depth 7). Same
Gemma4 assistant architecture — only the *training recipe* changed. New knobs in
`training_step.py`/`train_online.py`: `--soft-ce-weight`, `--hard-ce-weight`,
`--feature-l1-weight`. Result: warm-starting the accept-2.2 draft with the feature loss
drove `feat_l1` 1.83 → ~0.89 and lifted **vLLM accept 2.217 → 2.512** (rate 0.41 → 0.50) —
the ceiling was not a wall. Next: add soft-CE back on top (`soft_ce 1.0 + feature 0.9`) to
sharpen target-argmax agreement (the accept metric) toward vanilla's 3.58.

## Recommendations
1. **[done]** Hidden shift is folded directly into `training_step.py` (draft step-0 consumes
   the shifted `h_{t-1}`; feature labels use the unshifted hidden). The old `patch_hidden_shift`
   monkeypatch is now a no-op. Apply the same to the offline cache path
   (`training_step_from_cache`) if it is used.
2. **Feature distillation is essential for the tail** — pure logit-distillation caps accept
   ~2.2. Use `--feature-l1-weight 0.9` (+ soft- and/or hard-CE) to train the recurrent hidden.
3. Put a **vLLM accept-length eval in the training loop** — loss and HF are blind to the
   original shift bug; only a vLLM eval exposed it.
4. TTT (recurrent hidden + teacher-forced token) is correct EAGLE-3 — do NOT "fix" it with
   free-running tokens (that was a wrong lead). The tail is fixed by feature distillation +
   bigger batch + more steps, not by changing the token feed.

## Code changes (landed)
- `src/speculators/models/gemma4_mtp/training_step.py`: hidden shift folded in
  (`_shift_right`), EAGLE feature-distillation loss (`_feature_l1`, smooth-L1 of
  `backbone_hidden` vs the target's unshifted hidden), `MTPLossConfig.feature_l1_weight`,
  and a guard that skips the vocab-softmax when `soft_ce_weight == 0`.
- `scripts/gemma4_mtp/train_online.py`: `--soft-ce-weight`/`--hard-ce-weight`/
  `--feature-l1-weight` args; `patch_hidden_shift` is now a no-op; windowed (mean/N) loss log.

## Reproduction
Diagnostic scripts in the session scratchpad: `forward_parity_probe.py`, `aime_overfit_probe.py`,
`localize_forward.py`, `magnitude_probe.py`, `hidden_sensitivity.py`, `kv_sensitivity.py`,
`hf_specdecode.py` (dual convention), `dump_run.sh` + `analyze_dump.py`/`compare_hidden*.py`
(vLLM instrumentation; the vLLM patch was reverted after use), `shift_test.py`, `analyze_recurrence.py`.
Best-result checkpoints: `output/gemma4_26b_mtp_rinit_shiftfix_4gpu` (shift-fixed, 2.22),
`output/gemma4_26b_mtp_assistant_featdistill` (+ feature loss, 2.51).
Env: vLLM 0.24.0+cu129, torch 2.11.0+cu129, conda `speculator`.
