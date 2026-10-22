---
name: draft-opd
description: Draft-OPD (on-policy distillation for speculative draft models, arXiv 2605.29343) applied to our Gemma-4-31B-it DSpark drafts on the speculators + vLLM stack. Use when implementing or debugging the OPD post-training stage - the spec-decode rollout logger, error-position replay, the acceptance-aware KL loss, round-based weight sync - or when deciding what to run after the conv / block-size ablations. Triggers include Draft-OPD, OPD, on-policy distillation, rollout anchors, rejected-token replay, first-rejected-only, acceptance-aware KL.
---

# Draft-OPD on our stack

Paper: *Draft-OPD: On-Policy Distillation for Speculative Draft Models* (Lei et al.,
arXiv 2605.29343 v3, Sep 2026). Code: github.com/Simplified-Reasoning/Draft-OPD
(verl + vendored SGLang-with-DFlash; EAGLE3 variant; `feature/dflash-tv-loss`;
experimental `dev_vllm`). The plan and task list live in
`examples/train/opd/PLAN.md`; the loss / data-format details in `reference.md`
next to this file. We implement **Path A: port the algorithm into speculators +
our patched vLLM**, not their verl/SGLang stack (no DSpark, no Gemma 4, no
conv there).

## 1. The method in one screen

SFT on target trajectories plateaus because at inference the draft is judged on
blocks *it* proposed. OPD trains on those draft-induced states:

1. **Rollout with real speculative decoding** (target + current draft, block K).
   Keep the verified, target-quality continuation. For every draft step record
   an *anchor* `a_m` (verified prefix length), the drafted block
   `d_m = (d_{m,1..K})`, and the accepted count `r_m`.
2. **Replay** each anchor: student `q(d_{m,k} | c_m, d_{m,<k})` from the draft,
   teacher `p(d_{m,k} | c_m, d_{m,<k})` from one target forward over
   `context + drafted block` - i.e. the target is scored on the draft's proposal,
   including the rejected tail. (`c_m = (x, y_{<=a_m})`.)
3. **Acceptance-aware loss**: forward KL `KL(p||q)` on accepted positions
   `k <= r_m`; reverse KL `KL(q||p)` on rejected positions `k > r_m`, weighted
   `w_k = gamma^(k-1)`, gamma = 0.8, normalized; `L = (L_acc + L_rej)/2`.
   Decay on *rejected only* (applying it to accepted hurts, Table 6).
   **First-rejected-only** (rejected loss only on the first failed token) matches
   full Draft-OPD (5.94 vs 5.95 tau, Table 5) and needs the target scored only
   up to the first rejection -> use it; it removes most replay cost.
4. No serving change: the result is a drop-in checkpoint.

Paper numbers (DFlash, 5 layers, block 16, Qwen3-4B/8B, SpecForge SFT 6 ep then
OPD): tau 5.51 -> 5.96 (+8%), mean speedup +12-13% over DFlash, +23% over
EAGLE-3; gains hold at concurrency 32 in SGLang. Ablations: mixed KL > all-forward
> all-reverse; error anchors > random anchors; insensitive to gamma in 0.5-0.95
and lambda_rej in 0.5-1. Cost: 120 H200-hours for the 4B target (16K prompts x 8
epochs, batch 24 prompts, ~5.3K steps, 4k-token rollouts, AdamW 3e-4 cosine,
0.05 warmup) *including* the frozen target's rollout and verification.

## 2. Mapping onto our stack

| Paper component | Ours |
|---|---|
| Rollout engine (SGLang + DFlash) | our vLLM 0.26.0 eval env with the DSpark(+conv) serving patch (`examples/train/vllm_patches/`); rollouts = spec-decode generations over a prompt pool |
| Anchors / drafted blocks / r_m | **new**: a logging hook in `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` (draft ids are in `_generate_draft`, acceptance in the rejection sampler) -> per-request jsonl |
| Teacher p on draft-proposed prefixes | hidden-state server (`scripts/launch_vllm.py` + hs_connectors) prefill of `context + drafted block`, then the local `verifier_lm_head(verifier_norm(h))` as in `dflash/core.py:417-422` -> full-vocab p |
| Student q | the DSpark draft forward from the anchor (mask block over the context hidden states, exactly the training forward); q at slot k must include the Markov bias of the *drafted* token k-1 (`markov_head`), since that is what serving uses |
| Loss | **new** `OPDLoss` next to the TV/CE losses: forward KL (accepted) + gamma-weighted reverse KL (first rejected only); confidence-head labels unchanged |
| Weight sync to the rollout engine | **rounds**: train N steps -> checkpoint -> restart the rollout vLLM on it. vLLM has no cheap hot-swap; verl does per-step sync. Slightly off-policy between rounds; keep rounds short (<= 1/8 epoch) |
| Draft block layout | serve-time query block = `num_speculative_tokens` rows (anchor first) with `sample_from_anchor=True`; the replayed block must use the same layout |

Key facts about our draft that matter here: `sample_from_anchor=True` (slot k
predicts token k, k = block_size at serve), 32k draft vocab via `d2t/t2d`
(tokens outside it are guaranteed rejections - OPD cannot fix that; 11.2% of
GPQA prompt tokens, 3.4% AIME, 2.7% LCB), Markov head rank 256, confidence head
trained but not served, all-sliding 5 layers, optional conv (kernel 2, group 16).

## 3. Gotchas already known

- **The scheduler never sees the draft ids under async scheduling** (vLLM
  0.26 default): `scheduled_spec_decode_tokens` holds `-1` placeholders and
  `take_draft_token_ids()` is only called for structured outputs. The first
  smoke test logged all-`-1` blocks. Fix in place: the engine core calls
  `take_draft_token_ids()` right after `update_from_output` (the handler's host
  copy is ready, no extra sync) and hands it to `RolloutLogger.on_draft`; the
  next `on_step` substitutes the stored block for the placeholders
  (`vllm_patches/engine_core_rollout_log.patch`).
- **The draft-vocab mapping comes from the data dir, and it must be the
  checkpoint's.** `train.py` loads `d2t.npy`/`t2d.npy` from `--data-path`, or
  rebuilds them from `token_freq.pt` if absent. A rebuilt mapping differed from
  nemo782k's files (27,755/32,000 kept tokens overlap, different order), so the
  checkpoint's 32k lm_head rows were read through the wrong mapping: slot-0
  accuracy 0.00, accept_len 1.000, loss 42 from step 1 (pilot 12477930; the
  "control" on `gemma4_dspark_32` failed identically with its 349k-corpus
  mapping). `rollouts_to_dataset.py` now copies `d2t/t2d` from the
  `--token-freq` dataset and refuses to run without them. Any dataset used with
  a checkpoint must carry that checkpoint's mapping -- that includes "control"
  runs on other SFT subsets.
- **Anchor position**: the training anchor is the LAST VERIFIED token
  (`plen + anchor - 1`); slot 0 carries its embedding and sees base tokens
  strictly before it. `plen + anchor` is one late and mislabels the
  accepted/rejected slots.
- **Training flags live in the pydantic schema**
  (`src/speculators/train/config/schema.py`, `DFlashArgs` for DSpark knobs),
  not in `scripts/train.py`'s `parse_args()` -- that function is dead code;
  `main(TrainConfig.resolve())` builds the parser from the schema. A flag added
  only to `parse_args()` is "unrecognized" at runtime (cost one sweep).
- **Effective batch**: one micro-step is one 8192-token packed sequence
  (~10 rollouts of ~600 tokens). The 2-round pilot at ACCUM 1 / 1 rank /
  fresh Muon+AdamW warmed to 3e-4 (Muon 3e-3) wrecked the draft within ~100
  steps (parity 0.996 -> 0.65, rollout accept rate 0.41 -> 0.26). Size the
  batch (ranks x ACCUM) and LR like a fine-tune, not like the SFT recipe.
- **`finish=length` clips the last step**: a request hitting `max_tokens` keeps
  only the first `max_tokens` output ids, so the final step's `gen` is longer
  than what remains; the assembler compares on the kept prefix.

- **Replay cost is quadratic without prefix caching**: one full prefill of
  `context` per anchor. The hidden-state server runs `--no-enable-prefix-caching`
  (the connector must see every request executed). First-rejected-only cuts the
  scored suffix to <= r_m+1 tokens but the context prefill remains; either make
  the connector cache-safe (only new tokens' hidden states are needed) or cap
  rollouts at 2048 tokens and batch anchors of one rollout in one request by
  appending each block as a separate sequence. Decide this before writing code.
- **Student and serving must agree**: verify replayed q against the serving
  draft's logits on the same anchor (the `dspark-train-serve-parity` skill tells
  the story of how an off-by-one slot silently halved acceptance). Build that
  parity check first.
- **Rollout data is on-policy only for the checkpoint that produced it**: never
  mix rollouts across rounds; tag every record with the draft checkpoint id.
- **Do not use benchmark prompts for the pool** (AIME/GPQA/LCB are the eval);
  take prompts from the nemo782k sources (kimi-mtp, Nemotron stem/math/code)
  held out from SFT, and keep thinking mode off unless the eval is thinking.
- **The target's own sampling**: rollouts at temperature 0 (greedy verify) match
  our eval; the paper also reports T=0.6. Record the sampling params in the
  rollout header.
- **Memory**: the loss needs full-vocab p and q only on the scored positions
  (<= r_m+1 per anchor), far less than SFT's `[anchors x block, vocab]` logits.

## 4. Where things are

- Plan / task list / gates: `examples/train/opd/PLAN.md`.
- Loss math, data formats, API sketches: `.claude/skills/draft-opd/reference.md`.
- Serving patch the rollout logger extends: `examples/train/vllm_patches/`
  (applied in `/import/snvm-sc-scratch1/mengmengj/condaenvs/dspark-host`).
- SFT recipe and the checkpoints to post-train: `examples/train/README.md`
  (conv ep25, block-12 continuation once it lands).
- Cluster mechanics (sc3 B200, FIPS, GPU self-test, eval wrappers): `HANDOFF.md` §0.
