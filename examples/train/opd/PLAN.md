# Draft-OPD for Gemma-4-31B-it DSpark drafts — implementation plan

Branch `mengmengj/draft-opd` (off `dspark-conv` @ d638e4b). Method and stack
mapping: `.claude/skills/draft-opd/SKILL.md`; loss/data formats/gates:
`.claude/skills/draft-opd/reference.md`. Decision (2026-10-21): **Path A** —
port the algorithm onto speculators + our patched vLLM; first-rejected-only
variant; round-based weight sync.

## Goal and success criterion

Post-train the best SFT checkpoint with Draft-OPD and beat it on the standard
eval at matched k. Seed (decided 2026-10-26): **`conv_blk12/checkpoints/33`**
(block 12, README §0/§4.1) — `run_rounds.sh` now defaults to it, with k, block
size and γ taken from the seed's config (12/12/12); the 2026-10-21/22 pilots
and sweeps were on conv ep25 at k=8. Target **≥ +0.3 accept_len at k=12** (and
no loss at k=8) on aime/gpqa/lcb (paper: +8% τ on DFlash), with tok/s
following. Hard floor: no regression on the SFT val split.

## Phases

### Phase 1 — Rollout logger (vLLM hook)  ~2–3 days, 1 A100
Emit per draft step `(request id, step, draft_ids[K], accepted)` from the
serving path, assembled into the rollout jsonl of `reference.md` §2.
- [ ] Locate in the eval env's vLLM: where the DSpark speculator's block
      (`_generate_draft`) is stored for the next step and where the rejection
      sampler's accepted counts and request ids are available in the model
      runner (`v1/worker/gpu/model_runner.py`).
- [ ] `vllm_patches/dspark_rollout_log.patch`: env-gated (`DSPARK_ROLLOUT_LOG`)
      host-side logger, async jsonl writer, no change to the CUDA-graph path.
- [ ] `scripts/opd/assemble_rollouts.py`: join per-step rows with the API
      server's outputs (prompt/response ids, finish reason) into rollout records;
      assert the invariant `response_ids[a:a+r] == draft_ids[:r]`.
- [ ] `scripts/opd/rollout.py`: drive a prompt pool through the server
      (concurrency N, temperature 0, max_tokens 2048) and produce one jsonl.
- Gate 1: invariant holds on 100% of steps over 200 prompts; first-rejection
  histogram printed (App. D style) — also a useful diagnostic on its own.

### Phase 2 — Replay data path  ~2–3 days, 1–2 A100 (simplified 2026-10-21)
**Key simplification (first-rejected-only):** every teacher distribution the
loss needs is in ONE prefill of the verified rollout `prompt + output_ids`:
- accepted slots `k <= r`: the drafted token equals the rollout token, so the
  target state at `anchor_pos + k - 1` is exactly the replay state;
- the first rejected slot `r + 1`: `p(. | c, d_1..d_r)` is the target
  distribution at the last accepted position — the state that sampled the
  bonus token in the rollout (same tokens, causal model, identical state).
So no per-anchor re-prefill and no prefix-caching work: replay data costs the
same as SFT data, and the hidden-state server runs unchanged. Teacher forcing
matches serving on every scored slot (the previous token of each scored slot
is a verified token), so the Markov head sees the same inputs as at serve
time. Only the full-tail variant (slots > r+1 conditioned on rejected draft
tokens) would need extra target forwards — deferred.
- [x] (design) above; the rollout record already carries `anchor` and
      `accepted` per step.
- [x] `scripts/opd/rollouts_to_dataset.py`: rollout jsonl -> the trainer's
      arrow format (torch-formatted; `input_ids`, `loss_mask`, `seq_len`,
      `messages`) plus per-token `opd_anchor_mask` / `opd_accepted`
      (`anchor_pos = plen + anchor - 1`, the last verified token); copies the
      checkpoint dataset's `token_freq.pt` + `d2t/t2d`; drops full-accept
      blocks, `finish=length` tails and blocks that do not fit.
- [x] Trainer data path: `select_given_anchors`, `anchor_mask` through
      `_build_attention_mask`; `opd_accepted` -> per-block counts -> loss.
- [x] **Gate 2 passed** (pilot C, job 12488003, conv ep25, A100):
      `opd_acc_argmax_match = 0.997` on 957 accepted slots; forward KL on
      accepted slots 0.04, reverse KL on first-rejected slots 12.9. Control B
      (same rows, random anchors, SFT loss): slot-0 acc 0.95, accept_len 6.7.
      Gate 3 (greedy re-verification from p, q) is implied at this level and
      deferred.
- [x] (non-issue) the first micro-step's low tokens/s is its 25 s fetch wait
      on the hidden-state server for the first items; the batch is full.

### Phase 3 — Loss and trainer integration  ~2 days, 1 A100
- [ ] `src/speculators/models/dspark/opd_loss.py`: forward KL on accepted
      positions, γ-weighted reverse KL on the first rejected position, on the
      draft-vocab support via `d2t` with an out-of-vocab "rest" bucket; λ=1/1,
      γ=0.8; unit tests against a dense reference.
- [ ] Trainer: `--loss-fn opd` path that consumes replay samples (fixed anchors,
      given block tokens, accepted mask) instead of sampled anchors + mask
      blocks; confidence head unchanged; `--dflash-decay-gamma` not applied to
      OPD positions.
- [ ] `--from-pretrained`-style init from the SFT checkpoint with a fresh
      optimizer (AdamW 3e-4 cosine, 0.05 warmup per the paper) — the trainer's
      resume path restores the optimizer, so add a `--reset-optimizer` or copy
      only `model.safetensors` + config into the run dir.

### Phase 4 — Round loop + pilot  ~2–3 days, then 4 A100 or 8 B200/H200
- [ ] `examples/train/opd/run_round.sh`: start rollout server on ckpt[r] →
      rollout P prompts → stop → replay → train N steps → ckpt[r+1]; sbatch
      wrapper for sc3 with the usual gates (wandb, GPU self-test).
- [ ] Prompt pool: held-out prompts from the nemo782k sources (kimi-mtp,
      Nemotron stem/math/code), none of the eval benchmarks; `data/opd_prompts/`.
- [ ] Pilot: 1K prompts × 2 rounds on conv ep25; gate 4 (val accept_len ≥ SFT −
      0.05; first-rejection histogram shifts right).
- [ ] Full run: 16K prompts × 8 passes (paper), ~5 days on 8 GPUs; then the
      standard eval (`sbatch_sc3_eval.sh`) + ctx sweep vs the SFT checkpoint.
      Sizing on 8 B200 (4 HS engines + 4 ranks, `run_rounds.sh` env): one
      micro-step is one 8192-token packed sequence ≈ 10–13 rollouts of ~600
      tokens, so 4 ranks × ACCUM 1 ≈ 45 prompts per optimizer step (paper:
      24 prompts; ACCUM 1 with 2 ranks would match it). 16K × 8 passes / 45 ≈
      2,850 optimizer steps → `SCHED_TOTAL≈2850`, cosine, warmup 0.05,
      lr 3e-4 (paper). Rounds: 8 (one pass of the pool per round, P=16000);
      rollouts at concurrency 32 on one B200 ≈ 1–1.5 h per round.

### Deferred / open
- Draft vocab size: 32k leaves 11.2% of GPQA tokens unreachable; OPD cannot fix
  it. Consider 64k or full vocab for the SFT recipe before/alongside OPD.
- True per-step on-policy sync (verl-style) instead of rounds — only if the
  pilot shows round staleness matters.
- Thinking-mode rollouts (paper caps 4096); our eval is non-thinking at 2048.

### Finding (2026-10-21, pilot `opd_pilot_rounds2`, job 12492724)
2 rounds x 1,000 nemo782k prompts on 3 A100 ran end to end (1 h 10 min), but
training HURT the draft: round-1 rollout accept_rate 0.41 -> 0.26 (full
blocks 19.6% -> 3.8%), val accept_len 2.57 -> 1.95, parity metric 0.996 at
step 1 -> 0.645 after ~100 steps. Config was fresh AdamW warmed to 3e-4 with
~10 prompts per step (1 rank, ACCUM 1) and a loss dominated by the
reverse-KL term (13-22 nats) over a 0.04 forward-KL term, with full-accept
blocks dropped -> almost no anchoring signal, high LR. Next: sweep on the
round-0 dataset (no rollouts): LR 2e-5 + ACCUM 8, keep-full-accept blocks,
lambda_rej in {1, 0.3}; criterion: parity >= 0.99 and val accept_len not
below the pre-training value after one epoch; then redo 2 rounds with the
best setting and check that round-1 rollout acceptance RISES.

### Sweep result (2026-10-22, jobs 12515333/336/337/338, round-0 data, 1 epoch = 91 steps)
All at LR 2e-5, ACCUM 4, constant schedule, fresh optimizer, weights-only seed:

| variant | val accept_len | val parity | val rej-KL | note |
|---|---|---|---|---|
| V4 λ=0 (control, accepted-slot fwd KL only) | 4.163 | 0.990 | 13.97 | stable |
| V1 λ=1, keep full-accept | 4.114 | 0.978 | 13.68 | stable |
| V2 λ=0.3, keep full-accept | 4.119 | 0.979 | 13.67 | stable |
| V3 λ=1, drop full-accept | 3.105 | 0.971 | 13.70 | different (harder) val set; worse |

Reading: at this LR the loss is stable (no damage) but one epoch on 1K prompts
moves nothing measurable — the rejected-slot term barely decreases. Decision:
keep full-accept blocks (default now in `run_rounds.sh`), λ=1, LR 5e-5 as the
starting point for the next multi-round run; judge by round-to-round rollout
accept_rate. **No further runs until the B200 pool is accessible** (decision
2026-10-22); engineering continues: `sbatch_sc3_opd_full.sh` preset,
prompt-pool token filter, docs.

## Compute budget
Dev/pilot: 1–4 A100-80 (sc3-c98 / sc3-c81) — unaffected by the B200 outage.
Full OPD stage: ~700–1000 B200-hours (paper's 120 H200-h for a 4B target,
scaled by target size), i.e. 4–6 days on one 8-GPU node.

## Log
- 2026-10-21: branch created; skill + plan written; Path A chosen.
- 2026-10-28 pilots (2 x 4k, paired seed baseline on round-1 prompts 0.2112;
  round 0 reproduced the seed's 0.2149 in both): **A λ=0.05 reverse: round-1
  0.2146 (+0.0034)**; B forward KL on the rejected slot: 0.2050 (−0.006,
  worse — matches the paper's "mixed > all-forward"). A's val forward-KL on
  accepted slots still drifts up (0.086 -> 0.14 -> 0.20), so the full run v2
  (`opd_full_blk12_sci16k_v2`, λ=0.05, OOV mask, LR 5e-5, 8 x 16k, round-0
  rollouts reused from v1) runs with STOP_ON_DROP: a round whose paired
  acceptance falls >0.002 below the previous round ends the run.
- 2026-10-28: **the paper's objective as implemented degrades the draft.**
  Full run 13949602 (blk12 ep33, sci16k, k=12, LR 5e-5, λ=1, reverse KL on
  the first rejected slot): rollout accept_rate 0.2117 (ckpt 33) -> 0.2090
  (34) -> 0.1987 (35); val forward-KL on accepted slots 0.29 -> 0.49. Both
  runs that night died on $HOME quota (vLLM usage stats, then flashinfer JIT);
  all caches now live on scratch and the sampler JIT is off. Diagnostics
  (`opd_diag_r0`, LR≈0): teacher alignment is correct — teacher argmax ==
  drafted token on 99.7% of accepted slots and == bonus token on 98% of
  in-vocab rejected slots (gate 3, now a logged metric) — but **28% of
  first-rejected slots have a bonus token outside the 32k draft vocab** (36%
  at slot 0; renormalized targets make reverse KL train toward a token the
  target never produced), and the rejected term is **18 nats vs 0.08** for the
  accepted term, i.e. the gradient is ~200:1 the rejected slot. Fixes: OOV
  rejected slots dropped from the loss (`rej_valid`), `--opd-rej-kl
  {reverse,forward}`. Pilots (2 x 4k prompts, paired baseline at ckpt 33 on
  prompts 4000-7999 = 0.2112): A `opd_pilot_lam005` (λ=0.05, reverse),
  B `opd_pilot_fwd` (λ=1, forward KL on the rejected slot). Resume bug fixed
  (round r skipped when its checkpoint exists).
- 2026-10-27: B300 smoke (sc3-c130, 1 GPU, k=12, blk12 ep33): env runs on
  B300 unchanged; 40/40 GSM8K rollouts, invariant 100%, accept_rate 0.59.
  Prompt pool: Nemotron-v2 stem is entirely inside nemo782k (354,974/355,000),
  so the pool is `data/opd_prompts/sci16k.jsonl` = 12k OpenScienceReasoning-2
  + 4k Nemotron-v2 chat, hash-disjoint from nemo782k (`build_opd_pool.py`).
  Pilot 13936853 (8 B200, 2 x 1k, LR 5e-5): round-0 rollout accept_rate 0.2194
  -> round-1 0.2192 (no damage, no movement in 24 steps; parity 0.996 at step 1,
  rej-KL 15.4 -> 13.8). Fixes: data-parallel rollouts (8 servers), wandb
  allow_val_change on the hparams setter. Test2 13944090 passed 2 rounds.
  **Full run launched: job 13944091, sc3-cp84, `opd_full_blk12_sci16k`, 8 rounds
  x 16k, LR 5e-5 cosine 2850 steps, wandb `opdfullblk12v1`.** Block-8 control
  12398695 running on sc3-cp83 (moved into the reservation).
- 2026-10-26: seed switched to block-12 ep33 (k=12); README §0 defines the
  cumulative recipe and the bar OPD has to clear to become its third stage.
- 2026-10-21: Phase 1 done — gate 1 passed (smoke 12450272: 40/40 GSM8K
  rollouts, invariant on 100% of 1,886 steps; conv ep25 at k=8 on A100:
  accept_rate 0.70, 50.8% of blocks fully accepted, first-rejection mass
  8.9% at slot 1 decaying to 4.9% at slot 8). Three vLLM patches
  (scheduler hook, engine-core draft handoff, worker host copy) + the hard
  requirement `--no-async-scheduling` (async mode races the next step).
  Phase 2 simplified: no per-anchor replay prefill needed (see above).
- 2026-10-21: Phase 2 + loss done; end-to-end pilot through the real driver
  on 2 A100 (hidden-state server + 1 rank). Two data-path bugs found by the
  A/B/C runs and fixed: dataset must be torch-formatted (state.json), and the
  draft-vocab mapping must be the checkpoint's `d2t/t2d` (rebuilt from
  token_freq it differs -> slot-0 acc 0.00); anchor offset corrected to the
  last verified token. Gate 2 passed (argmax match 0.997).
