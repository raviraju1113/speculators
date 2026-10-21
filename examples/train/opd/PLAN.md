# Draft-OPD for Gemma-4-31B-it DSpark drafts — implementation plan

Branch `mengmengj/draft-opd` (off `dspark-conv` @ d638e4b). Method and stack
mapping: `.claude/skills/draft-opd/SKILL.md`; loss/data formats/gates:
`.claude/skills/draft-opd/reference.md`. Decision (2026-10-21): **Path A** —
port the algorithm onto speculators + our patched vLLM; first-rejected-only
variant; round-based weight sync.

## Goal and success criterion

Post-train the best SFT checkpoint (conv ep25 now; block-12 continuation if it
wins) with Draft-OPD and beat it on the standard eval at matched k:
target **≥ +0.3 accept_len at k=8** on aime/gpqa/lcb (paper: +8% τ on DFlash),
with tok/s following. Hard floor: no regression on the SFT val split.

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
- [ ] `scripts/opd/rollouts_to_dataset.py`: rollout jsonl -> the trainer's
      arrow format (`input_ids`, `loss_mask` over the response, `token_freq`
      untouched) plus per-row `opd_anchor_pos` / `opd_accepted` lists
      (`anchor_pos = len(prompt_ids) + anchor`); drop steps with
      `accepted == K` and `finish=length`-clipped last steps.
- [ ] Trainer data path: `anchored_block_indices` built from the given anchors
      (`max_anchors` subsampling) instead of sampled ones; `opd_accepted`
      threaded to the loss.
- [ ] **Parity (gate 2)**: student q from the SFT forward at a logged anchor ==
      the serving draft's logits (serve with `--no-async-scheduling`, log the
      block, replay in the trainer forward). Gate 3: greedy re-verification
      from (p, q) reproduces logged `accepted` on >= 99% of anchors.

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

### Deferred / open
- Draft vocab size: 32k leaves 11.2% of GPQA tokens unreachable; OPD cannot fix
  it. Consider 64k or full vocab for the SFT recipe before/alongside OPD.
- True per-step on-policy sync (verl-style) instead of rounds — only if the
  pilot shows round staleness matters.
- Thinking-mode rollouts (paper caps 4096); our eval is non-thinking at 2048.

## Compute budget
Dev/pilot: 1–4 A100-80 (sc3-c98 / sc3-c81) — unaffected by the B200 outage.
Full OPD stage: ~700–1000 B200-hours (paper's 120 H200-h for a 4B target,
scaled by target size), i.e. 4–6 days on one 8-GPU node.

## Log
- 2026-10-21: branch created; skill + plan written; Path A chosen.
- 2026-10-21: Phase 1 done — gate 1 passed (smoke 12450272: 40/40 GSM8K
  rollouts, invariant on 100% of 1,886 steps; conv ep25 at k=8 on A100:
  accept_rate 0.70, 50.8% of blocks fully accepted, first-rejection mass
  8.9% at slot 1 decaying to 4.9% at slot 8). Three vLLM patches
  (scheduler hook, engine-core draft handoff, worker host copy) + the hard
  requirement `--no-async-scheduling` (async mode races the next step).
  Phase 2 simplified: no per-anchor replay prefill needed (see above).
