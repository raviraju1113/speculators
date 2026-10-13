---
name: dspark-results-status
description: "DSpark official runs, headline numbers, all-sliding architecture verdict, and open threads as of 2026-09-14"
metadata:
  node_type: memory
  type: project
  originSessionId: 976456e9-265b-4d6a-afd3-854bb7665b28
  modified: 2026-09-21T15:56:58.794Z
---

Run table + commands now live in speculators/examples/train/README.md (rewritten
2026-09-14; cleanup commit 107688e archived cluster-era scripts). Branch
mengmengj/dspark-upstream pushed to origin (raviraju1113/speculators fork).

**Headline results**
- Checkpoint of record: nemo continued `checkpoints/24` (val 4.69) — k8 aime
  5.89 / gpqa 4.43 / lcb 5.25, 4.1–5.4× vs 59.5 tok/s TP=1 baseline.
- Scratch nemo782k ep25 == continued at matched exposure (curriculum
  irrelevant): k8 aime 5.99 / gpqa 4.22 / lcb 5.07.
- flag=True (sample_from_anchor) wins all matched comparisons; k=8 best;
  upstream vLLM main derives dspark_bonus_anchor from checkpoint config
  (deployment: recent vLLM + aime accept-gate ≥~5.5).

**Architecture verdict (2026-09, meeting 2026-09-15): all-sliding draft.**
AA-LCR paired bins 1k–112k (results/lcr_bins_*, plot_lcr_bins.py):
- all-sliding scratch: flat/rising 3.80→3.98 (k8), 3.76→3.82 (k7).
- fullattn2 (layers {0,3} full): identical in-window, decays to ~3.0 at 112k;
  ep15 == ep25 point-for-point → NOT undertraining. In-window bench parity
  confirmed (fullattn2 ep25 k8: 6.06/4.25/5.13 vs scratch 5.99/4.22/5.07).
- Mechanism: draft full layers get the LOCAL θ=10k RoPE (dflash core.py
  flattens rope_parameters to sliding_attention variant — logged warning);
  wavelengths top out ~63k, aliasing beyond; plus softmax dilution over 112k
  keys trained on ≤8k. Global context arrives pre-integrated in fused target
  hidden states (aux layers 17/29/47 are Gemma's full-attn layers, θ=1M).
- Ecosystem: RedHat Gemma DSpark = all-sliding 2048 (same as ours); RedHat
  Gemma DFlash = all-full + unscaled θ=10k (predicted to decay); RadixArk
  Kimi-K3-DSpark = all-full + YaRN ×16→1M (the "full attention done right"
  counterexample; Kimi target is all-full-attention). DSpark paper (DeepSeek,
  arXiv:2607.05147) uses full-attn DFlash backbone but only short-context evals.
- Principle: draft should only do attention it can do in-distribution at every
  serve length; delegate global context to target via hidden states.

**YaRN-at-serve result (2026-09-15, job 239): positional encoding was the
dominant failure.** fullattn2 ep25 served with flat zero-shot YaRN (factor 16
over original 8192; checkpoint `checkpoints/25_yarn` = symlinked weights +
edited config) recovers long context almost completely: k8 112k = 3.93 (was
3.00 unscaled; all-sliding 3.98). Small in-window tax at 1k-4k (~-0.15,
3.65@1k vs 3.79) = flat-YaRN cost on sliding layers (1k bin is the control).
k3 pass done 2026-09-15: 2.55@1k -> 2.50@112k (flat; unscaled ep25 k3 was
2.75 -> 2.42), so k3 pays a larger in-window tax (-0.2) but no longer decays.
Both YaRN series are in results/lcr_bins_accept_len.png (regenerated
2026-09-16; title now states the YaRN rescue). plot_lcr_bins.py defaults to
k8+k3 and has `--matched-k` (k7+k3 -> lcr_bins_accept_len_k7.png) ready for
the RedHat DFlash run. Run it with envs/dspark-host python (system python has
no matplotlib). Plot edits from 2026-09-16 are UNCOMMITTED.
Implication for Kimi-DSpark options: option 3 (mixed arch, short data,
serve-time YaRN) is LIVE -- attention-statistics OOD is secondary. Constraint
found: neither train code nor vLLM qwen3_dflash.py supports per-layer-type
rope in the draft (one flat rope_parameters, one shared rotary) -- mixed
sliding+full with per-type rope needs small patches in both. Kimi plan:
colleague is testing RadixArk Kimi-K3-DSpark checkpoint on long context;
options = (1) all-sliding no yarn, (2) full-attn + yarn + long data
(RadixArk recipe), (3) mixed + serve-time yarn on current data.

**Per-layer-type YaRN (2026-09-17): dspark-host venv PATCHED** (backups *.orig
next to each file): vllm/model_executor/models/qwen3_dflash.py (resolves
rope_parameters per layer_types entry; fused context-KV RoPE falls back to a
per-layer loop when cos/sin tables differ) and
vllm/transformers_utils/configs/speculators/base.py (transformers 5.14
convert_rope_params_to_dict flattens a nested dict on generic PretrainedConfig
-- SpeculatorsConfig now strips the strays and re-standardizes). Flat configs
resolve exactly as before. Checkpoint dir
fullattn1/checkpoints/25_yarn_full = symlinked weights + config with
transformer_layer_config.rope_parameters = {"full_attention": yarn x16 orig
8192, "sliding_attention": default} (HF v5 nested form). Bins TAG
fullattn1_ep25_yarn_full; plot view `--fullattn1`. Same trick for fullattn2
would be checkpoints/25_yarn_full with layers {0,3} yarn.

**Open threads**
- RedHat DFlash LCR bins run STILL not submitted (external replication of
  decay; predicted to decay like our unscaled fullattn2):
  `CKPT=/sms-scratch/checkpoints/gemma-4-31B-it-speculator.dflash
  TAG=redhat_dflash METHOD=dflash K_MAIN=7 SKIP_BASE=1 sbatch
  examples/train/sbatch_lcr_bins.sh`; then add --matched-k mode to
  plot_lcr_bins.py (k7+k3 only) and regenerate.
- Commit 25429ef unpushed (no git credentials on this node; push from
  bastion): `git push origin mengmengj/dspark-upstream`.
- fullattn1 (1 full layer {0} + 4 sliding, WSD 26ep) training DONE 2026-09-15
  22:45 (job 232). checkpoints/25 == checkpoint_best; val accept_len 4.50 vs
  fullattn2 4.75 (small train-side gap). Evals prepared 2026-09-16, user
  submits from bastion: `CONFIG=gemma4-31b-dspark-fullattn1-ep25.yaml sbatch
  examples/train/sbatch_eval.sh` and `CKPT=.../fullattn1/checkpoints/25
  TAG=fullattn1_ep25 SKIP_BASE=1 sbatch examples/train/sbatch_lcr_bins.sh`
  (unscaled RoPE, no YaRN). Node 098 cannot reach the SLURM controller and has
  no bastion hostname resolvable -- sbatch only from bastion; user wants
  commands as `sbatch --export=ALL,VAR=..` lines.
  RESULTS 2026-09-16: benches (job 251) k8 aime 5.76 / gpqa 4.11 / lcb 4.88
  (fullattn2 6.06/4.25/5.13, scratch 5.99/4.22/5.07) -> fullattn1 ~0.2-0.3
  BELOW both in-window (matches val gap; WSD vs linear is a confound). LCR
  bins (job 252): k3 FLAT/rising 2.78@1k -> 2.86@112k (fullattn2 k3 decays to
  2.42; all-sliding k3 2.95) -- one unscaled full layer did NOT decay at k3.
  k8 rerun (GPU_UTIL=0.88) done 2026-09-17: fullattn1 k8 unscaled is FLAT
  3.66@1k -> 3.80@112k (fullattn2 k8 decays 3.79 -> 3.00). One full layer at
  index 0 is NOT load-bearing for position -- weakens "any full layer decays".
  Per-layer YaRN (25_yarn_full, done 2026-09-21): k8 3.64 -> 3.97 (matches
  all-sliding 3.98 at 112k, +0.17 over unscaled at 96k/112k), k3 2.76 -> 2.83
  (-0.03..-0.08 vs unscaled, small). In-window tax at 1k: -0.02 (k8) vs -0.15
  for flat YaRN on fullattn2 -> the flat-YaRN tax was mostly on sliding layers
  (caveat: different checkpoint). Views regenerated; no-yarn title updated. Plot views: default = YaRN story (hides
  fullattn1); `--no-yarn` = architecture story incl. fullattn1, no YaRN ->
  lcr_bins_accept_len_noyarn.png; `--matched-k` = k7 for RedHat.
- PR to Ravi: cherry-pick train.py:527 distributed-unpack fix (from 189a77c)
  onto origin/main — main crashes on every training launch since #711.
- origin/main is 26 commits ahead of our last merge (AgentX eval, eagle3,
  "64k/128k inference support"); merge AFTER the meeting, gated by in-container
  `scripts/train.py --dry-run` (last merge silently broke schema Literal +
  distributed contract).
- MLA-draft 3-phase plan; block=12 ablation candidate; AgentX concurrency
  replay for perf team.

2026-08-19 disk incident + shm guard, node sick list (097/099 full /tmp, 101
bad GPUs), wandb/netrc/linger rules: see examples/train/README.md and
[[b200-box-setup]] [[dspark-run-recipe]].
