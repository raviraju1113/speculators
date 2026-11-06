# Draft-OPD on the Gemma-4-31B DSpark draft — status and lessons

Status (2026-11-04): **tried twice, works mechanically, not adopted — closed.** Eight rounds of
on-policy post-training raised acceptance on its own prompt pool by 22% and
lowered it on every standard benchmark by 9–18%. The served draft remains the
SFT checkpoint `conv_blk12/checkpoints/33` at k=12 (main README §0). This doc
is the summary; `PLAN.md` is the chronological log; the paper is
[arXiv 2605.29343](https://arxiv.org/abs/2605.29343).

## 1. What Draft-OPD does

SFT teaches the draft what the target says by reading finished answers. It
never shows the draft its own serving-time mistakes. OPD closes that gap:

1. Take a batch of prompts (16k per round) from a **prompt pool** — questions
   only, no answers.
2. Serve the target with the current draft on them, exactly as in production,
   up to 2048 tokens per answer.
3. Log every block: the 12 tokens the draft proposed, how many the target
   accepted, and the token the target chose at the first rejection.
4. Train the draft one pass over that log. Accepted slots: pull the draft's
   distribution toward the target's (forward KL). First rejected slot: pull it
   toward the token the target chose (reverse KL, weight λ_rej).
5. Save, measure acceptance on the same prompts with the new draft, repeat.

Speed in speculative decoding has two levers, tokens accepted per step
(accept_len, the paper's τ) and cost per step. OPD changes weights, not
architecture, so cost per step is fixed and **accept_len is the whole
effect**. The paper's headline is τ 5.51 → 5.96 (+8%) and 4.33× → 4.86×.

## 2. What we ran and what came out

Full run v2 (`opd_full_blk12_sci16k_v2`, job 14101064, 8×B200, 2 h 15 min):
seed blk12 ep33, k = block = 12, 8 rounds × 16k prompts, λ_rej 0.05 reverse KL,
out-of-vocab rejected slots masked, LR 5e-5 cosine, fully-accepted blocks kept.

**On the pool it works** — paired acceptance at k=12 on the same 16k prompts,
measured by the next round's rollout:

| ckpt | 33 (seed) | 34 | 35 | 36 | 37 | 38 | 39 | 40 |
|---|---|---|---|---|---|---|---|---|
| accept_rate | 0.212 | 0.226 | 0.227 | 0.232 | 0.238 | 0.246 | 0.251 | 0.257 |
| accepted/step | 2.54 | 2.72 | 2.72 | 2.78 | 2.86 | 2.95 | 3.01 | 3.09 |

**On the benchmarks it hurts, from round 1 on** (B200, TP=1, same flags;
`scripts/evaluate/experiments/results/gemma4-31b-dspark-opd-v2`):

| draft | k | AIME acc_len | GPQA | LCB | AIME tok/s | GPQA | LCB |
|---|---|---|---|---|---|---|---|
| blk12 ep33 (seed) | 12 | **7.20** | **4.58** | **5.79** | **395** | **254** | **313** |
| OPD 1 round (ckpt 34) | 12 | 7.17 | 4.56 | 5.74 | 388 | 249 | 307 |
| OPD 7 rounds (ckpt 40) | 12 | 6.29 | 4.21 | 4.79 | 343 | 232 | 257 |
| OPD 8 rounds (ckpt 41) | 12 | 6.19 | 4.21 | 4.73 | 338 | 231 | 255 |
| blk12 ep33 (seed) | 8 | **6.15** | **4.30** | **5.21** | **340** | **239** | **282** |
| OPD 8 rounds (ckpt 41) | 8 | 5.47 | 4.01 | 4.45 | 304 | 225 | 243 |

After 8 rounds: −14% / −9% / −18% accept_len at k=12, −11% / −7% / −15% at k=8,
tok/s down 10–14%. One round is already flat-to-negative, so there is no
stopping point at which OPD on this pool helps. The target's outputs did not
change (total completion tokens within +3% / 0% / +1%; temperature 0, exact
verification).

## 3. Why it dropped

Three things combine; the first is the main one.

**The pool is off-domain.** The pool is 12k science multiple-choice questions
(OpenScienceReasoning-2) + 4k general chat (Nemotron-v2 chat). The benchmarks
are competition math (AIME), graduate science (GPQA) and code (LCB), and the
SFT corpus nemo782k is math/code/STEM-heavy — i.e. the benchmarks are
in-domain for SFT and out-of-domain for the pool. A 5-layer draft has no
spare capacity: it learned the pool's distribution and gave up part of the
SFT one. The paper never meets this because its pool (GSM8K, MATH, AoPS,
CodeAlpaca) and its benchmarks (GSM8K, MATH-500, HumanEval, MBPP) are the
same domains, and it names MT-Bench as its only out-of-domain benchmark.

**Nothing in the loop anchors the draft.** Every training row of every round
is a rollout from the pool; the SFT data is never seen again. GPQA is science
multiple choice — the same format as 75% of the pool — and still fell 9%, so
part of the drop is plain forgetting from the objective, not only domain.
That is the share SFT replay would address.

**The draft's reduced vocabulary breaks the paper's rejected-slot loss.** The
draft predicts over the 32k most frequent tokens, not Gemma's 262k. At the
first rejected slot the signal is "the token the target chose"; in **28% of
those slots that token is outside the 32k** (36% at slot 0). The paper's loss
renormalises the target over the draft vocabulary, so for those slots it
trains toward a token the target never produced. And the rejected term is
~18 nats vs ~0.08 for the accepted term: at the paper's λ_rej = 1 the
gradient is ~200:1 the rejected slot, the noisiest signal in the data. With
the paper's loss as written, pool acceptance fell every round (v1: 0.212 →
0.209 → 0.199). Fixes: out-of-vocab rejected slots dropped from the loss
(`rej_valid`), λ_rej 0.05. These made OPD work on the pool; they do not
touch the domain problem.

## 4. Our setting vs the paper's

| | paper (Draft-OPD, arXiv 2605.29343) | ours (v2) | consequence |
|---|---|---|---|
| target | Qwen3-4B / 8B / 30B-A3B | Gemma-4-31B-it | — |
| draft | DFlash, 5 layers (8 for 30B), block 16, **full vocab** | DSpark, 5 all-sliding layers + DFlash2 conv, block 12, **32k draft vocab** | the out-of-vocab rejected-slot problem is ours alone |
| draft starting point | 6 SFT epochs | 33 SFT epochs (converged, §4.1 control) | less headroom for OPD, more fragile to a high LR |
| prompt pool | 16k: 2k GSM8K train, 5k MATH, 4k AoPS, 5k CodeAlpaca | 16k: 12k OpenScienceReasoning-2 MCQ, 4k Nemotron-v2 chat | **the main difference**: theirs is in-domain with their evals, ours is not |
| pool vs SFT data | disjoint | disjoint (hash-filtered against nemo782k) | same |
| eval benchmarks | GSM8K, MATH-500, HumanEval, MBPP, SWE-Lite; MT-Bench out-of-domain | AIME, GPQA, LiveCodeBench | ours are all out-of-domain for our pool |
| rounds / passes | 8 passes over the pool, ~5.3k steps, batch 24 | 8 rounds, 2,850 steps | matched scale |
| rollout length | 2048 (non-thinking) / 4096 (thinking) | 2048 | same regime |
| loss | forward KL accepted + reverse KL first rejected, λ_rej = 1 | same, λ_rej = 0.05, out-of-vocab rejected slots masked | λ=1 destroyed acceptance here |
| optimizer / LR | AdamW 3e-4 cosine, 5% warmup | Muon + AdamW, 5e-5 cosine | 3e-4 destroyed the draft here (2e-5…5e-5 stable) |
| draft ↔ engine sync | not stated | once per round | — |
| teacher logits | target forward per step | one prefill per rollout (replay-free; exact at temperature 0) | 18 B200 GPU-h for 31B vs ~120 H200 GPU-h for 4B |
| SFT replay in OPD | none | none | — |
| fully-accepted blocks | kept | kept (dropping them was worse) | same |

## 5. What worked, what did not

| tried | result | kept? |
|---|---|---|
| rollout logger + replay-free data path (one prefill per rollout) | exact: teacher argmax == drafted token on 99.7% of accepted slots, == bonus token on 98% of in-vocab rejected slots | yes |
| data-parallel rollouts (one vLLM server per GPU) | 8× rollout throughput; round ≈ 19 min on 8 B200 | yes |
| paper's LR 3e-4 | destroyed the draft (parity 0.645 after ~100 steps) | no → 2e-5…5e-5 |
| paper's loss as written (λ_rej = 1, reverse KL, no vocab mask) | pool acceptance fell every round (0.212 → 0.199 in 2) | no |
| forward KL on the rejected slot | −0.006 vs seed in the pilot (worse than reverse) | no |
| λ_rej 0.05 reverse KL + out-of-vocab rejected slots masked | pilot +0.0034; full run +22% on the pool over 8 rounds, monotone | yes (as the loss) |
| dropping fully-accepted blocks from training | worse: they carry the anchoring signal | no |
| external science/chat pool (sci16k) | +22% on the pool, −9…−18% on the benchmarks | **no** |
| STOP_ON_DROP guard on paired pool acceptance | never fired — pool acceptance is the wrong early-stop signal for this failure | keep, but add a benchmark check |
| 8 rounds before the first benchmark eval | the loss was visible after round 1; 7 rounds wasted | no → eval after every round |

## 6. Plan and order (decided 2026-11-02)

The Gemma DSpark line now exists to produce a recipe that transfers to a SOTA
DFlash2 draft for GLM-5.3. Two tracks; **A first, B only on spare B200 nodes.**

**Track A — OPD retry on blk12 ep33 (done, negative; verdict below).**
Settled the OPD recipe questions (pool domain, replay, rounds, λ_rej); the
answer transfers to any draft class: not for a converged seed.

*Status 2026-11-02 13:45:* code done and unit-tested (commit b1897ea), pool
built (16,000 = 972 GSM8K / 3,200 MATH / 6,828 AoPS / 5,000 CodeAlpaca after
the nemo782k filter removed 87% of GSM8K-train and 57% of MATH-train). Two
arms running, 2 rounds × 16k each, eval after every round:
chained on one node (GLM regen has priority — never touched): arm **c**
`opd_paper16k_assft` (ordinary loss on the same rollouts) job 15215073, then
arm **a** `opd_paper16k_replay` (OPD + 8k replay rows/round) job 15215074,
then arm **b** `opd_paper16k` (OPD, no replay) job 15215075. First attempts
ran on the idle H200 sc3-c127 (round-0 rollouts + ckpt 34 of arm c done
there); at 15:38 the chain was cancelled and resubmitted on the B200
sc3-cp88 when it freed, where c resumed at round 1.

*Moved off c127 (15:40):* arm c round 0 finished on c127 (checkpoints/34, pool acceptance
0.441 at k=12); per request the chain now runs on the B200 pool inside `sn50-hdi-0`
(the preset needs `--reservation`, added with `scontrol update`): arm **c** job 15215073 on
sc3-cp88 (resumed at round 1, same run dir), then **a** 15215074, then **b** 15215075
(`afterany` chain). Note: a second copy of arm c (15214442, submitted 15:37, pinned to c127)
started the moment the original was cancelled and was cancelled again — two jobs on one run
dir would corrupt `rounds/r1`; it had not produced rollouts yet.

*Arm b1, an accident worth keeping (15:05):* the first arm-b job ran with the
paper's **λ_rej = 1** — the preset never set it and the driver default is 1.0
(v2 had passed 0.05 explicitly; `run_rounds.sh` now defaults to 0.05). One
round on the in-domain pool took paired acceptance **0.441 → 0.395** and the
guard stopped it. Same failure as v1 on sci16k (0.212 → 0.209 → 0.199), now
on a pool where the seed is strong: the rejected term (~19 nats vs 0.05 for
the accepted one, 13% of first-rejected slots out of vocab here vs 28% on
sci16k) dominates at λ=1 whatever the pool. Its ckpt 34 is being evaluated as
`b1_34_k12` (run dir `opd_paper16k_lam1`) to see what λ=1 does to the
benchmarks in one round. Arm a's first attempt (15182106) hung at step 0: the replay term
had a data-dependent branch, so ranks with/without replay blocks emitted
different metric keys and the per-key `dist.reduce` desynced (NCCL watchdog
timeout). Fixed branch-free (commit 384feb2); its round-0 rollouts are
reused. Evals on idle 4×A100 nodes (accept_len is hardware-invariant; sc-c96
is excluded: its driver, CUDA 12.7, is too old for the eval env). Results land
in `results/gemma4-31b-dspark-opd-v3`.

*Paired pool acceptance at k=12 (same 16k prompts, measured by the next
round's rollout; seed 0.4413 / 5.30 accepted per step):*

| arm | after round 0 (ckpt 34) | ckpt 35 |
|---|---|---|
| b1: OPD, λ_rej = 1 (accident) | 0.3952 / 4.74 (−10%) — guard stopped it | – |
| c: ordinary loss on the rollouts | **0.4691 / 5.63** (+6.3%) | trained; no pool number (2 rounds = 1 paired measurement) |
| b: OPD λ 0.05, no replay | 0.4598 / 5.52 (+4.2%) | trained |
| a: OPD λ 0.05 + 8k replay rows | 0.4405 / 5.28 (−0.2%, flat) | trained |

*Benchmarks, k=12, accept_len AIME / GPQA / LCB (A100 evals, jobs
15435658–15435668; seed on the same hardware 7.19 / 4.64 / 5.80; results in
`results/gemma4-31b-dspark-opd-v3`):*

| checkpoint | AIME | GPQA | LCB | vs seed |
|---|---|---|---|---|
| c, 1 round (ordinary loss on the rollouts) | 7.20 | 4.63 | 5.76 | 0 / 0 / −1% |
| b, 1 round (OPD λ 0.05) | 7.00 | 4.58 | 5.72 | −3% / −1% / −1% |
| a, 1 round (OPD λ 0.05 + replay) | 6.90 | 4.55 | 5.67 | −4% / −2% / −2% |
| b, 2 rounds | 6.71 | 4.43 | 5.40 | −7% / −5% / −7% |
| a, 2 rounds | 6.50 | 4.39 | 5.27 | −10% / −5% / −9% |
| a, 2 rounds, k=8 | 5.74 | 4.10 | 4.80 | −7% / −5% / −8% (seed k=8 6.16 / 4.30 / 5.19) |
| b1, 1 round (OPD λ 1, accident) | 6.41 | 4.38 | 5.34 | −11% / −6% / −8% |
| c, 2 rounds | 6.88 | 4.46 | 5.52 | −4% / −4% / −5% |
| d: seed blk12 ckpt 28 itself (baseline for arm d) | 6.81 | 4.52 | 5.55 | −5% / −3% / −4% vs ckpt 33 |
| d, 1 round (paper's exact loss: λ 1, γ^(k−1)) on ckpt 28 | 5.97 | 4.16 | 5.09 | **−12% / −8% / −8% vs ckpt 28** |

**Track A verdict (2026-11-04): Draft-OPD does not help this draft, in-domain
pool or not. Closed.**

- The pool gain is real but does not transfer. OPD at λ 0.05 raised paired
  pool acceptance 4.2% in one round (arm b) and lowered every benchmark, by
  1–3% after one round and 5–7% after two. Each round makes it worse, as in
  v2 on the science pool.
- Plain SFT on the same on-policy rollouts (arm c) raised the pool 6.3% and
  left the benchmarks flat after one round (0 / 0 / −1%), then lost 4–5%
  after the second. So continued training on 16k self-generated rollouts
  is itself mildly harmful to a converged draft, and the acceptance-aware
  objective adds to it: at every round OPD (b) is below plain SFT (c) on
  every benchmark, and the gap widens with rounds (round 2: b −7/−5/−7%
  vs c −4/−4/−5%).
- SFT replay (arm a) did not protect the draft: it cancelled the pool gain
  and lost *more* on the benchmarks than no replay. Forgetting of the SFT
  distribution was the wrong model of the v2 drop.
- λ_rej = 1, the paper's setting, is strictly worse again (−6…−11% in one
  round): the ~19-nat rejected term dominates whatever the pool.

Why the paper sees +8% and we see a loss: their seed is a 6-epoch DFlash draft
with headroom; ours is converged (33 epochs; the block-8 control proved 8 more
SFT epochs change nothing). On a converged draft the on-policy objective
trades benchmark acceptance for pool acceptance. Nothing in the loop can fix
that — pool domain, replay and λ were all tried. For GLM-5.3: do not plan on
OPD as a stage; if its DFlash2 draft is trained to convergence the same
result is expected, and if it is stopped early (paper-style) OPD becomes a
question worth one pilot, not a line item.

*Arm d (2026-11-04, job 15659070, done): the paper's exact loss on an earlier
seed.* λ_rej 1, γ 0.8 as γ^(k−1) from the block start (`--opd-rej-decay
paper`), reverse KL, seed **blk12 checkpoint 28** (val 5.29 vs 5.55 at 33),
same pool, 2 rounds, no replay; first-rejected-only and the OOV mask kept
(the paper's own ablation shows first-rejected-only is equivalent; the mask
is required by our 32k vocab). Tests two things at once: whether the two
loss deviations mattered, and whether a less-converged seed has the
headroom the paper's 6-epoch draft has. Baseline = checkpoint 28 itself,
eval job 15658944; results `d_seed28_k12`, `d29_k12`.
**Result (15:20): pool acceptance 0.4206 (seed 28) → 0.3648 after one round
(−13%), guard stopped it** — the same collapse as λ=1 on seed 33 (−10%) and
v1 on sci16k, now with the paper's decay and a less-converged seed. The
paper's exact rejected term is the wrong regime for this draft at either
epoch; the decay changes nothing. **Benchmarks (18:20, A100, jobs 15658944
/ 15661164): checkpoint 28 itself 6.81 / 4.52 / 5.55; after one paper-loss
round 5.97 / 4.16 / 5.09, i.e. −12% / −8% / −8%** — the largest benchmark
loss of any arm, on a seed that did have headroom (ckpt 28 sits 3–5% below
ckpt 33). Headroom was not the missing ingredient; the λ=1 rejected term
is simply destructive for a 32k-vocab draft whose rejected-slot
distribution is far from the target's. Arm d closes the paper-exact
question: no setting of λ, γ, pool, replay or seed epoch turned OPD
positive on benchmarks.

*Arm e / f (launched 2026-11-05 20:30): does OPD work on an under-trained
seed?* The one hypothesis track A did not test. Arm d's "ep28" was not an
early checkpoint: the block-12 run is a warm start from conv ep25, so ep28 is a
draft that had already converged over 25 block-8 epochs, plus 3 epochs of
block-12 settling (its 5% deficit vs ep33 is the new slots, not headroom).

What the paper's seed is: the 6-epoch checkpoint of a run whose own SFT
baseline peaks after 10 epochs (DFlash 5.51 vs Draft-OPD 5.96 on Qwen3-4B),
described as "after an initial SFT warm-up, continued offline SFT quickly
plateaus". No dataset size, LR or batch is given for the seed SFT ("same
mixture as DFlash"), and the paper never varies the seed epoch. So epoch
counts and data sizes cannot be matched across the two setups; the only
usable comparison is *position on the draft's own validation curve*: how far
below the run's final accept_len the checkpoint sits, and whether the curve
is still rising there. The target model matters only through that curve.

Our 26-epoch block-8 run (`gemma4_31b_dspark_nemo782k_conv`, job 11154700,
val accept_len at 8 slots; checkpoint index = log epoch − 1):

| checkpoint | epoch | val accept_len | % of final | saved |
|---|---|---|---|---|
| 1 | 2 | 3.97 | 83% | yes |
| 2 | 3 | 4.07 | 85% | no |
| **8** | 9 | **4.23** | **89%** | yes |
| 9 | 10 | 4.26 | 89% | yes |
| 16 | 17 | 4.43 | 93% | yes |
| 20 | 21 | 4.59 | 96% | yes |
| 25 | 26 | 4.78 | 100% | yes |

Same shape as the paper's description: fast climb to epoch 3, then ~0.02 per
epoch, the last 0.2 from the LR anneal (LR 6e-4 linear decay over 26 epochs).
By the paper's own "on the plateau" criterion the analogue of their epoch-6
seed is somewhere between our checkpoints 8 and 16, not ep28/33.

Why checkpoint 8: past the warm-up, 0.55 below the final value with the curve
still rising, and saved while the LR was still high — the maximum-headroom
case. If OPD cannot beat plain SFT there, no seed will. Checkpoint 16 is the
closer "epoch 6 of 10" analogue and is the follow-up only if 8 shows a gain.

Design: one round each on the paper16k pool at k=8 (seed block 8), LR 5e-5
cosine over 375 steps, λ_rej 0.05 + OOV mask, no replay.
- arm **e** `opd_paper16k_ep8` (OPD) job 15955532, B200 sc3-cp84;
- arm **f** `opd_paper16k_ep8_assft` (ordinary loss on arm e's round-0
  rollouts, copied in) — submitted by the watcher as soon as e's rollouts
  are assembled. **Required**: a checkpoint saved at high LR improves from
  any low-LR training by annealing alone, so OPD must beat f, not the seed.
- baseline: checkpoint 8 itself at k=8, eval job 15955534 (`e_seed8_k8`);
  results `e9_k8`, `f9_k8` in `results/gemma4-31b-dspark-opd-v3`.
Read-out: e9 > f9 on the benchmarks → OPD needs an un-converged seed and the
GLM plan becomes "stop SFT early, then OPD"; e9 ≤ f9 → OPD is off the table
for any seed.

*Status 2026-11-06 09:30.* Arm e finished in 27 min (round of 16k rollouts
at k=8: pool accept_rate 0.4913, 3.93 accepted/step; the seed's own
rollout). **Benchmarks k=8 (A100): seed ckpt 8 5.47 / 3.92 / 4.57 → e9
5.60 / 3.93 / 4.63 (+2.4% / 0 / +1.2%)** — the first OPD arm that does not
lose on benchmarks, with a small gain on AIME and LCB. Whether that is OPD
or just annealing a high-LR checkpoint is exactly what arm f decides. Arm f
(15958487) died in the dataset build: the rollout-reuse path in
`run_rounds.sh` skipped the block that activates the serving env, so
`rollouts_to_dataset.py` had no `datasets` module and the pipe hid the
failure (fixed, commit 82b3a50). Resubmitted as job 16091568 on the same
reused rollouts; `f9_k8` eval follows automatically.

**Track B — a full DFlash2 draft on Gemma (if B200s are spare).** The GLM
draft will be the DFlash2 class (full vocab, selector, DFlash loss) and
nothing we measured was on that class; a Gemma run is the cheap dress
rehearsal. Train from scratch with the stage-4 schedule (block 8 → 12 warm
start), compare with conv ep25 / blk12 ep33 at the same k, then port OPD.
~12 days to an evaluated draft, ~17 with OPD, 8 B200. Needs a Gemma driver
for the DFlash2 class and a vLLM serving path; both are reused for GLM.

Bar to clear is unchanged (main README §0): beat blk12 ep33 at k=12 on
AIME/GPQA/LCB with no loss at k=8. A realistic win is +0.3…+0.5 accept_len,
the paper's +8%; three points is not on the table.

## 7. Pointers

- Log with every job id and diagnosis: `PLAN.md`.
- Driver: `run_rounds.sh` (defaults = v2 recipe), 8×B200 wrapper
  `sbatch_sc3_opd_full.sh`, pilot `sbatch_sc3_opd_pilot.sh`, smoke
  `sbatch_sc3_rollout_smoke.sh`.
- Pool builder: `scripts/opd/build_opd_pool.py`; pool
  `data/opd_prompts/sci16k.jsonl`.
- Loss: `src/speculators/models/dspark/metrics.py` (`opd_loss`), wiring in
  `core.py`; flags `--opd-lambda-rej`, `--opd-rej-kl`.
- Verdict eval: `scripts/evaluate/experiments/gemma4-31b-dspark-opd-v2.yaml`,
  results in `results/gemma4-31b-dspark-opd-v2`.
- Runs: `/import/ml-sc-scratch1/mengmengj/output_opd/opd_full_blk12_sci16k_v2`
  (checkpoints 33–41), v1 `opd_full_blk12_sci16k`, pilots `opd_pilot_*`.
