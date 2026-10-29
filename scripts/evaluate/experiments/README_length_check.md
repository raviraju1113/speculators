# Response-length / end-to-end check for DSpark drafts (2026-10-28)

Our own notes for the `gemma4-31b-length-check*.yaml` runs. The harness
itself is documented in [README.md](README.md) (config-driven runner) and
[../mtp_server_eval/README.md](../mtp_server_eval/README.md) (metrics); this
file is the *why* and the *reading guide* for one specific question.

## The question

A colleague observed, with a diffusion-style (block / parallel) draft, that
acceptance length and decode tok/s look good but **the target's replies get
longer**, so end-to-end time does not improve. Two sub-questions:

1. Does serving Gemma-4-31B-it with our DSpark drafts (and with Google's own
   MTP "assistant" draft) change *what* the target says, not just how fast?
2. If replies do get longer, is that a property of block/diffusion drafts in
   general, or a verification / numerics artifact?

### What the theory says

- **Greedy verification is lossless modulo numerics.** vLLM's DSpark path
  accepts a draft token iff it equals the target's argmax at that position
  (the confidence head is not used at serve time), so under `temperature 0`
  the baseline and the speculated run must produce the same token sequence.
  Any divergence comes from a different numerical path (batched verify of
  k+1 positions vs one-token decode), and once two runs diverge they are
  different samples of the same model, equally likely to be longer or
  shorter. A *systematic* lengthening under greedy would be a real bug.
- **Sampled verification is lossless in distribution** (rejection sampling
  reproduces the target's token distribution), but each reply's length is a
  random draw. A draft can only bias the length if the implementation is
  wrong; otherwise the lengths differ per prompt but match on average.
- So the expected result is: same mean length, balanced per-prompt
  direction, lower wall-clock. The colleague's effect, if real, should show
  up as a one-sided tok_ratio > 1 in the tables below.

### Why the existing tables cannot answer it

Every earlier eval (`gemma4-31b-assistant-ref`, `-dspark-conv-ep25`,
`-dspark-conv-blk12-ep33`, ...) ran **greedy with max_tokens 2048 and
max_model_len 4096**, and recorded completion tokens + e2e seconds but not
the finish reason. Reading those files back (compare_lengths.py on the stored
details) gives:

| k=8 draft | AIME tok ratio | GPQA | LCB | >1.5x / <0.67x on LCB |
|---|---|---|---|---|
| assistant (Google MTP) | 0.99 | 1.02 | 1.01 | 2.1% / 1.3% |
| conv ep25 (DSpark+DFlash2) | 0.99 | 1.02 | 1.00 | 2.2% / 1.9% |

Within a few percent, balanced direction, same for both draft types. But the
cap censors exactly the prompts that matter: ~10% of the baseline's LCB
replies ran into the 2048 cap, and a draft that turned a 2k reply into a 10k
runaway would be invisible. And greedy is not how the model is served.
The colleague's observation assumes **no cap**, so the rerun raises it.

## The rerun

Same prompts, same harness, paired per prompt. Changes vs the old tables:

| setting | old tables | length check (greedy) | length check (sampled) |
|---|---|---|---|
| yaml | `gemma4-31b-{assistant-ref,dspark-conv-ep25,dspark-conv-blk12-ep33}` | `gemma4-31b-length-check.yaml` | `gemma4-31b-length-check-sampled.yaml` |
| max_tokens / max_model_len | 2048 / 4096 | **16384 / 20480** | 16384 / 20480 |
| temperature | 0 | 0 | **1.0, top_p 0.95, top_k 64** (gemma generation_config; = what vLLM serves when the client sends nothing) |
| prompts | full (30 / 198 / 1055) | full | 100 per benchmark (aime all 30), same subset for every experiment (seed 42) |
| draws per prompt | 1 | 1 | **3** (`repeats: 3`, request seed 1234+rep) |
| hardware | B200 (blk12: A100) | B200 TP=1 | B200 TP=1 |
| experiments | — | baseline, assistant_k8, conv_ep25_k8, blk12_ep33_k12 | same |

A literal "no cap" is impossible (vLLM needs max_model_len, and a true runaway
never stops), so the cap is raised 8x above the old one and far above any
natural reply, and **cap hits become a metric**: a config that hits the cap
more often than the baseline is producing runaways.

Why both greedy and sampled: greedy is the controlled experiment (lengths must
match; any drift is an artifact of the speculative path). Sampled is the
production-relevant one and the most likely place the colleague saw the
effect; three draws per prompt separate a length bias from sampling noise.
**Open point: if production serves Gemma at a different temperature, rerun
the sampled yaml with that value** (the fields are `eval.temperature/top_p/top_k`).

### Added metrics

`run_vllm_eval.py` now records per request `finish_reason` (`length` = cap
hit) and `rep`, and per benchmark `mean_completion_tokens`, `cap_hits`,
`cap_hit_rate`, `total_e2e_s`, `mean_e2e_s`, plus the sampling settings.
The old fields (`decode_tok_s`, `accept_length`, ...) are unchanged, so
`compare_speedup.py` still works.

`../mtp_server_eval/compare_lengths.py` pairs every prompt between the
baseline and each config (repeats averaged) and prints per benchmark:

| column | meaning |
|---|---|
| tok_ratio | sum of config tokens / sum of baseline tokens (1.0 = same length) |
| med_ratio | median per-prompt ratio |
| >1.5x / <0.67x | share of prompts much longer / much shorter than baseline; balanced = numerics, one-sided = bias |
| cap b->c | cap hits baseline -> config |
| e2e_ratio | config wall-clock / baseline wall-clock (< 1 = faster end to end) |
| tok/s | end-to-end throughput |
| nocap_n / nocap_r | the same tok_ratio over prompts that hit the cap in neither run |

### How to read the outcome

- `tok_ratio ~ 1`, >1.5x and <0.67x shares similar, cap hits equal:
  the draft does not change lengths; e2e_ratio is then the honest speedup.
- `tok_ratio > 1` with a one-sided >1.5x share or extra cap hits under
  **greedy**: a verification / numerics bug in the DSpark serving path
  (look at the divergent prompts' outputs first).
- Same under **sampled** only: the rejection-sampling path is biased;
  compare against assistant_k8, which uses vLLM's stock MTP path, to tell a
  DSpark-specific problem from a vLLM-wide one.
- Greedy cap hits grow for *every* config including baseline: that is the
  model's own runaway rate at 16k, not a draft effect.

## What the literature says (checked 2026-10-28)

No paper reports a *systematic* lengthening of replies under speculative
decoding, for diffusion/block drafts or any other kind. What is documented:

- **Greedy divergence from numerics, not bias.** Verifying k+1 positions in
  one batch runs different kernels / reduction orders than one-token decode,
  so near-tied argmaxes flip and the two runs then continue as different
  samples of the same model. Koziev et al. (arXiv 2609.15504, *Orthrus*, a
  hybrid AR-diffusion drafter): bf16 exact-trajectory match on only ~45% of
  1,190 prompts, FP32 matches on all, no downstream degradation. The
  llama.cpp issue #25618 shows the same for `draft-dspark`, `draft-dflash`
  and `draft-mtp` alike, worse on quantized targets; lengths move both ways
  (50 vs 56 tokens, 301-374 token spread). A production write-up (vLLM,
  8B target + 1B draft, 2,000 prompts) saw 1.2% exact-match failures.
  Our old tables agree: ±2% paired length, balanced direction.
- **Stop-condition bugs that DO lengthen outputs, and scale with draft length.**
  Yang (Zenodo preprint, Oct 2026, *The "Lossless" Illusion*): logit
  processors see only committed tokens, so `min_tokens`, stop strings and
  thinking budgets inside an accepted batch are missed; on Qwen3-0.6B / vLLM
  V1 that produced duplicated `</think>` and replies running past the greedy
  stop in a few % of cases. Plus six vLLM / one SGLang distribution bugs in
  the sampled path. Synthetic / small-model evidence, single author, but the
  mechanism is real and longer blocks make it more likely. **If a colleague's
  setup used min_tokens, stop strings or a thinking budget, check this first.**
- **Lossy verification changes outputs by design.** Wang et al. (arXiv
  2607.26627): typical acceptance / threshold / interpolated verification
  loses accuracy, more on harder tasks (+0.4 pp GSM8K to +6.7 pp AIME gap).
  Not our case: vLLM's DSpark path is exact argmax / rejection sampling, the
  confidence head is unused at serve time, and the DSpark paper (arXiv
  2607.05147) requires the non-anticipating property for losslessness.
- **Nobody reports mean reply length.** DFlash (arXiv 2602.06036) and DSpark
  report acceptance length and tok/s only, with a 2048 cap; neither prints
  completion tokens or wall-clock vs baseline. That is the gap this check
  fills.

## Results (jobs finished 2026-10-28, read 2026-10-29)

Raw: `results/gemma4-31b-length-check{,-sampled}/<exp>/mtp_eval_{summary.json,details.jsonl}`
and `compare_lengths.json` next to them. Every table below is raw numbers
with the baseline as its own row: **total tok** = sum of completion tokens
over all replies, **mean tok** = per reply, **cap** = replies cut at 16384,
**wall-clock** = summed end-to-end seconds for the whole benchmark (the
number that answers "is it faster end to end"), **dec tok/s** = decode
throughput, **accept_len** = accepted tokens per step incl. the bonus token.

### Greedy (temperature 0, full datasets, B200 TP=1)

AIME, 30 prompts:

| config | total tok | mean tok | cap | wall-clock s | dec tok/s | accept_len |
|---|---|---|---|---|---|---|
| baseline (no draft) | 47,200 | 1,573 | 0 | 756 | 62.5 | – |
| assistant_k8 (Google MTP) | 43,309 | 1,444 | 0 | 169 | 258.8 | 5.68 |
| conv_ep25_k8 (DSpark+conv) | 38,594 | 1,286 | 0 | 114 | 343.1 | 6.11 |
| blk12_ep33_k12 (DSpark+conv) | 60,787 | 2,026 | 1 | 169 | 363.3 | 7.43 |

GPQA-Diamond, 198 prompts:

| config | total tok | mean tok | cap | wall-clock s | dec tok/s | accept_len |
|---|---|---|---|---|---|---|
| baseline | 159,369 | 805 | 1 | 2,531 | 63.1 | – |
| assistant_k8 | 177,714 | 898 | 2 | 750 | 239.7 | 5.35 |
| conv_ep25_k8 | 164,975 | 833 | 1 | 656 | 254.7 | 4.55 |
| blk12_ep33_k12 | 161,804 | 817 | 1 | 660 | 248.3 | 4.66 |

LiveCodeBench, 1,055 prompts:

| config | total tok | mean tok | cap | wall-clock s | dec tok/s | accept_len |
|---|---|---|---|---|---|---|
| baseline | 1,212,945 | 1,150 | 7 | 19,043 | 63.8 | – |
| assistant_k8 | 1,285,835 | 1,219 | 8 | 5,457 | 238.2 | 5.39 |
| conv_ep25_k8 | 1,286,868 | 1,220 | 7 | 4,810 | 270.6 | 5.33 |
| blk12_ep33_k12 | 1,277,000 | 1,210 | 10 | 4,367 | 296.1 | 5.95 |

Per-prompt view on LCB (paired by prompt id): replies more than 1.5x the
baseline's length: 4.5% / 5.4% / 5.1% of prompts (assistant / conv / blk12);
replies shorter than 2/3 of the baseline's: 3.0% / 3.3% / 4.1%. Median
per-prompt ratio 1.00 everywhere (most replies are token-identical).

### Sampled (T 1.0, top_p 0.95, top_k 64; 100 prompts per benchmark x 3 draws)

AIME, 30 prompts x 3 draws = 90 replies:

| config | total tok | mean tok | cap | wall-clock s | dec tok/s | accept_len |
|---|---|---|---|---|---|---|
| baseline | 145,202 | 1,613 | 1 | 2,321 | 62.7 | – |
| assistant_k8 | 172,315 | 1,915 | 1 | 782 | 221.7 | 5.19 |
| conv_ep25_k8 | 140,543 | 1,562 | 0 | 492 | 288.1 | 5.48 |
| blk12_ep33_k12 | 196,456 | 2,183 | 1 | 652 | 303.1 | 6.21 |

GPQA, 100 prompts x 3 = 300 replies:

| config | total tok | mean tok | cap | wall-clock s | dec tok/s | accept_len |
|---|---|---|---|---|---|---|
| baseline | 247,512 | 825 | 1 | 3,940 | 62.9 | – |
| assistant_k8 | 234,995 | 783 | 0 | 1,168 | 203.5 | 4.63 |
| conv_ep25_k8 | 234,369 | 781 | 0 | 1,075 | 220.5 | 4.02 |
| blk12_ep33_k12 | 232,651 | 776 | 0 | 1,019 | 231.0 | 4.26 |

LCB, 100 prompts x 3 = 300 replies:

| config | total tok | mean tok | cap | wall-clock s | dec tok/s | accept_len |
|---|---|---|---|---|---|---|
| baseline | 240,642 | 802 | 0 | 3,748 | 64.4 | – |
| assistant_k8 | 254,030 | 847 | 0 | 1,204 | 214.0 | 4.88 |
| conv_ep25_k8 | 247,201 | 824 | 0 | 956 | 263.0 | 4.97 |
| blk12_ep33_k12 | 241,362 | 804 | 0 | 848 | 289.7 | 5.50 |

Noise floor for the sampled AIME numbers: the baseline's own three draws
total 46.0k / 55.3k / 44.0k tokens, i.e. the same model varies by 25% from
draw to draw on 30 prompts; GPQA 93.6k / 77.0k / 76.9k; LCB 80.5k / 82.9k /
77.2k. blk12's three AIME draws: 53.9k / 62.0k / 80.5k (one draw carries
the excess).

Bootstrap 95% CIs on the total-token ratio vs baseline (resampling prompts),
for the record: greedy LCB 0.99–1.14 / 1.00–1.13 / 0.98–1.13 (assistant /
conv / blk12); sampled AIME 0.92–1.57 / 0.76–1.22 / 1.11–1.64; sampled GPQA
0.83–1.05 / 0.85–1.03 / 0.81–1.06; sampled LCB 0.98–1.13 / 0.93–1.13 /
0.94–1.07.

### Reading

1. **No systematic lengthening from the DSpark / diffusion-style drafts.**
   Under sampling, the production-like case, conv k8 writes 140.5k AIME
   tokens vs the baseline's 145.2k, 234.4k vs 247.5k on GPQA, 247.2k vs
   240.6k on LCB: within the model's own draw-to-draw spread, and GPQA is
   *shorter* for all three drafts (233-235k vs 248k). The one cell that
   stands out is blk12 on AIME, 196k vs 145k, but that is 30 prompts where
   the baseline itself swings 44k-55k between draws and one blk12 draw
   (80k) carries the excess; the same draft on LCB writes 241k vs 241k.
   Google's autoregressive MTP draft shows the same pattern (AIME 172k vs
   145k, GPQA 235k vs 248k, LCB 254k vs 241k), so nothing here separates
   block drafts from AR drafts.
2. **Greedy shows a small common shift for all three drafts alike:** on LCB
   the baseline writes 1.213M tokens, the drafts 1.277-1.287M (+5-6%), with
   4.5-5.4% of prompts getting a reply >1.5x the baseline's and 3.0-4.1%
   getting one <2/3 of it (a surplus of 16-20 prompts out of 1,055). Since
   the AR assistant draft shows it identically, it is a property of vLLM's
   speculative path vs single-token decode (batch-shape numerics; see the
   literature section), not of the draft architecture. Where the two paths
   diverge, either can be the one that runs away: AIME id 80 goes 1,730 ->
   16,384 tokens under blk12 k12, AIME id 70 goes 8,669 -> 1,391 under both
   k8 drafts.
3. **End-to-end wall-clock drops 3-6.6x in every cell:** greedy LCB 19,043 s
   baseline vs 5,457 / 4,810 / 4,367 s; sampled LCB 3,748 s vs 1,204 / 956 /
   848 s; and so on for every benchmark. The decode speedup survives the
   length drift almost intact. The colleague's "same end-to-end time" is not
   reproduced on this target with these drafts in vLLM.
4. **Cap hits at 16k are rare and balanced:** greedy LCB 7 baseline vs 7 / 8
   / 10; 0-1 everywhere else. The 2048 cap of the old tables was cutting 10%
   of LCB replies, so those tables understated everyone's length equally
   but hid nothing draft-specific.

Caveats: AIME has 30 prompts and is inconclusive on its own in every table;
the sampled set used 100 prompts per benchmark (3 draws); one target
(Gemma-4-31B-it), one engine (vLLM 0.26, exact verification). If the
colleague's effect came from a different serving stack (lossy / threshold
acceptance, stop-string or min_tokens processors, quantized target), those
are the knobs to compare next, not the draft architecture.

## Running

One SLURM job per experiment (the 16k cap makes the baseline's runaway
prompts slow; budget 12 h each), submitted from the repo root (the branch
was developed in a worktree at the time, hence `REPO=$PWD`):

```bash
cd /import/ml-sc-scratch1/mengmengj/speculators
for y in gemma4-31b-length-check gemma4-31b-length-check-sampled; do
  for e in baseline assistant_k8 conv_ep25_k8 blk12_ep33_k12; do
    REPO=$PWD ONLY=$e CONFIG=$y.yaml \
      sbatch --chdir=$PWD --reservation sn50-hdi-0 --time=12:00:00 -J ${y#gemma4-31b-}_$e examples/train/sbatch_sc3_eval.sh
  done
done
# results: scripts/evaluate/experiments/results/<yaml>/<experiment>/
cd scripts/evaluate/experiments
python ../mtp_server_eval/compare_lengths.py --dir results/gemma4-31b-length-check --baseline baseline
python ../mtp_server_eval/compare_lengths.py --dir results/gemma4-31b-length-check-sampled --baseline baseline
python ../mtp_server_eval/compare_speedup.py --dir results/gemma4-31b-length-check --baseline baseline
```

The B200 nodes sit inside the `sn50-hdi-0` reservation (we are members):
without `--reservation` the jobs pend on "Resources" forever.
Each job gets its own server port (`8400 + job id % 1000`), so several can
share a node. `blk12_ep33_k12` needs `--max-num-seqs 32
--max-num-batched-tokens 16384` (per-experiment `server.extra_args` in the
yaml), as in the blk12 yaml.

## Status

- 2026-10-28 15:00: harness changes + yamls committed on `mengmengj/glm-dflash2`
  (then a worktree; since 2026-10-29 checked out in the main `speculators` dir);
  8 jobs (greedy 14155310-13, sampled 14155314-17, sc3-cp84/85) submitted.
- 2026-10-28 20:00: all 8 finished (greedy baseline 6h15, others 45 min-2h50).
- 2026-10-29: results + reading above. Verdict: no draft-specific length
  effect; e2e speedup 3-6.6x. Open: compare against the colleague's serving
  stack settings if their observation persists.
