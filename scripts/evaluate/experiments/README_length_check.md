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
and `compare_lengths.json` next to them. tok_ratio = config tokens / baseline
tokens, paired per prompt; [..] = 95% bootstrap CI over prompts; cap = replies
cut at 16384 (baseline -> config); e2e = wall-clock ratio (<1 = faster).

### Greedy (temperature 0, full datasets, B200 TP=1)

| config | pooled tok_ratio | AIME (30) | GPQA (198) | LCB (1055) | LCB >1.5x / <0.67x | cap AIME/GPQA/LCB | e2e AIME/GPQA/LCB |
|---|---|---|---|---|---|---|---|
| assistant_k8 (Google MTP) | 1.062 | 0.92 [0.64,1.23] | 1.12 [0.99,1.32] | 1.06 [0.99,1.14] | 4.5% / 3.0% | 0->0, 1->2, 7->8 | 0.22 / 0.30 / 0.29 |
| conv_ep25_k8 (DSpark+conv) | 1.050 | 0.82 [0.60,1.03] | 1.04 [0.83,1.29] | 1.06 [1.00,1.13] | 5.4% / 3.3% | 0->0, 1->1, 7->7 | 0.15 / 0.26 / 0.25 |
| blk12_ep33_k12 (DSpark+conv) | 1.056 | 1.29 [0.74,2.17] | 1.02 [0.98,1.06] | 1.05 [0.98,1.13] | 5.1% / 4.1% | 0->1, 1->1, 7->10 | 0.22 / 0.26 / 0.23 |

Decode tok/s (accept_len) at the 16k cap: baseline 62.5 / 63.1 / 63.8;
assistant_k8 259 (5.68) / 240 (5.35) / 238 (5.39); conv_ep25_k8 343 (6.11) /
255 (4.55) / 271 (5.33); blk12_ep33_k12 363 (7.43) / 248 (4.66) / 296 (5.95).
Same as the 2048-cap tables within noise.

### Sampled (T 1.0, top_p 0.95, top_k 64; 100 prompts per benchmark, 3 draws)

| config | pooled tok_ratio | AIME (30x3) | GPQA (100x3) | LCB (100x3) | cap | e2e AIME/GPQA/LCB |
|---|---|---|---|---|---|---|
| assistant_k8 | 1.044 | 1.19 [0.92,1.57] | 0.95 [0.83,1.05] | 1.06 [0.98,1.13] | 1->1, 1->0, 0->0 | 0.34 / 0.30 / 0.32 |
| conv_ep25_k8 | 0.982 | 0.97 [0.76,1.22] | 0.95 [0.85,1.03] | 1.03 [0.93,1.13] | 1->0, 1->0, 0->0 | 0.21 / 0.27 / 0.26 |
| blk12_ep33_k12 | 1.059 | 1.35 [1.11,1.64] | 0.94 [0.81,1.06] | 1.00 [0.94,1.07] | 1->1, 1->0, 0->0 | 0.28 / 0.26 / 0.23 |

Noise floor: the baseline's own three draws differ in total AIME tokens by
1.26x (46.0k / 55.3k / 44.0k) and GPQA by 1.22x; LCB by 1.07x. Decode
speedups under sampling: assistant 3.2-3.5x, conv k8 3.5-4.6x, blk12 k12
3.7-4.8x (acceptance drops ~0.5 vs greedy, as expected).

### Reading

1. **No systematic lengthening from the DSpark / diffusion-style drafts.**
   Under sampling, the production-like case, conv k8 is 0.98 pooled, and GPQA
   is *shorter* for all three drafts (0.94-0.95). The only cell with a CI
   above 1 is blk12 on AIME (1.35), 30 prompts where the baseline's own
   draw-to-draw spread is 1.26x and one draw of blk12 (80k vs 54k / 62k
   tokens) carries the excess; LCB for the same draft is 1.00. Google's
   autoregressive MTP draft shows the same pattern (AIME 1.19, GPQA 0.95,
   LCB 1.06), so nothing here separates block drafts from AR drafts.
2. **Greedy shows a small common shift, +5-6% pooled, for all three drafts
   alike,** carried by LCB (+5-6%, CIs touching 1.0) where the share of
   replies >1.5x the baseline exceeds the <0.67x share by 1-2 points
   (16-20 prompts of 1055). Since the AR assistant draft shows it identically,
   it is a property of vLLM's speculative path vs single-token decode
   (batch-shape numerics; see the literature section), not of the draft
   architecture. Where the two paths diverge, either can be the one that runs
   away: AIME id 80 goes 1,730 -> 16,384 under blk12 k12, AIME id 70 goes
   8,669 -> 1,391 under both k8 drafts.
3. **End-to-end wall-clock is 3-6.6x lower in every cell** (e2e_ratio
   0.15-0.34), i.e. the decode speedup survives the length drift almost
   intact: pooled e2e speedup ~ decode speedup / 1.05. The colleague's
   "same end-to-end time" is not reproduced on this target with these
   drafts in vLLM.
4. **Cap hits at 16k are rare and balanced:** greedy LCB 7 baseline vs 7 / 8
   / 10; sampled 0-1 everywhere. The 2048 cap of the old tables was cutting
   10% of LCB replies, so those tables understated everyone's length equally
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
