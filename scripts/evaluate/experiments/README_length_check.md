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

## Running

One SLURM job per experiment (the 16k cap makes the baseline's runaway
prompts slow; budget 12 h each), submitted from the worktree so the main
checkout stays untouched for the jobs running from it:

```bash
cd /import/ml-sc-scratch1/mengmengj/speculators-glm
for y in gemma4-31b-length-check gemma4-31b-length-check-sampled; do
  for e in baseline assistant_k8 conv_ep25_k8 blk12_ep33_k12; do
    REPO=$PWD ONLY=$e CONFIG=$y.yaml \
      sbatch --chdir=$PWD --time=12:00:00 -J ${y#gemma4-31b-}_$e examples/train/sbatch_sc3_eval.sh
  done
done
# results: scripts/evaluate/experiments/results/<yaml>/<experiment>/
cd scripts/evaluate/experiments
python ../mtp_server_eval/compare_lengths.py --dir results/gemma4-31b-length-check --baseline baseline
python ../mtp_server_eval/compare_lengths.py --dir results/gemma4-31b-length-check-sampled --baseline baseline
python ../mtp_server_eval/compare_speedup.py --dir results/gemma4-31b-length-check --baseline baseline
```

Each job gets its own server port (`8400 + job id % 1000`), so several can
share a node. `blk12_ep33_k12` needs `--max-num-seqs 32
--max-num-batched-tokens 16384` (per-experiment `server.extra_args` in the
yaml), as in the blk12 yaml.

## Status

- 2026-10-28: harness changes + yamls committed on `mengmengj/glm-dflash2`;
  8 jobs submitted (see the job table in the commit / Slack thread). Results
  and the reading go here when they land.
