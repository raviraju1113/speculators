# Serving a speculative-decoding draft on Gemma-4-26B-A4B-it (MoE)

How to deploy the DSparkFlash draft model in front of the Gemma-4 26B MoE
target in vLLM, what to set, and what the settings are worth.

Numbers below are `sc1_delta`, single run per comparison, prefix caching off,
one discarded warmup per cell, in-run baseline. Measured on one A100 80GB,
vLLM 0.28.

---

## TL;DR

```bash
vllm serve google/gemma-4-26B-A4B-it \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
  --max-model-len 24576 --max-num-seqs 512 --max-num-batched-tokens 32768 \
  --async-scheduling \
  --speculative-config '{
      "model": "<draft-checkpoint>",
      "num_speculative_tokens": 7,
      "method": "dspark",
      "dspark_draft_topk": 64
  }'
```

Three settings carry almost all of the value:

| setting | value | why |
|---|---|---|
| `method` | `dspark` | required; without it vLLM rejects `dspark_draft_topk` |
| `num_speculative_tokens` | **7** | best under load. k=9 is faster for a single user but slower at concurrency ≥ 8 |
| `dspark_draft_topk` | **64** | **+109 tok/s at concurrency 128.** Without it the draft loses to a plain MTP assistant under load |

---

## What you get

| concurrency | no draft | with draft | speedup |
|--:|--:|--:|--:|
| 1 | 112.6 | 214.2 | **1.90×** |
| 8 | 362.9 | 672.6 | 1.85× |
| 32 | 625.8 | 1062.0 | 1.70× |
| 128 | 732.8 | 1023.6 | 1.40× |
| 256 | 728.2 | 1006.0 | 1.38× |

Speedup **falls as the server saturates**, and that is inherent, not a defect.

```
speedup = AL / (1 + Ω)
```

`AL` = acceptance length, tokens committed per target forward pass. `Ω` = what
the draft itself costs, as a fraction of target time. At concurrency 1 the GPU
is memory-bound with spare compute, so drafting is nearly free. At saturation
every draft cycle is taken from the batch it is meant to accelerate, Ω rises,
and the speedup compresses toward 1.

**Quote speedups with their concurrency.** A 1.9× headline is a single-user
number and will not survive contact with a loaded server.

---

## Choosing `num_speculative_tokens`

The checkpoint is `block_size=16` with `sample_from_anchor=False`, so its
**native depth is k=15**. Serving below that is allowed and usually better.

Single user:

| k | AL | tok/s |
|--:|--:|--:|
| 7 | 4.924 | 202.9 |
| **9** | 5.303 | **209.0** |
| 11 | 5.499 | 207.4 |
| 13 | 5.402 | 195.3 |
| 15 (native) | **5.545** | 195.4 |

Acceptance climbs monotonically to k=15 while throughput peaks at k=9 — past
that, the extra accepted tokens cost more to produce than they save.

Under load the optimum moves **down**:

| concurrency | k=7 | k=9 |
|--:|--:|--:|
| 1 | 214.2 | **220.0** |
| 8 | **672.6** | 672.1 |
| 32 | **1062.0** | 1021.8 |
| 256 | **1006.0** | 982.4 |

**Use k=7 for a shared endpoint; k=9 only if you are genuinely serving one
request at a time.**

---

## Why `dspark_draft_topk=64` matters so much

The draft carries a 262,144-entry vocabulary, which makes the Markov head's
`markov_w2` a 134 MB tensor re-streamed on every draft step. `dspark_draft_topk`
picks candidates from the base logits *first*, so the head scores 64 entries
instead of 262k.

| | c128 tok/s |
|---|--:|
| draft, no top-k | 980.7 |
| draft + `dspark_draft_topk=64` | **1089.5** |
| MTP assistant (for reference) | 1070.7 |

Acceptance is unchanged (4.949 → 4.947). The gain is pure Ω — which is also the
cleanest evidence that Ω, not acceptance, is the binding constraint here.

Note this is a **logit** top-k over the vocabulary. It is unrelated to
`topk_context`, which is the DSA lightning indexer's context selection.

---

## Comparison with the MTP assistant

Against `gemma-4-26B-A4B-it-assistant`, each at its best tested depth:

| concurrency | draft (k=7 + topk64) | MTP assistant (k=7) | margin |
|--:|--:|--:|--:|
| 1 | 214.2 | 210.6 | +1.7% |
| 32 | 1062.0 | 1005.1 | +5.7% |
| 256 | 1006.0 | 998.0 | +0.8% |

Ahead at every level, but by **0.8–5.7%**, and at saturation the margin is
inside measurement spread. Treat the two as comparable under heavy load and
pick on operational grounds.

Caveat: the assistant was still improving at k=7, the top of the tested range,
so its ceiling is not established. The margin above may narrow.

---

## Measurement notes, if you benchmark this yourself

These cost us four retracted conclusions:

1. **Disable prefix caching** (`--no-enable-prefix-caching`). With it on, the
   same arm measured 524 → 608 → 870 tok/s across runs — 66% drift that
   swallows any real effect.
2. **Measure every arm inside one run**, on one GPU, one at a time. Do not
   compare numbers from different runs.
3. **Noise floor is ±0.3 AL** at 20 samples, and ~0.4–1.0% on throughput at
   c128. Differences smaller than that are not differences.
4. **Training-run variance is ~0.10 AL** at fixed config. Two checkpoints
   trained identically but with different seeds differed by that much, so
   anything under it is not evidence of an architectural effect.
5. **An implausibly good number is a bug report.** An accept_len near its k+1
   ceiling, an accept_rate of exactly 1.000, or a negative loss term means
   broken plumbing, not a breakthrough.

---

## Known limitations

- **Dense Gemma-4-31B behaves differently.** Speedup there is ~1.37× at one
  user and goes **below 1.0 at 16 concurrent users** — the draft makes it
  slower than no draft. Do not carry these MoE settings to the dense model
  without re-measuring.
- **DFlash2 convolution checkpoints will not serve.** vLLM 0.28's
  `gemma4_dspark.py` has no conv support and fails with
  `ValueError: There is no module or parameter named 'layers.0.attention_conv'`.
- The draft is trained for this exact target. It will not transfer to another
  backbone.
