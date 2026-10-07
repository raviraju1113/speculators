# DSpark draft for gemma-4-31B-it — evaluation

1x A100-80, TP=1, `--enforce-eager`, greedy (temp 0.0), n=30 per benchmark.
`baseline` is the backbone with no draft, measured in the SAME run under the
same server settings; speedup is decode tok/s over it. Decode throughput
excludes prefill (measured from first token to last).

Rows are paired at matching `k` (`num_speculative_tokens`) so each comparison
holds the draft budget fixed.

> **Read `accept_len`, not `accept_rate`.** The eval defines
> `accept_rate = (accept_len - 1) / k`, so it is normalised by k and is NOT
> comparable across different k — the same checkpoint "improves" from 0.32 to
> 0.64 just by lowering k, while getting *slower*. `accept_len` (expected
> tokens per verifier step) is what determines speedup.

## AIME

| draft | k | accept_len | accept_rate | decode tok/s | speedup |
|---|---|---|---|---|---|
| backbone only (no draft) | — | — | — | 21.0 | **1.00x** |
| **ours** — 0.99 epoch | 8 | 3.569 | 0.3211 | 66.4 | **3.16x** |
| RedHat (published) | 8 | 4.562 | 0.4452 | 84.0 | **4.00x** |
| **ours** — 0.99 epoch | 3 | 2.907 | 0.6356 | 54.8 | **2.61x** |
| RedHat (published) | 3 | 2.254 | 0.4182 | 41.8 | **1.99x** |

## GPQA-Diamond

| draft | k | accept_len | accept_rate | decode tok/s | speedup |
|---|---|---|---|---|---|
| backbone only (no draft) | — | — | — | 21.2 | **1.00x** |
| **ours** — 0.99 epoch | 8 | 2.820 | 0.2276 | 52.9 | **2.50x** |
| RedHat (published) | 8 | 3.920 | 0.3650 | 72.7 | **3.43x** |
| **ours** — 0.99 epoch | 3 | 2.460 | 0.4868 | 46.8 | **2.21x** |
| RedHat (published) | 3 | 2.612 | 0.5374 | 48.5 | **2.29x** |

## LiveCodeBench

| draft | k | accept_len | accept_rate | decode tok/s | speedup |
|---|---|---|---|---|---|
| backbone only (no draft) | — | — | — | 21.9 | **1.00x** |
| **ours** — 0.99 epoch | 8 | 3.235 | 0.2793 | 59.4 | **2.71x** |
| RedHat (published) | 8 | 3.871 | 0.3589 | 70.9 | **3.24x** |
| **ours** — 0.99 epoch | 3 | 2.687 | 0.5624 | 50.0 | **2.28x** |
| RedHat (published) | 3 | 2.754 | 0.5848 | 50.4 | **2.30x** |

### Note on k=7

RedHat's checkpoint ships `speculative_tokens: 7` (its native layout:
`sample_from_anchor=False` → 1+7 = 8 slots = `block_size`). It is shown above at
k=8 for a like-for-like comparison with ours, which is technically out of spec
for that checkpoint. At its native k=7 it scores slightly **better** on two of
three benchmarks, so the k=8 rows if anything understate it:

| RedHat k=7 (native) | accept_len | decode tok/s | speedup |
|---|---|---|---|
| AIME | 4.626 | 85.2 | 4.06x |
| GPQA-Diamond | 4.012 | 74.8 | 3.53x |
| LiveCodeBench | 3.559 | 64.7 | 2.95x |

Our checkpoint has no k=7 row: it was trained with `sample_from_anchor=True`,
whose native layout is k=8.

## The bug fixed along the way

speculators defaults DSpark to `sample_from_anchor=True`; vLLM's speculators
loader hardcodes the opposite. The default config trains a draft that decodes
one slot off, loads with no error, and reports ~half its real acceptance.
Same weights, same harness, one-line vLLM change:

| AIME, ours k=8 | accept_len | accept_rate | decode tok/s | speedup |
|---|---|---|---|---|
| before fix | 1.838 | 0.1047 | 33.9 | 1.61x |
| after fix | **3.569** | **0.3211** | **66.4** | **3.16x** |

The run now training uses `--no-sample-from-anchor` and will decode correctly
on stock vLLM with no patch.
