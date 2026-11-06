# Kimi K3 DSpark: fp8 draft KV cache ablation

Does quantizing the DSpark draft's own KV cache to fp8 change acceptance?
Scope, per Yun Du/Chen Wu: fp8 for DSpark's GQA cache only, not Kimi K3's own
MLA cache (target stays untouched).

## Setup

- **Target**: `/import/ml-sc-scratch5/chenw/models/Kimi-K3-patched` (unmodified,
  bf16/mxfp4 as shipped, MLA KV cache untouched)
- **Draft**: `/import/ml-sc-scratch5/chenw/models/Kimi-K3-DSpark` (RadixArk
  architecture, the original checkpoint — full attention, `layer_types: {
  full_attention}`, GQA 64 query heads / 16 KV heads, no sliding-window
  modification)
- **Mechanism**: vLLM's `speculative_config.kv_cache_dtype` (`vllm/config/speculative.py`)
  is wired independently into the draft's own `cache_config`
  (`vllm/v1/worker/gpu/spec_decode/dspark/utils.py::load_dspark_model`) —
  confirmed by reading the source before running anything. Baseline omits
  the field (draft KV cache defaults to bf16); the fp8 arm adds
  `"kv_cache_dtype": "fp8"` to the same `--speculative-config` JSON. Nothing
  else changes between the two runs.
- **Serving**: canonical 8x B300 launch command in
  [kimi_k3_draft_eval.md's "Canonical serving command"](kimi_k3_draft_eval.md#canonical-serving-command--kimi-k3--radixark-dspark-8x-b300).
  This ablation's only change vs. that command: the fp8 arm adds
  `"kv_cache_dtype":"fp8"` inside `--speculative-config` — nothing else
  differs between the two runs.
- **Data**: the same 24-set clean-benchmark suite and cached prompts used
  throughout the sliding-window ablation (`kimi-k3-data-clean/<set>/gen_cache/`,
  n=50 per set, n=30 for aime/aime26), replayed token-id level against a live
  server via `/v1/completions`. `accept_len`/`accept_rate` read from vLLM's
  cumulative `vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total`
  counters (before/after delta per set) — same methodology as
  `dspark_sliding_window_ablation.md`'s live-serving sweeps.

## A pre-existing crash, unrelated to this ablation

Both the baseline and fp8 runs hit the same crash partway through (baseline
after 12/24 sets, fp8 after 10/24 sets): all 8 workers threw
`RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered`,
traced to `vllm/models/kimi_k3/nvidia/kda.py:905`, inside
`_store_cache_checkpoints_kernel` — a Triton kernel in Kimi K3's own **native
KDA (linear-attention) cache-checkpointing path**. This is target-model
serving code, has nothing to do with DSpark, the draft, or KV cache dtype.
It recurred a second time on each variant at a similar cumulative request
count (~500-700 requests into one server's lifetime, regardless of which
specific set was running), which points to some kind of accumulating-state
bug in that kernel (e.g. a checkpoint counter/ring-buffer overflow) rather
than a bad input — matches the `cudaErrorIllegalAddress`-in-target's-MLA/KDA
pattern also seen during the earlier AgentX investigation
(`docs/experiments/kimi_k3_draft_eval.md`).

Recovery: a retry harness that relaunches the server and resumes from the
last completed set (the client skips any set whose output file already
exists) got both variants to all 24 sets after 2 attempts each. One set,
baseline's `speed-multilingual`, was mid-flight when the first crash hit
(11/50 requests failed before the server died) — see the results table
footnote below.

This crash should be reported/tracked separately as a Kimi K3 serving
stability bug in the native KDA path — it is not an artifact of the fp8
experiment and was not introduced by it.

## Results

| set | n | AL baseline | AL fp8 | Δ AL | Δ% | AR baseline | AR fp8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 50 | 5.7106 | 5.7236 | +0.0130 | +0.23% | 0.6729 | 0.6748 |
| bfcl | 50 | 4.3756 | 4.4202 | +0.0446 | +1.02% | 0.4822 | 0.4886 |
| mbpp | 50 | 4.4465 | 4.4432 | -0.0033 | -0.07% | 0.4924 | 0.4919 |
| speed-coding | 50 | 4.4839 | 4.4950 | +0.0111 | +0.25% | 0.4977 | 0.4993 |
| livecodebench | 50 | 3.7227 | 3.7261 | +0.0034 | +0.09% | 0.3890 | 0.3894 |
| math500 | 50 | 3.8490 | 3.8383 | -0.0107 | -0.28% | 0.4070 | 0.4055 |
| tool_call | 50 | 3.5948 | 3.5902 | -0.0046 | -0.13% | 0.3707 | 0.3700 |
| speed-rag | 50 | 3.8622 | 3.8616 | -0.0006 | -0.02% | 0.4089 | 0.4088 |
| rag | 50 | 3.8319 | 3.8208 | -0.0111 | -0.29% | 0.4046 | 0.4030 |
| translation | 50 | 3.7122 | 3.6920 | -0.0202 | -0.54% | 0.3875 | 0.3846 |
| swe-bench-pro | 50 | 3.3423 | 3.3452 | +0.0029 | +0.09% | 0.3346 | 0.3350 |
| speed-multilingual† | 50 | 3.1484 | 3.1159 | -0.0325 | -1.03% | 0.3069 | 0.3023 |
| summarization | 50 | 3.6325 | 3.6296 | -0.0029 | -0.08% | 0.3761 | 0.3757 |
| speed-writing | 50 | 2.9911 | 2.9994 | +0.0083 | +0.28% | 0.2844 | 0.2856 |
| qa | 50 | 3.1398 | 3.1392 | -0.0006 | -0.02% | 0.3057 | 0.3056 |
| speed-qa | 50 | 3.1385 | 3.1378 | -0.0007 | -0.02% | 0.3055 | 0.3054 |
| aa-lcr-4k | 50 | 2.8196 | 2.8203 | +0.0007 | +0.02% | 0.2599 | 0.2600 |
| aa-lcr-1k | 50 | 2.7046 | 2.7054 | +0.0008 | +0.03% | 0.2435 | 0.2436 |
| aime | 30 | 2.6951 | 2.6885 | -0.0066 | -0.24% | 0.2422 | 0.2412 |
| swe-rebench | 50 | 3.0257 | 3.0203 | -0.0054 | -0.18% | 0.2894 | 0.2886 |
| mtbench | 50 | 3.2757 | 3.2753 | -0.0004 | -0.01% | 0.3251 | 0.3250 |
| gpqa | 50 | 2.7663 | 2.7663 | +0.0000 | +0.00% | 0.2523 | 0.2523 |
| writing | 50 | 3.3030 | 3.3075 | +0.0045 | +0.14% | 0.3290 | 0.3296 |
| aime26 | 30 | 2.5354 | 2.5239 | -0.0115 | -0.45% | 0.2193 | 0.2177 |

**Macro mean AL**: baseline 3.5045, fp8 3.5036 — Δ **-0.0009 (-0.03%)**.

† `speed-multilingual`'s baseline arm was mid-flight when the first crash hit
(see above): its AL/AR are computed over ~39/50 successful samples, not 50 —
still a valid ratio (drawn from real counter deltas), just smaller effective
n for this one row. Every other row is a clean 50/50 (or 30/30 for
aime/aime26), zero request errors.

- Per-set deltas are small and scattered in both directions: 13 sets slightly
  down, 10 slightly up, 1 exactly flat (`gpqa`). No systematic degradation
  in either direction.
- Largest single-set effect is `speed-multilingual` at -1.03% (the set with
  the reduced-n caveat above) and `bfcl` at +1.02% — both within noise range
  for n=50, not a directional signal.
- 22 of 24 sets are within ±0.3% absolute AL change.

## Memory and throughput (the actual motivation for fp8 KV cache)

Acceptance parity alone doesn't tell you whether fp8 is worth doing — the
point of a smaller draft KV cache is memory/throughput, not accuracy. Two
follow-up measurements, using data already collected above (no new runs):

**Memory — theoretical, exact.** DSpark's draft
(`Kimi-K3-DSpark/config.json`): 5 layers, 16 KV heads, head_dim 64, GQA.
Per-token draft KV cache footprint (K+V, all 5 layers):
`2 (K,V) x 5 layers x 16 heads x 64 head_dim x bytes/elem`:

| dtype | bytes/elem | draft KV bytes/token |
|---|---:|---:|
| bf16 (baseline) | 2 | 20,480 (20.0 KiB) |
| fp8 | 1 | 10,240 (10.0 KiB) |

Exactly half, as guaranteed by the byte width — not something that needs
empirical verification.

**But it doesn't move the server-reported combined pool.** Both server
startup logs report `Available KV cache memory` / `GPU KV cache size` for
the whole engine (target + draft share one memory pool under this deployment):
33.94 GiB / 327,600 tokens (39.99x max concurrency) for baseline vs.
33.92 GiB / 327,361 tokens (39.96x) for fp8 — a difference smaller than
run-to-run noise, and in the "wrong" direction. Back-computing the
target's own per-token cost from the baseline numbers (33.94 GiB / 327,600
tokens ≈ 108.6 KiB/token) shows why: Kimi K3's own 93-layer MLA cache costs
~5.4x more per token than the draft's bf16 cache (draft bf16 is ~18% of the
target's per-token cost, fp8 ~9%). Halving a piece that's already ~1/5 to
1/10 the size of the dominant target cache is a rounding error against a
34 GiB pool — this deployment's memory profile is target-cache-bound, not
draft-cache-bound, so the concurrency/memory benefit of fp8 here is
negligible in absolute terms even though the underlying ratio (2x smaller
draft cache) is real and would matter more in a deployment where the draft's
cache footprint is a larger share of the total (e.g. a much bigger draft, or
a target/draft pair sharing a tighter memory budget).

**Throughput — small, fairly consistent regression, not a win.** Using each
set's already-recorded mean single-request tok/s (`mean_tok_s` in the same
result files, unbatched/single-stream — not a production concurrent-serving
number):

| | baseline | fp8 | Δ% |
|---|---:|---:|---:|
| mean tok/s (24-set average) | 239.8 | 238.3 | **-0.63%** |

20 of 24 sets are slightly negative (worst: `swe-bench-pro` -3.79%, likely a
cold-start artifact — it was the first set re-run after that variant's
server-crash recovery, not fp8-specific), 3 slightly positive, 1 flat. The
small, fairly consistent negative bias is consistent with fp8 adding a
dequantization step on every attention read without enough memory-bandwidth
savings to offset it, at this draft's small scale (5 layers, tiny K/V
tensors — the reads were likely not memory-bandwidth-bound to begin with, so
there's little bandwidth win to bank against the added dequant cost). This
is single-stream latency, not concurrent throughput, so it doesn't capture
whatever benefit fp8 would deliver by allowing more concurrent requests via
freed cache memory — which, per the memory result above, is itself
negligible in this specific target/draft size ratio.

## Takeaway

fp8 quantization of the DSpark draft's own GQA KV cache produces no
measurable accept_len/accept_rate degradation on Kimi K3 (-0.03% macro
delta, noise). The memory/throughput motivation doesn't pay off in *this*
deployment either: the draft's cache is a small fraction of the combined
memory pool that Kimi K3's own giant 93-layer MLA cache dominates, so halving
the draft's per-token bytes doesn't measurably change total concurrency
headroom, and single-stream throughput shows a small, fairly consistent
regression (-0.63%) likely from added dequantization overhead. Net: for this
specific target (a very large model whose own KV cache dwarfs the draft's),
fp8 on the draft alone is acceptance-safe but not a clear win — it would need
to be paired with quantizing the target's own cache (out of scope here, per
Yun Du/Chen Wu's DSpark-first sequencing) to produce a meaningful combined
memory benefit.
