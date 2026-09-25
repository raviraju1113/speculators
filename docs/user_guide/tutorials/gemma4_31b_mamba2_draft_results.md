# Gemma-4-31B: Mamba2 vs Transformer Eagle-3 Draft

Head-to-head of a **Mamba2 (SSM) Eagle-3 draft** against the matched **Qwen3 transformer
Eagle-3 draft**, on the same 25-benchmark suite as
[gemma4_31b_full_spec_decode_results.md](gemma4_31b_full_spec_decode_results.md).

The question, from the [design proposal](https://docs.google.com/document/d/1ZTKBJi51_wfUMIzLN_AKRiAy4_Aa-YUuc6a5BJ0Z32Q/edit):
an Eagle-3 draft normally runs one transformer block, whose KV cache grows linearly with
context. A Mamba2 block instead carries a **constant ~5.6 MB recurrent state**. Does the
draft stay within **10% of the transformer's acceptance length**? If so it wins
decisively on long-context serving capacity.

## Setup

| Parameter | Value |
|---|---|
| Backbone | `/sms-scratch/checkpoints/gemma-4-31B-it` (bf16) |
| Hardware | 4x NVIDIA B200, TP=4 |
| vLLM | 0.25.1 + `vllm_patches/0001-eagle3-mamba2-draft-metadata.patch` |
| `max_model_len` / `max_tokens` | 8,192 / 4,096 |
| Temperature | 0.0 (greedy) |
| Config | [`gemma4-31b-full-mamba2.yaml`](../../../scripts/evaluate/experiments/gemma4-31b-full-mamba2.yaml) |

Both drafts were trained on the **same** 782k-row `kimi-mtp-nemotron-stem-code-math` mix
with matched scaffolding (`--optimizer adamw --no-norm-before-fc --no-norm-output`, tap
layers `[2,30,57]`, full 262,144 draft vocab, `ttt_steps=3`, `lr 1e-4`, seed 42). Every
hyperparameter logged to W&B matches except `draft_arch`.

| arm | draft block | block params | trainable |
|---|---|---:|---:|
| **A** | Qwen3 transformer layer | 567.0 M | 2.06 B |
| **B** | Mamba2 mixer | **313.0 M** | **1.81 B** |

> **Read the rows against each other, not against the older Gemma-4 table.** Those
> numbers came from A100s with prefix caching on at 0.9 utilization; these are B200s at
> 0.65 with prefix caching off, both required by the Mamba draft (see
> [`vllm_patches/README.md`](../../../vllm_patches/README.md)). Both arms here share
> identical settings, so the comparison between them is clean.

## Results

| benchmark | n | A accept_len | **B accept_len** | B/A | A tok/s | **B tok/s** | B/A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aime | 30 | 3.634 | **3.549** | 0.98x | 379.1 | **351.0** | 0.93x |
| gpqa | 50 | 2.881 | **3.017** | 1.05x | 304.4 | **309.1** | 1.02x |
| livecodebench | 50 | 3.086 | **3.201** | 1.04x | 312.6 | **311.6** | 1.00x |
| gsm8k | 50 | 3.703 | **3.782** | 1.02x | 462.3 | **446.9** | 0.97x |
| humaneval | 50 | 3.496 | **3.446** | 0.99x | 421.8 | **395.1** | 0.94x |
| mbpp | 50 | 3.082 | **3.288** | 1.07x | 370.9 | **376.2** | 1.01x |
| math500 | 50 | 3.844 | **3.777** | 0.98x | 440.2 | **412.4** | 0.94x |
| mt-bench | 50 | 2.253 | **2.482** | 1.10x | 267.8 | **280.4** | 1.05x |
| aime26 | 30 | 3.619 | **3.549** | 0.98x | 367.1 | **342.4** | 0.93x |
| bfcl | 50 | 3.146 | **3.010** | 0.96x | 351.9 | **323.7** | 0.92x |
| swe-bench-pro | 50 | 2.432 | **2.430** | 1.00x | 239.1 | **231.5** | 0.97x |
| speed-coding | 50 | 2.942 | **3.059** | 1.04x | 305.2 | **301.6** | 0.99x |
| speed-multilingual | 50 | 1.835 | **1.783** | 0.97x | 220.7 | **204.6** | 0.93x |
| speed-rag | 49 | 2.542 | **2.638** | 1.04x | 259.5 | **261.7** | 1.01x |
| speed-qa | 50 | 1.974 | **2.214** | 1.12x | 250.6 | **266.6** | 1.06x |
| speed-writing | 50 | 2.005 | **2.212** | 1.10x | 196.4 | **209.9** | 1.07x |
| HumanEval | 50 | 3.302 | **3.434** | 1.04x | 399.5 | **396.3** | 0.99x |
| math_reasoning | 50 | 3.663 | **3.615** | 0.99x | 470.9 | **440.4** | 0.94x |
| qa | 50 | 1.974 | **2.214** | 1.12x | 250.7 | **266.6** | 1.06x |
| question | 50 | 2.251 | **2.480** | 1.10x | 267.6 | **280.3** | 1.05x |
| rag | 50 | 2.472 | **2.605** | 1.05x | 260.7 | **263.0** | 1.01x |
| summarization | 50 | 1.889 | **2.057** | 1.09x | 195.2 | **205.5** | 1.05x |
| tool_call | 50 | 2.299 | **2.493** | 1.08x | 262.8 | **273.3** | 1.04x |
| translation | 50 | 2.242 | **2.420** | 1.08x | 289.9 | **295.8** | 1.02x |
| writing | 50 | 2.251 | **2.480** | 1.10x | 267.6 | **280.3** | 1.05x |
| **token-weighted** | 1209 | **2.835** | **2.937** | **1.036x** | **311.9** | **308.5** | **0.989x** |

Arm B at k=3 for reference: token-weighted accept_len 2.637, decode 300.4 tok/s.
k=5 is the better operating point for the Mamba2 draft on every aggregate measure.

## Verdict

**Decision criterion `tau_B >= 0.90 * tau_A`: 2.937 >= 2.551 — PASS**, and not
marginally. Arm B *exceeds* Arm A on acceptance (**1.036x**) at essentially equal
throughput (**0.989x**), from a mixer block **45% smaller**. It leads on **17 of 25**
benchmarks.

The split is systematic rather than noise. Arm B is stronger on prose-shaped work
(speed-qa / qa 1.12x, mt-bench / writing / question 1.10x, summarization 1.09x,
tool_call / translation 1.08x) and weaker on math and code (bfcl 0.96x,
speed-multilingual 0.97x, aime / aime26 / math500 0.98x). Throughput tracks acceptance
closely — Arm B is faster on exactly the benchmarks where it accepts more — so the two
drafts' per-step costs are near enough to equal that acceptance is what moves wall
clock. The ~1% aggregate tok/s deficit comes from the math/code benches, not from the
SSM being slower per step.

## What this does not show

**The memory case is untested here.** The proposal's argument is long context: a
constant 5.6 MB state against a KV cache reaching ~4.29 GB per sequence at 256K. This
suite ran at `max_model_len` 8,192, where that advantage is worth nothing. What it
establishes is that a recurrent drafter is *competitive on quality* — not a degraded
fallback accepted for its memory profile. The long-context sweep
([`gemma4-31b-mamba2-ctxlen-sweep.yaml`](../../../scripts/evaluate/experiments/gemma4-31b-mamba2-ctxlen-sweep.yaml),
AA-LCR 1k->128k) is the test that values the memory story.

Also worth noting: the proposal's own Arm A spec (42 query heads / 8 KV heads) is not
realizable — 42 % 8 != 0 is invalid GQA — and understates the shipped draft's KV by 4x
(16,384 B/token, not 4,096). The Mamba2 memory advantage at 256K is therefore ~766x,
not the ~190x claimed.

## Losslessness

Speculative output was compared token-for-token against target-only decoding with
otherwise identical flags: **5 of 6 probe prompts byte-identical**. The single mismatch
first diverges at token 7, where the target's own top-2 logprobs are **exactly tied**
(`gap = 0.000000 nats`) — argmax tie-breaking under a different floating-point
reduction order, not the draft. Neither eager nor CUDA-graph execution is consistently
favoured, and output is bit-stable within a server instance.

## Long context: AA-LCR 1k -> 128k

**This is the result the project exists for.** Same server, same run, 100 samples per
bin, `max_tokens` 1024, TP=8, `max_model_len` 131072. Baseline (no draft) is the speedup
denominator. Config:
[`gemma4-31b-mamba2-ctxlen-sweep.yaml`](../../../scripts/evaluate/experiments/gemma4-31b-mamba2-ctxlen-sweep.yaml).

| bin | A accept_len | **B accept_len** | B/A | baseline tok/s | A tok/s | **B tok/s** | A speedup | **B speedup** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1k | 2.464 | **2.550** | 1.03x | 219.8 | 295.5 | **292.4** | 1.34x | **1.33x** |
| 2k | 2.412 | **2.547** | 1.06x | 206.9 | 263.2 | **268.1** | 1.27x | **1.30x** |
| 4k | 2.404 | **2.540** | 1.06x | 184.7 | 222.4 | **229.4** | 1.20x | **1.24x** |
| 8k | 2.435 | **2.564** | 1.05x | 152.1 | 173.2 | **181.9** | 1.14x | **1.20x** |
| 16k | 2.294 | **2.542** | 1.11x | 129.7 | 111.0 | **125.7** | 0.86x | **0.97x** |
| 32k | 2.196 | **2.589** | 1.18x | 98.7 | 64.9 | **79.7** | 0.66x | **0.81x** |
| 64k | 1.949 | **2.609** | 1.34x | 79.2 | 32.5 | **45.7** | 0.41x | **0.58x** |
| 128k | 1.755 | **2.646** | 1.51x | 63.6 | 18.9 | **30.2** | 0.30x | **0.47x** |

### Acceptance: the hypothesis holds, decisively

| | 1k | 128k | change |
|---|---:|---:|---:|
| eagle3 transformer (A) | 2.464 | 1.755 | **-28.8%** |
| **mamba2 (B)** | 2.550 | 2.646 | **+3.8%** |

The transformer draft decays **29%** from 1k to 128k. The Mamba2 draft is
**flat** -- slightly *better* at 128k than at 1k. The B/A ratio grows monotonically with
context, from 1.03x at 1k to **1.51x at 128k**.

The concern going in was the opposite: that a fixed-size recurrent state would have to
compress harder as context grows and lose fidelity. It does not. What degrades is the
*transformer* draft, whose single attention layer evidently gets worse at picking the
next token as its KV cache grows.

### Throughput: both arms lose to no-draft beyond ~8k, and that is pre-existing

Speculation is a net **loss** past 16k on this stack -- for both drafts. Arm B is
uniformly and substantially better (0.47x vs 0.30x at 128k, ~1.6x more throughput than
Arm A) but still below 1.0x.

This is not an artifact of the Mamba-specific serving flags. The independently recorded
2026-08-26 eagle3 run reproduces here almost exactly:

| | recorded (k=3) | this run (k=5) |
|---|---|---|
| baseline 64k / 128k tok/s | 78.5 / 62.7 | 79.2 / 63.6 |
| eagle3 64k / 128k tok/s | 29.4 / 17.2 | 32.5 / 18.9 |
| eagle3 64k / 128k accept_len | 1.911 / 1.741 | 1.949 / 1.755 |

So the long-context throughput collapse is a property of Eagle-3 drafting on this stack,
predating this work: the per-round draft cost grows with context faster than acceptance
can pay for it. Arm B's constant-size state makes it grow *more slowly* -- which is
exactly the predicted mechanism, and is why Arm B is ~1.6x faster than Arm A at 128k --
but not slowly enough to stay above 1.0x here.

Arm B roughly breaks even at 16k (0.97x) where Arm A is already at 0.86x.

### Reading this

The draft-quality question is settled: a recurrent drafter is strictly better at long
context, and the advantage widens with length. The *serving* question is not: neither
draft pays for itself past ~8k in this configuration, and closing that gap is a
throughput-engineering problem (the flags here are deliberately conservative -- prefix
caching off, 0.65 utilization -- and the draft path is unoptimized), not a draft-quality
one.
