# Speculative-decoding experiments, by target model

Index of every experiment write-up in this directory, grouped by the **target
model** being accelerated. Each target has its own draft models, its own
baselines, and its own numbers — results are **not** comparable across targets.

Start here rather than in a results doc: several of them contain superseded
conclusions that later sections correct, and the "state" column below says
which to trust.

---

## gemma-4-26B-A4B-it — MoE, 30 layers

The most heavily worked target. Five draft families trained, 25+ benchmarks,
production traffic, and agentic load.

| doc | covers | state |
|---|---|---|
| [gemma4_26b_moe_results.md](gemma4_26b_moe_results.md) | **main doc, 19 sections.** Six-way draft comparison, 400k DSpark scale-up, DFlash warm start, the Ω diagnosis, AgentX, assistant-vs-DSparkFlash | current; **read §0 first** — it indexes by draft model and flags which sections are superseded |
| [dspark_topk_context_ab.md](dspark_topk_context_ab.md) | first A/B of DSA-style top-k *context* selection (indexer vs dense) on 641-token data | superseded by §18 of the main doc, which re-ran it at 11k context |

**Current best draft:** DSparkFlash (DSpark warm-started from stock DFlash)
served with `dspark_draft_topk=64` — 1.877× at batch 1, 1.536× at batch 128,
clean protocol. See main doc §0 and §19.

**Key finding:** acceptance length is nearly identical across the leading
drafts (4.84–4.92), so the whole ordering is decided by **Ω, the draft's own
serving cost** — not by drafting quality. Main doc §15.

---

## gemma-4-31B-it — dense

The MoE sibling's dense counterpart. Far less written up than the 26B, despite
carrying the majority of the eval configs in
`scripts/evaluate/experiments/*.yaml` (27 of 41 reference this backbone).

| doc | covers | state |
|---|---|---|
| [gemma4_31b_results.md](gemma4_31b_results.md) | MTP assistant k-sweep (4×A100, tp=4), and pointers to 31B material elsewhere | thin — 2 sections; most 31B runs are not written up |

**Gap worth knowing:** a published DSpark exists for this target
(`RedHatAI/gemma-4-31B-it-speculator.dspark`, downloaded to
`/nvmedata/hf_checkpoints/`) and has never been benchmarked here. Its config is
the reference recipe the 26B work kept comparing itself against
(5 layers, SWA 2048, block 8, `draft_vocab_size=32000`).

---

## Kimi K3

| doc | covers | state |
|---|---|---|
| [kimi_k3_draft_eval.md](kimi_k3_draft_eval.md) | **16 sections.** TorchSpec EAGLE3 vs published DSpark drafts, plus three architecturally distinct DSpark checkpoints (`Inferact/`, `lightseekorg/`, native vs converted) | current |
| [dspark_sliding_window_ablation.md](dspark_sliding_window_ablation.md) | **9 sections.** Can a full-attention DSpark checkpoint be forced into sliding-window attention at serve time with no retraining, and what does it cost? | current; the origin of the "the draft's useful context is local" finding that the 26B DSA work kept running into |

---

## Cross-target

| doc | covers | applies to |
|---|---|---|
| [gemma4_mtp_vllm_hidden_shift_bug.md](gemma4_mtp_vllm_hidden_shift_bug.md) | **12 sections.** Why from-scratch Gemma4-MTP drafts looked perfect in training and collapsed to accept_len ≈ 1.07 in vLLM: the trainer fed the draft the target hidden state from the wrong position. Root cause + fix | whole Gemma4 family; read before training any MTP draft |

---

## Targets with training recipes but no results doc

Example launchers exist in [`examples/train/`](../../examples/train/) for these,
but no evaluation has been written up:

| target | launchers |
|---|---|
| Qwen3-8B | `dflash_qwen3_8b_*.sh`, `eagle3_qwen3_8b_*.sh`, `peagle_qwen3_8b_*.sh` |
| Qwen3-0.6B | `dspark_qwen3_0_6b_sharegpt_online.sh` |
| Qwen3-5.9B | `mtp_qwen3_5_9b_gsm8k_online.sh` |
| Llama-3-8B | `eagle3_llama3_8b_ultrachat_offline_5k.sh` |
| GLM-5.2-FP8 | 2 configs in `scripts/evaluate/experiments/` |

---

## Before quoting any number from these docs

Three measurement effects here are larger than most results people try to
measure. All three are documented in main doc §15 and in
[`scripts/evaluate/mtp_server_eval/README.md`](../../scripts/evaluate/mtp_server_eval/README.md):

1. **Prefix caching makes throughput depend on run history** — three identical
   back-to-back runs drifted 524 → 608 → 870 tok/s. Anything measured before
   2026-10-04 carries this.
2. **Only compare arms started in the same run.** Cross-session comparison once
   produced a −0.495 accept_len "regression" where a matched control showed
   +0.005.
3. **accept_len noise is ±0.3 at `--num-samples 20`**, measured from the eval
   suite's own duplicate prompt sets. Per-benchmark deltas below ~0.3 are noise.
