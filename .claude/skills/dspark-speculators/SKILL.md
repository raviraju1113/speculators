---
name: dspark-speculators
description: Guide for implementing DSpark draft-model training support in the vllm-project/speculators codebase (or a fork of it). Use when adding the DSpark speculator type, Markov head, confidence head, DSpark losses, STS calibration, or DSpark checkpoint export; when extending DFlash code toward DSpark; or when training/validating a DSpark speculator for a target model such as Gemma 4. Triggers include mentions of DSpark, semi-autoregressive drafting, Markov head, confidence head, confidence-scheduled verification, or accepted-length parity testing.
---

# Implementing DSpark training in speculators

DSpark (DeepSeek, arXiv:2607.05147) = DFlash parallel backbone + a Markov sequential head (fixes suffix decay) + a confidence head (predicts per-position acceptance for scheduled verification). Speculators already has DFlash end-to-end, so DSpark is an extension, not a new pipeline. All exact equations, loss definitions, and pseudo-code are in [reference.md](reference.md). The authors' reference implementation is github.com/deepseek-ai/DeepSpec (MIT — porting logic is fine, keep attribution).

## Division of labor — do not implement serving

- speculators (this repo): architecture, training, calibration, checkpoint export. NO inference code.
- vLLM ≥ 0.25.0: serves DSpark natively. Our only serving obligation is exporting a checkpoint whose `speculator_config` matches vLLM's DSpark schema. Validate against a public DSpark speculator checkpoint (e.g. `siro1/glm-5.2-dspark-spec-v1` on HF) before designing config keys.
- Data prep: reuse `prepare_data.py` + `scripts/response_regeneration` unchanged.

## Before writing code

1. Check whether DSpark training already landed upstream or in a PR: search vllm-project/speculators PRs/issues for "dspark" (Red Hat previewed DSpark checkpoints, so a branch likely exists). Adopt/extend rather than duplicate.
2. Map the DFlash touch points in this repo: `grep -ri dflash --include="*.py" -l`. Every file that registers, configures, trains, or exports DFlash is a file DSpark will touch. Read the DFlash draft model class and its loss before anything else.
3. Check whether DFlash's existing loss already includes a distribution-matching (TV or KL vs target) term and position weighting `w_k = exp(-(k-1)/γ)`. DSpark reuses both if present.

## Implementation order

Work in this order; each step is independently testable.

### 1. Model: `dspark` draft model class

Subclass/compose the DFlash draft model. Add:
- **Markov head**: embedding table `W1 ∈ R^{V×r}` and projection `W2 ∈ R^{r×V}`, default `r=256`. Transition bias `B(x_{k-1}, ·) = W1[x_{k-1}] @ W2` added to the backbone's base logits at position k. See reference.md §1.
- **Confidence head**: one linear layer + sigmoid over `[h_k ; W1[x_{k-1}]]` → scalar `c_k ∈ (0,1)`. Reuses the Markov embedding `W1`. See reference.md §2.
- **Anchor-as-first-position**: input is anchor + (γ−1) mask tokens producing γ logits (DFlash originally feeds anchor + γ masks and predicts only masks). Check what the repo's DFlash does before changing.
- Embeddings and LM head: shared with the verifier and frozen — same as DFlash. Never add them to the optimizer.

Unit test: forward pass shapes; Markov bias changes logits at positions ≥1 but not the backbone hidden states; confidence output in (0,1).

### 2. Training: losses and teacher forcing

Extend the train loop with `--speculator-type dspark`:
- **Teacher forcing, no sequential loop**: at train time the Markov head consumes the ground-truth previous token `x*_{k-1}`, so all γ positions train in parallel. The left-to-right sampling loop exists only at inference (vLLM's problem).
- **Loss** = `0.1·L_ce + 0.9·L_tv + 1.0·L_conf`, each position-weighted by `w_k = exp(-(k-1)/γ)`. Definitions in reference.md §3. The TV and confidence-label terms need the target's full next-token distribution: apply the frozen shared LM head to the verifier hidden states already streamed during online training.
- **Confidence labels are soft**: `c*_k = 1 − ½·||p_draft_k − p_target_k||₁`, computed with a detached draft distribution.
- New CLI flags: `--markov-rank` (256), loss weight overrides. Reuse `--block-size`, `--max-anchors`, `--num-layers`, `--target-layer-ids`, `--sliding-window` plumbing from DFlash unchanged.
- Support `--from-pretrained <dflash_checkpoint>` warm start: load backbone weights, init Markov/confidence heads fresh. This is the de-risked path — for Gemma 4 31B-it, warm-start from `RedHatAI/gemma-4-31B-it-speculator.dflash`.

Sanity test: overfit 32 samples — L_ce → ~0, L_tv → small, confidence AUC > 0.9 on train.

### 3. STS calibration (new post-training script)

Small standalone script, run after training on a held-out split. For k = 1..γ in order: 1-D grid search a temperature `T_k` applied to `c_k` minimizing the Expected Calibration Error of the cumulative product `∏_{i≤k} c_i^{calibrated}`, holding earlier temperatures fixed. Order-preserving by construction. Write `[T_1..T_γ]` into the checkpoint config. Algorithm in reference.md §4.

### 4. Checkpoint export

Extend the speculators config/export for `dspark`: backbone (DFlash fields) + Markov head weights + confidence head weights + STS temperatures + block size. Match vLLM ≥ 0.25.0's expected schema — diff your config.json against the public DSpark speculator checkpoint field-by-field, and confirm `vllm serve <checkpoint>` loads it.

### 5. Metrics and eval

Add to the eval flow: accepted length τ per decoding round (include the bonus token — the paper does), position-wise conditional acceptance (denominator = cases where positions 1..k−1 all accepted), confidence head AUC and ECE per position.

## Validation gates (in order)

1. Unit tests (step 1–2 above).
2. `vllm serve` loads the exported checkpoint; speculative metrics appear in logs.
3. **Parity run**: train DSpark for Qwen3-8B on target-regenerated data and compare accepted length against DeepSpec's released `deepseek-ai/dspark_qwen3_8b_block7` (paper Table 1: GSM8K 6.17, HumanEval 5.52, MT-Bench 3.72, with block 7, 5 layers, temp 1.0). Within ~5% ⇒ port is correct. Expect DSpark > DFlash by ~15-18% at block 7.
4. Only then run the Gemma 4 target.

## Data recipe (no new code)

1. Prompts: `mlabonne/open-perfectblend` (HF). 2. Regenerate ALL assistant responses with the target model via `scripts/response_regeneration` (target's recommended sampling params; thinking mode matching serve-time). 3. Feed JSONL to `prepare_data.py`. Raw ShareGPT without regeneration is only for pipeline smoke tests — it trains on the wrong distribution and caps acceptance far below paper numbers.

## Pitfalls

- Do NOT implement sequential sampling in the training loop — teacher forcing only.
- Do NOT let shared embeddings / LM head receive gradients.
- TV loss needs full-vocab target probabilities; if only top-k logprobs are streamed from vLLM, extend extraction or compute the LM-head projection locally from hidden states.
- Confidence labels use total variation of full distributions, not a 0/1 accepted flag.
- Position weights matter: forgetting `w_k` skews training toward tail positions and hurts accepted length.
- STS calibrates the cumulative product left-to-right, not each `c_k` independently.
- Paper/DeepSpec numbers assume temperature 1.0 evaluation and bonus token included in τ.
