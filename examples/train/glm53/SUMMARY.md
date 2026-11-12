# GLM-5.3 DFlash2 draft — what we did and what we learned

*2026-10-30 → 2026-11-12 · branch `ravir/glm53-dflash2-kimi-recipe`*

## TL;DR

- **Our best draft from our own training reaches 77–80% of incoai's acceptance.** It was trained off-policy for 10 epochs, then fine-tuned for 2 epochs on GLM-5.3's own answers.
  - 22-bench FP8: accept_len 2.99 vs 3.89 (2.01× vs 2.53× speedup).
  - 8-bench suite: 3.14 vs 3.95.
- **Why we trail incoai:**
  - **Serving stack: 0%.** The gap is the same on SGLang and on vLLM.
  - **Architecture: about 11%.** This covers incoai's target layers, its draft shape, and block size 8.
  - **Everything else: training budget and breadth.**
    - On the *identical* architecture, incoai's weights score 3.95 and our 2-epoch run scores 2.88.
    - Our data is **not** the problem: fine-tuning incoai on it *improves* incoai.
- **New best draft overall:** incoai warm-started and fine-tuned for half an epoch on our clean on-policy data. It scores **4.04 vs incoai's 3.95 (+2.3%)**, up on 6 of 7 benchmarks and higher at every draft position.
  - Path: `glm53_dflash2/output/incoai_ft_onpolicyA_half/checkpoints_vllm/final`, served with k=7.
- **The trainer's validation `accept_len` is a poor serving proxy for DFlash2.** On that fine-tune it fell 5.7% while serving rose 2.3% (see [§4.4](#44-incoai-scored-by-our-pipeline-and-fine-tuned-on-our-data)). Rank drafts by serving evals.
- **Recommendation:** use "incoai warm start + our on-policy fine-tune" as the production recipe.
  - Next: a full epoch, then add the multilingual regen ([§5](#5-multilingual-prompts-for-on-policy-regeneration)) and function-calling data.

All paths below are relative to `/import/ml-sc-scratch1/ravir/spec_decoding_work/` unless they are absolute or repo paths.

---

## 1. Timeline

| when | what | outcome |
|---|---|---|
| Oct 30 – Nov 3 | **Stage 1:** Kimi-K3-recipe DFlash2, off-policy, 10 epochs | val accept_len 3.307 (epoch 9); 22-bench 2.83 |
| Nov 3 – 4 | **Stage 2a:** on-policy FT on the first GLM regen snapshot | **contaminated** (looping shard-B rows), superseded |
| Nov 4 – 6 | **Stage 2b:** on-policy FT on clean shard-A data ("ftA") | 22-bench 2.99 / 2.01×; our best self-trained draft |
| Nov 10 | Gap plan agreed; multilingual prompt set built (152k prompts) | handed to a colleague for GLM-5.3 regeneration |
| Nov 10 – 11 | **Phase 0:** does the serving stack hide the gap? (SGLang vs vLLM) | no |
| Nov 10 – 11 | **Phase 1:** architecture ablations A0–A3 | architecture ≈ 11% |
| Nov 11 | **Phase 2:** incoai scored by our val pipeline, then fine-tuned on our data | our data helps; new best draft 4.04 |
| Nov 12 | All GPUs released | — |

## 2. Shared setup

**Training**
- 3 B200 nodes per run:
  - 2 nodes serve the bf16 GLM-5.3 verifier (TP8×PP2, vLLM 0.26 + `vllm_pp_aux` plugin, `--block-size 64`).
  - 1 node runs the 8-rank speculators trainer.
- Hidden states move between them through Mooncake.
- Launcher: `launch_glm53_dflash2_salloc.sh` → `run_glm53_dflash2_3node.sh`.
- The ablations ran from frozen code snapshot `glm53_dflash2/code_295f434`.

**Serving eval**
- Our harness: `run_experiments.py` + `run_vllm_eval.py`, via `run_eval_glm53_1node.sh`.
- Server: official FP8 GLM-5.3, TP8 on **one** 8×B200 node, vLLM 0.33. bf16 doesn't fit one node, and DFlash2 can't run with pipeline parallelism.
- Sampling: T 1.0, top-p 0.95, `reasoning_effort=max`, max_tokens 4096.
- Two benchmark suites:

| suite | benchmarks | prompts per benchmark | config |
|---|---|---|---|
| 22-bench | full suite | 50 | `eval_glm53_full22_fp8_v033.yaml` |
| 8-bench (ablation) | aime, gpqa, livecodebench, gsm8k, math500, humaneval, mbpp, mt-bench | 50 (aime 30) | `eval_glm53_ablations_fp8_v033.yaml` |

- **gsm8k is excluded from 8-bench means.** It swung ±13% between two identical runs. Without it, a repeat run of A0 agrees within 0.2% (2.583 vs 2.588).
- `accept_len` is vLLM's mean accepted tokens per verify step, +1 for the bonus token.
- Per-position acceptance is draft-step-weighted from each server's `SpecDecoding metrics` lines.

**Draft architectures**

| | ours (Kimi recipe, A0) | incoai (A3 = incoai's exact architecture) |
|---|---|---|
| decoder layers | 5 | 6 |
| attention | 64 heads × head_dim 64, 16 KV heads | 64 heads × head_dim 128, 8 KV heads |
| MLP / sliding window | 12288 / 2048 | 12288 / 2048 |
| target layers tapped (speculators aux ids) | 7 20 44 57 71 | 6 20 34 48 62 76 (= incoai's `target_layer_ids` + 1) |
| block size / served k | 7 / 6 | 8 / 7 |
| config | `glm53_dflash2_draft_config.json` | `glm53_dflash2_draft_config_incoai_shape.json` |

## 3. Stages 1–2: our own training lineage

Full details, the per-benchmark 22-bench tables and the contamination analysis are in [`results/README.md`](results/README.md).

| draft | recipe | 22-bench accept_len / speedup |
|---|---|---|
| `ours_ep9` | Kimi K3 recipe: off-policy gemma-4-31B answers on nemo782k (782k rows), 10 epochs, AdamW 5e-5, 16k sequences, forward KL; best = epoch 9 (val accept_len 3.307) | 2.83 / 1.90× |
| `ftA` half-epoch | continued FT of `ours_ep9` on **clean** GLM-5.3 on-policy regen (shard A, 141,818 rows), LR 2e-5 | 2.98 / 2.00× |
| `ftA_ep2best` | same run, best checkpoint (end of epoch 2) | **2.99 / 2.01×** |
| incoai | released GLM-5.3-DFlash2 | 3.89 / 2.53× |

**Lessons from stages 1–2:**
- On-policy fine-tuning gives a fast +5%, then plateaus after about half an epoch.
- The first on-policy snapshot was 56% shard-B rows generated on 2-node PP=2 pairs. Those rows loop until max_tokens, so we threw that run away. Only shard A (single-node generation) is clean.

## 4. Gap analysis: why incoai is ~25% better

The question: is the gap architecture, training method, or data? We split it into three independent tests.

### 4.1 Phase 0 — the serving stack isn't hiding the gap

We served both drafts on the same FP8 B200 node, on the model card's 5 tasks, with SGLang git main (DFLASH, FA4 draft attention) and with vLLM 0.33:

| accept_len | ours vLLM | ours SGLang | incoai vLLM | incoai SGLang | incoai card (GB300) |
|---|---|---|---|---|---|
| gsm8k | 4.24 | 3.96 | 5.42 | 5.10 | 5.94 |
| math500 | 4.06 | 3.93 | 4.82 | 5.34 | 6.02 |
| humaneval | 3.65 | 3.45 | 4.67 | 4.96 | 5.48 |
| mbpp | 3.15 | 3.10 | 4.11 | 4.48 | 4.95 |
| mt-bench | 2.35 | 2.32 | 2.96 | 3.22 | 4.19 |
| **mean** | **3.49** | **3.35** | **4.40** | **4.62** | 5.32 |

- SGLang moves us −4% and incoai +5%, so our share of incoai is 79% on vLLM and 73% on SGLang. vLLM isn't shortchanging us; the gap is in the weights.
- The remaining distance to incoai's card numbers is most likely their prompt sets and sample counts.
- We dropped the planned bf16 check: §4.4 scores incoai on bf16 hidden states anyway.
- Raw results: [`results/p0-sglang-card-fp8/`](results/p0-sglang-card-fp8/). Runner: `sglang_eval/run_sglang_card_eval.sh`.

### 4.2 Phase 1 — architecture ablations

Every run trained from scratch on the same clean shard-A data for 2 epochs at LR 5e-5. Each step adds one piece of incoai's architecture.

| run | change | val accept_len ep1 → ep2 | 8-bench serving (excl. gsm8k) | step | share of incoai |
|---|---|---|---|---|---|
| A0 | control (our architecture) | 2.004 → 2.135 | 2.583 (repeat run 2.588) | | 65% |
| A1 | + incoai's 6 target layers | 2.055 → 2.192 | 2.672 | +3.4% | 68% |
| A2 | + incoai draft shape (6 layers, head_dim 128, 8 KV heads) | 2.195 → 2.354 | 2.833 | +6.0% | 72% |
| A3 | + block 8 (k=7) = incoai's exact architecture | 2.199 → 2.362 | 2.878 | +1.6% | 73% |
| ftA_ep2best | our architecture, our full recipe (10 + 2 epochs) | 2.630 (ep2) | 3.141 | | 80% |
| incoai | incoai's weights | 3.526 (see §4.4) | 3.949 | | 100% |

Per-benchmark serving accept_len:

| | aime | gpqa | lcb | gsm8k* | math500 | humaneval | mbpp | mt-bench | mean excl. gsm8k |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 2.89 | 1.88 | 2.58 | 3.67 | 3.31 | 2.86 | 2.59 | 1.97 | 2.583 |
| A1 | 2.99 | 1.94 | 2.63 | 3.30 | 3.43 | 2.96 | 2.72 | 2.03 | 2.672 |
| A2 | 3.17 | 2.06 | 2.77 | 3.70 | 3.63 | 3.22 | 2.83 | 2.15 | 2.833 |
| A3 | 3.23 | 2.06 | 2.79 | 3.68 | 3.74 | 3.28 | 2.92 | 2.11 | 2.878 |
| ftA_ep2best | 3.50 | 2.31 | 3.04 | 4.26 | 4.03 | 3.63 | 3.14 | 2.34 | 3.141 |
| incoai | 4.39 | 2.78 | 3.90 | 5.44 | 4.89 | 4.60 | 4.10 | 2.97 | 3.949 |
| incoai + our FT | 4.38 | 2.80 | 3.92 | 5.46 | 5.05 | 4.81 | 4.32 | 3.00 | **4.041** |

\* gsm8k is shown but too noisy to compare.

**Findings:**
- **Target layers: +3%.**
- **Draft shape: +6%.** Its gain is largest at the later draft positions.
- **Block 8: +1–2%.** A3's per-position acceptance matches A2's, and the extra 7th token is accepted 3.4% of the time. Block 8 can only pay off once the first 6 tokens are usually accepted.
- **All of incoai's architecture together: +11%.** Data and training on the *same* architecture gave +22% (A0 → ftA).
- Validation and serving agree on the *direction* of every architecture step.

### 4.3 Where the gap shows: acceptance along the block

Unconditional per-position acceptance from the vLLM server logs (8-bench run):

| position | 1 | 2 | 3 | 4 | 5 | 6 | 7 | server accept_len |
|---|---|---|---|---|---|---|---|---|
| A0 | .645 | .381 | .217 | .119 | .064 | .033 | | 2.46 |
| A3 | .678 | .427 | .260 | .159 | .097 | .058 | .034 | 2.71 |
| ftA_ep2best | .692 | .467 | .319 | .224 | .163 | .119 | | 2.98 |
| incoai | .751 | .566 | .435 | .340 | .272 | .221 | .182 | 3.77 |
| incoai + our FT | .754 | .574 | .444 | .351 | .283 | .232 | .192 | 3.83 |

- incoai is only 6–9 points ahead at position 1, but its acceptance decays far more slowly: 0.18 at position 7, vs 0.03 for A3.
- Staying correct deep into the block is exactly what more training buys: A0 → ftA moves positions 4–6 the most.

### 4.4 incoai scored by our pipeline and fine-tuned on our data

- We imported incoai's HF/vLLM checkpoint as a speculators checkpoint with `import_dflash2_from_hf.py`. Its embed/lm_head come from the GLM-5.3 verifier, and it maps 1:1 onto the A2/A3 shape.
- Two runs on the A2 nodes, both with the same validation split and the same bf16 hidden states as A0–A3:
  - **(a) val-only:** one step at LR 0.
  - **(b) fine-tune:** half an epoch (1,824 steps) on our shard-A data at LR 2e-5.

| trainer validation | incoai as released | incoai + ½ epoch our data | ftA_ep2best (ours) | A3 ep2 |
|---|---|---|---|---|
| accept_len (1−TV based) | **3.526** | 3.325 (−5.7%) | 2.630 | 2.362 |
| greedy acc pos 1 / 4 / 7 | .825 / .405 / .248 | .830 / .411 / .255 | .755 / – / – | .738 / – / – |
| teacher-forced selector acc | .742 | .743 | .663 | .620 |
| top-16 oracle accept length | 5.77 | 5.86 | 4.76 | 4.61 |
| val loss | 1.854 | 1.520 | 2.095 | 2.194 |
| **FP8 serving, 8-bench excl. gsm8k** | **3.949** | **4.041 (+2.3%)** | 3.141 | 2.878 |

- **(a) The gap is the same in bf16, teacher-forced validation** (ours = 75% of incoai) as in FP8 serving. So it isn't FP8 or the serving stack; it's the weights.
- **(b) Our on-policy data helps a strong draft.**
  - Serving improves on 6 of 7 benchmarks (aime flat), and acceptance is higher at every position.
  - Greedy accuracy, candidate recall and the oracle all improve too.
- **The trainer's `accept_len` dropped anyway.**
  - It is the cumulative product of (1 − TV distance) between the *unary* head's distribution and the target's: an analytical T=1 acceptance that **ignores the candidate selector** (`src/speculators/models/dspark/metrics.py`, around lines 188–251).
  - Our CE-style loss sharpens the unary distribution, which lowers that overlap even as the real (selector-based) serving acceptance rises.
  - Use serving evals, or greedy position accuracy and selector metrics, to rank DFlash2 drafts.

### 4.5 Decomposition

| factor | effect | evidence |
|---|---|---|
| serving stack (vLLM vs SGLang), FP8 vs bf16 | ~0% | §4.1; §4.4(a) ratio identical in bf16 val |
| target layers | +3% | A0 → A1 |
| draft shape | +6% | A1 → A2 |
| block 8 | +1–2% | A2 → A3 |
| **training budget / breadth** | **the rest: 2.88 → 3.95 (+37%) on identical architecture** | A3 vs incoai; same architecture, our data improves incoai (+2.3%) |
| our data quality | not negative | §4.4(b) |

What exactly incoai did differently in training is unknown. Candidates: much more data, more epochs, broader or multilingual prompts, a loss closer to the sampling distribution.

## 5. Multilingual prompts for on-policy regeneration

`multilingual_data/prompts_v2/`, 152,496 prompts: `prompts_multilingual_all.jsonl`, per-language files, `README.md` and `manifest.json`.
- Builder: `multilingual/build_multilingual_prompts.py`, commit `25eac5c`.
- Handed to a colleague for GLM-5.3 regeneration on single nodes (not 2-node PP, which is what corrupted shard B).
- Motivation: speed-multilingual (0.58 of incoai) and translation (0.67) were our worst 22-bench ratios.

| language | prompts | sources |
|---|---|---|
| es, fr, de, ja | 15,000 each | Nemotron-Post-Training-v2 10k (stratified by category) + WildChat-4.8M 5k |
| it | 7,500 | Nemotron v2 5k + WildChat 2.5k |
| zh | 25,000 | WildChat-4.8M |
| ru, ar, ko, pt, vi | 7,500 each | WildChat-4.8M |
| th | 7,500 | WangchanThaiInstruct 7,014 + WildChat 486 |
| translation | 14,996 | OPUS-100 pairs as translation prompts |

- 14,259 duplicates removed.
- 7 prompts dropped for overlapping eval benchmarks.
- Nemotron v2 covers only es/fr/de/it/ja, and its raw split is about 60% math, hence the stratified sampling and the other sources.

## 6. Artifacts

Under `glm53_dflash2/`:

| what | path |
|---|---|
| **best draft (incoai + ½ epoch our data), vLLM format, k=7** | `output/incoai_ft_onpolicyA_half/checkpoints_vllm/final`; weights symlink to `checkpoints/0/model.safetensors`, currently mode 600 (run `chmod 644` before sharing) |
| best self-trained draft (ftA epoch 2), vLLM format, k=6 | `output/glm53_dflash2_onpolicyA_ft3ep_lr2e-5/snapshots_vllm/ep2_end_best` |
| Kimi-recipe epoch 9 (speculators) | `output/glm53_dflash2_gemma782k_kimirecipe_v2/checkpoints/8` |
| A0–A3 final drafts (vLLM format) | `output/glm53_abl_{A0_control,A1_incoai_taps,A2_incoai_shape,A3_incoai_shape_block8}/checkpoints_vllm/final` |
| incoai as a speculators checkpoint | `output/incoai_imported/speculators` |
| clean on-policy training data | `data/glm53_onpolicy_snap20pct_shardA_s16k` (141,818 rows) |
| run dirs / logs | `runs/<run>_<timestamp>/`, `logs/` |
| full eval outputs incl. server logs | `eval/glm53-ablations-fp8-v033/`, `eval/p0-sglang-card-fp8/`, `eval/glm53-full22-fp8-v033/` |
| SGLang env with DFlash2 (git main 6047c17, torch 2.14.1+cu130) | `/import/ml-sc-scratch1/ravir/envs/sglang_dflash2` |

**Raw numbers in this repo:**
- [`results/glm53-ablations-fp8-v033/`](results/glm53-ablations-fp8-v033/): summaries, per-prompt details and `per_position.txt` for every 8-bench eval.
- [`results/p0-sglang-card-fp8/`](results/p0-sglang-card-fp8/)
- [`results/val_metrics_gap_analysis.json`](results/val_metrics_gap_analysis.json): every validation metric for A0–A3, ftA and both incoai runs.
- To refresh, run `results/collect_gap_analysis.py` with the `glm53_dflash2` env python.

**Code (commits on this branch):**

| commit | what |
|---|---|
| `295f434` | driver overrides `TARGET_LAYER_IDS` / `DRAFT_CONFIG` / `BLOCK_SIZE` for the ablations |
| `25eac5c` | multilingual prompt builder |
| `c156b98`, `0358aeb` | 8-bench ablation eval config |
| `e4cd08d` | SGLang card-reproduction eval (`sglang_eval/`); `export_dflash2_vllm.py` now writes `sample_from_anchor: false` explicitly |
| `46f56f5` | `import_dflash2_from_hf.py` (HF/vLLM DFlash2 → speculators) + 8-bench reference entries |

## 7. Gotchas worth remembering

- **gsm8k at n=50 is noise** (±13% between identical runs). Exclude it, or run twice.
- **Trainer `accept_len` ≠ serving** for DFlash2 (§4.4). It was fine for ranking architecture ablations, but wrong for comparing training recipes.
- **Never generate on-policy data on 2-node PP=2 pairs.** Those rows loop until max_tokens; that was the shard-B contamination.
- **SGLang DFlash2 needs git main.** Build it with `SGLANG_BUILD_RUST_EXTS=none` (no cargo on the cluster).
- **vLLM 0.28's DFlash2 is broken** (accept_len ~1.4); use 0.33.
- **vLLM PP hidden-state extraction:**
  - needs the `vllm_pp_aux` plugin and `--block-size 64`;
  - an lcm workaround silently corrupted ~30% of hidden states under load, so always load-test extraction.
- **sc3 compute nodes:**
  - bash 5.2 rejects regexes inside `"${VAR:-...}"` defaults;
  - `~/.local/bin/env` shadows `/usr/bin/env`;
  - system python3 is 3.6.
- **Daily B200 reservation `sn50-hdi-1`** runs 09:00–00:00. New jobs must start inside the window; running jobs continue past it (FLEX).

## 8. Recommended next steps

1. **Adopt "incoai warm start + our on-policy FT"** as the production draft (4.04 today).
   - Run a full epoch (3,648 steps, ~6.5 h on 3 nodes) to see whether it keeps climbing. Our own drafts plateaued after half an epoch of on-policy FT.
2. **Add the multilingual regen** (§5) and the function-calling data to that fine-tune. The weakest 22-bench categories are coverage problems, not on-policy problems.
3. **A from-scratch draft of our own on the incoai architecture** needs a much larger budget: the 10-epoch off-policy stage + FT, ~4–5 days on 3 nodes. It would likely still trail option 1. Worth it only if we need independence from incoai's weights.
4. **If we keep training our own drafts, try a loss closer to the sampling distribution** (TV/KL on the unary head) and add a selector-aware validation metric. That way, validation tracks serving again.
