---
name: dflash2-port-assessment
description: "What DFlash2 (z-lab, Aug 2026) adds over DFlash/DSpark, where the code lives upstream, and the 2026-09-23 assessment of porting conv/selector onto our Gemma4 DSpark draft"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7bb84d2c-495c-4925-8fbf-bccba4d7f8e8
  modified: 2026-09-28T01:52:20.446Z
---

DFlash2 (z-lab/inco.ai, Aug 2026; blog inco.ai/blog/dflash2, HF z-lab/Qwen3.8-27B-DFlash2)
= DFlash + (a) two-tap grouped dynamic causal conv wrapping attn+MLP per draft layer
(conv_kernel_size 2, conv_group_size 16, intra-block only, identity-init) and (b) a
candidate selector: top-16 unary candidates per slot, bilinear score
unary + <pred_codebook[prev] * Hproj(h), succ_codebook[cand]>, rank 256, sequential
walk in a Triton kernel. Blog ablation (Qwen3.5-4B): selector alone +0.34 (T0)/+0.47 (T1),
conv alone "close to 15-layer DFlash"; both 4.92 -> 5.97. Confidence head optional (vLLM
PR 55668) = same as DSpark's.

Code: upstream speculators main has src/speculators/models/dflash2/ (#1006, 7a58fc5) and a
`_make_decoder_layer` factory in dflash/core.py; our fork mengmengj/dspark-upstream is
~95 commits / 200 files behind and has NO dflash2. vLLM main: qwen3_dflash2.py
(DFlashGroupedConv, CandidateSelector) + v1/worker/gpu/spec_decode/dflash2/speculator.py;
gemma4_dspark.py has no conv hook, so a Gemma DSpark+conv checkpoint needs a vLLM patch.
vLLM DSpark speculator already has `dspark_draft_topk` (top-k pruned Markov walk).

Prior art for DSpark+conv: TIE-Pilot/dspark-attnconv-block7-qwen3-4b (arXiv:2608.27339):
attention-style Markov head + kernel-2/group-16 short conv + slot embeddings = +3.8% over
official DSpark on Qwen3-4B. Small gain once a Markov head already exists.

**State 2026-09-25:** branch `mengmengj/dspark-conv` (off dspark-upstream at 8fefab7,
which committed the YaRN plot edits). Cherry-picked upstream 7a58fc5 (#1006) as b2d79f3;
the auto-merge silently dropped the `is_dflash_family` num_layers default in schema.py
-> restored in 14d4409 (lesson: cherry-picks of upstream schema.py need the unit tests,
not just a clean apply). Gate passed in container: tests/unit/models + tests/unit/train
green (3 GPU-only tests need --device), `scripts/train.py --dry-run` with the
checkpoint-of-record recipe OK. 2026-09-27: variant DONE (034b0fc): DSparkSpeculatorConfig.conv_kernel_size/
conv_group_size (0 = off), CLI --dspark-conv-kernel-size/--dspark-conv-group-size,
speculators_model_type stays "dspark". Preset run_nemo782k_conv_local.sh +
sbatch_nemo782k_conv.sh = scratch recipe with DSPARK_CONV_KERNEL=2 (26 ep, SCHED_TOTAL
38700). Dry-run OK: 20 conv tensors, 72.5M params. USER DECISION: train FROM SCRATCH
(not warm-start) for a 1:1 comparison vs gemma4_31b_dspark_nemo782k_scratch. User
submits from bastion: `sbatch examples/train/sbatch_nemo782k_conv.sh`. LAUNCHED 2026-10-13 as SLURM job 319 on node 097 (wandb nemoconv782k, online); user
cancelled it at ~10:33 (micro-step ~2970, epoch 0, no checkpoint) to move machines.
Early metrics looked normal (loss 0.57, accept_len 1.58 at step 2965). Preset now
defaults WANDB_RUN_ID=nemoconv782k_v2 for the from-scratch relaunch (wandb resume
would drop steps below 2970 under the old id). Orphan /dev/shm/hidden_states_*conv
may remain on 097. Full handoff: examples/train/HANDOFF.md. Serving a conv
checkpoint still needs a vLLM gemma4_dspark.py patch (DFlashGroupedConv exists in
qwen3_dflash2.py); compare on val accept_len first.
Upstream remote `upstream` = vllm-project/speculators is now configured in the repo.

**Assessment 2026-09-23:** test conv first (orthogonal to Markov head, warm-startable from
checkpoints/24 because identity-init, long-context safe since intra-block, measurable via
train-side val accept_len before any vLLM work). Skip the selector: functionally overlaps
the Markov head and needs a serving rewrite. Gate: val accept_len gain >= ~0.1-0.2 at
matched exposure before writing the Gemma4 vLLM conv patch. [[dspark-results-status]]
[[dspark-run-recipe]]
