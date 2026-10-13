---
name: dspark-run-recipe
description: "Current DSpark training/eval recipe on the B200 fleet — flag=TRUE decision, presets, sbatch wrappers, sizing rules"
metadata: 
  node_type: memory
  type: project
  originSessionId: 976456e9-265b-4d6a-afd3-854bb7665b28
  modified: 2026-08-20T00:48:33.548Z
---

DSpark (gemma-4-31B-it) training recipe, current as of 2026-08-18:

- **Train with `--sample-from-anchor` (SAMPLE_FROM_ANCHOR=1). Decided 2026-08-14** after a
  controlled ablation at matched steps: True wins every same-k comparison (+0.2–1.3
  accept_len, biggest on math), native k=8 adds throughput, outputs stabler under
  speculation. Serving True checkpoints REQUIRES the algos.py patch
  (`dspark_bonus_anchor = not sample_from_anchor`) — baked into the
  `localhost/mengmengj-vllm-official` image and `/sms-scratch/mengmengj/envs/dspark-host`.
- Launch: preset script exports env → `examples/train/run_local_podman.sh` → podman.
  Via SLURM: `sbatch examples/train/sbatch_nemo782k*.sh` from the BASTION (self-healing
  wrappers). Full map in `examples/train/README.md`.
- Sizing: global batch ≈512 = 5.6 conv/step ×4 ranks × ACCUM 23 (349k corpus) or
  7.4 conv/step × 4 × ACCUM 18 (nemo782k). `SCHED_TOTAL` must be set explicitly on any
  data change or extension (driver's built-in epoch length is stale). Extensions are warm
  restarts: same RUN_NAME, larger EPOCHS, recomputed SCHED_TOTAL.
- SAVE_BEST must stay 0 (=1 disables sub-epoch saving, trainer.py:517).
  CHECKPOINT_FREQ=0.25; checkpoints 13GB; TRAIN_DATA_RATIO=0.998.
- Eval: 1-GPU bare-metal (dspark-host venv), `run_experiments.py` + yaml in
  scripts/evaluate/experiments; rates come from vLLM counters (k-correct by
  construction); compare accept_len only within same k; across k compare tok/s.
- Never delete pinned wandb runs — deleted ids are burned server-side and crash resumes
  (cost us a day). [[b200-box-setup]] [[dspark-results-status]]
