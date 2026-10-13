# Memory index

- [B200 box setup](b200-box-setup.md) — 5-node fleet 097-101, SLURM from bastion, linger requirement, podman/NFS flags, image tar, dspark-host venv
- [DSpark run recipe](dspark-run-recipe.md) — flag=TRUE decision + patch requirement, presets/sbatch wrappers, ACCUM/SCHED_TOTAL sizing, SAVE_BEST=0
- [DFlash2 port assessment](dflash2-port-assessment.md) — what DFlash2 adds (conv + selector), where upstream code lives, our fork lacks it, conv-first test plan
- [Gemma4 DSpark data provenance](gemma4-dspark-data-provenance.md) — both corpora on-policy (Gemma-regenerated), 782k = kimi-regen + Nemotron-v2 stem/math/code regen by Ravi
- [DSpark results & status](dspark-results-status.md) — official runs + headline numbers, all-sliding verdict + NEW: serve-time YaRN rescues full-attn decay (Kimi option 3 live), open threads incl. yarn k3 pass, redhat bins, fullattn1, PR to Ravi
