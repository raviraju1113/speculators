# DSpark training on the B200 fleet — script map

Everything here targets `google/gemma-4-31B-it` with a DSpark draft (block 8,
5 layers, Markov + confidence heads). Environment, gotchas, and history live in
`.claude/skills/` (dspark-gemma4-handoff, dspark-train-serve-parity) and
`DSPARK_GEMMA4_RUNBOOK.md` (cluster era). This file maps the scripts.

## The four official runs

| run (checkpoints under `/sms-scratch/mengmengj/output/`) | flag | data | epochs | preset | wandb id | status / result |
|---|---|---|---|---|---|---|
| `gemma4_31b_dspark_accum23x4_lr6e4` | **False** | kimi-regen 349k | 15 | `run_accum23x4_lr6e4_local.sh` | `po2pjgk1` | done; plateaued ~ep9; native k=7 mean accept_len 4.18 |
| `gemma4_31b_dspark_accum23x4_lr6e4_anchorT` | **True** | kimi-regen 349k | 15 | `run_accum23x4_anchorT_local.sh` | `anchort23x4v2` (+`txf4742q`,`anchort23x4v1` fragments) | done; plateaued ~ep9; native k=8 mean 4.55; **flag=True wins every matched comparison** |
| `gemma4_31b_dspark_anchorT_nemo782k` | True | nemo782k (continued from anchorT ep14) | 15→25 (+ext to 32) | `run_anchorT_nemo782k_local.sh` | `anchortnemo782k` | val 4.19→4.69 and climbing — data scaling works; extension in flight |
| `gemma4_31b_dspark_nemo782k_scratch` | True | nemo782k, from scratch | 16 | `run_nemo782k_scratch_local.sh` | `nemoscratch782k` | in flight; ablation: does old-corpus pretraining matter at matched exposure? |

Shared recipe: 4 vLLM engines + 4 training ranks on one 8×B200 node, lr 6e-4
linear + 4% warmup, global batch ≈512 via accumulation (ACCUM 23 on the 349k
corpus, 18 on nemo782k — sized by conversations/step), quarter-epoch saves,
`SAVE_BEST=0`, `TRAIN_DATA_RATIO=0.998`. Extensions = warm restarts: relaunch
with larger `EPOCHS` + recomputed `SCHED_TOTAL` (comments in each preset).

**Standing decision (2026-08-14): train with `--sample-from-anchor` (True).**
Same-k it adds +0.2–1.3 accept_len (huge on math), native k=8 adds throughput
on top, and its outputs are stabler under speculation. Serving True checkpoints
REQUIRES the algos.py patch (see dspark-train-serve-parity skill) — baked into
`localhost/mengmengj-vllm-official` and the `dspark-host` venv.

## Launch stack (current, B200 fleet)

- `run_local_podman.sh` — the launcher every preset execs: builds the inner
  script, podman-runs `mengmengj-vllm-official` with the required flags
  (`--security-opt label=disable` for NFS, `--pids-limit=-1`, nofile=min(hard,
  524288)), mounts `/sms-scratch` + `$HOME`, appends to `logs/<JOBNAME>.txt`,
  auto-selects wandb online/offline by `~/.netrc` presence.
- `dspark_online_gemma4_31b.sh` — the driver (runs inside the container):
  vLLM hidden-state server + torchrun training, port probing, idle-GPU guard,
  resume-from-newest-checkpoint, `SAMPLE_FROM_ANCHOR` → flag mapping.
- `sbatch_nemo782k.sh` / `sbatch_nemo782k_scratch.sh` / `sbatch_anchorT.sh` —
  SLURM wrappers (submit FROM THE BASTION). Self-heal per node: podman runtime
  dir for sessionless batch jobs, image auto-load from
  `/sms-scratch/mengmengj/docker/mengmengj-vllm-official.tar`, off-SLURM
  squatter wait loop, `podman wait` to hold the allocation, graceful stop on
  scancel.
- `sbatch_eval_final.sh` / `sbatch_eval_anchorT_final.sh` / `sbatch_eval_nemo.sh`
  — 1-GPU eval jobs (bare-metal `dspark-host` venv, no podman) running
  `scripts/evaluate/experiments/run_experiments.py` configs.

### Node rules learned the hard way
1. `loginctl enable-linger` ONCE per node before any direct (non-SLURM) run —
   logout kills rootless containers otherwise. Done: 097, 098, 101.
2. SLURM only sees SLURM: check `squeue -w <node>` AND `nvidia-smi` before
   claiming a node; jupyter/ray squatters are invisible to sinfo.
3. wandb credential = `~/.netrc`, node-local. Copy via
   `install -m 600 ~/.netrc /sms-scratch/mengmengj/.netrc.tmp` + `mv` on target.
4. Never delete pinned wandb runs — deleted ids are burned and crash resumes.

## Eval / analysis (in `scripts/evaluate/experiments/`)
- `gemma4-31b-dspark-{meeting,final,anchorT-final,nemo}.yaml`, ablation yamls —
  run configs; baseline + RedHat measured on this hardware in
  `results/gemma4-31b-dspark-meeting/results_table_b200.csv`.
- `plot_flag_comparison.py` — the True/False × epoch chart (len + rate panels).
- `analyze_k_inversion.py` — rate@k3 vs native sanity sweep.
- `verify_results.py`-style audit: served checkpoint + k are recorded in each
  experiment's `server.log`; grep before trusting any table.
- `summarize_run.py` (here) — training-log summarizer.

## Historical (SambaNova SLURM-cluster era, kept for reference)
`submit_docker.sh`, `run_bal_accum_lr6e4.sh`, `run_dp_*.sh`, `run_full_*.sh`,
`sngpu_*.sh`, `container_setup.sh`, `check_new_vm.sh`, `test_vllm_dp.sh` —
sngpu/cluster-specific; superseded by the launch stack above.
`run_overfit_*.sh` — the overfit-gate runs (keep: cheapest way to validate any
architecture/config change; see runbook §overfit gate).
Non-gemma examples (`eagle3_*`, `dflash_*`, `peagle_*`, `mtp_*`) — upstream
examples, untouched.
