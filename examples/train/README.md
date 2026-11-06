# DSpark training on the B200 fleet — script map

Everything here targets `google/gemma-4-31B-it` with a DSpark draft (block 8,
5 layers, Markov + confidence heads, draft vocab 32k). Environment, gotchas,
and history live in `.claude/skills/` (dspark-gemma4-handoff,
dspark-train-serve-parity) and `DSPARK_GEMMA4_RUNBOOK.md` (cluster era).
Obsolete cluster-era and completed one-off scripts are in `archive/`.

## Official runs & checkpoints (under `/sms-scratch/mengmengj/output/`)

| run | data | epochs | preset | wandb | result |
|---|---|---|---|---|---|
| `..._accum23x4_lr6e4` | kimi-regen 349k | 15 | `run_accum23x4_lr6e4_local.sh` | `po2pjgk1` | flag=False baseline; k=7 mean accept_len 4.18 |
| `..._accum23x4_lr6e4_anchorT` | kimi-regen 349k | 15 | `run_accum23x4_anchorT_local.sh` | `anchort23x4v2` | flag=True; k=8 mean 4.55; **True wins every matched comparison** |
| `..._anchorT_nemo782k` | nemo782k, continued from anchorT ep14 | to 24 | `run_anchorT_nemo782k_local.sh` | `anchortnemo782k` | **checkpoint of record: `checkpoints/24`** (val 4.69) — k8 aime 5.89 / gpqa 4.43 / lcb 5.25, 4.1–5.4× vs 59.5 tok/s baseline |
| `..._nemo782k_scratch` | nemo782k, from scratch | 26 | `run_nemo782k_scratch_local.sh` | `nemoscratch782k` | `checkpoint_best -> 25`; == continued run at matched exposure (curriculum irrelevant); k8 aime 5.99 / gpqa 4.22 / lcb 5.07 |
| `..._nemo782k_fullattn2` | nemo782k, from scratch, full-attn layers {0,3} | 26 | `run_nemo782k_fullattn2_local.sh` | `fullattn2nemo` | `checkpoint_best -> 25`; in-window == scratch (k8 aime 6.06/gpqa 4.25/lcb 5.13) but **decays on long context** — see architecture verdict |
| `..._nemo782k_fullattn1` | nemo782k, from scratch, full-attn layer {0} | 26 (WSD) | `run_nemo782k_fullattn1_local.sh` | `fullattn1nemo` | pending launch (node 100 needs `/dev/shm/hidden_states_*` cleanup first) |

**Standing decisions**
- (2026-08-14) Train with `--sample-from-anchor` (True): +0.2–1.3 accept_len
  same-k, native k=8, stabler outputs. Serving True checkpoints needs the
  bonus-anchor fix — baked into the `dspark-host` venv / local image; upstream
  vLLM main (post-0.26.0) derives it from checkpoint config.
- (2026-09) **Architecture: all-sliding draft (window 2048), no full-attention
  layers.** AA-LCR bins (1k–112k): all-sliding flat 3.8→4.0 accept_len;
  full-attention variants decay to ~3.0 at 112k regardless of training dose
  (ep15 == ep25). Cause: draft full layers run the local θ=10k RoPE on contexts
  they never trained on, while the target's fused hidden states already carry
  the global context. RedHat's Gemma DSpark is also all-sliding; RadixArk's
  Kimi draft is full-attn only because it ships YaRN ×16.

## Training

Stack: preset → `run_local_podman.sh` (podman launcher: label=disable,
pids-limit=-1, nofile, mounts, wandb on/offline by `~/.netrc`) →
`dspark_online_gemma4_31b.sh` (in-container driver: 4 vLLM hidden-state
engines + 4 training ranks, port probing, idle-GPU guard, resume from newest
checkpoint, hard-fail if `/dev/shm` <200G free). SLURM wrappers submit FROM THE
BASTION, self-heal per node (podman runtime dir, image auto-load, squatter
wait), hold the allocation via `podman wait`, and turn scancel into a graceful
5-min stop.

Shared recipe: lr 6e-4 + 4% warmup, global batch ≈512 via accumulation
(ACCUM_STEPS 18 on nemo782k), quarter-epoch saves, `SAVE_BEST=0`,
`TRAIN_DATA_RATIO=0.998`, 8k train seqs.

```bash
# New run, 26 epochs with WSD (warmup -> plateau -> 15% decay tail; safe to
# stop/resume mid-plateau without warm-restart LR dips):
sbatch [-w trn-b200x8-XXX] examples/train/sbatch_nemo782k_fullattn1.sh
# (preset sets EPOCHS=26 SCHED_TOTAL=38700 SCHEDULER_TYPE=wsd WSD_DECAY_RATIO=0.15)

# Extend/continue an existing linear-schedule run (warm restart — expect an
# LR climb + temporary val dip): raise EPOCHS and recompute SCHED_TOTAL
# (total steps for the FULL trajectory; comments in each preset):
EPOCHS=26 SCHED_TOTAL=38700 sbatch examples/train/sbatch_nemo782k_fullattn2.sh

# Scheduler knobs any preset understands (via run_local_podman.sh passthrough):
#   SCHEDULER_TYPE=linear|cosine|constant|wsd|none   WSD_DECAY_RATIO=0.15
#   FULL_ATTENTION_INDICES="0 3"   SAMPLE_FROM_ANCHOR=1
```

Monitoring: `summarize_run.py logs/<JOBNAME>.txt`; wandb ids above. WSD sanity:
lr pins at 6.0e-4 after warmup and stays there until the final 15% of steps.

### Node rules learned the hard way
1. `loginctl enable-linger` ONCE per node before any non-SLURM run.
2. SLURM only sees SLURM: check `squeue -w <node>` AND `nvidia-smi` — off-SLURM
   squatters are invisible and the wrappers/driver wait on idle GPUs.
3. wandb credential = `~/.netrc`, node-local. Stage via
   `install -m 600 ~/.netrc /sms-scratch/mengmengj/.netrc.tmp` + `mv` on target.
4. Never delete pinned wandb runs — deleted ids are burned and crash resumes.
5. Clean `/dev/shm/hidden_states_*` orphans per node before launching; the
   driver refuses to fall back to NFS (it once filled /sms-scratch to 100%).
6. Sick nodes excluded in every wrapper: 097 (full /tmp), 099 (full /tmp,
   flashinfer JIT fails), 101 (bad GPUs).

## Evaluation

All results land in `scripts/evaluate/experiments/results/`. Accept stats are
vLLM counter deltas: accept_rate = (accept_len − 1)/k.

```bash
# Standard benchmarks (aime/gpqa/livecodebench), 1 GPU, TP=1, ~1.5h/experiment.
# Write a yaml in scripts/evaluate/experiments/ (copy
# gemma4-31b-dspark-fullattn2-ep25.yaml), then:
CONFIG=<your>.yaml sbatch examples/train/sbatch_eval.sh

# AA-LCR context-length bins (1k..112k paired bins, the long-context accept_len
# sweep), 2 GPUs TP=2 at 131072 max-model-len, ~1.5h per k pass:
CKPT=<checkpoint dir> TAG=<name> SKIP_BASE=1 \
  sbatch examples/train/sbatch_lcr_bins.sh
#   defaults: k8 + k3 passes, method dspark. SKIP_BASE=1 skips the no-draft
#   baseline (checkpoint-independent; exists as lcr_bins_scratch_best_base).
#   Non-dspark drafts / matched-k: METHOD=dflash K_MAIN=7; SKIP_K3=1 for
#   add-on passes. E.g. RedHat's all-full-attention DFlash:
#     CKPT=/sms-scratch/checkpoints/gemma-4-31B-it-speculator.dflash \
#     TAG=redhat_dflash METHOD=dflash K_MAIN=7 SKIP_BASE=1 sbatch ...

# Chart it (auto-discovers results/lcr_bins_*):
python scripts/evaluate/experiments/plot_lcr_bins.py

# Other sweeps: sbatch_ctx_sweep.sh (synthetic-padding length sweep, TP=1),
# sbatch_lcr_sweep.sh (older 4-bucket AA-LCR real-document sweep, TP=2).
```

Plot/analysis scripts (in `scripts/evaluate/experiments/`):
`plot_lcr_bins.py` (accept_len vs context per checkpoint/k — the architecture
verdict chart), `plot_ctx_sweep.py`, `plot_nemo_vs_scratch.py`,
`plot_flag_comparison.py`, `analyze_k_inversion.py`, `tabulate_results.py`.
Hardware reference table: `results/gemma4-31b-dspark-meeting/results_table_b200.csv`.
Served checkpoint + k are recorded in each experiment's `server.log` — grep
before trusting any table.

## Validation on a new box / after config changes
`run_overfit_1gpu_offline.sh` (+ `run_overfit_extend.sh`) — the overfit gate:
cheapest way to validate any architecture/config change end-to-end (see
runbook §overfit gate). Quick checks after code changes: `DRY_RUN=1` on any
preset writes the generated inner script without launching (inspect the exact
train command), and `scripts/train.py --dry-run` inside the container
initializes + saves a checkpoint without training — it catches CLI/schema
breaks (the parser is GENERATED from
`src/speculators/train/config/schema.py` — edit there, not just
trainer.py/train.py).

## Historical
`archive/` — SambaNova-cluster-era launchers (`submit_docker.sh`, `sngpu_*`,
`run_dp_*`, `run_full_*`), completed one-off eval wrappers
(`sbatch_eval_{final,anchorT_final,nemo,nemo_full,scratch,scratch_ep25}.sh`,
`sbatch_anchorT.sh`), pre-DSpark gemma examples, `NEXT_STEPS.md`.
Non-gemma examples (`eagle3_*`, `dflash_*`, `peagle_*`, `mtp_*`,
`dspark_qwen3_0_6b_*`) — upstream examples, untouched.
