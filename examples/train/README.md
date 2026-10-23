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
| `..._nemo782k_fullattn1` | nemo782k, from scratch, full-attn layer {0} | 26 (WSD) | `run_nemo782k_fullattn1_local.sh` | `fullattn1nemo` | done 2026-09-15; `checkpoint_best -> 25` (val 4.50 vs fullattn2 4.75); evals: `gemma4-31b-dspark-fullattn1-ep25.yaml` + LCR bins `TAG=fullattn1_ep25` |
| `..._nemo782k_conv` | nemo782k, from scratch, all-sliding + **DFlash2 local conv** in every draft layer (`DSPARK_CONV_KERNEL=2`, group 16; +72.5M params) | 26 | `run_nemo782k_conv_local.sh` / `sbatch_nemo782k_conv.sh` | `nemoconv782k_v2` | done 2026-10-20 on sc3-cp99 (job 11154700, 5 d 7 h; first attempt `nemoconv782k` on lax01 cancelled at step ~2970). `checkpoint_best -> 25`, val 4.779 vs scratch 4.711 (+0.07). Evals 2026-10-21 (`gemma4-31b-dspark-conv-ep25.yaml`, served with `vllm_patches/` conv patch): k8 aime **6.12** / gpqa **4.32** / lcb **5.19** vs scratch 5.99 / 4.22 / 5.07 (+0.10..0.13, tok/s +3-4%); k5 +0.04..0.07; k3 ~0. ctx sweep 1k-32k flat, +0.0..+0.26 vs scratch, no decay. Small, consistent gain; LCR bins not run yet |

**Google `gemma-4-31B-it-assistant` (native MTP, 4 layers, full 262k vocab) vs
our conv ep25 — accept_len / accept_rate, same harness, both on B200 TP=1
(`gemma4-31b-assistant-ref.yaml`, 2026-10-21; an A100 rerun agrees to ±0.03):**

Baseline (no draft): 62.5 / 62.7 / 63.4 tok/s on AIME / GPQA / LCB.

| k | bench | assistant accept_len / rate / tok/s (×base) | conv ep25 accept_len / rate / tok/s (×base) | Δ accept_len | Δ tok/s |
|---|---|---|---|---|---|
| 8 | AIME | 5.75 / 0.593 / 256 (4.1×) | **6.12 / 0.640 / 341 (5.5×)** | +0.38 | +33% |
| 8 | GPQA | **5.03 / 0.504** / 225 (3.6×) | 4.32 / 0.415 / **241 (3.8×)** | −0.71 | +7% |
| 8 | LCB | 5.19 / 0.524 / 235 (3.7×) | 5.19 / 0.524 / **284 (4.5×)** | 0.00 | +21% |
| 5 | AIME | 4.79 / 0.757 / 235 (3.8×) | **4.83 / 0.766 / 264 (4.2×)** | +0.04 | +12% |
| 5 | GPQA | **4.28 / 0.655 / 211 (3.4×)** | 3.76 / 0.551 / 206 (3.3×) | −0.52 | −2% |
| 5 | LCB | **4.40 / 0.680** / 220 (3.5×) | 4.32 / 0.663 / **233 (3.7×)** | −0.08 | +6% |
| 3 | AIME | 3.56 / 0.854 / **196 (3.1×)** | 3.55 / 0.851 / 194 (3.1×) | −0.01 | −1% |
| 3 | GPQA | **3.34 / 0.780 / 184 (2.9×)** | 3.05 / 0.682 / 167 (2.7×) | −0.29 | −9% |
| 3 | LCB | **3.40 / 0.799 / 185 (2.9×)** | 3.34 / 0.779 / 180 (2.8×) | −0.06 | −3% |

At each model's best k (8), conv is faster on all three benchmarks (+33% AIME,
+21% LCB, +7% GPQA) because DSpark drafts the block in one pass while the
assistant's 4-layer MTP head drafts k tokens sequentially; the assistant wins
only at k=3, where acceptance dominates per-step cost. Conv trails on GPQA
accept_len at every k — the domain where 11.2% of tokens fall outside our 32k
draft vocab (3.4% AIME, 2.7% LCB).

**AA-LCR long context, k=8 accept_len — conv ep25 vs the all-sliding scratch
ep25 (same architecture minus the convs):**

Paired bins: the same questions at every length; n = 100 per bin, 67 at 96k,
37 at 112k. Scratch numbers are from `lcr_bins_scratch_best_{k8,k3}` on the
lax01 box (checkpoint 25, the last of the 26 from-scratch epochs).

| k | draft | 1k | 2k | 4k | 8k | 16k | 32k | 64k | 96k | 112k |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | scratch ep25 | 3.80 | 3.73 | 3.51 | 3.68 | 3.71 | 3.68 | 3.77 | 3.92 | 3.98 |
| 8 | **conv ep25** | 3.81 | 3.91 | 3.63 | 3.84 | 3.82 | 3.87 | 3.89 | 3.99 | 4.10 |
| 8 | Δ conv − scratch | +0.01 | +0.18 | +0.12 | +0.16 | +0.11 | +0.19 | +0.12 | +0.07 | +0.12 |
| 3 | scratch ep25 | 2.90 | 2.89 | 2.84 | 2.85 | 2.86 | 2.86 | 2.89 | 2.95 | 2.94 |
| 3 | **conv ep25** | 2.90 | 2.93 | 2.82 | 2.91 | 2.90 | 2.90 | 2.93 | 2.97 | 2.98 |
| 3 | Δ conv − scratch | 0.00 | +0.04 | −0.02 | +0.06 | +0.04 | +0.04 | +0.04 | +0.02 | +0.04 |

accept_rate follows accept_len exactly ((accept_len − 1)/k): scratch k8
0.35 → 0.37, conv k8 0.35 → 0.39. Both drafts are flat to 112k (no
long-context decay — the full-attention variants dropped to ~3.0 there,
`notes/dspark-results-status.md`); conv sits +0.1…+0.2 above scratch at k=8
from 2k onward, +0.0…+0.06 at k=3, consistent with the in-window benchmarks.

Decode speed (note only, not comparable across rows): scratch was served with
TP=2 on B200 (k8 72.8 tok/s at 1k → 39.1 at 112k; k3 58 → 29; no-draft
baseline 25.6), conv with TP=4 on four A100-80 (sc3-c98: k8 58–60 tok/s up to
32k → 41/31/28 at 64k/96k/112k; k3 45 → 21; baseline 18). The drop beyond 64k
is long-prompt prefill, present in both. Google assistant bins:
`lcr_bins_assistant_{k8,k3}` (same wrapper, `METHOD=` empty).

**Standing decisions**
- (2026-08-14) Train with `--sample-from-anchor` (True): +0.2–1.3 accept_len
  same-k, native k=8, stabler outputs. Serving True checkpoints needs the
  bonus-anchor fix — baked into the `dspark-host` venv / local image; upstream
  vLLM main (post-0.26.0) derives it from checkpoint config.
- (2026-09-17) `dspark-host` venv also carries a per-layer-type RoPE patch for
  the DFlash/DSpark draft (`qwen3_dflash.py`, `configs/speculators/base.py`;
  `*.orig` backups alongside): a checkpoint config may set
  `transformer_layer_config.rope_parameters` in the Transformers-v5 nested
  form `{"full_attention": {yarn...}, "sliding_attention": {default...}}` so
  serve-time YaRN applies to the full-attention layers only. Flat configs are
  unaffected. Example: `..._fullattn1/checkpoints/25_yarn_full` (symlinked
  weights + edited config).
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

```bash
# SambaNova sc3 B200 cluster (2026-10-15+): bare-metal conda, /import paths,
# W&B gate on the compute node. See HANDOFF.md §0.
sbatch examples/train/sbatch_sc3_nemo782k_conv.sh

# Eval on sc3 (twins of sbatch_eval / sbatch_ctx_sweep / sbatch_lcr_bins; all
# source sc3_eval_prelude.sh: eval conda env, GPU self-test, and a gate that
# refuses to serve a conv checkpoint on a vllm without the gemma4_dspark conv
# patch -- unpatched vllm silently drops the conv weights):
CONFIG=gemma4-31b-dspark-conv-ep25.yaml sbatch examples/train/sbatch_sc3_eval.sh   # aime/gpqa/lcb, k8/k5/k3
sbatch examples/train/sbatch_sc3_ctx_sweep.sh          # 1k..32k, k8   (CKPT/TAG/K overridable)
sbatch examples/train/sbatch_sc3_lcr_bins.sh           # AA-LCR bins, needs aa-lcr-*.jsonl copied first
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
