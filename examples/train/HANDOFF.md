# Gemma-4-31B DSpark: machine-move handoff (2026-10-13)

Everything needed to continue the Gemma 4 DSpark draft work on another machine.
Read this first; `README.md` (run table + script map) and `notes/` (the working
notes that were in Claude's memory on the old box) carry the detail.

> **Draft-OPD (2026-10-21):** on-policy post-training of the DSpark drafts is
> being built on branch `mengmengj/draft-opd` — method, stack mapping and
> gotchas in `.claude/skills/draft-opd/`, phased plan and status in
> `examples/train/opd/PLAN.md`, code under `scripts/opd/`, `examples/train/opd/`
> and `vllm_patches/*rollout_log*`.

## 0. 2026-10-15: moved to the SambaNova sc3 B200 cluster

The conv run was relaunched from scratch on the SambaNova SLURM cluster
(partition `gpuonly`, nodes `sc3-cp[77-108]`, 8x B200-180GB each, driver
595.71, 256 CPUs, 2.3 TB RAM, `/dev/shm` 1.2 TB, node-local `/scratch` 11 TB
exposed as `$SLURM_TMPDIR=/scratch/jobs/<jobid>`). No podman: training runs
bare-metal in the cluster conda env. Running as job 11154700 on sc3-cp99
(started 2026-10-15 14:30, after 11151719/11153559/11154196 died on the FIPS
and GPU-2 issues below), wandb `nemoconv782k_v2`.

| Item | Path on this cluster |
|---|---|
| repo | `/import/ml-sc-scratch1/mengmengj/speculators` (branch `mengmengj/dspark-conv`) |
| conda env | `/import/ml-sc-scratch1/mengmengj/condaenvs/dspark` (torch 2.11+cu130, vllm 0.26.0, editable speculators; activate via `/import/snvm-sc-scratch1/mengmengj/miniconda3/etc/profile.d/conda.sh`) |
| target model | `/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it` (has `chat_template.jinja`) |
| nemo782k data | `/import/ml-sc-scratch1/mengmengj/datasets/kimi_nemotron_782k` (copied with symlinks resolved, 5.9 GB, 13 arrow shards + d2t/t2d) |
| outputs | `/import/ml-sc-scratch1/mengmengj/output/<RUN_NAME>` (shared NFS, 93% full -- prune epoch dirs) |
| wandb | `~/.netrc` on the shared NFS home, so every node is logged in |

Launch (from any login node that runs `sbatch`, e.g. sc-vnc10; no ssh to the
compute node is needed or possible -- they are password-only):

```bash
cd /import/ml-sc-scratch1/mengmengj/speculators && mkdir -p logs
sbatch examples/train/sbatch_sc3_nemo782k_conv.sh      # any idle gpuonly node
tail -f logs/dspark_nemo_conv-<jobid>.out              # driver banner + trainer
tail -f output/gemma4_31b_dspark_nemo782k_conv/logs/vllm.log
```

`sbatch_sc3_nemo782k_conv.sh` is the sc3 twin of `sbatch_nemo782k_conv.sh`:
same recipe and wandb id, but `/import` paths, conda instead of the image,
`--gres=gpu:b200m180:8 --exclusive --mem=0 --time=8-00:00:00` (the scratch
run took 132 h), and a hard W&B gate -- it runs `wandb.Api().viewer` on the
compute node and exits 3 if the API is unreachable rather than training
offline. Resubmit as-is to resume from the newest checkpoint. To look inside a
running allocation without ssh: `srun --jobid=<id> --overlap -N1 -n1 --mem=0
bash -c 'nvidia-smi; ...'`.

**FIPS gotcha, already fixed in the conda env (2026-10-15).** sc3-cp nodes
have kernel `fips_enabled=1`; the login nodes do not, so everything imports
fine on sc-vnc10 and the first job (11151719) died ~2 min in with
`crypto/fips/fips.c:154: OpenSSL internal error: FATAL FIPS SELFTEST FAILURE`
as soon as `vllm serve` imported cv2. The `opencv-python-headless` wheel bundles
a RedHat FIPS build of OpenSSL 1.1.1k (no `.hmac`), whose self-test aborts.
Fix applied: the two bundled libs under
`condaenvs/dspark/lib/python3.12/site-packages/opencv_python_headless.libs/`
were replaced by conda-forge OpenSSL 1.1.1w (env `condaenvs/ossl11`), patchelf'd
to the original SONAMEs/NEEDED; originals are in
`condaenvs/dspark/opencv_fips_orig_libs/`. Reinstalling opencv brings the crash
back. Verified on sc3-cp87: `vllm serve --help` ok, 2-rank NCCL all_reduce ok
bare-metal (no container needed, unlike sc3-c81 in August), vLLM loads the
verifier. The nodes do have rootless podman 5.8 + CDI (`/var/run/cdi/nvidia.yaml`)
and `cuda-docker-run-wrapper` (sudo, password) if a container is ever wanted.

**Bad GPU: sc3-cp78 GPU 2** (UUID `GPU-449b0e9b-84d5-3100-f326-16b722108077`,
serial 1792626025693) fails `torch.cuda.init()` with `cudaErrorDevicesUnavailable`
while idle (0 MiB, compute mode Default, no ECC errors). Job 11153559 died 3 min
in when DP engine 2 landed on it. Submit with `--exclude=sc3-cp78` until the
admins fix it; the wrapper now self-tests every GPU before launching and exits 4
naming the node.

**Eval on this cluster (prepared 2026-10-20, for the epoch-25 conv checkpoint):**
`sbatch_sc3_eval.sh` (+ `gemma4-31b-dspark-conv-ep25.yaml`, aime/gpqa/lcb at
k8/k5/k3), `sbatch_sc3_ctx_sweep.sh` (1k-32k) and `sbatch_sc3_lcr_bins.sh`
(AA-LCR bins, TP=2), all via `sc3_eval_prelude.sh`. Eval env: conda clone
`/import/snvm-sc-scratch1/mengmengj/condaenvs/dspark-host` (carries
`algos.patch` and the opencv FIPS fix; serving patches go there, never into the
training env). Two blockers before any conv checkpoint can be evaluated:
1. **vLLM conv patch -- DONE 2026-10-21** in the eval env and saved under
   `vllm_patches/` (`dflash2_conv.py`, `qwen3_dspark_conv.patch`,
   `algos_conv.patch`, parity test). It lives in `qwen3_dspark.py`, not
   `gemma4_dspark.py`: speculators DSpark checkpoints are always routed to
   `Qwen3DSparkModel`. Gate job `sbatch_sc3_conv_verify.sh`. First chain
   (2026-10-21 00:20): verify 12241538 -> eval 12241539 + ctx sweep 12241540.
   Training result for the record: val accept_len ep25 conv **4.779** vs scratch
   **4.711** (+0.068, shrinking as the LR annealed; below the ~0.1 gate), the
   user chose to evaluate anyway. **Eval results (2026-10-21, jobs 12241539/40,
   `results/gemma4-31b-dspark-conv-ep25`, `results/ctx_sweep_conv_ep25_k8`):**
   k8 aime 6.122 / gpqa 4.322 / lcb 5.191 vs scratch 5.993 / 4.222 / 5.071
   (+0.10..+0.13 accept_len, decode tok/s +3-4%); k5 +0.04..+0.07; k3 +0.00..+0.03;
   context sweep 1k-32k +0.00..+0.26, flat (no decay, conv is intra-block).
   **Assistant 1-on-1 (2026-10-21):** `gemma4-31b-assistant-ref.yaml` (baseline +
   google assistant k3/k5/k8, conv-eval settings) queued for B200 as job 12386739
   (pool drained for "network" since 05:59); accept_len-only copy
   `gemma4-31b-assistant-ref-a100.yaml` running on sc3-c98 (job 12398234, no
   baseline: A100 single-stream decode is ~21 tok/s). Earlier assistant table
   (`results/gemma4-31b-3bench`) was 4xA100 TP=4 and 50-sample GPQA/LCB subsets.
   Note the draft's 32k vocab leaves 11.2% of GPQA prompt tokens unpredictable
   (3.4% AIME, 2.7% LCB) -- the likely cause of the GPQA gap vs the assistant.
   **Block-size ablation (queued 2026-10-21, `sbatch_sc3_conv_continue.sh`):**
   warm-start from conv `checkpoints/25` for 8 more epochs (EPOCHS=34,
   SCHED_TOTAL=50600, warm-restart linear schedule): block 12 / gamma 12
   (`..._conv_blk12`, wandb `convblk12v1`) and the matched block-8 / gamma 4
   control (`..._conv_blk8ctrl`, wandb `convblk8ctrlv1`). Compare the two
   continuations on serve-time tok/s (accept_len is not comparable across k);
   driver gained a `DECAY_GAMMA` passthrough (`--dflash-decay-gamma`, default 4.0).
2. **AA-LCR bin files** (`scripts/evaluate/mtp_server_eval/data/aa-lcr-*.jsonl`)
   are not on this cluster; rsync them from the lax01 repo (command in the
   script header). aime/gpqa/livecodebench are in git and present.

The lax01 image/podman notes below (sections 4-5) do not apply here.

## 1. Code: branches to pull

Fork: `https://github.com/raviraju1113/speculators` (remote `origin`). Upstream
`vllm-project/speculators` is remote `upstream` (we are ~95 commits behind its
main; do NOT merge it blindly -- the last merge silently broke the train schema,
and a cherry-pick of one upstream commit dropped a hunk; always run
`tests/unit/models tests/unit/train/config` and `scripts/train.py --dry-run`
after any upstream pull).

| Branch | Contains | State on 2026-10-13 |
|---|---|---|
| `mengmengj/dspark-upstream` | checkpoint-of-record recipe, all evals/plots, YaRN per-layer serving story | 2 commits unpushed from the B200 node (`25429ef`, `8fefab7`) |
| `mengmengj/dspark-conv` | the above + upstream DFlash2 (#1006, cherry-picked) + DSpark optional DFlash2 conv + conv preset + this handoff (docker/, vllm_patches/, notes/) | never pushed; 7+ commits ahead of dspark-upstream |

Push from a machine with GitHub credentials (the B200 nodes have none):
```bash
git push origin mengmengj/dspark-upstream mengmengj/dspark-conv
```

## 2. Where the work stands

- **Checkpoint of record**: `output/gemma4_31b_dspark_anchorT_nemo782k/checkpoints/24`
  (val accept_len 4.69; k8 aime 5.89 / gpqa 4.43 / lcb 5.25; 4.1-5.4x over the
  59.5 tok/s TP=1 baseline). Architecture: 5 all-sliding (window 2048) Qwen3-style
  layers, hidden 5376, block 8, sample_from_anchor=True, Markov rank 256 vanilla,
  confidence head, draft vocab 32000, aux layers 1/17/29/47/58.
- **Architecture verdict**: all-sliding draft. Full-attention draft layers decay on
  long context (AA-LCR 112k) unless served with YaRN; per-layer-type YaRN at serve
  time recovers it (`notes/dspark-results-status.md`).
- **In flight: DSpark + DFlash2 local conv** (branch `mengmengj/dspark-conv`).
  Two-tap grouped dynamic causal convs around attention and MLP in each of the 5
  layers (+72.5M params), no selector. Trained FROM SCRATCH with the exact
  `nemo782k_scratch` recipe for a 1:1 comparison. First launch (SLURM job 319,
  node 097, wandb `nemoconv782k`) was cancelled at micro-step ~2970 (epoch 0, no
  checkpoint) for the machine move. **Relaunch from scratch** with the preset,
  which now defaults to wandb id `nemoconv782k_v2`:
  `sbatch examples/train/sbatch_nemo782k_conv.sh` (from a SLURM submit host) or
  `bash examples/train/run_nemo782k_conv_local.sh` (bare node with podman).
  Decision rule: val accept_len vs the scratch run at matched epochs; a gain under
  ~0.1 is not worth the serving patch. Serving a conv checkpoint needs a vLLM
  `gemma4_dspark.py` conv wrap that is NOT written yet (`vllm_patches/README.md`).
- **Open threads** (details in `notes/dspark-results-status.md`): RedHat DFlash
  LCR-bins run (external replication of the decay); PR to Ravi for the
  `train.py` distributed-unpack fix; merge of upstream main gated by dry-run;
  Kimi-K3 DSpark long-context options; block=12 ablation; MLA-draft plan.

## 3. Data (all ON-POLICY: every assistant turn was generated by gemma-4-31B-it)

| Corpus | Path on the old box | Size | Notes |
|---|---|---|---|
| 349k kimi-regen | `/sms-scratch/mengmengj/data/gemma4_dspark` (arrow + d2t/t2d) | 3.3 GB | kimi-mtp prompts regenerated by Gemma 4 (chenw's `kimi-regen-gemma4-31b/train_regen.jsonl`) |
| 782k kimi+nemotron | `/sms-scratch/mengmengj/data/kimi_nemotron_782k` -> **symlinks** into `/sms-scratch/ravira/datasets/kimi-mtp-nemotron-stem-code-math/` | 5.9 GB (resolved) | ~291k kimi-regen rows + ~491k Nemotron-v2 stem/math/code prompts regenerated by Gemma 4 (Ravi, 2026-07-30..08-01, `/sms-scratch/ravira/speculators/output/nemotron_v2/*.jsonl`). Copy with `cp -L` / `rsync -L` to resolve the symlinks. |
| AA-LCR eval prompts | `/sms-scratch/mengmengj/data/aa_lcr` | 36 MB | long-context bins 1k-112k |
| Target model | `/sms-scratch/checkpoints/gemma-4-31B-it` (+ `chat_template.jinja`) | 59 GB | or re-download google/gemma-4-31B-it |

## 4. Environment

- **Training**: podman image `localhost/mengmengj-vllm-official` = official
  `vllm/vllm-openai:v0.26.0-cu129-ubuntu2404` + `docker/Dockerfile` (applies the
  `dspark_bonus_anchor` patch). Tarball: `/sms-scratch/mengmengj/docker/mengmengj-vllm-official.tar`
  (12 GB, `podman load -i`). Python packages live in the thin venv
  `/sms-scratch/mengmengj/envs/dspark` (`--system-site-packages`, editable
  `speculators` + `hs_connectors` + wandb) -- rebuild on a new box with
  `SETUP=1 bash examples/train/run_local_podman.sh` (edit `VENV`/`REPO` paths).
- **Serving/eval**: bare-metal venv `/sms-scratch/mengmengj/envs/dspark-host`
  (vllm 0.26.0 + the three patches in `vllm_patches/`, originals as `*.orig`).
  Evals: `scripts/evaluate/experiments/run_experiments.py` + the `gemma4-31b-*.yaml`
  configs; LCR bins via `sbatch_lcr_bins.sh`; plots via `plot_lcr_bins.py`.
- **Driver**: CUDA 13 stack needs driver >= 580 (B200 boxes: 595.71).
- **Node rules** (`README.md` §"Node rules", `notes/b200-box-setup.md`):
  `loginctl enable-linger` once per node; `~/.netrc` is node-local (stage it or
  wandb goes offline); clean `/dev/shm/hidden_states_*` orphans before launching
  (097 may still hold `hidden_states_gemma4_31b_dspark_nemo782k_conv`); SLURM
  does not see off-SLURM GPU squatters -- the wrappers wait on idle GPUs.

## 5. What to copy to the new machine

If the new machine mounts the same `/sms-scratch`: nothing but `~/.netrc`,
`~/.claude/projects/<project-key>/memory/` (or just use `notes/`), and
`loginctl enable-linger`.

If it does not, in priority order:

| Item | Path | Size |
|---|---|---|
| repo (or `git pull` after pushing) | `/sms-scratch/mengmengj/speculators` | small |
| data, symlinks resolved | `data/gemma4_dspark`, `data/kimi_nemotron_782k` (`-L`), `data/aa_lcr` | 3.3 + 5.9 GB + 36 MB |
| target model | `/sms-scratch/checkpoints/gemma-4-31B-it` | 59 GB |
| checkpoint of record | `output/gemma4_31b_dspark_anchorT_nemo782k/checkpoints/24` | 13 GB |
| scratch control for the conv A/B | `output/gemma4_31b_dspark_nemo782k_scratch/checkpoints/25` (+ `run.yaml`) | 13 GB |
| container image tar + Dockerfile | `/sms-scratch/mengmengj/docker/` | 12 GB |
| serving venv (patched) | `/sms-scratch/mengmengj/envs/dspark-host` | 8.3 GB (or rebuild: vllm==0.26.0 + `vllm_patches/`) |
| wandb credential | `~/.netrc` | - |
| Claude memory | `~/.claude/projects/-sms-scratch-mengmengj/memory/` (same content as `notes/`) | tiny |

Optional: the other fullattn1/fullattn2 checkpoints (`checkpoints/25`,
`25_yarn`, `25_yarn_full`, 13 GB each) only if the YaRN/long-context work
continues; all their numbers are already in `scripts/evaluate/experiments/results/`
(tracked in git).

## 6. Claude Code memory on the new machine

The memory directory is keyed by the working directory path. If the repo lands at
a different path, copy `notes/*.md` into
`~/.claude/projects/<new-project-key>/memory/` (the key is the cwd with `/`
replaced by `-`), keeping `MEMORY.md` as the index. The tracked skills under
`.claude/skills/` travel with the repo.
