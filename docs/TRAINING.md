# Speculator Training Flow (SambaNova cluster setup)

This is my end-to-end setup for training speculative-decoding drafts (EAGLE3 / MTP /
D-Flash / P-EAGLE) against a frozen verifier, running on the `sngpu` scheduler inside
the project Docker image. It's meant to get a colleague from a fresh checkout to a
trained + evaluated draft without rediscovering the cluster-specific gotchas.

If you just want the package overview / model formats, see [README.md](../README.md).
This doc is the *operational* recipe.

---

## 0. TL;DR

```bash
# 1. Prepare data (tokenize + loss-mask + token-freq)  — CPU, cheap
python scripts/prepare_data.py \
    --model google/gemma-4-31B-it \
    --data ultrachat --max-samples 50000 --seq-length 8192 \
    --output ultrachat_gemma4_31b_50k_seq_len_8k/

# 2a. Online hidden states: start the verifier vLLM server (separate GPUs/node)
python scripts/launch_vllm.py google/gemma-4-31B-it \
    --hidden-states-path /shared/hidden_states -- --tensor-parallel-size 4

# 2b. …OR generate them offline first (see §3).

# 3. Train (submits a bare sngpu node -> project docker -> torchrun/FSDP)
export WANDB_API_KEY=...
bash scripts/submit_train.sh                       # on-policy EAGLE3, 2-layer draft
NUM_LAYERS=1 bash scripts/submit_train.sh          # 1-layer variant

# 4. Evaluate acceptance / speedup
cd scripts/evaluate/experiments
python run_experiments.py --config gemma4-kimi-mtp-300k.yaml
```

Everything below explains the pieces, the knobs, and why the wrappers exist.

---

## 1. Environment

### Creating the conda environment

Do this once inside the container (or on any box with the toolchain). It builds the
`speculators` env the wrappers expect:

```bash
conda create -n speculators python=3.11 -y
conda activate speculators

# Install the package (editable) from the repo root
pip install -e .

# Extra deps for the evaluation scripts (vllm, guidellm, plotting, yaml, …)
pip install -r scripts/evaluate/requirements.txt
```

`pip install -e .` is editable so host code edits take effect immediately (this is what
makes the mounted-checkout overlay in `launch_interactive_docker.sh` work — no reinstall
needed after editing `src/`). The eval requirements are only needed if you'll run §5.

### Where training runs

Training runs **inside the project Docker image**, not directly on the node. The
`sngpu` scheduler has no `--shm-size`/`--ipc` knob, and the DataLoader passes large
per-sample hidden-state tensors through `/dev/shm`; without a big shared-memory segment
you hit `unable to allocate shared memory ... Resource temporarily unavailable`. So the
pattern is always: **ask sngpu for a bare node, then launch the container with
`--shm-size` + `--ipc=host` ourselves.**

| Piece | Value (adjust paths to your user) |
|---|---|
| Docker image (batch) | `sc-artifacts2.sambanovasystems.com/sw-docker-scratch/speculators:ngc-24.12` |
| Docker image (interactive) | `…/speculators:ngc-14.12.t3` |
| Conda env | `speculators` |
| `conda.sh` | `/import/ml-sc-scratch1/ravir/miniconda3/etc/profile.d/conda.sh` |
| `HF_HOME` | `/import/ml-sc-scratch1/ravir/cache` |
| WandB project | `eagle3-speculators` (needs `WANDB_API_KEY` in your env) |

> ⚠️ The conda path, `HF_HOME`, and default `NODELIST` in `submit_train.sh` are
> hard-coded to my scratch. Override them (see the env-var table in §4) or they'll
> point at paths you can't write.

### Interactive shell (debugging / iterating on code)

```bash
# Allocate an interactive node first:
sngpu --cpu 92 --mem 900000 --gpu 4 --gputype a100m80 --time 5:59:00 --interactive
# Then, from inside the allocation:
bash launch_interactive_docker.sh
```

`launch_interactive_docker.sh` **stages just the code tree** (`src/`, `scripts/`, …)
onto job-local scratch (`$SLURM_TMPDIR`) and mounts *that* over the image's baked-in
`/workspace/speculators`, symlinking data/checkpoint dirs back to `/import`. This is
because:

- The full checkout is ~2.3 TB (dataset + `.git` + checkpoints); `cuda-docker-run-wrapper`
  refuses NFS `-v` mounts outside `$SLURM_TMPDIR`, but only the few MB of *code* needs to
  be live.
- Without the overlay you silently run the **stale** `scripts/train.py` baked into the
  image against your patched `src/speculators`.

Re-run the script after editing code on the host to re-sync (it's a copy, not a live mount).

---

## 2. Data preparation

`scripts/prepare_data.py` turns a raw dataset into the training format:

1. applies the verifier's chat template + tokenizes,
2. builds a **loss/assistant mask** (only assistant tokens contribute to loss),
3. records **token-frequency stats** (`token_freq.pt`, used to build the pruned draft
   vocab mappings `d2t.npy` / `t2d.npy`).

```bash
python scripts/prepare_data.py \
    --model google/gemma-4-31B-it \
    --data ultrachat \                 # shortcut name, HF id, or local .jsonl (repeatable)
    --max-samples 50000 \
    --seq-length 8192 \
    --output ultrachat_gemma4_31b_50k_seq_len_8k/
```

Output lands in `--output/` as `*.arrow` + `token_freq.pt`. Re-running is a no-op unless
you pass `--overwrite`.

**Shortcut:** you can skip this step and pass `--data` straight to `train.py`, which
tokenizes on the fly into `--data-path` (the cheap step only — hidden states are still
generated online). Handy for quick runs; prefer the explicit `prepare_data.py` step for
anything you'll reuse.

### Off-policy / regenerated responses (optional)

To train on responses regenerated by the *target* model (off-policy tokens), use the
regeneration pipeline, then train with `--use-off-policy-tokens`:

```bash
scripts/response_regeneration/run_all.sh \
    --model google/gemma-4-31B-it --dataset magpie --tp-size 4 --gpus 0,1,2,3
```

`run_all.sh` starts a vLLM server, regenerates, and tears the server down (it reaps the
whole process group and refuses to start on GPUs still held by a stale server — set
`RESPONSE_REGEN_KILL_STALE=1` to auto-clean). Output JSONL goes to your invocation dir
(or `$SLURM_TMPDIR` — **copy it to persistent storage before the job ends**).

---

## 3. Hidden states (verifier features)

The draft is trained to match the verifier's hidden states at specific layers. There are
two ways to supply them:

### Online (default) — generate on demand from a live vLLM endpoint

Start the verifier with the hidden-states connector:

```bash
python scripts/launch_vllm.py google/gemma-4-31B-it \
    --hidden-states-path /shared/hidden_states \
    -- --tensor-parallel-size 4 --gpu-memory-utilization 0.9
```

- `--hidden-states-path` **must be reachable from the training node** (same node or a
  shared network drive) — vLLM writes the features there and the DataLoader reads them.
- Default target layers are `[2, L/2, L-3, L]`. If you override `--target-layer-ids`,
  you **must pass the identical list to `train.py`** or the features won't line up.
- In `train.py` this is controlled by `--vllm-endpoint`, `--on-missing generate`
  (default), and `--on-generate {delete,cache}`. Use `--on-generate cache` for
  **hybrid** training: generate on epoch 1, reuse from disk after.

### Offline

Generate hidden states ahead of time and point `--hidden-states-path` at the dump; run
training with `--on-missing skip` (the `submit_train.sh` default) so it never calls
vLLM.

---

## 4. Training

`scripts/train.py` is the general entry point (EAGLE3 / D-Flash / P-EAGLE / MTP). It runs
under `torchrun` and shards with FSDP.

### The submit wrapper (batch runs)

`scripts/submit_train.sh` is the way I launch batch jobs. It generates two git-ignored
scripts and submits them:

- `.job_<name>.sh` — **outer**, runs on the bare sngpu node, docker-runs the inner script
  with `--shm-size 16G --ipc=host`.
- `.job_<name>_inner.sh` — **inner**, runs in the container: `conda activate` +
  `torchrun … scripts/train.py …`. Your `WANDB_API_KEY` is baked in here (git-ignored,
  redacted from stdout).

```bash
export WANDB_API_KEY=...                              # kept out of git
bash scripts/submit_train.sh                          # default: on-policy EAGLE3, 2-layer
NUM_LAYERS=1 bash scripts/submit_train.sh             # 1-layer draft
OFF_POLICY=1 NUM_LAYERS=1 bash scripts/submit_train.sh# off-policy baseline
DRY_RUN=1 bash scripts/submit_train.sh                # print the scripts, don't submit

# Power-user: forward a full train.py arg list verbatim (torchrun preamble still added)
bash scripts/submit_train.sh --speculator-type mtp --verifier-name-or-path google/gemma-4-31B-it ...
```

Key env knobs (see the header of `submit_train.sh` for the full list):

| Var | Meaning | Default |
|---|---|---|
| `NUM_LAYERS` | draft decoder layers (the main experimental variable) | `2` |
| `OFF_POLICY` | `1` adds `--use-off-policy-tokens` | `0` |
| `EPOCHS` / `LR` / `SEQ_LEN` | training length / LR / seq len | `5` / `1e-4` / `8192` |
| `VERIFIER` | verifier model id/path | `google/gemma-4-31B-it` |
| `DATA_PATH` / `HIDDEN_STATES_PATH` | prepared dataset / HS dir | ultrachat 50k |
| `GPU` / `NPROC` / `CUDA_DEVICES` | GPUs requested / procs / device list | `4` / `4` / `0,1,2,3` |
| `NODELIST` | sngpu node | `sc3-c98` |
| `SAVE_PATH` / `RUN_NAME` | checkpoint dir / WandB run name | derived from above |

Watch progress: `tail -f <JOBNAME>.out`.

### Running train.py directly

```bash
# multi-GPU (FSDP)
torchrun --standalone --nproc_per_node=4 scripts/train.py \
    --verifier-name-or-path google/gemma-4-31B-it \
    --data-path ultrachat_gemma4_31b_50k_seq_len_8k/ \
    --speculator-type eagle3 --num-layers 2 --draft-arch llama \
    --epochs 5 --lr 1e-4 --total-seq-len 8192 \
    --on-missing skip --logger wandb --run-name my_run

# single GPU
python scripts/train.py ...
```

### How the draft is defined (pick exactly one)

- `--from-pretrained <path|hf-id>` — finetune an existing draft, **or** (config-only
  dir) init fresh weights from a saved speculator config. Takes precedence over everything.
- `--draft-config <config>` — use a decoder config as the draft `transformer_layer_config`;
  build the rest from CLI args.
- decoder-shaping flags (`--num-layers`, `--draft-arch`, `--draft-hidden-act`,
  `--sliding-window[-indices]`) — synthesize the decoder from the verifier config.

These are mutually exclusive; `train.py` errors if you mix them. **MTP** is special: it
reuses the verifier's own decoder config and extracts the native MTP head weights, so the
shaping flags and `--draft-config` don't apply.

### Useful flags

- `--dry-run` — build + init the draft, save a checkpoint to `--save-path`, exit before
  training. Validate the config/weights in vLLM, then feed it back via `--from-pretrained`.
- `--speculator-type {eagle3,dflash,peagle,mtp}` — algorithm.
- `--draft-vocab-size` (+ `token_freq.pt`) — pruned draft vocab via `d2t`/`t2d` mappings.
- `--optimizer {adamw,muon}` — Muon applies to 2-D weight matrices, AdamW to the rest.
- `--scheduler-type {linear,cosine,constant,none}`, `--checkpoint-freq` (`<1` = sub-epoch),
  `--save-best`, `--no-resume-from-checkpoint`.
- `--draft-attn-impl {simple_flex_attention,sdpa,eager}` — use `sdpa`/`eager` on hardware
  without flex-attention. Note: with `simple_flex_attention`, `--total-seq-len` must be a
  multiple of 128.

Training auto-resumes from the last checkpoint in `--save-path` unless
`--no-resume-from-checkpoint`. FSDP full-state-dict checkpoints are written per epoch
(or per `--checkpoint-freq`), and `checkpoint_best` symlinks the lowest val-loss epoch.

### Gemma-4 MTP (standalone path)

`scripts/gemma4_mtp/train.py` is a separate DDP loop for TTT multi-step distillation of
the Gemma-4 native MTP assistant (freezes the target + the assistant's tied
head/embeddings, trains the decoder + projections). Optionally precompute target signals
with `prepare_cache.py` to avoid loading the full target during training:

```bash
torchrun --nproc_per_node=4 scripts/gemma4_mtp/train.py \
    --target /path/to/gemma4/target --assistant /path/to/gemma4/assistant \
    --data train_regenerated.jsonl --output ./out/gemma4_mtp \
    --epochs 1 --bf16 --ttt-steps 5
```

### Simultaneous multi-draft training (single node, 4 GPUs)

All registered draft types can be trained **at the same time** on one 4-GPU node by
sharing the frozen target: one vLLM hidden-states server feeds every generic
`scripts/train.py` trainer, and the draft models themselves are small enough to
co-locate two per GPU. Ready-made pipelines (Gemma4-26B-A4B target):

- [`examples/train/gemma4_26b_tri_draft_online.sh`](../examples/train/gemma4_26b_tri_draft_online.sh)
  — EAGLE3 + DSpark + MTP (one trainer per GPU).
- [`examples/train/gemma4_26b_penta_draft_online.sh`](../examples/train/gemma4_26b_penta_draft_online.sh)
  — all five types at once:

| GPU | Component | Observed footprint |
|---|---|---|
| 0 | shared vLLM hidden-states server (target, layer ids `2 15 27` +last) | ~68 GB |
| 1 | EAGLE3 + P-EAGLE trainers (co-located) | ~45 GB |
| 2 | DSpark + DFlash trainers (co-located) | ~62 GB |
| 3 | Gemma4-MTP fine-tune (own live target + draft on one GPU) | ~78 GB |

Design points (validated: all trainer GPUs at 100% util, server fetch wait 2–5%):

- **Shared hidden-state cache** (`--on-generate cache`, one `--hidden-states-path` for
  all trainers): each sample's features are generated once and reused by every trainer
  and epoch. Budget ~26 MB/sample at seq 4096 (30k samples ≈ 780 GB). Never mix
  `--on-generate delete` between trainers sharing a cache — they race on files.
- **Per-draft configuration is independent** (lr, layers, block size, losses, epochs —
  each leg is its own process with its own optimizer/checkpoints); only
  `--target-layer-ids` must match the server, and `--total-seq-len` must fit under the
  server's `--max-model-len`.
- **Trainings are isolated**, so simultaneity is purely a wall-clock win — this is
  different from `train_online.py`'s YAML `drafts:` mode, which trains sequentially.
- The Gemma4-MTP leg runs single-GPU (with one visible device, `train_online.py` puts
  target and draft on the same GPU) and needs no server.

First results (5k samples × 2 epochs, aime, greedy, k=5 / DSpark k=8; baseline ≈126 tok/s):
from-scratch DSpark **1.51×** (accept_len 3.00), EAGLE3 **1.40×** (2.88); the vanilla
assistant reference is 2.03× (4.83). Fine-tuning the assistant at the random-init lr
6e-4 *degraded* it to 1.26× — fine-tune with lr ≈ 5e-5 instead. Eval configs:
`scripts/evaluate/experiments/gemma4-26b-tri-draft-eval.yaml` (+ `gemma4-26b-dspark-eval.yaml`).

---

## 5. Evaluation

Acceptance rate / throughput / speedup live under
[scripts/evaluate/experiments/](../scripts/evaluate/experiments/). The runner launches one
vLLM server per experiment (backbone alone, or with a draft via `--speculative-config`),
runs the benchmark, stops the server, and prints a speedup table (first experiment =
baseline).

```bash
cd scripts/evaluate/experiments
python run_experiments.py --config gemma4-kimi-mtp-300k.yaml
python run_experiments.py --config gemma4-kimi-mtp-300k.yaml --dry-run   # print commands only
```

Point a config's `experiments[].draft` at the `checkpoint_best` dir your training run
produced. Benchmarks/samples/temperature are set in the YAML; greedy (`temperature: 0.0`)
gives canonical acceptance. See [scripts/evaluate/experiments/README.md](../scripts/evaluate/experiments/README.md)
for the config schema.

---

## 6. Gotchas / lessons learned

- **`scripts/train.py` and torchrun:** after the distributed refactor (#656),
  `maybe_setup_distributed()` returns nothing (topology comes from getter functions).
  Upstream `scripts/train.py` still unpacked a 4-tuple from it and crashed on every
  launch (`TypeError: cannot unpack non-iterable NoneType`); fixed here by calling it
  bare. With the fix, both `torchrun` and plain single-GPU `python` launches work.
- **Server context headroom:** hidden-state generation sends `prompt + 1` tokens, so
  samples truncated to exactly `--seq-length` are rejected by a server whose
  `--max-model-len` equals it (400s, silent sample skips). Give the server headroom:
  `max-model-len = seq_length + 256`.
- **FlashInfer JIT picks up `$PATH`'s nvcc:** an old system `/usr/bin/nvcc` (CUDA 10.x)
  fails with `Unknown option '-generate-dependencies-with-compile'` and kills vLLM
  startup. Export the toolkit matching your torch build first, e.g.
  `export CUDA_HOME=/usr/local/cuda-12.9 PATH="$CUDA_HOME/bin:$PATH"`.
- **DSpark serving needs vLLM ≥ 0.28** (`method: dspark`, `Qwen3DSparkModel`); on this
  box that's the venv `/nvmedata/chenw/envs/speculator-vllm028`. Training works in the
  regular `speculator` env; only serving/eval needs the newer vLLM.
- **Fine-tune lr ≠ from-scratch lr:** lr 6e-4 (the random-init recipe) wrecked the
  vanilla Gemma4 assistant in a 2-epoch fine-tune (vLLM accept_len 4.83 → 2.98);
  use ~5e-5 for fine-tuning an already-good draft.
- **Shared memory:** always run training in the container with `--shm-size` + `--ipc=host`
  (the submit wrapper does this). Bare sngpu containers OOM `/dev/shm`.
- **Stale code:** when iterating, make sure you're running the mounted checkout, not the
  image's baked-in copy (`launch_interactive_docker.sh` handles this).
- **Target layer ids** must match between `launch_vllm.py` and `train.py`.
- **Hidden-states path** must be visible to the training node (shared FS or same node).
- **`$SLURM_TMPDIR` is node-local** — copy regen output / anything you want to keep to
  persistent storage before the job ends.
- **`WANDB_API_KEY`** is read from your shell env and only baked into the git-ignored
  inner script — rotate it rather than committing it.

---

## 7. Kimi K3 (branch `feature/kimi-k3`)

Work-in-progress port of this flow to Kimi K3 as the target. Status: assets staged
locally, training **not** started yet.

### Target model

- `/import/ml-sc-scratch5/chenw/models/Kimi-K3-patched` — weights symlinked from
  raghup's read-only checkpoint (~1.5 TB mxfp4, 96 shards); modeling code copied +
  HF-patched. Built by `MoEInfer/kimi_k3/scripts/prepare_model.sh`
  (`/import/snvm-sc-scratch1/chenw/MoEInfer/kimi_k3/`).
- Architecture (`KimiK3ForConditionalGeneration` → text `KimiLinearForCausalLM`,
  trust_remote_code): 93 layers, hidden 7168, 896 routed experts, vocab 163 840,
  1M ctx. Hybrid attention: KDA linear layers with a full MLA layer every 4th
  (`linear_attn_config.full_attn_layers`); MLA `q_lora_rank` 1536 / `kv_lora_rank` 512.
  `num_nextn_predict_layers = 0` — **no MTP head ships with the checkpoint**.
- Serving: TP8 on the 8×B300 node via the `kimi_k3` conda env
  (`/import/ml-sc-scratch6/chenw/conda_env/kimi_k3`: vLLM 0.24, torch 2.13+cu130,
  fla-core) with the KDA recurrent-state patch from
  `MoEInfer/kimi_k3/patches/vllm/apply_vllm_patch.py`. The target needs the whole
  node, so hidden-state generation and training must run as separate phases.

### Existing EAGLE3 draft (baseline to beat)

Trained by fengluh with **TorchSpec** (3-node B200 run, `kimi_k3_eagle3_mla`); all
assets copied under `/import/ml-sc-scratch5/chenw/models/kimi-k3-draft-torchspec/`:

| item | path (relative to that dir) |
|---|---|
| distcp checkpoint (iter 39388, model+optimizer) | `iter_0039388/` |
| draft config | `kimi_k3_eagle3_mla.json` |
| TorchSpec run config | `config.yaml` |
| TorchSpec source (the training code) | `TorchSpec/` |
| on-policy regen train data (~2.4 GB jsonl) | `data/train_regen_code_math_merged_converted.jsonl` |
| eval conversations | `data/eval_conversations.jsonl` |
| converted HF / vLLM exports | `hf/`, `hf-vllm/` |

Draft = 1-layer **DeepSeek-V3-style MLA** EAGLE3 head (`Eagle3DeepseekV2ForCausalLM`,
`model_type: deepseek_v3`), full 163 840 vocab (no t2d/d2t pruning), aux hidden-state
layers `[48, 68, 88]`, TTT length 4. Reported at iter 39388: eval token-acc 0.594,
simulated acceptance length **1.44**.

Convert distcp → HF (CPU, kimi_k3 env):

```bash
cd /import/ml-sc-scratch5/chenw/models/kimi-k3-draft-torchspec/TorchSpec
PYTHONPATH=. /import/ml-sc-scratch6/chenw/conda_env/kimi_k3/bin/python tools/convert_to_hf.py \
    --input-dir ../iter_0039388 --config ../kimi_k3_eagle3_mla.json \
    --output-dir ../hf --force            # add --vllm for the vLLM-shaped export
```

### Training data staged

- `lightseekorg/kimi-mtp-dataset` (HF hub) — the `kimi_mtp` entry in
  `src/speculators/data_generation/configs.py` (~477k ShareGPT-style rows, has
  multimodal turns; see the note there).
- `/import/ml-sc-scratch5/chenw/models/kimi-k3-data/kimi-mtp-nemotron-stem-code-math/`
  — ravira's 5.9 GB HF-arrow nemotron STEM/code/math mix.
- The on-policy regen jsonl above (what the TorchSpec baseline was trained on).

### Planned (not started)

1. **MTP speculator via this repo**: add a `kimi_linear` entry to the MTP registry
   (`src/speculators/models/mtp/model_definitions.py`). The draft layer can reuse
   transformers' `deepseek_v3` MLA decoder layer (same trick TorchSpec used) instead
   of the trust_remote_code Kimi classes.
2. **DSpark draft** for K3; evaluate with the Kimi-K3-DSpark acceptance suite already
   wired into `scripts/evaluate/mtp_server_eval/`.
3. Baseline-eval the converted TorchSpec EAGLE3 draft with vLLM spec-decode on the
   B300 node before training anything new.
