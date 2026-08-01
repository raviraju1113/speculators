# DSpark for Gemma-4-31B-it — runbook

Working notes for training a DSpark draft for `google/gemma-4-31B-it` on the
SambaNova cluster. Branch: `mengmengj/dspark-upstream`. Last updated 2026-07-31.

## 0. Status

| Item | State |
|---|---|
| Upstream merge (DSpark code) | ✅ commit `57e6f09` |
| Conda env | ✅ `/import/ml-sc-scratch1/mengmengj/condaenvs/dspark` (torch 2.11.0+cu130, vllm 0.26.0, transformers 5.14.1) |
| Data prep (full) | ✅ 349,389 examples, `datasets/gemma4_dspark` |
| Data prep (overfit split) | ✅ `datasets/gemma4_dspark_256` (or `_32`) |
| Job scripts | ✅ this directory |
| Usable GPU node found | ✅ `sc3-c98` only (`sc3-c81` still unprobed) |
| Recon on sc3-c98 | ✅ driver 595.71.05, /dev/shm 504 GB, W&B reachable |
| Overfit gate | 🟡 partial — 22/50 epochs before TIMEOUT; learning but not converged |
| Container / NCCL fix | ⬜ script ready, not yet run (§8) |
| Full run | ⬜ |
| Eval | ⬜ |

**Overfit progress so far** (job 58571742, TIMEOUT at 1 h, resumes from epoch 21):

| Metric | Early | Epoch ~21 | Gate |
|---|---|---|---|
| `position_1_acc` | 0.184 | 0.354 | > 0.95 |
| `accept_len` | 1.195 | 1.435 | climbing |

Rising steadily, not plateaued — most likely step-starved (256 samples packed into
8192-token batches ≈ 25 optimizer steps/epoch). `ON_GENERATE=cache` has since been
set in `run_overfit_nonccl.sh`; epochs were ~3.2 min each and were dominated by
vLLM regenerating hidden states, not by training.

**Key background:** DSpark was already merged upstream (PR #677, 2026-06-29) five
days after this fork branched, so no algorithm code needed writing.
`RedHatAI/gemma-4-31B-it-speculator.dspark` exists on HF for the same target —
use it as the config oracle and the number to beat.

## 1. Which nodes can we use

Our env ships `torch 2.11.0+cu130`, which requires a **CUDA 13 driver (≈580+)**.
A cu128 rebuild would need ≥570.26. Surveyed 2026-07-31:

| Node | GPUs | Driver / CUDA | Verdict |
|---|---|---|---|
| **sc3-c98** | 4× A100-80 | **595.71.05 / 13.2** | ✅ **the only usable node** |
| sc3-c81 | 4× A100-80 | **unknown** | ❓ probe it — same hw/family as c98 |
| sc3-c97 | 4× A100-80 | 565.57.01 / 12.7 | ✗ too old |
| sc-c82 | 4× A100-80 | 565.57.01 / 12.7 | ✗ too old |
| sc-c96 | 8× A100-80 | 565.57.01 / 12.7 | ✗ too old (chenw's box; his cuda-compat hack existed for this reason) |
| sc-c120 | 2× H100-80 | 565.57.01 / 12.7 | ✗ too old |
| sc3-c126–129 | 8× H200-141 | unknown | ✗ reserved / maint / drain |

Two escalations worth pushing, both bigger wins than a reservation:
1. **Driver bump on sc-c96 / sc3-c97 / sc-c82** → unblocks ~20 A100s.
2. **The 349-job single-user backlog** in `gpuonly` → a QOS `MaxJobs` cap would
   unblock the whole team.

## 2. How many GPUs each step needs

Verifier `gemma-4-31B-it` is **62.6 GB**, so it fits on one A100-80 with ~17 GB
left for KV, or splits across two with generous headroom.

| Step | GPUs | Layout | Notes |
|---|---|---|---|
| recon (env/driver) | **1** | — | validates everything except NCCL |
| recon (+NCCL) | 2 | — | optional; the overfit gate tests NCCL anyway |
| **overfit gate** | **3** | vLLM TP=2 + 1 train rank | comfortable minimum |
| overfit gate (squeezed) | **2** | vLLM TP=1 + 1 train rank | `VLLM_TP=1 TRAIN_GPUS_N=1`; 62.6 GB + ~17 GB KV, viable for short prefill-only requests |
| **full run** | **4** | vLLM TP=2 + 2 train ranks | the `full` preset |
| full run (squeezed) | 3 | vLLM TP=2 + 1 train rank | `TRAIN_GPUS_N=1` |
| eval, per config | 1–2 | verifier + draft | serving only |

**Absolute floor for training is 2 GPUs.** There is no 1-GPU training path:
the verifier and at least one training rank cannot share one card.

## 3. Commands, in order

All paths absolute — `sbatch` copies job scripts to a spool dir, so
`$(dirname "$BASH_SOURCE")` does not point at the repo.

`sngpu` forwards the job command to `sbatch --wrap`, which accepts only
`bash <one-script-path>` — **no arguments, no `env VAR=... bash ...` prefix.**
That is why the presets live in wrapper scripts.

```bash
cd /import/ml-sc-scratch1/mengmengj/speculators
conda activate /import/ml-sc-scratch1/mengmengj/condaenvs/dspark
mkdir -p logs
```

### 3.1 Probe the last unknown node (2 min)

```bash
bash examples/train/sngpu_driver_survey.sh
squeue -u mengmengj
grep -H . logs/driver_*.txt
```
If `sc3-c81` reports ≥580, we have two usable nodes instead of one.

### 3.2 Recon on sc3-c98 — 1 GPU, backfills easily (10 min)

```bash
sngpu --jobname recon98a --partition gpuonly --nodelist sc3-c98 \
  --gpu 1 --gputype a100m80 --cpu 8 --mem 32000 --time 00:10:00 \
  --output ./logs/recon_sc3c98_1gpu.txt \
  -- bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/sngpu_recon.sh
```

**Gate — read the log, do not trust the exit code.** `sngpu_recon.sh` is a
diagnostic: it reports problems in its output and still exits 0. Want to see:
- `cuda_available: True` with **no** "driver too old" line ← the result that unblocks everything
- torch 2.11.0+cu130, vllm 0.26.0, `dspark module: .../mengmengj/speculators/...`
- `/dev/shm` large (755 GB on sc-c96, so bare-metal conda is fine — no docker/shm workaround needed)
- `api.wandb.ai` reachable; if not, use `WANDB_MODE=offline` + `wandb sync` later

### 3.3 One-time W&B setup (login node)

```bash
pip install wandb
wandb login          # writes ~/.netrc, which compute nodes inherit
```
Never pass `WANDB_API_KEY` on the sngpu line — job args appear in listings/logs.

### 3.4 Overfit gate — 3 GPUs (~1 h) ← the important one

```bash
sngpu --jobname dspark_overfit --partition gpuonly --nodelist sc3-c98 \
  --gpu 3 --gputype a100m80 --cpu 24 --mem 200000 --time 01:00:00 \
  --output ./logs/dspark_overfit.txt \
  -- bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/run_overfit_sc3c98.sh
```

**Gates** (in the log and W&B project `gemma4-dspark`):
- `loss` → near zero
- `position_1_acc` > 95%
- `confidence_pred_mean` moves off 0.5
- `accept_len` climbing

A high plateau means `--target-layer-ids`, `mask_token_id`, or the vocab mapping
is wrong. Debug here — never in a multi-hour run. This step also writes
`d2t.npy` / `t2d.npy` into the data dir for the first time, and implicitly proves
NCCL works (vLLM TP=2 uses it).

### 3.5 Full run — 4 GPUs

Build a subset first; `run_full_sc3c98.sh` points at `gemma4_dspark_100k`:

```bash
MODEL=/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it
DATA=/import/ml-sc-scratch5/chenw/datasets/kimi-regen-gemma4-31b/train_regen.jsonl
python scripts/prepare_data.py --model "$MODEL" --data "$DATA" \
  --seq-length 8192 --max-samples 100000 \
  --output /import/ml-sc-scratch1/mengmengj/datasets/gemma4_dspark_100k

sngpu --jobname dspark_full --partition gpuonly --nodelist sc3-c98 \
  --gpu 4 --gputype a100m80 --cpu 92 --mem 900000 --time 48:00:00 \
  --output ./logs/dspark_full.txt \
  -- bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/run_full_sc3c98.sh
```

Start smaller (20–30k) if you want a number the same day — hidden-state
generation, not the draft's backward pass, sets the wall clock.

### 3.6 Serve + eval

```bash
vllm serve /import/ml-sc-scratch5/chenw/models/gemma-4-31B-it \
  --speculative-config '{"method":"dspark","model":"<ckpt>/checkpoint_best","num_speculative_tokens":7}' \
  --chat-template /import/ml-sc-scratch1/mengmengj/spec_eval/gemma4_chat_template.jinja
```
Gemma-4 ships no chat template in `tokenizer_config.json` (it's a separate
`chat_template.jinja`), so vLLM 400s on chat completions without `--chat-template`.

Then write `scripts/evaluate/experiments/gemma4-31b-dspark.yaml` (copy an existing
`gemma4-31b*.yaml`) and compare against: baseline, `RedHatAI/...speculator.dspark`,
your EAGLE-3-regen, and the `gemma-4-31B-it-assistant` MTP path
(measured 2026-07-06: AR 0.87, accept-len 3.6, 174 tok/s).

## 4. Training config (matches RedHat's shipped checkpoint)

| Flag | Value |
|---|---|
| `--speculator-type` | `dspark` |
| `--block-size` | 8 |
| `--num-layers` | 5 |
| `--draft-vocab-size` | 32000 |
| `--target-layer-ids` | `1 17 29 47 58` — **identical on vLLM and train sides** |
| `--sliding-window` | 2048 (all draft layers sliding by default; opt out with `--full-attention-indices`) |
| `--markov-rank` / `--markov-head-type` | 256 / `vanilla` |
| `--loss-fn` | `{"ce": 0.1, "tv": 0.9}` |
| `--confidence-head-alpha` | 1.0 |
| `--max-anchors` | 3072 (256 for overfit) |
| `--total-seq-len` | 8192 (multiple of 128 required by flex attention) |
| `--on-missing` / `--on-generate` | `generate` / `delete` |

Flags that **no longer exist** after the upstream merge: `--data`,
`--max-samples`, `--overwrite-data`, `--sliding-window-indices`. Subset via
`prepare_data.py --max-samples` instead.

## 5. Code reading guide (~3-4 h, in this order)

Each block makes the next one legible. Answer the questions rather than skimming —
they are the things you will need when the overfit gate misbehaves.

### Block 1 (~30 min) — the DSpark delta

`src/speculators/models/dspark/model_definitions.py` (whole file, ~100 lines)
- Where does the "previous token" come from at train time? (teacher forcing off
  `input_ids`; there is no sequential loop in training)
- `MarkovHead` takes both `verifier_vocab_size` and `draft_vocab_size` — why does
  `W1` index one and `W2` project to the other?
- `ConfidenceHead` input dim is `hidden + markov_rank`: what is concatenated, and
  what is detached before it?

`src/speculators/models/dspark/core.py:32-56` and `:112+`
- how the heads attach; note `DSparkDraftModel(DFlashDraftModel)` — everything
  else is inherited, hence Block 2

### Block 2 (~60 min) — the inherited backbone

`src/speculators/models/dflash/core.py:271-373` (the real forward)
- what an "anchor" is, and why `--max-anchors` controls memory
- lines 299-306: how mask tokens are built and where the anchor token is spliced in
- lines 327-333: target distributions computed locally from
  `verifier_last_hidden_states` via the frozen LM head — why the `torch.roll(...,1)`?
  (this is why DeepSpec's offline target cache is unnecessary here)
- line 362: `aligned_loss_mask[:, ::block_size] = 0` — what does it zero, and how
  does `sample_from_anchor` change it?

`src/speculators/models/dflash/model_definitions.py` — `Qwen3DFlashAttention.forward`:
the KV-injection trick (`k_ctx` from `target_hidden`). This one function explains
the architecture.

### Block 3 (~45 min) — loss and metrics (what you watch during training)

`src/speculators/models/dspark/metrics.py:48-150`
- `accept_rate = sum_v min(q_v,p_v) = 1 - d_TV`: why is this the confidence
  head's label instead of a 0/1 accepted flag?
- how `accept_len` is computed, and whether it includes the bonus token (matters
  for comparing against the paper's 6.05 / 5.64)
- what `confidence_cumprod_bias` measures, and why STS would target it

`src/speculators/models/metrics.py` — `loss_function`, `resolve_loss_config`,
`dflash_loss_decay` (line 286).

**Discrepancy to resolve:** the paper uses `w_k = exp(-(k-1)/gamma)` with
**gamma = block_size = 8**; the code uses the same formula but
`--dflash-decay-gamma` defaults to **4.0**, decaying twice as fast and
down-weighting exactly the tail positions the Markov head exists to rescue.
Decide whether to pass `--dflash-decay-gamma 8` for parity.

### Block 4 (~45 min) — plumbing you will debug

- `src/speculators/train/data.py:203-264` — `ArrowDataset` online path; trace one
  sample from `__getitem__` through `on_missing=generate` to `on_generate=delete`
- `scripts/launch_vllm.py` + `hs_connectors/src/hs_connectors/transfer.py:132-155`
  — server side and the file handoff
- `src/speculators/train/trainer.py:424-530`, `:639` — loop, and how
  `_sum`/`_total` metrics reduce across ranks
- `src/speculators/train/config/schema.py` — every CLI flag (`DataArgs` ~195,
  `DFlashArgs` ~437, `DSparkArgs` ~464)

### Capstone (~20 min) — paper vs implementation

Diff `.claude/skills/dspark-speculators/reference.md` §1-4 against the code.
Three known deltas to confirm:
1. position-weight gamma: 4.0 (code) vs block_size=8 (paper)
2. `sample_from_anchor`: upstream defaults `True`, RedHat's checkpoint says `false`
3. STS calibration (§4) is absent upstream — only calibration *metrics* exist

## 6. Open items / known risks

- [ ] **`HS_PATH` is on NFS.** `dspark_online_gemma4_31b.sh` defaults it under
      `/import`. At 64 KB/token, 10k tok/s is ~640 MB/s sustained write+read —
      NFS likely won't hold that. Move it to node-local scratch before the full
      run (states are transient under `--on-generate delete`).
- [ ] Add a 20–30k-sample preset for a same-day first result.
- [ ] Patch dead `--data`/`--max-samples` flags in
      `examples/train/eagle3_online_gemma4_31b.sh` — that script is currently
      broken on this branch and it's colleagues' entry point.
- [ ] Re-add a `--data` shim to the merged `train.py` if the team wants the old
      one-command flow back (~40 lines).
- [ ] STS calibration (skill doc §3) is the only DSpark piece missing upstream.
      Defer until eval numbers exist — and first check whether vLLM 0.26 even
      consumes per-position temperatures (RedHat's config has no such field).
- [ ] Queue contention is the schedule's dominant term, not engineering.

## 7. Container path (the candidate NCCL fix)

### Why

On bare metal, NCCL segfaults in `ncclNetPluginInit` (`plugin/net.cc:216`) during
`ncclCommInitRank`. It kills vLLM TP>=2 and any `torchrun` launch. Tried and did
NOT help: `NCCL_NET_PLUGIN=none`, `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=lo`.

Ravi runs vLLM **TP=4 on this same node** inside the NGC container, so the
container is the prime suspect for a fix — the host's broken `libnccl-net.so`
simply is not visible inside it.

**We do not strictly need this.** The working bare-metal layout (vLLM TP=1 +
training launched as plain `python`) trains fine; the container buys *throughput*
(TP=2/4 for hidden-state generation), not correctness.

### Facts established 2026-07-31

- `nvcr.io/nvidia/pytorch:25.12-py3` is **public** — `docker manifest inspect`
  succeeds with no NGC auth.
- `docker` exists on the login node (`/usr/bin/docker`), but compute nodes pull
  their own image; a login-node pull does not warm their cache.
- **`sngpu --image` works only with `--interactive`.** In batch it fails
  (`boot_docker.sh` needs sudo and a TTY). This is the crux for long runs.
- A container is **ephemeral**: `pip install`s inside it vanish when the session
  ends. Only installs into our conda env on `/import` persist — which is why
  phase 1 below (reuse the existing env, zero installs) is the path to prefer.
- Ravi's docker notes: the image defaults to a **64 MB `/dev/shm`**, far too small
  for the DataLoader; he passes `--ipc=host --ulimit memlock=-1
  --ulimit stack=67108864`. Bare metal has 504 GB, so this is a container-only
  problem we do not otherwise have.

### Procedure

```bash
# 2 GPUs to get the NCCL answer; 1 GPU still answers everything else
sngpu --interactive --time 00:30:00 --cpu 24 --mem 200000 --gpu 2 \
  --nodelist sc3-c98 --image nvcr.io/nvidia/pytorch:25.12-py3
# inside the container shell:
bash /import/ml-sc-scratch1/mengmengj/speculators/examples/train/container_setup.sh
```

`container_setup.sh` runs three phases, cheapest first:

| Phase | Question | Outcome |
|---|---|---|
| 0 | in a container? `/dev/shm` size? host `libnccl-net.so` visible? | absent plugin ⇒ strong signal the container fixes it |
| 1 | does our `/import` conda env import torch/vLLM/speculators in here? | if yes, **no installs needed at all** |
| 2 | **does a 2-rank NCCL all_reduce pass?** | the answer that matters |
| 3 | fallback only if 2 fails | build on the image's own torch (`pip install --no-deps vllm==0.26.0`) — means the bad NCCL is in our pip torch, not the host |

### If it works, long runs still need more work

Because `--image` is interactive-only, a multi-hour training job cannot simply
"use the container". Two options:

1. **Port Ravi's docker-in-batch pattern** (`scripts/submit_train.sh` on
   `origin/ravir/eagle3_branch`): request a *bare* sngpu node, then have the job
   script `docker run` the image itself with `--shm-size 16G --ipc=host`. ~1 h of
   work. This is the only way to get a container into a batch job here.
2. **Stay bare metal** at TP=1 / single rank and accept lower hidden-state
   throughput. Costs nothing, already working.

Decide based on whether the final run is online-at-scale (throughput matters, do
option 1) or offline-at-30k (generation is a one-time cost, option 2 is fine).

### Scheduling reality (2026-07-31)

One user holds the `gpuonly` partition: ~36 pending **1-GPU** jobs at higher
priority than ours, plus 3 running interactive `bash` sessions on sc3-c98 with
`TIME_LIMIT=10-16:00:00` declared (actual runtime <1 h per the user).
Consequences:

- SLURM's `StartTime` estimate is fiction — it assumes declared limits.
- Real risk is **starvation of multi-GPU requests**: freed GPUs are absorbed
  one-at-a-time by his 1-GPU jobs, so 2 rarely free simultaneously.
- Our priority rises with age (QOS 444444 + age), so we do climb.
- Most useful asks: shorter declared `--time` (enables backfill for everyone),
  and keeping sc3-c98 clear since it is the only driver-compatible node.

## 8. Realistic timeline

~20–40 GPU-hours of work. On a free 4-GPU node: **3–4 working days** to a trained
checkpoint plus a measured acceptance length against RedHat's. Paper-parity on
all 349k conversations with tuning and ablations is a multi-week program, not a
one-week one.
