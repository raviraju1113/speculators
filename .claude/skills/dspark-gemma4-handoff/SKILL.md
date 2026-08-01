---
name: dspark-gemma4-handoff
description: Portable, machine-independent handoff for training a DSpark speculative-decoding draft for google/gemma-4-31B-it with the speculators library. Use when picking this work up on a NEW machine or VM - it lists the assets to copy, the exact environment build, the training commands for both 1-GPU (offline) and multi-GPU (online) layouts, the validated config, and every gotcha already paid for. Triggers include DSpark, Gemma 4 draft model, speculators training, hidden-state generation, overfit gate, acceptance length.
---

# DSpark for Gemma-4-31B-it — portable handoff

Everything needed to continue this work on a machine that is NOT the SambaNova
cluster. Cluster-specific items are quarantined in the last section.

## 1. Goal and where we are

Train a **DSpark** draft (DFlash backbone + Markov head + confidence head) for
`google/gemma-4-31B-it`, then measure accepted length vs. existing baselines.

**No algorithm code needs writing.** DSpark is already upstream in
`vllm-project/speculators` (PR #677, merged 2026-06-29) plus ~10 follow-up fixes.
`RedHatAI/gemma-4-31B-it-speculator.dspark` exists on HF for this exact target —
use it as the config oracle and as the number to beat.

Status as of handoff: environment built, data prepared (349,389 conversations),
overfit gate partially run (22/50 epochs, `position_1_acc` 0.18 → 0.35 and still
rising, i.e. step-starved rather than broken). Not yet: converged overfit, full
run, eval.

## 2. Assets to bring to the new machine

| Asset | Where it is now | Size | Notes |
|---|---|---|---|
| **Regenerated training data** | `/import/ml-sc-scratch5/chenw/datasets/kimi-regen-gemma4-31b/train_regen.jsonl` | 1.7 GB | **The irreplaceable one.** Gemma-4's own responses re-generated over the kimi-mtp corpus; cost an 8-GPU multi-hour job. Copy this. |
| Prepared arrow dataset | `/import/ml-sc-scratch1/mengmengj/datasets/gemma4_dspark` | 3.4 GB | Optional — regenerable from the JSONL in ~1-2 h of CPU |
| Overfit splits | `datasets/gemma4_dspark_{32,256}` | small | Trivially rebuilt |
| Target model | `google/gemma-4-31B-it` | 62.6 GB | Re-download from HF, or copy |
| Repo | fork branch `mengmengj/dspark-upstream` | — | Or just clone `vllm-project/speculators` main; it has DSpark |

If the new machine has internet, only `train_regen.jsonl` really needs copying.

## 3. Environment (exact)

Requires an NVIDIA driver **>= 580** (CUDA 13) for `torch 2.11+cu130`. A cu128
build would need >= 570.26. Check first: `nvidia-smi`.

```bash
conda create -p <envdir>/dspark python=3.12 -y
conda activate <envdir>/dspark
pip install -U pip
pip install "vllm==0.26.0"                       # pulls torch 2.11.0+cu130
# hs_connectors is an in-repo uv WORKSPACE member. pip does not understand
# [tool.uv.workspace], goes looking on PyPI, and fails with
# "Could not find a version that satisfies the requirement hs-connectors".
# Install the local member FIRST:
pip install -e ./hs_connectors
pip install -e .                                 # speculators
pip install wandb && wandb login                 # writes ~/.netrc
```

Verify: `python -c "import speculators.models.dspark as d, vllm, hs_connectors; print(d.__file__, vllm.__version__)"`
Confirmed-good versions: torch 2.11.0+cu130, vllm 0.26.0, transformers 5.14.1,
datasets 5.0.0, wandb 0.28.1.

## 4. Data prep

```bash
python scripts/prepare_data.py \
  --model <path-to>/gemma-4-31B-it \          # LOCAL path: has chat_template.jinja
  --data <path-to>/train_regen.jsonl \
  --seq-length 8192 \
  --output <data>/gemma4_dspark
# small splits for the overfit gate
python scripts/prepare_data.py ... --max-samples 256 --output <data>/gemma4_dspark_256
```

- Produces `*.arrow` + `token_freq.pt`. `d2t.npy`/`t2d.npy` are built by the first
  training run from `token_freq.pt`.
- Expect ~251 warnings of "No assistant response spans found" out of 349,389 rows
  (0.07%) — rows with an empty `conversations` list. Harmless; they are dropped.
  The warning suggests raising `--seq-length`, which is a red herring.
- `train.py` no longer has `--data`, `--max-samples`, `--overwrite-data`, or
  `--sliding-window-indices` (removed/renamed upstream). Subset here, not there.

## 5. GPU layouts — pick by what you have

Verifier is **62.6 GB**; the training process needs **~42 GB** (~2.4B trainable
params at AdamW fp32 master weights + frozen embeddings).

| GPUs | Mode | Layout |
|---|---|---|
| **1** | **offline only** | vLLM alone dumps hidden states → exits → training alone. Phases split in TIME. |
| 2 | online | vLLM TP=1 + 1 training rank (plain `python`, no NCCL) |
| 3 | online | vLLM TP=2 + 1 training rank |
| 4+ | online | vLLM TP=2 + 2 training ranks (needs working NCCL) |

**Online can never fit on 1 GPU** — 62.6 + 42 > 80 GB, regardless of dataset size.

Hidden states cost **64.5 KB/token** (5 aux layers + last, bf16, hidden 5376),
≈ 26 MB per conversation:

| Samples | Offline dump |
|---|---|
| 32 | 0.8 GB |
| 256 | 6.6 GB |
| 30k | 774 GB |
| 100k | 2.6 TB |
| 349k (full) | ~9 TB |

So: **offline for overfit + tuning** (pays generation once, makes each rerun
nearly free — online with `--on-generate delete` regenerates every epoch),
**online for the final full-scale run** (storage).

## 6. Running it

Ready-made scripts in `examples/train/`:
- `run_overfit_1gpu_offline.sh` — 1 GPU, offline, 200 epochs. Start here.
- `dspark_online_gemma4_31b.sh` — parameterized online run (`MODE=overfit|full`).
- `run_overfit_nonccl.sh` — 2 GPUs, online, no NCCL anywhere.

Minimal online invocation, if writing from scratch:

```bash
# GPU 0: verifier + hidden-state streaming
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py <model> \
  --hidden-states-path <hs> --target-layer-ids 1 17 29 47 58 \
  -- --tensor-parallel-size 1 --max-model-len 8192 \
     --gpu-memory-utilization 0.95 --no-enable-prefix-caching --port 8000 &

# GPU 1: training. plain `python` for ONE rank; torchrun only for >1 (see gotchas)
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
  --verifier-name-or-path <model> --data-path <data> \
  --vllm-endpoint http://localhost:8000/v1 \
  --speculator-type dspark --block-size 8 --num-layers 5 \
  --draft-vocab-size 32000 --target-layer-ids 1 17 29 47 58 \
  --sliding-window 2048 --markov-rank 256 --markov-head-type vanilla \
  --enable-confidence-head --confidence-head-with-markov \
  --loss-fn '{"ce": 0.1, "tv": 0.9}' --confidence-head-alpha 1.0 \
  --max-anchors 3072 --total-seq-len 8192 --epochs 5 --lr 3e-4 \
  --on-missing generate --on-generate delete \
  --logger wandb --save-path <ckpt>
```

Offline variant: dump first with `scripts/data_generation_offline.py
--preprocessed-data <data> --endpoint http://localhost:8000/v1 --output <hs>
--model <model> --validate-outputs`, kill vLLM, then train with
`--hidden-states-path <hs> --on-missing skip` and no server.

## 7. Config (matches RedHat's shipped checkpoint)

| Flag | Value |
|---|---|
| `--speculator-type` | `dspark` |
| `--block-size` | 8 |
| `--num-layers` | 5 |
| `--draft-vocab-size` | 32000 |
| `--target-layer-ids` | `1 17 29 47 58` — **identical on vLLM and training sides** |
| `--sliding-window` | 2048 (all draft layers sliding by default; opt out via `--full-attention-indices`) |
| `--markov-rank` / `--markov-head-type` | 256 / `vanilla` |
| `--loss-fn` | `{"ce": 0.1, "tv": 0.9}` |
| `--confidence-head-alpha` | 1.0 |
| `--max-anchors` | 3072 (256 for overfit) |
| `--total-seq-len` | 8192 (must be a multiple of 128 for flex attention) |
| `--lr` | 3e-4 full / 1e-3 overfit — **both unvalidated, see open questions** |

## 8. Gotchas already paid for (all portable)

- **`--max-model-len` must be >= `prepare_data --seq-length` (8192).** Sizing it
  from sampled conversation lengths is wrong: 4096 was tried and 4114-token
  requests returned HTTP 400.
- **KV cache at TP=1:** 62.6 GB of weights leave ~10.9 GiB at
  `--gpu-memory-utilization 0.92`, but an 8192 context needs ~11.9 GiB → engine
  init fails with a `ValueError`. Use **0.95**. With TP>=2 this stops mattering.
- **`--no-enable-prefix-caching`** — hidden-state extraction wants every request
  actually executed.
- **`torchrun` sets `LOCAL_RANK`**, and `maybe_setup_distributed()`
  (`src/speculators/train/distributed.py:143-154`) then calls
  `init_process_group("nccl")` **even for world_size=1**. For a single training
  rank, launch `python scripts/train.py` directly — that skips NCCL entirely.
- **NCCL:** if it segfaults in `ncclNetPluginInit` (`plugin/net.cc:216`) during
  `ncclCommInitRank`, that is a broken/mismatched host network plugin. Tried and
  did NOT help: `NCCL_NET_PLUGIN=none`, `NCCL_IB_DISABLE=1`,
  `NCCL_SOCKET_IFNAME=lo`. Workarounds that DO work: single-GPU-per-role (TP=1 +
  plain `python`), or run inside the NGC container
  (`nvcr.io/nvidia/pytorch:25.12-py3`, public) where the host plugin is invisible.
  Diagnose with `NCCL_DEBUG=INFO`, which names the library before it dies.
- **`--include-last-layer` is on by default** in `launch_vllm.py`: passing 5
  target layer ids yields 6 streamed tensors (5 aux + last). The last supplies
  `verifier_last_hidden_states`. Do not add layer 60 manually.
- **Gemma-4 ships no chat template in `tokenizer_config.json`** — it is a separate
  `chat_template.jinja` in the model dir. `transformers` picks it up (so
  `prepare_data.py` is fine), but **`vllm serve` needs `--chat-template <file>`**
  or chat completions 400.
- **Distributed test footgun:** call `torch.cuda.set_device(rank)` BEFORE
  `init_process_group`, else both ranks initialize NCCL on device 0 and segfault.
- **`mp.spawn` + heredoc** does not work: the child re-imports `__main__` and
  cannot reopen `<stdin>`. Put such tests in a real file.

## 9. Gates and metrics

Metric names (after `_sum`/`_total` reduction, see `train/utils.py:71`): `loss`,
`accept_len`, `accept_rate`, `position_{1..7}_acc`, `full_acc`,
`confidence_loss`, `confidence_abs_error`, `confidence_pred_mean`,
`confidence_cumprod_bias`. There is **no AUC metric** despite what the paper
write-ups suggest.

**Overfit gate:** `loss` → ~0, `position_1_acc` > 0.95, `confidence_pred_mean`
moves off 0.5, `accept_len` climbing. A high plateau means `--target-layer-ids`,
`mask_token_id`, or the vocab mapping is wrong. Still-rising-at-the-end means
step-starved: 256 samples packed into 8192-token batches is only ~25 optimizer
steps/epoch, so use many epochs (200) — cheap once hidden states are cached.

**Numbers to beat** (same target, gemma-4-31B-it):

| Config | accept_len | accept_rate | speedup |
|---|---|---|---|
| colleague's EAGLE-3 (kimi 300k) | 2.40–2.84 | 0.43–0.61 | 1.47–1.80× |
| `gemma-4-31B-it-assistant` (native MTP) | ~3.6 | ~0.87 | — |
| DSpark paper, Gemma4-12B block 7 | 3.35–6.05 | — | — |

Serving for eval:
```bash
vllm serve <model> --chat-template <chat_template.jinja> \
  --speculative-config '{"method":"dspark","model":"<ckpt>/checkpoint_best","num_speculative_tokens":7}'
```

## 10. Open questions (not yet answered)

1. **Learning rate.** Upstream examples use 3e-4; upstream schema default is 1e-4;
   colleague found 1e-4 >> 1e-5 for a 1-layer EAGLE-3 (different architecture).
   The 1e-3 used for the overfit gate is an unvalidated choice. Sweep {1e-4, 3e-4}.
2. **Position-weight gamma.** The paper uses `w_k = exp(-(k-1)/gamma)` with
   **gamma = block_size = 8**; the code's `--dflash-decay-gamma` defaults to
   **4.0**, decaying twice as fast and down-weighting exactly the tail positions
   the Markov head exists to fix. Try 8.
3. **`sample_from_anchor`.** Upstream defaults `True`; RedHat's shipped checkpoint
   says `false`. Affects whether you get `block_size` or `block_size-1`
   speculative tokens.
4. **STS calibration** (paper §4, sequential temperature scaling) is the only
   DSpark component absent upstream — only calibration *metrics* exist. Check
   whether vLLM even consumes per-position temperatures before building it;
   RedHat's config has no such field.

## 11. SambaNova-cluster-specific — re-derive on a new machine

Ignore all of this elsewhere; it is recorded so nothing is lost.

- Only node `sc3-c98` had a usable driver (595.71.05); `sc-c96`, `sc3-c97`,
  `sc-c82`, `sc-c120` were all 565.57.01 (too old). H200 nodes reserved.
- `sngpu` forwards the job command to `sbatch --wrap`, which accepts only
  `bash <one-script-path>` — no arguments and no `env VAR=... bash ...` prefix.
  Hence the wrapper scripts in `examples/train/`.
- `sngpu --image` works only with `--interactive`; batch fails (needs sudo/TTY).
  For containers in batch, request a bare node and `docker run` from the job
  script with `--shm-size 16G --ipc=host`.
- `/dev/shm` was 504 GB bare-metal, so the docker shm workaround was
  container-only.
- Queue was dominated by one user's ~36 pending 1-GPU jobs declaring
  `TIME_LIMIT=10-16:00:00`, which starves multi-GPU requests and makes SLURM's
  `StartTime` estimates fiction.
