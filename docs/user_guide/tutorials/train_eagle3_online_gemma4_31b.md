# Train Eagle-3 for Gemma-4-31B-it (Online, Disaggregated)

End-to-end recipe for training an Eagle-3 draft for `google/gemma-4-31B-it` with
**online, disaggregated** training: vLLM serves the frozen backbone and streams
hidden states on demand, while a separate process trains the draft (no offline
hidden-state dump). This tutorial also documents how to run the **CUDA-13 vLLM
stack on an older (CUDA 12.7 / driver 565.x) GPU node** via NVIDIA forward
compatibility — the setup that made this actually run here.

For a ready-to-run script see
[`examples/train/eagle3_online_gemma4_31b.sh`](https://github.com/vllm-project/speculators/blob/main/examples/train/eagle3_online_gemma4_31b.sh).

## Overview

- **Target (frozen):** `google/gemma-4-31B-it` (~59 GB bf16; a Gemma-4 VLM, `model_type: gemma4`).
- **Algorithm:** **Eagle-3, trained from scratch.** MTP does *not* apply — this
  model has no native MTP head. The published reference draft is
  `RedHatAI/gemma-4-31B-it-speculator.eagle3`.
- **Data:** `lightseekorg/kimi-mtp-dataset` (ShareGPT `{from, value}` conversations,
  Kimi-K2.5-regenerated). The Eagle-3 preprocessing consumes this format directly.
- **Validated env:** 8× A100 80 GB, driver **565.57.01 (CUDA 12.7)**, conda env
  `gemma4-spec`, `vllm 0.25.1`, `torch 2.11.0+cu130`.

> **Why "from scratch":** the trainable draft (an FC layer + a Qwen3-style decoder
> layer + projections) is randomly initialized. It reuses the target's
> `embed_tokens` + `lm_head` (frozen), and the target itself is only used to emit
> hidden states/logits via vLLM.

## Step 0: The driver / CUDA constraint (read first)

Gemma-4 requires a recent vLLM (`0.25.x`), which ships a **CUDA-13** `torch`
(`2.11.0+cu130`). CUDA 13 normally needs an **NVIDIA driver ≥ 580**. If your node's
driver is older (here: **565.57.01**, which caps at CUDA 12.7), `torch.cuda`
reports *"NVIDIA driver on your system is too old"* and there is **no cu126 build
of this vLLM** to fall back to.

Check first:

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader   # e.g. 565.57.01
cat /proc/driver/nvidia/version | head -1                     # kernel module branch
```

- **Driver ≥ 580:** you're done with this step — skip to Step 1 and use the
  original multi-GPU script; none of the forward-compat workarounds below are needed.
- **Driver 525–579 on datacenter GPUs (A100/H100):** use **CUDA forward
  compatibility** (Step 0a). This is what the rest of this tutorial assumes.

### Step 0a: Install the CUDA-13 forward-compat `libcuda` (no root)

On datacenter GPUs, NVIDIA's `cuda-compat` package provides a userspace
`libcuda.so` (a 580.x driver library) that talks to the older kernel driver.
Anaconda's `cuda-compat` is an empty stub — pull NVIDIA's RPM and extract the lib:

```bash
COMPAT=/import/ml-sc-scratch1/chenw/cuda-compat-13.0   # any persistent path
mkdir -p "$COMPAT" && cd "$COMPAT"
REPO=https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64
RPM=cuda-compat-13-0-580.173.02-1.el8.x86_64.rpm       # rhel8; pick el8/el9 to match your OS
curl -fsSL -O "$REPO/$RPM"
rpm2cpio "$RPM" | cpio -idm                              # no root needed
cp -a usr/local/cuda-13.0/compat/libcuda.so* "$COMPAT/"

# Decisive test: does the 565 kernel driver serve the CUDA-13 API through it?
LD_LIBRARY_PATH="$COMPAT" python - <<'PY'
import ctypes
l = ctypes.CDLL("libcuda.so.1"); assert l.cuInit(0) == 0
v = ctypes.c_int(); l.cuDriverGetVersion(ctypes.byref(v))
print("CUDA driver API:", v.value, "(>=13000 = usable)")
PY
```

If that prints `13000`, forward compat works on your node. **Caveat:** it enables
**single-GPU** CUDA only — **NCCL multi-GPU collectives segfault** under
forward-compat (see Step 3). Plan for a single inference GPU + single training GPU.

## Step 1: Environment (one-time)

Everything below runs in **one conda env**:

```text
name   : gemma4-spec
prefix : /import/snvm-sc-scratch2/chenw/miniconda3/envs/gemma4-spec
python : 3.10.14
key pkgs: vllm 0.25.1, torch 2.11.0+cu130, transformers >=5.10.2
```

Gemma-4 needs `transformers ≥ 5.x` and a Gemma-4-capable vLLM. Activate the env
(or create it first — `conda create -n gemma4-spec python=3.10 -y`), then install:

```bash
CONDA=/import/snvm-sc-scratch2/chenw/miniconda3
conda activate gemma4-spec          # env prefix: $CONDA/envs/gemma4-spec
pip install -U pip
pip install -U "vllm==0.25.1"                       # brings torch 2.11.0+cu130
pip install -e /import/ml-sc-scratch1/chenw/speculators
pip install -U "transformers>=5.10.2"
```

Building vLLM compiles `llguidance` (Rust, no wheels). If your **home directory has
a small quota**, point rustup/cargo at scratch first so the toolchain download
doesn't fail with "Disk quota exceeded":

```bash
export CARGO_HOME=/path/on/scratch/.cargo
export RUSTUP_HOME=/path/on/scratch/.rustup
export PATH="$CARGO_HOME/bin:$PATH"
```

### Dependency fixes required on this stack

The CUDA-13 torch pulls `numpy 2.x`, which breaks some older transitively-imported
packages. Apply once:

```bash
pip uninstall -y flash-attn        # 2.5.x built vs old torch -> ABI 'undefined symbol' at import
pip uninstall -y bitsandbytes      # 0.43 imports the removed triton.ops (Triton 3.x)
pip install -U wandb               # 0.17 uses the removed np.float_ (numpy 2.0)
```

Rationale: vLLM's rotary path only suppresses `ModuleNotFoundError`, so a *broken*
`flash-attn` (present but unimportable) crashes it — absent is better (falls back
to native rotary + `TRITON_ATTN`). `accelerate` guards `bitsandbytes`/`wandb`
behind `is_*_available()`, so removing/upgrading them makes the import chain clean.

## Step 2: Data

```text
backbone : /import/ml-sc-scratch5/chenw/models/gemma-4-31B-it
dataset  : /import/ml-sc-scratch5/chenw/datasets/kimi-mtp-dataset/data/train-00000-of-00001.jsonl
```

The dataset is ShareGPT format (`{"conversations": [{"from","value"}, ...]}`);
`--data` tokenizes it on the fly (loss-masked to assistant tokens), so there is
**no separate `prepare_data.py` step**.

### What training actually consumes (why the responses matter)

Eagle-3 online training is **self-distillation** — there is **no ground-truth
label**. For each `prompt+completion`, vLLM does **one forward pass** over the
existing tokens (teacher-forcing; the request sends all tokens as input and asks
for just "1 output token" — it does **not** generate) and returns, per position:

1. the target's **auxiliary hidden states** (layers `[2, 30, 57]`) → the draft's
   *input*, and
2. the target's **next-token distribution** (`lm_head(last_hidden)`) → the
   distillation *target* (soft labels).

The draft is trained (KL) to match that distribution, unrolled `--ttt-steps`
(Training-Time Test) so it also learns to consume its *own* predictions like it
will at inference.

Because the target's distribution is computed **conditioned on the completion
tokens**, the completion's *distribution* matters: only the **prompts are
model-agnostic**; ideally the **responses come from the target itself**.

### Option 1 — use kimi as-is (baseline)

Simplest; good first run. Its responses were generated by *Kimi-K2.5*, so they're
slightly off-distribution for gemma-4 → typically **lower acceptance**, but it
trains fine. Train in the default **on-policy** mode (Step 3).

### Option 2 — regenerate responses with gemma-4 (recommended)

Reuse the kimi **prompts** but replace every response with **gemma-4's own
output**, so the draft trains on in-distribution tokens (best acceptance):

> The built-in `scripts/response_regeneration` only supports a few *named* HF
> datasets with a flat prompt field — **not** a local ShareGPT/conversations
> jsonl. For the kimi data, extract the first human turn as the prompt and call
> the chat endpoint yourself:

```bash
# 1. Serve gemma-4 for GENERATION (plain vllm serve, single GPU, same env as Step 3)
CUDA_VISIBLE_DEVICES=0 vllm serve /import/ml-sc-scratch5/chenw/models/gemma-4-31B-it \
  --host 127.0.0.1 --port 8001 --api-key "" \
  --max-model-len 8192 --enforce-eager --gpu-memory-utilization 0.93 &
until curl -sf http://127.0.0.1:8001/v1/models >/dev/null; do sleep 5; done

# 2. prompt -> gemma-4 response, write the SAME conversations format (greedy = the
#    target's argmax path). ~pseudocode; loop over the kimi jsonl:
python - <<'PY'
import json
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="EMPTY")
M = "/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it"
IN  = "/import/ml-sc-scratch5/chenw/datasets/kimi-mtp-dataset/data/train-00000-of-00001.jsonl"
OUT = "/import/ml-sc-scratch1/chenw/gemma4_regenerated.jsonl"
with open(IN) as f, open(OUT, "w") as g:
    for i, line in enumerate(f):
        if i >= 5000: break
        conv = json.loads(line)["conversations"]
        prompt = next((m["value"] for m in conv if m["from"] in ("human", "user")), None)
        if not prompt: continue
        r = c.chat.completions.create(model=M, messages=[{"role":"user","content":prompt}],
                                      max_tokens=2048, temperature=0.0)
        g.write(json.dumps({"conversations": [
            {"from":"human","value":prompt},
            {"from":"gpt","value": r.choices[0].message.content or ""}],
            "source":"gemma4-regenerated"}) + "\n")
PY
```

Then in Step 3, point `--data` at `gemma4_regenerated.jsonl` **and add
`--use-off-policy-tokens`** (see the training command note). Regeneration is a
one-time pass, but note it *does* generate many tokens per prompt — slow on a
single GPU (plan for it, or use `--limit` for a first batch).

## Step 3: Launch — the runtime environment

Every command (vLLM **and** training) must export the following. Bake them into a
launcher so both inherit them:

```bash
ENV=/import/snvm-sc-scratch2/chenw/miniconda3/envs/gemma4-spec
COMPAT=/import/ml-sc-scratch1/chenw/cuda-compat-13.0

# env/lib FIRST (its libstdc++ has GLIBCXX_3.4.29 for libzmq; system /lib64 lacks it),
# then the forward-compat libcuda.
export LD_LIBRARY_PATH="$ENV/lib:$COMPAT:$LD_LIBRARY_PATH"
# ninja + nvcc on PATH for vLLM's runtime JIT; CUDA_HOME so it finds headers.
export PATH="$ENV/lib/python3.10/site-packages/ninja/data/bin:$ENV/lib/python3.10/site-packages/nvidia/cu13/bin:$PATH"
export CUDA_HOME="$ENV/lib/python3.10/site-packages/nvidia/cu13"
export PYTHONNOUSERSITE=1              # ignore a broken ~/.local (e.g. half-installed pyarrow)
export VLLM_USE_FLASHINFER_SAMPLER=0  # native sampler; the flashinfer JIT toolchain mismatches here
export WANDB_MODE=disabled            # no login/network at runtime
```

### GPU layout — single GPU (forward-compat nodes)

Because NCCL segfaults under forward compat, run **one** inference GPU and **one**
training GPU (no TP/DP, no distributed training). On an 80 GB A100 the 59 GB
weights leave ~12.5 GB for KV cache (`--enforce-eager` avoids graph-capture memory
and a kernel-load path):

```bash
# vLLM (GPU 0): serve backbone + stream hidden states
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py \
  /import/ml-sc-scratch5/chenw/models/gemma-4-31B-it \
  -- --tensor-parallel-size 1 --data-parallel-size 1 \
     --max-model-len 4096 --port 8000 \
     --enforce-eager --gpu-memory-utilization 0.93 &
# wait for "Application startup complete"
until curl -sf http://localhost:8000/health >/dev/null; do sleep 5; done
```

### Train (GPU 1) — plain `python`, NOT torchrun

`speculators` sets `is_distributed = "LOCAL_RANK" in os.environ`, and **torchrun
always sets `LOCAL_RANK`** — even for `--nproc_per_node 1`. That triggers
`init_process_group("nccl")`, which **segfaults** under forward compat. Launch
single-GPU training with plain `python` so no process group (and no FSDP) is created:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
  --verifier-name-or-path /import/ml-sc-scratch5/chenw/models/gemma-4-31B-it \
  --data /import/ml-sc-scratch5/chenw/datasets/kimi-mtp-dataset/data/train-00000-of-00001.jsonl \
  --data-path ./output/gemma4_31b_eagle3 \
  --max-samples 20000 \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path ./output/gemma4_31b_eagle3/checkpoints \
  --speculator-type eagle3 \
  --draft-vocab-size 32000 \
  --epochs 3 --lr 1e-4 \
  --total-seq-len 4096 \
  --on-missing generate --on-generate delete
```

> **Keep `--total-seq-len` == vLLM `--max-model-len`.** If training tokenizes
> longer than vLLM's context, every over-length sample is rejected with HTTP 400
> ("maximum context length is 4096 tokens"). Both are 4096 here.

> **Using regenerated data (Option 2)?** Point `--data` at
> `gemma4_regenerated.jsonl` and add **`--use-off-policy-tokens`**. In the TTT
> unroll, on-policy (default) feeds the draft's *own* argmax'd token at each step;
> off-policy feeds the *ground-truth* token instead — and for target-regenerated
> data those ground-truth tokens *are* gemma-4's own outputs, which is what its
> `--help` means by "required for regenerated data".

### On a driver ≥ 580 (no forward compat)

Skip all the extra env vars and use the multi-GPU layout from the script: 6
inference GPUs (`--tensor-parallel-size 2 --data-parallel-size 3`) + 1 training GPU
via `torchrun --standalone --nproc_per_node 1`, at `--total-seq-len 8192` /
`--max-model-len 8192`.

## Step 4: What healthy training looks like

```
Training epoch 1/3 started
train/loss=30.8 ...              # starts high (large-vocab distillation)
train/cond_acc_2=0.00 -> 0.47    # conditional acceptance climbs as it learns
train/loss ... -> ~10-14         # loss falls over the first few hundred steps
```

`nvidia-smi` should show GPU 0 pinned by vLLM (~79 GB) and GPU 1 at ~97 % util
(training). Checkpoints land in `./output/gemma4_31b_eagle3/checkpoints/`.

## Step 5: Evaluate

```bash
DRAFT=./output/gemma4_31b_eagle3/checkpoints/checkpoint_best \
  bash examples/evaluate/eval_gemma4_31b.sh
```

See [Evaluating Performance](evaluating_performance.md).

## Common issues & solutions

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` False, "driver too old" | CUDA-13 stack on ≤12.7 driver | Forward-compat `libcuda` (Step 0a); do **not** downgrade torch to cu126 (breaks vLLM) |
| `ImportError: libcudart.so.13` on `import vllm` | torch downgraded to cu126, mismatched vLLM | Reinstall the cu13 torch (`pip install --force-reinstall torch==2.11.0`) |
| NCCL `ncclInitKernelsForDevice` **SIGSEGV** | NCCL under forward compat | Single-GPU only; train with `python`, not `torchrun` |
| `GLIBCXX_3.4.29 not found` (libzmq) | system libstdc++ shadowing env's | Put `$ENV/lib` first on `LD_LIBRARY_PATH` |
| `FileNotFoundError: 'ninja'` / flashinfer JIT build fails | JIT toolchain not on PATH / header mismatch | Add ninja+nvcc to PATH + `CUDA_HOME`; set `VLLM_USE_FLASHINFER_SAMPLER=0` |
| `KeyError: 'rope_type'` building the draft | verifier has nested per-attention rope (Gemma-4); flat Qwen3 draft can't read it | Flatten it in `create_transformer_layer_config` (`scripts/train.py`) — pick a full-attention entry, force `rope_type: "default"` |
| `ModuleNotFoundError: triton.ops` / `np.float_` removed | numpy-2.0 fallout | Remove `bitsandbytes`; upgrade `wandb` (Step 1) |
| HTTP 400 "maximum context length is 4096" per sample | `--total-seq-len` > vLLM `--max-model-len` | Make them equal |

## Notes & next steps

- **Data provenance & acceptance:** the kimi responses are *Kimi-K2.5*'s, not
  gemma-4's — fine for a baseline, but regenerate with the target for best
  acceptance (Step 2, Option 2). Background on why: [Response
  Regeneration](response_regeneration.md).
- **The clean fix for everything above** is a GPU driver upgrade to **≥ 580** — it
  removes the compat lib, the single-GPU restriction, and the NCCL workaround, and
  restores the full multi-GPU / 8192-context recipe.
