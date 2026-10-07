# Gemma 4 31B-it — online draft training + eval

End-to-end recipe for training a speculative-decoding draft for
`google/gemma-4-31B-it` and evaluating it, on a single multi-GPU box.

- **Backbone (target):** `google/gemma-4-31B-it` — a ~62.6 GB MoE **VLM**.
- **Algorithm:** **EAGLE-3** (or DFlash). MTP does **not** apply — this model has
  no native MTP head; its published speculator is
  `RedHatAI/gemma-4-31B-it-speculator.eagle3`.
- **Data:** `lightseekorg/kimi-mtp-dataset` (ShareGPT-style conversations).

## 0. Hardware / driver requirement (read first)

Gemma 4 needs a **recent vLLM** (e.g. `vllm 0.25.x`), which pins **torch ≥2.12**,
which ships only **CUDA 12.8 / 13.0** builds. That means a minimum **NVIDIA driver
of ≥ 570.26** (for the cu128 build; newer still for cu130).

- **A driver on CUDA ≤ 12.7 (e.g. 565.x) cannot run this** — torch fails with
  "NVIDIA driver on your system is too old", and there is **no cu126/cu124 build
  of torch 2.12** to fall back to. This is a hardware/driver constraint, not
  something an env/pip change can work around; use a node with a ≥570 driver (or
  have the driver updated).
- GPUs: 8× A100 (40 GB or 80 GB). 80 GB is comfortable; see the GPU-layout note
  in step 3.

Check before anything else:

```bash
nvidia-smi | grep -i "Driver Version"     # want a CUDA >= 12.8 driver (>= 570.26)
python -c "import torch, vllm; print(torch.__version__, vllm.__version__, 'cuda_ok:', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count())"
# want: cuda_ok: True   gpus: 8
```

If `cuda_ok` is `False` with a "driver too old" warning, either move to a ≥570
node, or use CUDA forward-compatibility (below).

### 0b. Old driver? CUDA forward-compatibility (A100 / datacenter GPUs)

A100s support **CUDA forward compatibility**, so the cu13 vLLM/torch stack *can*
run on a 565 / CUDA-12.7 driver by loading a newer `libcuda` from NVIDIA's
`cuda-compat` package. This is what actually runs Gemma-4 on this box. Set **all**
of these before serving/training (and `source`/`conda activate` the env):

```bash
ENV=/import/snvm-sc-scratch2/chenw/miniconda3/envs/gemma4-spec
COMPAT=/path/to/cuda-compat-13.0     # forward-compat libcuda.so.580.x, extracted
                                     # from NVIDIA's rhel8 cuda-compat-13-0 rpm
                                     # (conda's cuda-compat is an empty stub)

# env lib FIRST (its libstdc++ has GLIBCXX_3.4.29 that libzmq needs), then compat libcuda
export LD_LIBRARY_PATH="$ENV/lib:$COMPAT:${LD_LIBRARY_PATH:-}"
# ninja + nvcc: vLLM JIT-compiles a flashinfer sampler kernel at runtime
export PATH="$ENV/lib/python3.10/site-packages/ninja/data/bin:$ENV/lib/python3.10/site-packages/nvidia/cu13/bin:$PATH"
export CUDA_HOME="$ENV/lib/python3.10/site-packages/nvidia/cu13"
export PYTHONNOUSERSITE=1             # ignore a broken ~/.local pyarrow that breaks `import transformers`
export VLLM_USE_FLASHINFER_SAMPLER=0
# one-time: `pip uninstall flash-attn` (its .so has a torch ABI mismatch; vLLM falls back to native rotary)
```

The run scripts apply all of this **automatically** when you activate the
`gemma4-spec` env and set `CUDA_COMPAT`:

```bash
conda activate gemma4-spec
CUDA_COMPAT=/path/to/cuda-compat-13.0 bash examples/train/eagle3_online_gemma4_31b.sh
CUDA_COMPAT=/path/to/cuda-compat-13.0 bash examples/evaluate/eval_gemma4_31b.sh
```

They derive the env from the active conda env (`$CONDA_PREFIX`) and force
single-GPU serving; on a proper ≥570 driver, just omit `CUDA_COMPAT`.

**Limitation — NCCL is broken under forward-compat**: multi-GPU collectives
segfault, so run vLLM **single-GPU only** (`--tensor-parallel-size 1
--data-parallel-size 1 --enforce-eager`); the 31B fits on one 80 GB A100. For
throughput, run **N independent single-GPU servers** (one per GPU) and fan the
client out across them (the `--endpoint a b c ...` round-robin) — this is exactly
what `run_regen_8gpu.sh` and the data-regeneration flow do. The clean fix is a
sysadmin driver bump to ≥580 (then drop the compat lib and use multi-GPU/TP).

## 1. Environment (one-time)

Gemma 4 needs **transformers ≥5.x** (`model_type: gemma4`) and a **vLLM build with
Gemma-4 support**. The scripts activate a conda env named `gemma4-spec` by default
(override with `CONDA_ENV=...`).

```bash
CONDA=/import/snvm-sc-scratch2/chenw/miniconda3

# Option A — reuse/rename an existing env (e.g. the old `llava` env).
# NOTE: `conda rename` clones then deletes, and the old torch/transformers are
# too old for Gemma 4 anyway, so most packages get replaced. It also stays on
# that env's Python (3.10), which the newest vLLM may reject.
conda deactivate 2>/dev/null || true
"$CONDA/bin/conda" rename -n llava gemma4-spec
conda activate gemma4-spec

# Option B — fresh env (recommended: newer Python, no stale deps)
# conda create -n gemma4-spec python=3.11 -y && conda activate gemma4-spec

# Install the stack (same for either option)
pip install -U pip
pip install -U "vllm>=0.18"                        # brings a compatible torch build
pip install -e /import/ml-sc-scratch1/chenw/speculators
pip install -U "transformers>=5.10.2,<5.13.0" pyyaml requests
```

**Verify** (reproduces the call that fails on an old transformers):

```bash
python - <<'PY'
import transformers, torch
print("transformers", transformers.__version__, "| torch", torch.__version__)
from transformers import AutoConfig
print("gemma4 ->", AutoConfig.from_pretrained(
    "/import/ml-sc-scratch5/chenw/models/gemma-4-31B-it").model_type)   # expect: gemma4
PY
python -c "import vllm; print('vllm', vllm.__version__)"
```

If `AutoConfig` still fails on `gemma4`, install transformers from source:
`pip install -U "git+https://github.com/huggingface/transformers.git"` (then
recheck it's still compatible with the installed vLLM).

## 2. Local assets (already downloaded)

```text
backbone : /import/ml-sc-scratch5/chenw/models/gemma-4-31B-it
dataset  : /import/ml-sc-scratch5/chenw/datasets/kimi-mtp-dataset/data/train-00000-of-00001.jsonl
```

## 2b. (Optional) Regenerate training data with the target

Aligning the training answers to Gemma-4's own outputs can raise draft
acceptance. `run_regen_multigpu.sh` starts one single-GPU vLLM server per GPU
(the only layout that works under forward-compat — NCCL is broken) and
regenerates the text conversations in the kimi dataset, round-robin across all
servers, resumable. Multimodal (`llava_instruct`) + tool (`continual_tool_kimi`)
rows are skipped automatically.

A ready submit wrapper with the sc-c96 paths + forward-compat env baked in is
[`scripts/response_regeneration/submit_regen_sc-c96.sh`](../../scripts/response_regeneration/submit_regen_sc-c96.sh):

```bash
# preview on the login node (no GPUs — prints the resolved commands and exits):
DRY_RUN=1 bash scripts/response_regeneration/submit_regen_sc-c96.sh

# submit the 8-GPU job (job command goes after `--`; logs -> ./logs/regen_gemma4.txt):
mkdir -p logs
sngpu --jobname regen_gemma4 --partition gpuonly --nodelist sc-c96 \
  --gpu 8 --gputype a100m80 --cpu 32 --mem 128000 \
  --output ./logs/regen_gemma4.txt --time 48:00:00 \
  -- bash /import/ml-sc-scratch1/chenw/speculators/scripts/response_regeneration/submit_regen_sc-c96.sh
```

(This node's `sngpu` has no `--bash` flag — pass the job command after `--`, or
use `--filepath <script>`. Use an **absolute** path to the wrapper: a batch job
copies a `--filepath` script to a spool dir, so relative references break.)

Output → `/import/ml-sc-scratch5/chenw/datasets/kimi-regen-gemma4-31b/train_regen.jsonl`
(over-length / failed rows → `.errors.jsonl`). It goes to a **dedicated dir**, so
**resubmit the exact same command to resume** — `--resume` skips already-done rows
(matched by `metadata.idx`), so an interrupted run picks up where it left off. Do
**not** point `OUTFILE` into the source dataset dir. Then point training's `DATA`
at the regenerated file instead of the original.

Context length matters: the wrapper defaults to `COMPAT_MAX_MODEL_LEN=8192` and
`MAX_TOKENS=4096`. A shorter context (e.g. the bare 4096 forward-compat default)
leaves too little input budget and makes many long multi-turn conversations fail
with "maximum context length exceeded" — bump these, not lower them. Other knobs:
`CONCURRENCY`, `SKIP_SOURCES=''` (keep multimodal/tool rows).

## 3. Train (disaggregated online EAGLE-3)

```bash
conda activate gemma4-spec
bash examples/train/eagle3_online_gemma4_31b.sh
```

vLLM serves the backbone + streams hidden states on the inference GPUs; training
consumes them on a separate GPU and deletes them after use (**no offline dump**).
`--data` tokenizes the local JSONL on the fly (no separate `prepare_data.py`).

**GPU layout** (gemma-4-31B is 62.6 GB; heads 32/16 → valid tensor-parallel sizes
are only {1, 2, 4, 8}, so a clean 7-way split isn't possible):

- **Default (40 GB or 80 GB):** 6 inference GPUs (TP=2 × DP=3) + 1 training GPU.
- **Full 7+1 (≥80 GB only):** edit the script to `VLLM_TP=1; VLLM_DP=7;
  TRAIN_GPUS=7; MAX_MODEL_LEN=4096` (weights leave tight KV headroom).

`SPECULATOR_TYPE=dflash bash examples/train/eagle3_online_gemma4_31b.sh` trains
DFlash instead.

## 4. Evaluate (speedup vs. baseline)

```bash
# published draft
bash examples/evaluate/eval_gemma4_31b.sh
# a freshly-trained checkpoint
DRAFT=./output/gemma4_31b_eagle3/checkpoints/checkpoint_best bash examples/evaluate/eval_gemma4_31b.sh
```

Serves baseline (backbone alone) vs. draft and prints the per-benchmark speedup.
See [scripts/evaluate/README.md](../../scripts/evaluate/README.md) for details.
