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

If `cuda_ok` is `False` with a "driver too old" warning, stop here — fix the
driver/node first; the rest of this guide won't run until that passes.

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
