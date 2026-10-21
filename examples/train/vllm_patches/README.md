# Local vLLM 0.26.0 patches carried by the serving env (`dspark-host`)

These are the diffs applied on top of `vllm==0.26.0` in
`/sms-scratch/mengmengj/envs/dspark-host` (bare-metal eval venv). Originals are
kept next to each patched file as `*.orig`. The training image
(`localhost/mengmengj-vllm-official`, built from `../docker/Dockerfile`) carries
ONLY `algos.patch`, applied at build time.

| Patch | What | Why |
|---|---|---|
| `algos.patch` | `transformers_utils/configs/speculators/algos.py`: `dspark_bonus_anchor = not sample_from_anchor` (read from the checkpoint config) | Upstream 0.26.0 hardcodes `True`; our checkpoints train with `sample_from_anchor=True`. Serving them unpatched is silently wrong (k-1 effective draft). Upstream main derives it from the config since ~0.27. |
| `base.patch` | `transformers_utils/configs/speculators/base.py`: strip strays that transformers 5.14 `convert_rope_params_to_dict` leaves when a nested per-layer-type `rope_parameters` dict is flattened; re-standardize | Needed to serve checkpoint configs that use the HF-v5 nested form `{"full_attention": {...}, "sliding_attention": {...}}` (serve-time YaRN on full-attention draft layers only). Flat configs unaffected. |
| `qwen3_dflash.patch` | `model_executor/models/qwen3_dflash.py`: resolve `rope_parameters` per `layer_types` entry; fused context-KV RoPE falls back to a per-layer loop when cos/sin tables differ | Same feature: per-layer-type RoPE in the DFlash/DSpark draft (fullattn1 `checkpoints/25_yarn_full`). |

Apply on a fresh `vllm==0.26.0` install:

```bash
V=$(python -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
for p in algos base qwen3_dflash; do
  f=$(grep -m1 '^+++ ' $p.patch | awk '{print $2}' | sed 's|^vllm/||')
  cp "$V/$f" "$V/$f.orig"
  patch "$V/$f" < $p.patch
done
```

## DFlash2 local-conv serving (2026-10-21, sc3 eval env `condaenvs/dspark-host`)

Serves speculators DSpark checkpoints trained with `conv_kernel_size > 0`
(`gemma4_31b_dspark_nemo782k_conv`). Note the checkpoint is routed to
`Qwen3DSparkModel` (`qwen3_dspark.py`) -- `update_dspark` in `algos.py` sets
`architectures=["Qwen3DSparkModel"]` for every speculators DSpark checkpoint,
whatever the target -- so the patch lives there, NOT in `gemma4_dspark.py`
(that file serves Gemma-4-layer drafts, which ours is not: the draft layers
are Qwen3-style, `transformer_layer_config.model_type: qwen3`).

| File | What |
|---|---|
| `dflash2_conv.py` | new module `vllm/model_executor/models/dflash2_conv.py`: `DFlashGroupedConv` + Triton kernel, backported from vLLM main `qwen3_dflash2.py`; CPU reference path kept for parity tests |
| `qwen3_dspark_conv.patch` | `DFlash2DSparkDecoderLayer` (conv `prepare` -> attn -> `finish`, same for MLP, inside the residual branches, exactly as `speculators.models.dflash2.model_definitions.Qwen3DFlash2DecoderLayer`); `Qwen3DSparkModel` rebuilds its layers with it when `config.conv_kernel_size > 0`, popping the attention registrations first |
| `algos_conv.patch` | pass `conv_kernel_size` / `conv_group_size` through `update_dspark` (otherwise the conv weights are dropped at load: `AutoWeightsLoader` ignores names with no parameter) |
| `test_dflash2_conv_parity.py` | parity vs the training conv on the checkpoint's real weights, blocks 8/5/3, plus a cross-block leak check; CPU exact (0 error), CUDA kernel at bf16 tolerance |

Serve-time conv `block_size` is the per-request query count (`num_speculative_tokens`
with `sample_from_anchor`, `1 + N` with the bonus-anchor layout), since the draft
rows are contiguous per-request blocks with the anchor first -- not the training
`block_size`. Gate job before any conv eval: `examples/train/sbatch_sc3_conv_verify.sh`
(parity on GPU, serve, confirm "DFlash2 local conv enabled" in the log, 5-prompt
AIME smoke with an accept_len floor).

Apply on a fresh 0.26.0: copy `dflash2_conv.py` next to `qwen3_dspark.py`, then
`patch` with `qwen3_dspark_conv.patch` and `algos_conv.patch` (after `algos.patch`).
