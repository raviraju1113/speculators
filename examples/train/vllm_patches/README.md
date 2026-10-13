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

Not yet written: a `gemma4_dspark.py` patch to serve DSpark checkpoints that
carry the DFlash2 local convolution (`conv_kernel_size > 0`). The conv module
already exists in vLLM main as `DFlashGroupedConv` in
`model_executor/models/qwen3_dflash2.py`; the Gemma 4 DSpark decoder layer needs
to be wrapped the same way `DFlash2Qwen3DecoderLayer` wraps the Qwen3 one.
