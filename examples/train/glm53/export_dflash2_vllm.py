#!/usr/bin/env python
"""Export a speculators DFlash2 checkpoint as a vLLM-native DFlash2 draft.

vLLM (0.28) serves DFlash2 drafts as ``DFlash2DraftModel`` with a
``dflash_config`` block (the layout of the incoai GLM-5.3-DFlash2 card), but
its speculators-format loader only registers eagle3/peagle/dflash/dspark, so a
``speculators_model_type: dflash2`` checkpoint is rejected as is. The tensor
names already match vLLM's qwen3_dflash2 module one for one, so only the
config changes: ``model.safetensors`` is symlinked, not copied. The draft's own
embed_tokens/lm_head ship with it and vLLM loads them when present.

The aux-layer translation is the one vLLM applies to speculators ``dflash``
checkpoints (transformers_utils/configs/speculators/algos.py::update_dflash):
``eagle_aux_hidden_state_layer_ids`` = the speculators ids, and
``dflash_config.target_layer_ids`` = ids - 1 (vLLM #40727), with
``causal = not sliding_window_non_causal``.

Serve with ``speculative_config: {method: dflash}`` and k = block_size - 1
(the checkpoint must be trained with sample_from_anchor=false: vLLM's DFlash2
block is the 1+N bonus-anchor layout).

    python examples/train/glm53/export_dflash2_vllm.py \
        --checkpoint .../checkpoints/8 --output .../checkpoints_vllm/8
"""

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True, help="speculators dflash2 checkpoint dir")
    ap.add_argument("--output", required=True, help="vLLM draft dir to write")
    args = ap.parse_args()

    src = Path(args.checkpoint).resolve()
    out = Path(args.output)
    cfg = json.loads((src / "config.json").read_text())
    if cfg.get("speculators_model_type") != "dflash2":
        raise SystemExit(f"not a dflash2 checkpoint: {src}")
    if cfg.get("sample_from_anchor"):
        raise SystemExit("sample_from_anchor=true: vLLM's DFlash2 block would read it one slot off")

    verifier = cfg["speculators_config"]["verifier"]["name_or_path"]
    vcfg = json.loads((Path(verifier) / "config.json").read_text())
    aux = cfg["aux_hidden_state_layer_ids"]

    draft = dict(cfg["transformer_layer_config"])
    draft.update(
        architectures=["DFlash2DraftModel"],
        model_type="qwen3",
        dtype=cfg.get("dtype", "bfloat16"),
        draft_vocab_size=cfg["draft_vocab_size"],
        num_target_layers=vcfg["num_hidden_layers"],
        eagle_aux_hidden_state_layer_ids=aux,
        is_causal=not cfg.get("sliding_window_non_causal", True),
        use_cache=False,
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        eos_token_id=vcfg.get("eos_token_id"),
        pad_token_id=vcfg.get("pad_token_id"),
        dflash_config={
            "block_size": cfg["block_size"],
            "conv_group_size": cfg["conv_group_size"],
            "conv_kernel_size": cfg["conv_kernel_size"],
            "mask_token_id": cfg["mask_token_id"],
            "selector_rank": cfg["selector_rank"],
            "selector_top_k": cfg["selector_top_k"],
            "target_layer_ids": [i - 1 for i in aux],
            "causal": not cfg.get("sliding_window_non_causal", True),
            # Explicit for SGLang, whose DSpark/NVFP4 config readers default it to True.
            "sample_from_anchor": False,
        },
    )
    if cfg.get("target_hidden_size") is not None:
        draft["target_hidden_size"] = cfg["target_hidden_size"]

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(draft, indent=2) + "\n")
    link = out / "model.safetensors"
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(src / "model.safetensors", link)
    print(f"[export] {src} -> {out} (k = {cfg['block_size'] - 1}, aux {aux}, verifier {verifier})")


if __name__ == "__main__":
    main()
