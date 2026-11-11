#!/usr/bin/env python
"""Import a vLLM/HF-layout DFlash2 draft (e.g. incoai/GLM-5.3-DFlash2) as a speculators
dflash2 checkpoint, so our trainer can score it (validation) or warm-start from it.

The inverse of export_dflash2_vllm.py. The HF draft ships no embed_tokens / lm_head (it shares
the target's at serve time), so both are copied from the verifier. The speculators config comes
from a TEMPLATE checkpoint of ours with the same decoder shape; block_size and the aux layer ids
are overwritten from the HF draft's dflash_config (aux id = dflash target_layer_id + 1, the
convention vLLM and SGLang use). Every HF tensor must exist in the template with the same shape.

    python examples/train/glm53/import_dflash2_from_hf.py \
        --hf-draft /import/ml-sc-nlpcheckpoints-scratch3/mengmengj/glm-5.3-dflash2-incoai \
        --template .../output/glm53_abl_A2_incoai_shape/checkpoints/0 \
        --verifier /import/ml-sc-nlpcheckpoints-scratch3/mengmengj/glm-5.3-bf16 \
        --output   .../output/incoai_imported/speculators
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hf-draft", required=True, type=Path)
    ap.add_argument("--template", required=True, type=Path, help="our speculators dflash2 checkpoint, same shape")
    ap.add_argument("--verifier", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    hf_cfg = json.loads((args.hf_draft / "config.json").read_text())
    dcfg = hf_cfg["dflash_config"]
    cfg = json.loads((args.template / "config.json").read_text())
    tcfg = cfg["transformer_layer_config"]
    for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
              "head_dim", "intermediate_size", "vocab_size"):
        if hf_cfg[k] != tcfg[k]:
            raise SystemExit(f"shape mismatch on {k}: hf {hf_cfg[k]} vs template {tcfg[k]}")
    cfg["block_size"] = dcfg["block_size"]
    cfg["aux_hidden_state_layer_ids"] = [i + 1 for i in dcfg["target_layer_ids"]]
    for k in ("conv_group_size", "conv_kernel_size", "selector_rank", "selector_top_k", "mask_token_id"):
        cfg[k] = dcfg[k]
    tcfg["rope_parameters"] = hf_cfg.get("rope_parameters", tcfg["rope_parameters"])

    tmpl = {}
    with safe_open(str(args.template / "model.safetensors"), "pt") as f:
        for k in f.keys():
            tmpl[k] = tuple(f.get_slice(k).get_shape())
    tensors = {}
    with safe_open(str(args.hf_draft / "model.safetensors"), "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k not in tmpl:
                raise SystemExit(f"HF tensor {k} not in template")
            if tuple(t.shape) != tmpl[k]:
                raise SystemExit(f"shape mismatch {k}: hf {tuple(t.shape)} vs template {tmpl[k]}")
            tensors[k] = t.contiguous()
    wmap = json.loads((args.verifier / "model.safetensors.index.json").read_text())["weight_map"]
    for ours, theirs in (("embed_tokens.weight", "model.embed_tokens.weight"), ("lm_head.weight", "lm_head.weight")):
        with safe_open(str(args.verifier / wmap[theirs]), "pt") as f:
            t = f.get_tensor(theirs)
        if tuple(t.shape) != tmpl[ours]:
            raise SystemExit(f"verifier {theirs} shape {tuple(t.shape)} vs template {tmpl[ours]}")
        tensors[ours] = t.to(tensors["norm.weight"].dtype).contiguous()
    missing = set(tmpl) - set(tensors)
    if missing:
        raise SystemExit(f"template tensors not provided: {sorted(missing)[:10]}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    shutil.copy(args.template / "config.py", args.output / "config.py")
    save_file(tensors, str(args.output / "model.safetensors"), metadata={"format": "pt"})
    print(f"[import] {args.hf_draft} -> {args.output}: {len(tensors)} tensors, block {cfg['block_size']}, "
          f"aux {cfg['aux_hidden_state_layer_ids']}")


if __name__ == "__main__":
    main()
