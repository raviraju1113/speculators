#!/usr/bin/env python3
"""Offline eval of a DSpark draft (e.g. RadixArk/Kimi-K3-DSpark) for Kimi K3.

Loads the HF-exported DSpark checkpoint into this repo's DSparkDraftModel
(the weight names match the export 1:1; embed/lm_head/verifier_norm come from
the verifier) and replays captured target hidden states through the repo's own
training-time forward, which computes:

- ``accept_rate``: analytical per-position acceptance = 1 - TV(draft, target)
- ``accept_len``: expected accepted draft tokens per block (cumprod of
  accept_rate over draft slots) — the DSpark analog of simulated acceptance len
- ``position_{k}_acc`` / ``full_acc``: top-1 agreement vs target argmax

Samples must be collected with the DRAFT's aux layer set: dflash
``target_layer_ids`` are 0-based layer outputs, so pass ids+1 to
``launch_vllm.py`` (e.g. [7,23,51,67,83] -> ``--target-layer-ids 8 24 52 68 84``,
final layer auto-appended). Files use the ``hs_stack`` schema written by
prep_and_collect.py / sweep_collect.py.

Run on 1 free GPU (kimi_k3 env):
    PYTHONPATH=src:hs_connectors/src python \
        scripts/evaluate/kimi_k3_offline_eval/run_dspark_eval.py \
        --draft /import/ml-sc-scratch5/chenw/models/Kimi-K3-DSpark \
        --model /import/ml-sc-scratch5/chenw/models/Kimi-K3-patched \
        --samples <collection dir>
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Qwen3Config

from speculators.models.dspark.core import DSparkDraftModel


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True, help="HF DSpark export dir")
    ap.add_argument("--model", required=True, help="Kimi-K3 dir (verifier)")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--max-anchors", type=int, default=3072)
    ap.add_argument("--output-json", default=None)
    ap.add_argument(
        "--truncate-k",
        type=int,
        default=None,
        help=(
            "Post-hoc: pool only the first K of the already-computed 7 "
            "slots' accept_rate/accept_len. NOTE this does NOT simulate a "
            "smaller drafting block -- within-block attention here is "
            "non-causal (create_anchor_block_mask_mod, dflash/attention.py), "
            "so slot 0-2's hidden states already reflect having attended to "
            "slots 3-6. Use --block-size-override for the real ablation."
        ),
    )
    ap.add_argument(
        "--block-size-override",
        type=int,
        default=None,
        help=(
            "Actually run the draft with a smaller block_size (e.g. 3 instead "
            "of the trained 7): changes the synthetic same-block attention "
            "window itself (fewer mask positions to attend to), the "
            "ground-truth targets pulled per block, and the RoPE positions "
            "assigned to synthetic slots. No weights are block_size-shaped, "
            "so this is safe against the loaded checkpoint. Reuses the "
            "already-collected hidden states -- no new server/collection."
        ),
    )
    return ap.parse_args()


def build_model(draft_dir: str, verifier: str, max_anchors: int) -> DSparkDraftModel:
    rc = json.load(open(Path(draft_dir) / "config.json"))
    tl = Qwen3Config(
        vocab_size=rc["vocab_size"],
        hidden_size=rc["hidden_size"],
        intermediate_size=rc["intermediate_size"],
        num_hidden_layers=rc["num_hidden_layers"],
        num_attention_heads=rc["num_attention_heads"],
        num_key_value_heads=rc["num_key_value_heads"],
        head_dim=rc["head_dim"],
        hidden_act=rc["hidden_act"],
        rms_norm_eps=rc["rms_norm_eps"],
        max_position_embeddings=rc["max_position_embeddings"],
        rope_theta=rc["rope_parameters"]["rope_theta"],
        rope_scaling={
            k: v for k, v in rc["rope_parameters"].items() if k != "rope_theta"
        },
        layer_types=rc["layer_types"],
        attention_bias=rc["attention_bias"],
        tie_word_embeddings=False,
    )
    model = DSparkDraftModel.from_training_args(
        verifier_config=tl,
        verifier_name_or_path=verifier,
        trust_remote_code=True,
        target_layer_ids=rc["dflash_config"]["target_layer_ids"],
        block_size=rc["block_size"],
        max_anchors=max_anchors,
        draft_vocab_size=rc["vocab_size"],
        markov_rank=rc["markov_rank"],
        markov_head_type=rc["markov_head_type"],
        enable_confidence_head=rc["enable_confidence_head"],
        confidence_head_with_markov=rc["confidence_head_with_markov"],
        loss_fn='{"ce": 0.1, "tv": 0.9}',
        confidence_head_alpha=1.0,
        total_seq_len=32768,
        mask_token_id=rc["dflash_config"]["mask_token_id"],
    )
    sd = load_file(Path(draft_dir) / "model.safetensors")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    core_missing = [
        m
        for m in missing
        if not m.startswith(
            ("embed_tokens", "lm_head", "verifier_lm_head", "verifier_norm", "t2d", "d2t")
        )
    ]
    if core_missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={core_missing}, unexpected={unexpected}")
    return model


def main():
    args = parse_args()
    device = "cuda"
    model = build_model(args.draft, args.model, args.max_anchors)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    trained_block_size = model.block_size
    if args.block_size_override is not None:
        if args.block_size_override > trained_block_size:
            raise ValueError(
                f"--block-size-override {args.block_size_override} exceeds the "
                f"trained block_size {trained_block_size}"
            )
        model.block_size = args.block_size_override
        print(
            f"Overriding block_size {trained_block_size} -> {model.block_size} "
            "(same weights; smaller synthetic same-block attention window)"
        )

    train_kwargs, _ = DSparkDraftModel.get_trainer_kwargs(
        loss_fn='{"ce": 0.1, "tv": 0.9}', max_anchors=args.max_anchors
    )

    sample_files = sorted(
        Path(args.samples).glob("sample_*.safetensors"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    print(f"{len(sample_files)} samples")

    sums: dict[str, float] = {}
    with torch.no_grad():
        for f in sample_files:
            d = load_file(f)
            stack = d["hs_stack"]  # [S, L(=5 aux + final), H]
            seq_len, num_layers, hidden = stack.shape
            aux = (
                stack[:, :-1, :]
                .reshape(seq_len, (num_layers - 1) * hidden)
                .unsqueeze(0)
                .to(device, torch.bfloat16)
            )
            last = stack[:, -1, :].unsqueeze(0).to(device, torch.bfloat16)
            input_ids = d["input_ids"].unsqueeze(0).to(device)
            loss_mask = d["loss_mask"].unsqueeze(0).to(device)
            _, _loss, metrics = model(
                hidden_states=aux,
                input_ids=input_ids,
                loss_mask=loss_mask,
                verifier_last_hidden_states=last,
                document_ids=torch.zeros_like(input_ids),
                truncate_k=args.truncate_k,
                **train_kwargs,
            )
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + float(v)

    results = {
        "num_samples": len(sample_files),
        "max_anchors": args.max_anchors,
        "trained_block_size": trained_block_size,
        "eval_block_size": model.block_size,
    }
    for k in sorted(sums):
        if k.endswith("_sum"):
            base = k[:-4]
            total = sums.get(base + "_total", 0.0)
            if total > 0:
                results[base] = sums[k] / total
    print(json.dumps(results, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
