#!/usr/bin/env python3
"""Phase 2 of the Kimi-K3 offline draft eval: teacher-forced TTT metrics.

Loads the TorchSpec EAGLE3 draft from its native FSDP/distcp checkpoint,
wraps it in TorchSpec's Eagle3Model TTT harness, and replays the cached
target hidden states from phase 1 — reproducing exactly the
`eval/avg_acc` / `eval/simulated_acc_len` metrics TorchSpec logged during
training (acc_i = draft-argmax vs target-argmax at TTT step i;
sim_acc_len = sum_k prod_{i<=k} acc_i).

Run on 1 GPU (kimi_k3 env):
    PYTHONPATH=<TorchSpec>:src:hs_connectors/src python \
        scripts/evaluate/kimi_k3_offline_eval/run_ttt_eval.py \
        --checkpoint .../iter_0039388 \
        --draft-config .../kimi_k3_eagle3_mla.json \
        --model /import/ml-sc-scratch5/chenw/models/Kimi-K3-patched \
        --samples .../eval_hs_samples --ttt-length 4
"""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dist_cp
from safetensors.torch import load_file

from speculators.utils.loading import load_model_layers


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="distcp dir (iter_xxx)")
    ap.add_argument("--draft-config", required=True)
    ap.add_argument("--model", required=True, help="Kimi-K3 dir (lm_head/norm)")
    ap.add_argument("--samples", required=True, help="phase-1 output dir")
    ap.add_argument("--ttt-length", type=int, default=4)
    ap.add_argument("--norm-eps", type=float, default=1e-5)
    ap.add_argument("--output-json", default=None)
    ap.add_argument(
        "--attention-backend",
        default="sdpa",
        choices=["sdpa", "flex_attention"],
        help=(
            "Draft attention backend. sdpa's TTT cached path materializes full "
            "attention weights (O(S^2) memory) -- use flex_attention for long "
            "sequences (what TorchSpec training used)."
        ),
    )
    ap.add_argument(
        "--aux-cols",
        default=None,
        help=(
            "Comma-separated column indices into the saved hs_stack to use as "
            "the aux concat (e.g. '0,3,6'). Default: all but the last column "
            "(or the legacy pre-concatenated 'hidden_states' tensor)."
        ),
    )
    return ap.parse_args()


def load_draft_state_dict(checkpoint: str) -> dict[str, torch.Tensor]:
    """Load the model weights from a TorchSpec FSDP distcp checkpoint.

    Keys in the distcp store are torchspec-native (e.g. ``midlayer.*``), so no
    remapping is needed — unlike the convert_to_hf export.
    """
    model_dir = Path(checkpoint) / "model"
    reader = dist_cp.FileSystemReader(str(model_dir))
    meta = reader.read_metadata()
    state = {
        k: torch.empty(t.size, dtype=t.properties.dtype)
        for k, t in meta.state_dict_metadata.items()
        if hasattr(t, "size")
    }
    dist_cp.load(state, storage_reader=reader)
    prefixes = ("model_state.", "model.", "draft_model.", "module.")
    out = {}
    for k, v in state.items():
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]  # noqa: PLW2901
                    changed = True
        out[k] = v
    return out


def main():
    args = parse_args()
    from torchspec.models.draft.auto import AutoDraftModelConfig, AutoEagle3DraftModel
    from torchspec.models.eagle3 import Eagle3Model, compute_lazy_target_padded

    device = "cuda"
    config = AutoDraftModelConfig.from_file(args.draft_config)
    draft = AutoEagle3DraftModel.from_config(
        config, attention_backend=args.attention_backend
    )

    state = load_draft_state_dict(args.checkpoint)
    missing, unexpected = draft.load_state_dict(state, strict=False)
    print(f"draft loaded: missing={missing}, unexpected={unexpected}")
    if any("midlayer" in k or "fc" in k for k in missing):
        raise RuntimeError(f"core draft weights missing: {missing}")

    draft = draft.to(device=device, dtype=torch.bfloat16).eval()
    model = Eagle3Model(
        draft, length=args.ttt_length, attention_backend=args.attention_backend
    )

    verifier = load_model_layers(["lm_head.weight", "model.norm.weight"], args.model)
    lm_head_weight = verifier["lm_head.weight"].to(device, torch.bfloat16)
    norm_weight = verifier["model.norm.weight"].to(device, torch.bfloat16)

    def rms_norm(x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + args.norm_eps)).to(x.dtype) * norm_weight

    sample_files = sorted(
        Path(args.samples).glob("sample_*.safetensors"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    print(f"{len(sample_files)} eval samples")

    all_acces, all_vlosses = [], []
    with torch.no_grad():
        aux_cols = (
            [int(c) for c in args.aux_cols.split(",")] if args.aux_cols else None
        )
        for f in sample_files:
            d = load_file(f)
            input_ids = d["input_ids"].unsqueeze(0).to(device)
            loss_mask = d["loss_mask"].unsqueeze(0).to(device)
            if "hs_stack" in d:
                stack = d["hs_stack"]  # [S, L, H]
                cols = aux_cols if aux_cols is not None else list(
                    range(stack.shape[1] - 1)
                )
                seq_len = stack.shape[0]
                aux = (
                    stack[:, cols, :]
                    .reshape(seq_len, len(cols) * stack.shape[2])
                    .unsqueeze(0)
                    .to(device, torch.bfloat16)
                )
            else:
                aux = d["hidden_states"].unsqueeze(0).to(device, torch.bfloat16)
            last = d["last_hidden_states"].unsqueeze(0).to(device, torch.bfloat16)

            target = compute_lazy_target_padded(
                rms_norm(last), lm_head_weight, args.ttt_length
            )
            _plosses, vlosses, acces, _counts = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                target=target,
                loss_mask=loss_mask,
                hidden_states=aux,
            )
            all_acces.append(torch.stack([a.float() for a in acces]))
            all_vlosses.append(torch.stack([v.float() for v in vlosses]))

    avg_acces = torch.stack(all_acces).mean(dim=0)
    avg_vloss = torch.stack(all_vlosses).mean(dim=0)
    avg_acc = avg_acces.mean().item()
    cumulative, sim_acc_len = 1.0, 0.0
    for i in range(avg_acces.shape[0]):
        cumulative *= avg_acces[i].item()
        sim_acc_len += cumulative

    results = {
        "num_samples": len(sample_files),
        "ttt_length": args.ttt_length,
        "eval/avg_acc": avg_acc,
        "eval/simulated_acc_len": sim_acc_len,
        **{f"eval/acc_{i}": avg_acces[i].item() for i in range(avg_acces.shape[0])},
        **{f"eval/vloss_{i}": avg_vloss[i].item() for i in range(avg_vloss.shape[0])},
    }
    print(json.dumps(results, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
