"""Parity test: the vLLM backport of the DFlash2 grouped conv vs the speculators
training implementation, on the real conv weights of a checkpoint.

    python test_dflash2_conv_parity.py <checkpoint_dir> [--device cuda]

Checks, per draft layer and for both convs (attention_conv, mlp_conv):
  1. vLLM reference (CPU) path == speculators grouped_dynamic_conv, and
  2. (on CUDA) the Triton kernel == the CPU reference,
for random bf16 inputs laid out as contiguous blocks (block sizes 8, 5, 3 --
the k values used in the evals; the kernel must never read across a block).
Exit code 0 iff every comparison is within tolerance.
"""

import argparse
import json
import sys

import torch
from safetensors import safe_open

from speculators.models.dflash2.model_definitions import grouped_dynamic_conv
from vllm.model_executor.models.dflash2_conv import _grouped_conv, dflash2_grouped_conv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rows", type=int, default=40)
    args = ap.parse_args()

    cfg = json.load(open(f"{args.ckpt}/config.json"))
    taps, group = int(cfg["conv_kernel_size"]), int(cfg["conv_group_size"])
    hidden = int(cfg["transformer_layer_config"]["hidden_size"])
    num_groups = hidden // group
    torch.manual_seed(0)
    worst = 0.0
    failed = False
    with safe_open(f"{args.ckpt}/model.safetensors", "pt") as f:
        layers = sorted({k.split(".")[1] for k in f.keys() if k.startswith("layers.")}, key=int)
        for li in layers:
            for conv in ("attention_conv", "mlp_conv"):
                base = f.get_tensor(f"layers.{li}.{conv}.base_kernel").float()       # [2, taps, hidden]
                wproj = f.get_tensor(f"layers.{li}.{conv}.kernel_projection.weight").float()
                for block in (8, 5, 3):
                    rows = (args.rows // block) * block
                    x = torch.randn(rows, hidden) * 0.5
                    kern = (x @ wproj.T).view(rows, 2, taps, num_groups)
                    for side in (0, 1):
                        delta = kern[:, side]                                          # [rows, taps, groups]
                        ref_spec = grouped_dynamic_conv(
                            x, delta, base[side], block_size=block, group_size=group
                        )
                        ref_vllm = _grouped_conv(x, delta, base[side], block, num_groups, group, taps)
                        err = (ref_spec - ref_vllm).abs().max().item()
                        worst = max(worst, err)
                        if err > 1e-4:
                            print(f"FAIL layer {li} {conv} side {side} block {block}: cpu ref vs speculators {err:.3e}")
                            failed = True
                        if args.device == "cuda":
                            dev = torch.device("cuda")
                            xb, db, bb = (t.to(dev, torch.bfloat16).contiguous() for t in (x, delta, base[side]))
                            out = dflash2_grouped_conv(xb, db, bb, block, group).float().cpu()
                            # bf16 inputs on the kernel side vs fp32 reference: compare at bf16 resolution
                            ref_bf = grouped_dynamic_conv(
                                xb.float().cpu(), db.float().cpu(), bb.float().cpu(), block_size=block, group_size=group
                            )
                            rel = ((out - ref_bf).abs() / (ref_bf.abs() + 1e-2)).max().item()
                            if rel > 2e-2:
                                print(f"FAIL layer {li} {conv} side {side} block {block}: triton vs ref rel {rel:.3e}")
                                failed = True
            print(f"layer {li}: ok (worst cpu-ref abs err so far {worst:.2e})")
    # Block-boundary check: changing row 0 of block 2 must not affect any row of block 1.
    x = torch.randn(16, hidden); delta = torch.zeros(16, taps, num_groups); base0 = torch.zeros(taps, hidden); base0[:] = 1.0
    y1 = _grouped_conv(x, delta, base0, 8, num_groups, group, taps)
    x2 = x.clone(); x2[8] += 10.0
    y2 = _grouped_conv(x2, delta, base0, 8, num_groups, group, taps)
    leak = (y1[:8] - y2[:8]).abs().max().item()
    if leak > 0:
        print(f"FAIL: block boundary leak {leak}"); failed = True
    else:
        print("block boundary: no leak across blocks")
    print("PARITY " + ("FAILED" if failed else "PASSED"), f"(device={args.device}, taps={taps}, group={group})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
