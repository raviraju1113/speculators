#!/usr/bin/env python
"""Gate for the TP8 x PP2 verifier: fetch hidden states for a few dataset rows
through vLLM + Mooncake and check every captured layer arrived intact.

train.py only warns and drops a sample when a fetch fails, so a broken PP
path (aux layers lost on stage 0) would otherwise look like a healthy run.

Checks per row: token ids round-trip, hidden_states is [T, n_layers * H] with
n_layers = len(target_layer_ids) + 1 (launch_vllm appends the last layer), no
NaN/inf, and no all-zero layer block (an unfilled receive buffer). Per-layer
RMS should grow with depth.

With --verifier (real weights only), also decode the captured last layer
(pre-norm) through the verifier's final RMSNorm + lm_head and require its
top-1 to match the next token on at least --min-agree of the loss-masked
positions: misplaced cache slots or misordered PP stages drop this to ~0.

    python examples/train/glm53/probe_hidden_states.py --data-path DS \
        --vllm-endpoint http://HEAD:8000/v1 --mooncake-master TRAIN:50051 \
        --num-layers 6 --hidden-size 6144
"""

import argparse
import sys

import openai
import torch
from datasets import load_from_disk

from hs_connectors import MooncakeBackend
from speculators.data_generation.offline import check_hidden_states
from speculators.data_generation.vllm_client import generate_hidden_states
from speculators.train.data import build_client_item


def _load_head(verifier: str) -> dict:
    """Final RMSNorm weight + lm_head of the verifier, read straight from its shards."""
    import json
    from pathlib import Path

    from safetensors import safe_open

    cfg = json.loads((Path(verifier) / "config.json").read_text())
    index = json.loads((Path(verifier) / "model.safetensors.index.json").read_text())["weight_map"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"eps": cfg.get("rms_norm_eps", 1e-5)}
    for key, name in (("norm", "model.norm.weight"), ("lm_head", "lm_head.weight")):
        with safe_open(str(Path(verifier) / index[name]), framework="pt") as f:
            out[key] = f.get_tensor(name).to(dev, torch.float32)
    return out


def _top1_agreement(head: dict, h_last: torch.Tensor, input_ids, loss_mask) -> float:
    """Fraction of loss-masked positions whose verifier top-1 is the next token."""
    h = h_last.to(head["norm"].device, torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + head["eps"]) * head["norm"]
    pred = (h @ head["lm_head"].T).argmax(-1)
    ids = torch.as_tensor(input_ids, device=pred.device)
    mask = torch.as_tensor(loss_mask, device=pred.device).bool()
    # position t predicts token t+1; score positions whose next token is trained on
    sel = mask[1:]
    return (pred[:-1][sel] == ids[1:][sel]).float().mean().item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--vllm-endpoint", required=True)
    ap.add_argument("--mooncake-master", required=True)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--hidden-size", type=int, default=6144)
    ap.add_argument("--rows", type=int, nargs="+", default=[0, 1, 2, 160])
    ap.add_argument("--verifier", default=None, help="checkpoint dir for the lm_head agreement check")
    ap.add_argument("--min-agree", type=float, default=0.3)
    args = ap.parse_args()
    head = _load_head(args.verifier) if args.verifier else None

    ns = argparse.Namespace(
        mooncake_master=args.mooncake_master,
        mooncake_metadata_server="P2PHANDSHAKE",
        mooncake_protocol="tcp",
    )
    transfer = MooncakeBackend.from_train_args(ns, args.data_path)
    transfer.setup()
    client = openai.OpenAI(base_url=args.vllm_endpoint, api_key="EMPTY", max_retries=0)
    model = client.models.list().data[0].id
    ds = load_from_disk(args.data_path)

    ok = True
    for row in args.rows:
        item = ds[row]
        handle = generate_hidden_states(client, model, build_client_item(item), timeout=600)
        hs = transfer.get_generated(handle)
        check_hidden_states(hs, item["input_ids"].tolist())
        h = hs["hidden_states"]
        T = len(item["input_ids"])
        want = (T, args.num_layers * args.hidden_size)
        blocks = h.float().reshape(h.shape[0], -1, args.hidden_size) if h.dim() == 2 else h.float()
        rms = blocks.pow(2).mean(dim=(0, 2)).sqrt().tolist()
        zero = [i for i, r in enumerate(rms) if r == 0.0]
        finite = bool(torch.isfinite(h.float()).all())
        good = tuple(h.reshape(h.shape[0], -1).shape) == want and not zero and finite
        agree_msg = ""
        if head is not None and good:
            agree = _top1_agreement(head, blocks[:, -1], item["input_ids"], item["loss_mask"])
            agree_msg = f" lm_head top-1 agreement={agree:.3f}"
            good &= agree >= args.min_agree
        ok &= good
        print(f"row {row}: T={T} hidden_states {tuple(h.shape)} {h.dtype} want {want} "
              f"finite={finite} zero_layers={zero} rms/layer={[round(r, 3) for r in rms]}"
              f"{agree_msg} -> {'OK' if good else 'FAIL'}", flush=True)
        transfer.delete(handle)
    print("PROBE", "PASSED" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
