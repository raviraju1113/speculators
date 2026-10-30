#!/usr/bin/env python
"""Gate for the TP8 x PP2 verifier: fetch hidden states for a few dataset rows
through vLLM + Mooncake and check every captured layer arrived intact.

train.py only warns and drops a sample when a fetch fails, so a broken PP
path (aux layers lost on stage 0) would otherwise look like a healthy run.

Checks per row: token ids round-trip, hidden_states is [T, n_layers * H] with
n_layers = len(target_layer_ids) + 1 (launch_vllm appends the last layer), no
NaN/inf, and no all-zero layer block (an unfilled receive buffer). Per-layer
RMS should grow with depth.

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--vllm-endpoint", required=True)
    ap.add_argument("--mooncake-master", required=True)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--hidden-size", type=int, default=6144)
    ap.add_argument("--rows", type=int, nargs="+", default=[0, 1, 2, 160])
    args = ap.parse_args()

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
        ok &= good
        print(f"row {row}: T={T} hidden_states {tuple(h.shape)} {h.dtype} want {want} "
              f"finite={finite} zero_layers={zero} rms/layer={[round(r, 3) for r in rms]} "
              f"-> {'OK' if good else 'FAIL'}", flush=True)
        transfer.delete(handle)
    print("PROBE", "PASSED" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
