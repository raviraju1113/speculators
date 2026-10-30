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


def _load_test(args, head) -> int:
    """Fetch many rows concurrently, as training does, and report per-row health.

    For every row: NaN layers / token span if any, else lm_head agreement. A
    load-dependent bug shows up as NaN rows or as a low-agreement tail that
    the same rows do not have when re-fetched one at a time (--rows).
    """
    import random
    import threading
    from concurrent.futures import ThreadPoolExecutor

    ds = load_from_disk(args.data_path)
    rows = random.Random(0).sample(range(len(ds)), args.random_rows)
    ns = argparse.Namespace(mooncake_master=args.mooncake_master,
                            mooncake_metadata_server="P2PHANDSHAKE", mooncake_protocol="tcp")
    local = threading.local()
    head_lock = threading.Lock()

    def fetch(row):
        if not hasattr(local, "client"):
            local.transfer = MooncakeBackend.from_train_args(ns, args.data_path)
            local.transfer.setup()
            local.client = openai.OpenAI(base_url=args.vllm_endpoint, api_key="EMPTY", max_retries=0)
            local.model = local.client.models.list().data[0].id
        item = ds[row]
        handle = generate_hidden_states(local.client, local.model, build_client_item(item), timeout=1800)
        hs = local.transfer.get_generated(handle)
        local.transfer.delete(handle)
        h = hs["hidden_states"].float().reshape(-1, args.num_layers, args.hidden_size)
        bad = ~torch.isfinite(h)
        if bad.any():
            tok = bad.any(dim=(1, 2)).nonzero().flatten()
            return row, len(h), "NAN", (bad.any(dim=(0, 2)).nonzero().flatten().tolist(),
                                        int(tok.min()), int(tok.max()), int(tok.numel()))
        if head is None:
            return row, len(h), "OK", None
        with head_lock:
            agree = _top1_agreement(head, h[:, -1], item["input_ids"], item["loss_mask"])
        return row, len(h), "OK", agree

    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(fetch, rows))
    nan = [r for r in results if r[2] == "NAN"]
    for row, T, _, (layers, t0, t1, n) in nan:
        print(f"NaN row {row}: T={T} layers={layers} tokens {t0}..{t1} ({n} tokens)", flush=True)
    agrees = sorted(r[3] for r in results if r[2] == "OK" and r[3] is not None)
    if agrees:
        q = lambda p: agrees[min(len(agrees) - 1, int(p * len(agrees)))]  # noqa: E731
        low = [r for r in results if r[2] == "OK" and r[3] is not None and r[3] < args.min_agree]
        print(f"agreement over {len(agrees)} rows: p1={q(0.01):.3f} p5={q(0.05):.3f} "
              f"p50={q(0.5):.3f}; rows below {args.min_agree}: "
              f"{[(r[0], r[1], round(r[3], 3)) for r in low][:20]}", flush=True)
    print(f"LOAD TEST rows={len(results)} nan={len(nan)} "
          f"low_agree={sum(1 for a in agrees if a < args.min_agree)}", flush=True)
    return 0 if not nan else 1


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
    ap.add_argument("--concurrency", type=int, default=0,
                    help="load test: fetch --random-rows rows with this many requests in flight")
    ap.add_argument("--random-rows", type=int, default=512)
    args = ap.parse_args()
    head = _load_head(args.verifier) if args.verifier else None
    if args.concurrency:
        return _load_test(args, head)

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
