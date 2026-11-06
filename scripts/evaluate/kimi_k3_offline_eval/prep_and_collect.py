#!/usr/bin/env python3
"""Phase 1 of the Kimi-K3 offline draft eval: tokenize + collect hidden states.

Tokenizes eval conversations with TorchSpec's KimiK3Parser (the exact
tokenization + loss-mask logic the TorchSpec EAGLE3 draft was trained/evaled
with), sends each sample's token ids to a running `extract_hidden_states` vLLM
server (scripts/launch_vllm.py), and writes one consolidated safetensors file
per sample:

    input_ids           [S]        int64
    loss_mask           [S]        int64
    hidden_states       [S, 3*H]   bf16   (aux layers concat, e.g. 48/68/88)
    last_hidden_states  [S, H]     bf16   (final layer, pre-final-norm)

Run (kimi_k3 env, server already up):
    PYTHONPATH=src:hs_connectors/src:<TorchSpec> python \
        scripts/evaluate/kimi_k3_offline_eval/prep_and_collect.py \
        --eval-jsonl .../eval_conversations.jsonl \
        --model /import/ml-sc-scratch5/chenw/models/Kimi-K3-patched \
        --base-url http://127.0.0.1:8082/v1 \
        --output .../eval_hs_samples
"""

import argparse
import fcntl
import json
import time
from pathlib import Path

import openai
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from speculators.data_generation.vllm_client import generate_hidden_states


def load_hs_file(handle: str, timeout_s: float = 120.0) -> dict[str, torch.Tensor]:
    """Load a connector-written hs file, honoring its async-write lock."""
    lock_path = handle + ".lock"
    deadline = time.monotonic() + timeout_s
    while not (Path(lock_path).exists() or Path(handle).exists()):
        if time.monotonic() > deadline:
            raise TimeoutError(f"hidden-states file never appeared: {handle}")
        time.sleep(0.2)
    if Path(lock_path).exists():
        with open(lock_path) as lf:  # writer holds LOCK_EX until file is complete
            fcntl.flock(lf, fcntl.LOCK_SH)
    return load_file(handle)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--model", required=True, help="Kimi-K3 model dir (tokenizer)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8082/v1")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--chat-template", default="kimi-k3")
    return ap.parse_args()


def main():
    args = parse_args()
    from torchspec.data.parse import create_parser
    from torchspec.data.template import TEMPLATE_REGISTRY

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    parser = create_parser(tokenizer, TEMPLATE_REGISTRY.get(args.chat_template))

    client = openai.Client(base_url=args.base_url, api_key="none")
    model_id = client.models.list().data[0].id

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    with Path(args.eval_jsonl).open() as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            conversation = row.get("conversations") or row.get("messages")
            input_ids, loss_mask = parser.parse(conversation, args.max_length)
            input_ids = input_ids.squeeze(0) if input_ids.dim() > 1 else input_ids
            loss_mask = loss_mask.squeeze(0) if loss_mask.dim() > 1 else loss_mask

            out_path = out_dir / f"sample_{i}.safetensors"
            if out_path.exists():
                print(f"[{i}] exists, skipping")
                n_ok += 1
                continue

            handle = generate_hidden_states(
                client,
                model_id,
                {"input_ids": input_ids.tolist()},
            )
            data = load_hs_file(handle)
            # "hidden_states" is [S, num_requested_layers, H]; requested order is
            # target_layer_ids (+ appended last layer), so index -1 on dim 1 is
            # the final layer.
            hs = data["hidden_states"]
            if hs.shape[0] != input_ids.shape[0]:
                raise ValueError(
                    f"sample {i}: hs seq {hs.shape[0]} != input len {input_ids.shape[0]}"
                )
            if not torch.equal(data["token_ids"], input_ids.to(torch.int64)):
                raise ValueError(f"sample {i}: token_ids mismatch with input_ids")
            seq_len, num_layers, hidden = hs.shape
            save_file(
                {
                    "input_ids": input_ids.to(torch.int64),
                    "loss_mask": loss_mask.to(torch.int64),
                    # Full stack [S, L, H] in requested-id order (last = final
                    # layer). The eval selects which columns form the aux concat,
                    # so one collection can test several capture-offset combos.
                    "hs_stack": hs.contiguous(),
                    "last_hidden_states": hs[:, -1, :].contiguous(),
                },
                out_path,
            )
            Path(handle).unlink(missing_ok=True)
            n_ok += 1
            print(f"[{i}] ok: seq={input_ids.shape[0]}, "
                  f"masked={int(loss_mask.sum())}, aux={tuple(hs.shape)}")

    print(f"done: {n_ok} samples in {out_dir}")


if __name__ == "__main__":
    main()
