#!/usr/bin/env python3
"""Carve the on- vs off-policy ablation splits out of kimi_nemotron_782k.

Writes, into --out-dir:

  off_policy.jsonl   train prompts WITH the corpus's original (non-target) responses,
                     ShareGPT {from,value} -- the off-policy arm, and also the input
                     to scripts/response_regeneration (which drops the assistant turns
                     and regenerates them with the target -> the on-policy arm).
  heldout_eval.jsonl held-out prompts in mtp_server_eval's {benchmark,id,prompt}
                     format (benchmark "kn782k_heldout"), for in-distribution
                     acceptance -- never seen by either arm.
  split_stats.json   counts and filter reasons, for the write-up.

Only single-turn rows (optional system + one user + one assistant) are kept: for
multi-turn rows the later user turns were written against the ORIGINAL replies, so
regenerating the replies would make the on-policy conversations incoherent -- a
confound unrelated to the policy question. Rows that hit the source's 8192-token
truncation are dropped so neither arm trains on cut-off responses.

    python export_prompts.py --num-train 100000 --num-heldout 1000 --out-dir <dir>
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

SRC = "/import/ml-sc-scratch1/mengmengj/datasets/kimi_nemotron_782k"
_TO_FROM = {"user": "human", "assistant": "gpt", "system": "system"}


def single_turn(messages) -> tuple[list[dict] | None, str]:
    roles = [m["role"] for m in messages]
    if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
        return None, "multi_turn_or_odd_roles"
    if any(not m["content"].strip() for m in messages if m["role"] != "system"):
        return None, "empty_turn"
    return [{"from": _TO_FROM[m["role"]], "value": m["content"]} for m in messages], ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=SRC)
    p.add_argument("--num-train", type=int, required=True)
    p.add_argument("--num-heldout", type=int, default=1000)
    p.add_argument("--max-seq-len", type=int, default=8192,
                   help="drop rows whose source seq_len reached this (truncated)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    ds = load_from_disk(args.src).select_columns(["messages", "seq_len"])
    ds = ds.with_format(None)
    order = list(range(len(ds)))
    random.Random(args.seed).shuffle(order)

    need = args.num_train + args.num_heldout
    kept: list[tuple[int, list[dict]]] = []
    reasons: Counter = Counter()
    for i in order:
        row = ds[i]
        if row["seq_len"] >= args.max_seq_len:
            reasons["truncated_at_source"] += 1
            continue
        conv, why = single_turn(row["messages"])
        if conv is None:
            reasons[why] += 1
            continue
        kept.append((i, conv))
        if len(kept) == need:
            break
    if len(kept) < need:
        raise SystemExit(f"only {len(kept)} usable rows, need {need}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    heldout, train = kept[: args.num_heldout], kept[args.num_heldout :]
    with open(out / "off_policy.jsonl", "w") as f:
        for i, conv in train:
            f.write(json.dumps({"id": f"kn782k_{i}", "conversations": conv},
                               ensure_ascii=False) + "\n")
    with open(out / "heldout_eval.jsonl", "w") as f:
        for i, conv in heldout:
            prompt = next(t["value"] for t in conv if t["from"] == "human")
            f.write(json.dumps({"benchmark": "kn782k_heldout", "id": f"kn782k_{i}",
                                "prompt": prompt}, ensure_ascii=False) + "\n")
    stats = {"src": args.src, "seed": args.seed, "scanned": sum(reasons.values()) + need,
             "num_train": len(train), "num_heldout": len(heldout),
             "dropped": dict(reasons)}
    (out / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
