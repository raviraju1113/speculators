#!/usr/bin/env python3
"""Dump the prompts of a prepared speculators dataset into a regeneration pool.

The arrow shards written by ``scripts/prepare_data.py`` keep a plain-text
``messages`` column next to the tokenized ids. That column is all a new target
needs: this script keeps the system/user turns of every row, drops the old
assistant answers (they belong to the previous target), and writes one JSONL
row per conversation in the ShareGPT layout the regeneration client reads
(``scripts/response_regeneration/script_multiendpoint.py --input-jsonl``).

Every row gets a stable ``id`` (``<tag>:<shard>:<row>``) so the client's
``--resume`` skips rows that are already regenerated, and
``scripts/merge_dedup_jsonl.py`` can merge several pools by prompt hash when
more data is added later.

Usage:
    python examples/train/glm53/extract_prompts.py \
        --dataset /import/ml-sc-scratch1/mengmengj/datasets/kimi_nemotron_782k \
        --tag nemo782k --output data/glm53/prompts_nemo782k.jsonl
"""

import argparse
import glob
import json
import os
import sys

import pyarrow as pa
import pyarrow.ipc as ipc

_TO_FROM = {"system": "system", "user": "human"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="prepare_data.py output dir (data-*.arrow shards)")
    p.add_argument("--tag", required=True, help="id prefix, e.g. nemo782k")
    p.add_argument("--output", required=True, help="output JSONL path")
    p.add_argument("--limit", type=int, default=None, help="stop after N rows (smoke tests)")
    p.add_argument(
        "--single-turn-only",
        action="store_true",
        help="keep only the first user turn of each row (drop multi-turn context)",
    )
    return p.parse_args()


def read_shard(path: str) -> pa.Table:
    with pa.memory_map(path) as src:
        try:
            return ipc.open_stream(src).read_all()
        except pa.ArrowInvalid:
            return ipc.open_file(src).read_all()


def main() -> int:
    args = parse_args()
    shards = sorted(glob.glob(os.path.join(args.dataset, "data-*.arrow")))
    if not shards:
        print(f"no data-*.arrow shards under {args.dataset}", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    written = skipped = 0
    turn_hist: dict[int, int] = {}
    with open(args.output, "w", encoding="utf-8") as out:
        for shard_idx, shard in enumerate(shards):
            table = read_shard(shard)
            if "messages" not in table.column_names:
                print(f"{shard}: no 'messages' column", file=sys.stderr)
                return 1
            for row_idx, messages in enumerate(table.column("messages").to_pylist()):
                if args.limit is not None and written >= args.limit:
                    break
                conv = []
                for m in messages or []:
                    role, content = m.get("role"), m.get("content")
                    if role not in _TO_FROM:
                        continue  # old assistant answers are regenerated
                    if not isinstance(content, str) or not content.strip():
                        continue  # empty system turns would reject the whole row
                    conv.append({"from": _TO_FROM[role], "value": content})
                    if args.single_turn_only and role == "user":
                        break
                if not any(t["from"] == "human" for t in conv):
                    skipped += 1
                    continue
                n_user = sum(t["from"] == "human" for t in conv)
                turn_hist[n_user] = turn_hist.get(n_user, 0) + 1
                out.write(
                    json.dumps(
                        {
                            "id": f"{args.tag}:{shard_idx:05d}:{row_idx}",
                            "source": args.tag,
                            "conversations": conv,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
            if args.limit is not None and written >= args.limit:
                break

    print(f"wrote {written} rows to {args.output} (skipped {skipped} without a user turn)")
    print("user turns per row:", dict(sorted(turn_hist.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
