#!/usr/bin/env python
"""Dump the gemma-4-31B-it regenerated nemo782k corpus as GLM-ready JSONL.

The off-policy stand-in while the GLM-5.3 regeneration (run_regen_glm53.sh)
runs: the same 782,413 prompts, but every final assistant turn was written by
gemma-4-31B-it, not by GLM-5.3. The arrow shards hold Gemma token ids, so the
draft dataset has to be re-tokenized from the ``messages`` column with the GLM
chat template (prepare_offpolicy_gemma.sh does that).

Rows are written as {"id", "conversations": [{"role", "content"}, ...]}, the
schema scripts/prepare_data.py reads. Rows whose last turn is not a non-empty
assistant reply are dropped (they would train on nothing under the last-turn
loss mask).

    python examples/train/glm53/build_offpolicy_gemma_jsonl.py \
        --dataset /import/ml-sc-scratch1/mengmengj/datasets/kimi_nemotron_782k \
        --output  data/glm53/gemma782k_offpolicy.jsonl
"""

import argparse
import json

from datasets import load_from_disk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, help="gemma arrow dataset dir")
    ap.add_argument("--output", required=True, help="output JSONL path")
    ap.add_argument("--tag", default="nemo782k", help="id prefix")
    args = ap.parse_args()

    ds = load_from_disk(args.dataset).select_columns(["messages"])
    kept = dropped = multi = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for i, row in enumerate(ds):
            conv = [{"role": m["role"], "content": m["content"] or ""} for m in row["messages"]]
            if not conv or conv[-1]["role"] != "assistant" or not conv[-1]["content"].strip():
                dropped += 1
                continue
            multi += sum(t["role"] == "assistant" for t in conv) > 1
            out.write(json.dumps({"id": f"{args.tag}:{i}", "conversations": conv}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"[build] kept {kept} ({multi} multi-turn), dropped {dropped} -> {args.output}")


if __name__ == "__main__":
    main()
