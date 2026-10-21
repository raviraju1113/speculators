#!/usr/bin/env python3
"""Make the two arms' training files cover exactly the same prompts.

Drops a prompt from BOTH arms when its on-policy regeneration errored or hit the
max_tokens cap (a capped Llama-3.1-8B reply at T=0.6 is almost always a
repetition loop -- 7/64 in the smoke test, mostly math). Dropping from both keeps
the prompt set identical, so response text stays the only variable.

    python finalize_arms.py --data-dir <abl>/data --max-tokens 6144
    # -> off_policy.final.jsonl, on_policy.final.jsonl, finalize_stats.json
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--max-tokens", type=int, default=6144)
    args = p.parse_args()
    d = Path(args.data_dir)

    on: dict[str, dict] = {}
    capped = 0
    for line in open(d / "on_policy.jsonl"):
        row = json.loads(line)
        if (row.get("metadata") or {}).get("error") is not None:
            continue
        usage = row["metadata"].get("usage") or {}
        if usage.get("completion_tokens", 0) >= args.max_tokens:
            capped += 1
            continue
        on[row["id"]] = row

    n_off = kept = 0
    with open(d / "off_policy.final.jsonl", "w") as f_off, \
         open(d / "on_policy.final.jsonl", "w") as f_on:
        for line in open(d / "off_policy.jsonl"):  # off-policy order = canonical order
            n_off += 1
            row = json.loads(line)
            if row["id"] not in on:
                continue
            kept += 1
            f_off.write(line)
            f_on.write(json.dumps({"id": row["id"],
                                   "conversations": on[row["id"]]["conversations"]},
                                  ensure_ascii=False) + "\n")

    stats = {"prompts": n_off, "regenerated_ok": len(on) + capped,
             "dropped_capped": capped, "dropped_missing_or_error": n_off - len(on) - capped,
             "kept_both_arms": kept}
    (d / "finalize_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
