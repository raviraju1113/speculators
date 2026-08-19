#!/usr/bin/env python3
"""Merge multiple conversation JSONL files, dropping duplicate prompts.

The prompt key is the first `human` turn of each record's `conversations` list.
The first occurrence of a prompt wins, so pass files in priority order.

Usage:
    python scripts/merge_dedup_jsonl.py a.jsonl b.jsonl -o merged.jsonl
"""

import argparse
import hashlib
import json
import sys


def prompt_key(record, normalize=True):
    """Return a hash of the first human turn, or None if there isn't one."""
    for turn in record.get("conversations", []):
        if turn.get("from") == "human":
            text = turn.get("value", "")
            if normalize:
                text = " ".join(text.split()).lower()
            return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="input JSONL paths, in priority order")
    parser.add_argument("-o", "--output", required=True, help="output JSONL path")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="match prompts byte-for-byte instead of normalizing whitespace and case",
    )
    parser.add_argument(
        "--keep-no-prompt",
        action="store_true",
        help="keep records with no human turn instead of dropping them",
    )
    parser.add_argument(
        "--keep-fields",
        nargs="+",
        metavar="FIELD",
        help=(
            "emit only these top-level fields. Use this when inputs carry "
            "differently-shaped metadata, which makes HF datasets fail to infer "
            "one Arrow schema (e.g. --keep-fields id conversations)"
        ),
    )
    args = parser.parse_args()

    seen = set()
    written = 0
    with open(args.output, "w") as out:
        for path in args.inputs:
            total = kept = dupes = skipped = bad = 0
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        bad += 1
                        continue

                    key = prompt_key(record, normalize=not args.exact)
                    if key is None:
                        if not args.keep_no_prompt:
                            skipped += 1
                            continue
                    elif key in seen:
                        dupes += 1
                        continue
                    else:
                        seen.add(key)

                    if args.keep_fields:
                        record = {k: record[k] for k in args.keep_fields if k in record}

                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1
                    written += 1

            print(
                f"{path}: read {total}, kept {kept}, dropped {dupes} duplicates"
                + (f", {skipped} without a human turn" if skipped else "")
                + (f", {bad} unparseable" if bad else ""),
                file=sys.stderr,
            )

    print(f"wrote {written} records to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
