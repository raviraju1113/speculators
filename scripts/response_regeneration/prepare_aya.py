#!/usr/bin/env python3
"""Convert downloaded Aya parquet shards into conversation JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Aya parquet shards into conversation JSONL."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/import/ml-sc-scratch5/chenw/datasets/aya_dataset/data"),
        help="Directory containing parquet shards (default: %(default)s)",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional single parquet file to convert instead of a directory",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(
            "/import/ml-sc-scratch5/chenw/datasets/aya_dataset/aya_conversations.jsonl"
        ),
        help="Path to the output JSONL file (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to convert",
    )
    return parser.parse_args()


def iter_parquet_rows(input_path: Path) -> Iterator[dict]:
    if input_path.is_dir():
        parquet_files = sorted(input_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {input_path}")
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            for row in df.to_dict(orient="records"):
                yield row
    else:
        df = pd.read_parquet(input_path)
        for row in df.to_dict(orient="records"):
            yield row


def row_to_conversation(row: dict) -> dict | None:
    prompt = row.get("inputs")
    answer = row.get("targets")
    if not isinstance(prompt, str) or not isinstance(answer, str):
        return None
    if not prompt.strip() or not answer.strip():
        return None

    record_id = row.get("user_id")
    if not isinstance(record_id, str) or not record_id.strip():
        payload = f"{prompt}\n{answer}"
        record_id = f"aya_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"
    else:
        record_id = f"aya_{record_id}"

    metadata = {
        "language": row.get("language"),
        "language_code": row.get("language_code"),
        "annotation_type": row.get("annotation_type"),
        "user_id": row.get("user_id"),
    }

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ],
        "source": "Aya",
        "id": record_id,
        **metadata,
    }


def convert_dataset(input_path: Path, output_jsonl: Path, limit: int | None = None) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(iter_parquet_rows(input_path)):
            if limit is not None and idx >= limit:
                break
            convo = row_to_conversation(row)
            if convo is None:
                skipped += 1
                continue
            handle.write(json.dumps(convo, ensure_ascii=False) + "\n")
            written += 1

    if skipped:
        print(f"Skipped {skipped} rows with missing or blank input/output values")
    return written


def main() -> None:
    args = parse_args()
    input_path = args.input_file or args.input_dir
    written = convert_dataset(input_path, args.output_jsonl, limit=args.limit)
    print(f"Wrote {written} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
