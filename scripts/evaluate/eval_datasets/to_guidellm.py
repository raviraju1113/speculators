#!/usr/bin/env python3
"""Convert DeepSpec-style ``{"turns": [...]}`` eval datasets to GuideLLM prompts.

``run_eval.sh MODE=throughput`` drives GuideLLM, which reads a single text
column (``prompt`` by default) via ``--data-args '{"data_files": "<subset>.jsonl"}'``.
The benchmark files in this directory instead store a ``turns`` list (one entry
per user turn), so this utility emits GuideLLM-ready ``{"prompt": ...}`` JSONL,
using the first user turn as the prompt (the standard single-prompt
acceptance-rate setup).

Usage:
    # Convert every *.jsonl in this dir into ./guidellm/<name>.jsonl
    python scripts/evaluate/eval_datasets/to_guidellm.py

    # Then benchmark against the converted prompts:
    MODE=throughput BASE_URL=http://localhost:8000 \\
      DATASET=scripts/evaluate/eval_datasets/guidellm \\
      SUBSETS=gsm8k,humaneval,math500 \\
      ./scripts/evaluate/mtp_server_eval/run_eval.sh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent


def turns_to_prompt(row: dict) -> str | None:
    """Return the first user turn as the GuideLLM prompt, or None to skip."""
    turns = row.get("turns")
    if isinstance(turns, list) and turns and isinstance(turns[0], str):
        return turns[0]
    # Some rows may already carry a prompt column.
    prompt = row.get("prompt")
    return prompt if isinstance(prompt, str) and prompt else None


def convert_file(src: Path, dst: Path) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            prompt = turns_to_prompt(json.loads(stripped))
            if prompt is None:
                skipped += 1
                continue
            fout.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
            written += 1
    return written, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_ROOT,
        help="Directory of DeepSpec-style turns JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "guidellm",
        help="Directory to write GuideLLM prompt JSONL files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Dataset stems to convert (default: every *.jsonl in --input-dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.datasets:
        sources = [args.input_dir / f"{name}.jsonl" for name in args.datasets]
    else:
        sources = sorted(args.input_dir.glob("*.jsonl"))

    for src in sources:
        if not src.exists():
            print(f"skip missing: {src}")
            continue
        dst = args.output_dir / src.name
        written, skipped = convert_file(src, dst)
        print(f"{src.name}: wrote {written} prompts (skipped {skipped}) -> {dst}")


if __name__ == "__main__":
    main()
