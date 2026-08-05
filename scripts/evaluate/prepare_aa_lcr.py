#!/usr/bin/env python3
"""Build AA-LCR long-context prompts for speculative-decoding acceptance eval.

Source: https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR
Used by Inferact/Kimi-K3-DSpark (~95k-token multi-document prompts, 100 rows).

Writes ``{"turns": [<prompt>]}`` JSONL (eval_datasets format). Defaults to the
scratch tree (not the git checkout)::

    /import/ml-sc-scratch5/chenw/datasets/eval/turns/aa-lcr.jsonl

Usage::

    python scripts/evaluate/prepare_aa_lcr.py
    python scripts/evaluate/prepare_aa_lcr.py \\
        --output-path /import/ml-sc-scratch5/chenw/datasets/eval/turns/aa-lcr.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path


REPO_ID = "ArtificialAnalysis/AA-LCR"
CSV_NAME = "AA-LCR_Dataset.csv"
ZIP_NAME = "extracted_text/AA-LCR_extracted-text.zip"

DEFAULT_OUTPUT = Path(
    "/import/ml-sc-scratch5/chenw/datasets/eval/turns/aa-lcr.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output turns JSONL (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap (useful for smoke tests).",
    )
    return parser.parse_args()


def _download() -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    csv_path = Path(
        hf_hub_download(REPO_ID, filename=CSV_NAME, repo_type="dataset")
    )
    zip_path = Path(
        hf_hub_download(REPO_ID, filename=ZIP_NAME, repo_type="dataset")
    )
    return csv_path, zip_path


def _load_documents(zf: zipfile.ZipFile, category: str, set_id: str) -> list[str]:
    """Load document texts for one AA-LCR document set."""
    prefix = f"{category}/{set_id}/"
    # Zip members may or may not include a top-level folder.
    names = [
        n
        for n in zf.namelist()
        if n.endswith(".txt") and (prefix in n or n.startswith(prefix))
    ]
    if not names:
        # Fallback: any path ending with /{category}/{set_id}/file.txt
        names = [
            n
            for n in zf.namelist()
            if n.endswith(".txt") and f"/{category}/{set_id}/" in f"/{n}"
        ]
    names = sorted(names)
    docs: list[str] = []
    for name in names:
        with zf.open(name) as handle:
            text = handle.read().decode("utf-8", errors="replace").strip()
        if text:
            docs.append(text)
    return docs


def build_prompt(question: str, documents: list[str]) -> str:
    parts = [
        "You are given multiple documents. Read them carefully and answer the "
        "question. Reason across documents when needed; do not rely on a single "
        "span that merely restates the question.",
        "",
    ]
    for i, doc in enumerate(documents, start=1):
        parts.append(f"=== Document {i} ===")
        parts.append(doc)
        parts.append("")
    parts.append("=== Question ===")
    parts.append(question.strip())
    parts.append("")
    parts.append("=== Answer ===")
    return "\n".join(parts)


def convert(output_path: Path, max_rows: int | None) -> int:
    csv_path, zip_path = _download()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with (
        zip_path.open("rb") as raw_zip,
        zipfile.ZipFile(raw_zip) as zf,
        csv_path.open(encoding="utf-8") as csv_f,
        output_path.open("w", encoding="utf-8") as out,
    ):
        reader = csv.DictReader(csv_f)
        for row in reader:
            if max_rows is not None and written >= max_rows:
                break
            question = (row.get("question") or "").strip()
            category = (row.get("document_category") or row.get("category") or "").strip()
            set_id = (
                row.get("document_set_id")
                or row.get("set_id")
                or row.get("document_set")
                or ""
            ).strip()
            if not question or not category or not set_id:
                continue
            docs = _load_documents(zf, category, set_id)
            if not docs:
                print(f"warning: no documents for {category}/{set_id}; skipping")
                continue
            prompt = build_prompt(question, docs)
            out.write(
                json.dumps(
                    {
                        "turns": [prompt],
                        "id": row.get("question_id") or row.get("id") or str(written),
                        "source": "ArtificialAnalysis/AA-LCR",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return written


def main() -> None:
    args = parse_args()
    n = convert(args.output_path, args.max_rows)
    print(f"wrote {n} AA-LCR prompts -> {args.output_path}")


if __name__ == "__main__":
    main()
