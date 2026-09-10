#!/usr/bin/env python3
"""Build a context-length sweep from AA-LCR for speculative-decoding acceptance eval.

Motivation
----------
The existing eval corpus can't answer "how does acceptance change with context
length": it is dense below 2k tokens, dense at 16k+, and nearly empty between
(44 prompts across 2k-16k). The one long-context set on disk,
``speed-low-entropy``, is repetitive code boilerplate, so comparing it against
the short qualitative slices confounds prompt *entropy* with prompt *length*.

This script isolates length. Source: https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR
(100 multi-document QA rows, all >=71k tokens). For each target length we keep
the instruction header and the question *fixed* and truncate only the document
block, so every bin holds the same 100 questions at a different context length —
a paired design. Acceptance differences across bins are then attributable to
length rather than to domain, task, or entropy.

Writes one ``{"benchmark", "id", "prompt"}`` JSONL per bin into
``mtp_server_eval/data/aa-lcr-<bin>.jsonl``, matching the format the mtp runners
read. Each bin is a separate benchmark name, so the existing per-benchmark
counter scraping in ``run_vllm_eval.py`` reports acceptance per length with no
harness change.

Bins above the source length
----------------------------
The longest AA-LCR document set is ~123k tokens, so a 128k bin is unreachable
by truncation. With ``--allow-short`` such a row emits its *full* untruncated
document set instead of being skipped, so the bin holds the longest prompts the
dataset can supply (~89k-123k tokens here). Those rows are marked
``"truncated": false``, and ``actual_tokens`` records the real length — read the
bin as "max available context", not as an exact 128k point.

Usage::

    python scripts/evaluate/prepare_aa_lcr_sweep.py
    python scripts/evaluate/prepare_aa_lcr_sweep.py --lengths 1024,4096,16384 --max-rows 20
    python scripts/evaluate/prepare_aa_lcr_sweep.py --tokenizer /sms-scratch/checkpoints/gemma-4-31B-it
    python scripts/evaluate/prepare_aa_lcr_sweep.py --lengths 65536,131072 --allow-short

Note the server's ``max_model_len`` must cover the largest bin plus
``max_tokens`` (e.g. bin 32768 + 4096 generation => ~40960).
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

DEFAULT_TOKENIZER = "/sms-scratch/checkpoints/gemma-4-31B-it"
DEFAULT_LENGTHS = "1024,2048,4096,8192,16384,32768"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "mtp_server_eval" / "data"

HEADER = (
    "You are given multiple documents. Read them carefully and answer the "
    "question. Reason across documents when needed; do not rely on a single "
    "span that merely restates the question."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lengths",
        default=DEFAULT_LENGTHS,
        help=f"comma-separated target prompt token counts (default: {DEFAULT_LENGTHS})",
    )
    p.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="HF tokenizer used to measure/truncate (must match the served backbone)",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--max-rows", type=int, default=None, help="cap rows (useful for smoke tests)"
    )
    p.add_argument(
        "--tolerance",
        type=int,
        default=32,
        help="accept prompts within +/- this many tokens of target (default: 32)",
    )
    p.add_argument(
        "--allow-short",
        action="store_true",
        help=(
            "when a row's documents cannot fill the target bin, emit the full "
            "untruncated document set instead of skipping the row"
        ),
    )
    return p.parse_args()


def _download() -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    csv_path = Path(hf_hub_download(REPO_ID, filename=CSV_NAME, repo_type="dataset"))
    zip_path = Path(hf_hub_download(REPO_ID, filename=ZIP_NAME, repo_type="dataset"))
    return csv_path, zip_path


def _load_documents(zf: zipfile.ZipFile, category: str, set_id: str) -> list[str]:
    """Load document texts for one AA-LCR document set.

    Zip members are nested under a top-level ``lcr/`` folder, so match on the
    ``{category}/{set_id}/`` infix rather than a strict prefix.
    """
    needle = f"/{category}/{set_id}/"
    names = sorted(
        n for n in zf.namelist() if n.endswith(".txt") and needle in f"/{n}"
    )
    docs = []
    for name in names:
        with zf.open(name) as handle:
            text = handle.read().decode("utf-8", errors="replace").strip()
        if text:
            docs.append(text)
    return docs


class DocPool:
    """Tokenized AA-LCR document sets, cached and reusable across rows.

    Several CSV rows share one document set (different questions over the same
    documents), and tokenizing a ~120k-token set takes seconds, so cache the
    token ids per ``(category, set_id)``.
    """

    def __init__(self, zf: zipfile.ZipFile, tok) -> None:
        self._zf = zf
        self._tok = tok
        self._ids: dict[tuple[str, str], list[int]] = {}

    def ids(self, category: str, set_id: str) -> list[int] | None:
        """Token ids of one set's assembled document block, or None if empty."""
        key = (category, set_id)
        if key not in self._ids:
            docs = _load_documents(self._zf, category, set_id)
            text = "\n\n".join(
                f"--- Document {i} ---\n{d}" for i, d in enumerate(docs, start=1)
            )
            self._ids[key] = (
                self._tok(text, add_special_tokens=False).input_ids if docs else []
            )
        return self._ids[key] or None


def _assemble(doc_text: str, question: str) -> str:
    """Build the full prompt. Question stays last so the draft model always
    conditions on it regardless of how much document context precedes it."""
    return (
        f"{HEADER}\n\n"
        f"=== Documents ===\n{doc_text}\n\n"
        f"=== Question ===\n{question.strip()}\n\n"
        f"=== Answer ===\n"
    )


def build_at_length(
    tok,
    doc_ids: list[int],
    question: str,
    target: int,
    tol: int,
    allow_short: bool = False,
):
    """Truncate the document block so the assembled prompt lands within `tol`
    tokens of `target`.

    Returns ``(prompt, actual_len, truncated)``, or None if the target is
    unreachable. A source shorter than `target` yields None unless
    `allow_short`, which instead emits the full untruncated document set with
    ``truncated=False``."""
    # Fixed overhead: everything except the document text.
    overhead = len(tok(_assemble("", question), add_special_tokens=False).input_ids)
    budget = target - overhead
    if budget <= 0:
        return None  # question + scaffolding alone already exceeds the target
    if len(doc_ids) < budget:
        if not allow_short:
            return None  # source document set can't fill this bin
        prompt = _assemble(tok.decode(doc_ids), question)
        return prompt, len(tok(prompt, add_special_tokens=False).input_ids), False

    # Decode->re-tokenize is not exactly length-preserving; correct iteratively.
    take = budget
    prompt, actual = None, None
    for _ in range(12):
        prompt = _assemble(tok.decode(doc_ids[:take]), question)
        actual = len(tok(prompt, add_special_tokens=False).input_ids)
        if abs(actual - target) <= tol:
            return prompt, actual, True
        take += target - actual
        take = max(1, min(take, len(doc_ids)))
    return (prompt, actual, True) if prompt else None


def main() -> None:
    args = parse_args()
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    csv_path, zip_path = _download()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    writers = {
        L: (args.out_dir / f"aa-lcr-{L // 1024}k.jsonl").open("w", encoding="utf-8")
        for L in lengths
    }
    counts = dict.fromkeys(lengths, 0)
    n_short = dict.fromkeys(lengths, 0)
    actuals: dict[int, list[int]] = {L: [] for L in lengths}
    skipped_rows = 0

    with zip_path.open("rb") as raw, zipfile.ZipFile(raw) as zf, csv_path.open(
        encoding="utf-8"
    ) as csv_f:
        pool = DocPool(zf, tok)
        for n_row, row in enumerate(csv.DictReader(csv_f)):
            if args.max_rows is not None and n_row >= args.max_rows:
                break
            question = (row.get("question") or "").strip()
            category = (row.get("document_category") or "").strip()
            set_id = (row.get("document_set_id") or "").strip()
            if not (question and category and set_id):
                skipped_rows += 1
                continue
            doc_ids = pool.ids(category, set_id)
            if not doc_ids:
                print(f"warning: no documents for {category}/{set_id}; skipping row")
                skipped_rows += 1
                continue

            qid = row.get("question_id") or str(n_row)
            for L in lengths:
                built = build_at_length(
                    tok, doc_ids, question, L, args.tolerance, args.allow_short
                )
                if built is None:
                    continue
                prompt, actual, truncated = built
                bench = f"aa-lcr-{L // 1024}k"
                writers[L].write(
                    json.dumps(
                        {
                            "benchmark": bench,
                            "id": f"{bench}__{category}_{set_id}_q{qid}",
                            "prompt": prompt,
                            "target_tokens": L,
                            "actual_tokens": actual,
                            "truncated": truncated,
                            "document_category": category,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                counts[L] += 1
                actuals[L].append(actual)
                if not truncated:
                    n_short[L] += 1

    for f in writers.values():
        f.close()

    print(f"\ntokenizer: {args.tokenizer}")
    if skipped_rows:
        print(f"skipped {skipped_rows} unusable source row(s)")
    print(f"{'benchmark':<16}{'n':>5}{'mean_tok':>10}{'min':>8}{'max':>8}{'short':>7}")
    for L in lengths:
        a = actuals[L]
        if not a:
            print(f"aa-lcr-{L // 1024}k".ljust(16) + f"{0:>5}   (target unreachable)")
            continue
        print(
            f"{f'aa-lcr-{L // 1024}k':<16}{counts[L]:>5}"
            f"{sum(a) / len(a):>10.0f}{min(a):>8}{max(a):>8}{n_short[L]:>7}"
        )
    print(f"\nwrote -> {args.out_dir}")
    for L in lengths:
        if n_short[L]:
            print(
                f"NOTE: aa-lcr-{L // 1024}k has {n_short[L]}/{counts[L]} row(s) below "
                f"target ({min(actuals[L])}-{max(actuals[L])} tokens) — the source "
                "documents run out. Treat the bin as 'max available context'."
            )
    ns = {counts[L] for L in lengths if counts[L]}
    if len(ns) > 1:
        print(
            "NOTE: bins have differing n, so the design is not fully paired; "
            "compare with care."
        )


if __name__ == "__main__":
    main()
