#!/usr/bin/env python3
"""Turn MTP eval summaries into readable Markdown / CSV tables.

Reads one or more labelled ``mtp_eval_summary.json`` files (the same inputs as
``mtp_server_eval/compare_speedup.py``) and emits a per-benchmark comparison
table with the speedup of every config relative to the baseline (decode tok/s
ratio). Prints Markdown to stdout and, optionally, writes ``.md`` / ``.csv``
files for pasting into a doc or opening in a spreadsheet.

Usage:
    # scan the runner's output dir (auto-labels by subdir; baseline first)
    python tabulate_results.py --dir ./results/gemma4-31b --baseline baseline

    # or list summaries explicitly (first = baseline)
    python tabulate_results.py \
        baseline=./results/gemma4-31b/baseline/mtp_eval_summary.json \
        assistant_k5=./results/gemma4-31b/assistant_k5/mtp_eval_summary.json

    # also write files
    python tabulate_results.py --dir ./results/gemma4-31b --out-dir ./results/gemma4-31b
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Columns rendered in the Markdown table (header, summary key, format spec).
COLUMNS = [
    ("n", "n", "d"),
    ("decode tok/s", "decode_tok_s", ".1f"),
    ("e2e tok/s", "e2e_tok_s", ".1f"),
    ("ttft (s)", "mean_ttft_s", ".3f"),
    ("accept_len", "accept_length", ".3f"),
    ("accept_rate", "accept_rate", ".4f"),
]


def load_summary(path: Path) -> dict:
    """Return {benchmark: row_dict} for one mtp_eval_summary.json."""
    rows = json.loads(Path(path).read_text())
    return {r["benchmark"]: r for r in rows}


def tok_of(row: dict) -> float | None:
    """Decode-only throughput, falling back to older field names."""
    return (
        row.get("decode_tok_s")
        or row.get("mean_output_tok_s")
        or row.get("e2e_tok_s")
    )


def fmt(value, spec: str) -> str:
    if value is None:
        return "—"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def collect(args: argparse.Namespace) -> list[tuple[str, dict]]:
    """Ordered [(label, summary_dict)]: --dir subdirs first, then label=path args."""
    labelled: list[tuple[str, dict]] = []
    if args.dir:
        for sub in sorted(Path(args.dir).iterdir()):
            f = sub / "mtp_eval_summary.json"
            if f.exists():
                labelled.append((sub.name, load_summary(f)))
    for spec in args.configs:
        label, _, path = spec.partition("=")
        labelled.append((label or Path(path).parent.name, load_summary(Path(path))))
    return labelled


def order_labels(labels: list[str], baseline: str) -> list[str]:
    """Baseline first, remaining configs in their original order."""
    return [baseline] + [l for l in labels if l != baseline]


def build_rows(labelled, benches, ordered, base) -> list[dict]:
    """Flat list of rows (one per benchmark x config) with a computed speedup."""
    by_label = dict(labelled)
    out = []
    for bench in benches:
        b_tok = tok_of(base.get(bench, {}) or {})
        for label in ordered:
            r = by_label[label].get(bench)
            if not r:
                continue
            tok = tok_of(r)
            speedup = tok / b_tok if (b_tok and tok) else None
            out.append({"benchmark": bench, "config": label, "speedup": speedup, **r})
    return out


def to_markdown(rows: list[dict], benches: list[str], ordered: list[str]) -> str:
    headers = ["benchmark", "config", *(h for h, _, _ in COLUMNS), "speedup"]
    lines = ["| " + " | ".join(headers) + " |"]
    align = ["---", "---"] + ["---:"] * (len(COLUMNS) + 1)
    lines.append("| " + " | ".join(align) + " |")
    by_key = {(r["benchmark"], r["config"]): r for r in rows}
    for bench in benches:
        first = True
        for label in ordered:
            r = by_key.get((bench, label))
            if not r:
                continue
            cells = [
                bench if first else "",
                label,
                *(fmt(r.get(key), spec) for _, key, spec in COLUMNS),
                "1.00×" if r["speedup"] and abs(r["speedup"] - 1) < 1e-9
                else (f"**{r['speedup']:.2f}×**" if r["speedup"] else "—"),
            ]
            lines.append("| " + " | ".join(cells) + " |")
            first = False
    return "\n".join(lines)


def write_csv(rows: list[dict], path: Path) -> None:
    fields = ["benchmark", "config", *(k for _, k, _ in COLUMNS), "speedup"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "speedup": None if r["speedup"] is None else round(r["speedup"], 4)})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "configs", nargs="*", help="label=path/to/mtp_eval_summary.json (first = baseline)"
    )
    ap.add_argument("--dir", help="scan subdirs of DIR for mtp_eval_summary.json")
    ap.add_argument("--baseline", help="label/subdir to use as baseline (default: first)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        help="write results_table.md and results_table.csv here",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    labelled = collect(args)
    if not labelled:
        raise SystemExit("no summaries found (pass label=path args or --dir)")

    labels = [l for l, _ in labelled]
    base_label = args.baseline or labels[0]
    if base_label not in labels:
        raise SystemExit(f"baseline {base_label!r} not among {labels}")

    ordered = order_labels(labels, base_label)
    benches = sorted({b for _, s in labelled for b in s})
    base = dict(labelled)[base_label]
    rows = build_rows(labelled, benches, ordered, base)

    md = to_markdown(rows, benches, ordered)
    print(f"baseline = {base_label} (speedup = decode tok/s ÷ baseline decode tok/s)\n")
    print(md)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        md_path = args.out_dir / "results_table.md"
        csv_path = args.out_dir / "results_table.csv"
        md_path.write_text(
            f"# Speedup vs `{base_label}` (decode tok/s ratio)\n\n{md}\n"
        )
        write_csv(rows, csv_path)
        print(f"\nwrote {md_path}\nwrote {csv_path}")


if __name__ == "__main__":
    main()
