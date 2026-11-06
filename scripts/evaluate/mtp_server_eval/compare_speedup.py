#!/usr/bin/env python3
"""Compare MTP eval summaries and print a speedup table.

Takes one or more labelled ``mtp_eval_summary.json`` files and prints, per
benchmark, each config's output throughput and accept length, plus the speedup
of every config relative to the baseline (the first one given).

Usage:
    python compare_speedup.py baseline=/path/baseline/mtp_eval_summary.json \
                              mtp=/path/mtp/mtp_eval_summary.json
    # sweep dirs (auto-labels by subdir name):
    python compare_speedup.py --dir ./results/sweep --baseline baseline
"""

import argparse
import json
from pathlib import Path


def load_summary(path):
    """Return {benchmark: {out_tok_s, accept_len, n}}."""
    rows = json.loads(Path(path).read_text())
    return {r["benchmark"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "configs",
        nargs="*",
        help="label=path/to/mtp_eval_summary.json (first = baseline)",
    )
    ap.add_argument("--dir", help="scan subdirs of DIR for mtp_eval_summary.json")
    ap.add_argument(
        "--baseline", help="label/subdir to use as baseline (default: first)"
    )
    args = ap.parse_args()

    labelled = []  # (label, summary_dict)
    if args.dir:
        base = Path(args.dir)
        for sub in sorted(base.iterdir()):
            f = sub / "mtp_eval_summary.json"
            if f.exists():
                labelled.append((sub.name, load_summary(f)))
    for spec in args.configs:
        label, _, path = spec.partition("=")
        labelled.append((label or Path(path).parent.name, load_summary(path)))

    if not labelled:
        ap.error("no summaries found (pass label=path args or --dir)")

    # Choose baseline.
    labels = [l for l, _ in labelled]
    base_label = args.baseline or labels[0]
    if base_label not in labels:
        ap.error(f"baseline {base_label!r} not among {labels}")
    base = dict(labelled)[base_label]

    def tok_of(row):
        # prefer decode-only throughput; fall back to older field names
        return (
            row.get("decode_tok_s")
            or row.get("mean_output_tok_s")
            or row.get("e2e_tok_s")
        )

    benches = sorted({b for _, s in labelled for b in s})
    print(f"baseline = {base_label}\n")
    for bench in benches:
        print(f"=== {bench} ===")
        print(
            f"  {'config':<18}{'decode_tok/s':>14}{'accept_len':>12}"
            f"{'accept_rate':>13}{'speedup':>10}"
        )
        b_tok = tok_of(base.get(bench, {}) or {})
        for label, s in labelled:
            r = s.get(bench)
            if not r:
                continue
            tok = tok_of(r)
            al = r.get("accept_length")
            ar = r.get("accept_rate")
            al_s = "n/a" if al is None else f"{al:.3f}"
            ar_s = "n/a" if ar is None else f"{ar:.4f}"
            sp = f"{tok / b_tok:.2f}x" if (b_tok and tok) else "n/a"
            print(f"  {label:<18}{tok:>14.1f}{al_s:>12}{ar_s:>13}{sp:>10}")
        print()


if __name__ == "__main__":
    main()
