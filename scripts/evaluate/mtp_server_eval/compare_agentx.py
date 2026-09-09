#!/usr/bin/env python3
"""Compare AgentX matrices and print a per-concurrency speedup table.

Takes one or more labelled ``matrix.tsv`` files (written by ``run_agentx.sh``)
and prints, per concurrency level, each config's decode throughput and accept
length, plus the speedup of every config relative to the baseline (the first one
given). The AgentX counterpart to ``compare_speedup.py``, which does the same for
the static-prompt benchmarks.

Usage:
    python compare_agentx.py baseline=/path/baseline/matrix.tsv \
                             eagle3=/path/eagle3/matrix.tsv
    # sweep dirs (auto-labels by subdir name):
    python compare_agentx.py --dir ./results/agentx-gemma4 --baseline baseline
"""

import argparse
import csv
from pathlib import Path

MATRIX_NAME = "matrix.tsv"
NA = "NA"


def load_matrix(path):
    """Return {users: {column: raw str}} from a run_agentx.sh matrix.tsv."""
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            users = (row.get("users") or "").strip()
            if users:
                rows[users] = row
    return rows


def _num(row, key):
    """Matrix cells are strings and may be ``NA``; return a float or None."""
    try:
        return float((row.get(key) or "").strip())
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "configs",
        nargs="*",
        help=f"label=path/to/{MATRIX_NAME} (first = baseline)",
    )
    ap.add_argument("--dir", help=f"scan subdirs of DIR for {MATRIX_NAME}")
    ap.add_argument(
        "--baseline", help="label/subdir to use as baseline (default: first)"
    )
    args = ap.parse_args()

    labelled = []  # (label, {users: row})
    if args.dir:
        base = Path(args.dir)
        for sub in sorted(base.iterdir()):
            f = sub / MATRIX_NAME
            if f.exists():
                labelled.append((sub.name, load_matrix(f)))
    for spec in args.configs:
        label, _, path = spec.partition("=")
        labelled.append((label or Path(path).parent.name, load_matrix(path)))

    if not labelled:
        ap.error("no matrices found (pass label=path args or --dir)")

    # Choose baseline.
    labels = [l for l, _ in labelled]
    base_label = args.baseline or labels[0]
    if base_label not in labels:
        ap.error(f"baseline {base_label!r} not among {labels}")
    base = dict(labelled)[base_label]

    # Concurrency levels are numeric; sort as ints so 16 follows 8, not 1.
    levels = sorted({u for _, m in labelled for u in m}, key=int)

    print(f"baseline = {base_label}\n")
    for users in levels:
        print(f"=== users={users} ===")
        print(
            f"  {'config':<18}{'decode_tok/s':>14}{'accept_len':>12}"
            f"{'accept_rate':>13}{'speedup':>10}{'valid':>8}"
        )
        b_tok = _num(base.get(users, {}) or {}, "decode_tok_s")
        for label, m in labelled:
            r = m.get(users)
            if not r:
                continue
            tok = _num(r, "decode_tok_s")
            al = _num(r, "accept_len")
            ar = _num(r, "accept_rate")
            tok_s = NA if tok is None else f"{tok:.1f}"
            al_s = "n/a" if al is None else f"{al:.3f}"
            ar_s = "n/a" if ar is None else f"{ar:.4f}"
            sp = f"{tok / b_tok:.2f}x" if (b_tok and tok) else "n/a"
            valid = (r.get("valid") or NA).strip() or NA
            print(
                f"  {label:<18}{tok_s:>14}{al_s:>12}{ar_s:>13}{sp:>10}{valid:>8}"
            )
        print()

    invalid = [
        (label, users)
        for label, m in labelled
        for users, r in m.items()
        if (r.get("valid") or "").strip().lower() == "false"
    ]
    if invalid:
        print(
            "!! submission_valid=false for: "
            + ", ".join(f"{label}@users={u}" for label, u in invalid)
        )
        print("   These cells broke an AgentX scenario rule (e.g. a short smoke")
        print("   duration via --unsafe-override) and are not comparable results.")


if __name__ == "__main__":
    main()
