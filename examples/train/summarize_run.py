#!/usr/bin/env python3
"""Summarize a speculators training run into a pasteable table.

Parses the job stdout log for train/ and val/ metrics and prints:
  * a full metric table (train vs val) including EVERY block position 0..N-1,
    not just the couple that happen to get logged inline
  * per-epoch validation history (accept_len / accept_rate / loss)
  * run facts: epochs completed, checkpoint_best, silently dropped samples

Covers the TRAINING side only. Serving numbers -- decode tok/s, speedup vs
baseline per benchmark -- come from scripts/evaluate/experiments/run_experiments.py
after the checkpoint is exported (that is the table format Ravi reports).

Usage:
    python examples/train/summarize_run.py <job-log> [--ckpt-dir DIR] [--markdown]
    python examples/train/summarize_run.py logs/dspark_full_lr3e4.txt --markdown
"""

from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from pathlib import Path

# Metrics are emitted by a rich logger and wrap across lines, e.g.
#   "                    train/loss=0.474, train/ce_loss=0.594,"
# so scan the whole file for key=value pairs rather than parsing line blocks.
METRIC_RE = re.compile(r"(train|val)/([A-Za-z_0-9]+?)(_epoch)?=([-+0-9.eE]+)")
EPOCH_RE = re.compile(r"\bepoch=(\d+)")
DROP_RE = re.compile(r"Failed to load/cache hidden states")


def parse(path: Path):
    """Return (final_train, final_val, val_history, n_drops)."""
    text = path.read_text(errors="replace")

    final_train: dict[str, float] = OrderedDict()
    final_val: dict[str, float] = OrderedDict()
    # val metrics are emitted once per epoch; keep them keyed by epoch so we can
    # show the trajectory, not just the endpoint.
    val_history: dict[int, dict[str, float]] = OrderedDict()

    cur_val: dict[str, float] = {}
    for line in text.splitlines():
        for split, name, _suffix, value in METRIC_RE.findall(line):
            try:
                val = float(value)
            except ValueError:
                continue
            if split == "train":
                final_train[name] = val
            else:
                final_val[name] = val
                cur_val[name] = val
        # An "epoch=N" on a val line closes that epoch's block.
        m = EPOCH_RE.search(line)
        if m and cur_val:
            val_history[int(m.group(1))] = dict(cur_val)
            cur_val = {}

    return final_train, final_val, val_history, len(DROP_RE.findall(text))


def positions(metrics: dict[str, float]) -> list[int]:
    idx = []
    for k in metrics:
        m = re.fullmatch(r"position_(\d+)_acc", k)
        if m:
            idx.append(int(m.group(1)))
    return sorted(idx)


def fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.4f}"


def emit(rows, headers, markdown: bool):
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    sep = "|" if markdown else " "

    def row(cells):
        body = sep.join(f" {str(c):<{widths[i]}} " for i, c in enumerate(cells))
        return f"|{body}|" if markdown else body

    print(row(headers))
    if markdown:
        print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    else:
        print("-" * (sum(widths) + 3 * len(widths)))
    for r in rows:
        print(row(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, help="job stdout log")
    ap.add_argument("--ckpt-dir", type=Path, default=None,
                    help="checkpoint dir, for epochs completed + checkpoint_best")
    ap.add_argument("--markdown", action="store_true", help="markdown tables")
    args = ap.parse_args()

    if not args.log.exists():
        raise SystemExit(f"no such log: {args.log}")

    train, val, history, drops = parse(args.log)
    if not train and not val:
        raise SystemExit("no train/ or val/ metrics found -- did training start?")

    print(f"\n=== run: {args.log} ===\n")

    # ---- run facts -------------------------------------------------------
    if args.ckpt_dir and args.ckpt_dir.exists():
        epochs = sorted(int(p.name) for p in args.ckpt_dir.iterdir() if p.name.isdigit())
        best = args.ckpt_dir / "checkpoint_best"
        best_txt = best.resolve().name if best.exists() else "-"
        print(f"epochs written   : {len(epochs)} (last = {epochs[-1] if epochs else '-'})")
        print(f"checkpoint_best  : epoch {best_txt}  (selected by lowest val loss)")
    print(f"validation epochs: {len(history)}")
    # Non-zero here means samples were silently skipped -- the dataloader returns
    # None on a failed hidden-state fetch and training continues regardless.
    flag = "  <-- INVESTIGATE" if drops else ""
    print(f"dropped samples  : {drops}{flag}\n")

    # ---- per-position table ---------------------------------------------
    pos = positions(train) or positions(val)
    if pos:
        rows = [[f"position_{i}", fmt(train.get(f"position_{i}_acc")),
                 fmt(val.get(f"position_{i}_acc"))] for i in pos]
        rows.append(["full_acc", fmt(train.get("full_acc")), fmt(val.get("full_acc"))])
        print("per-position acceptance (final epoch)\n")
        emit(rows, ["position", "train", "val"], args.markdown)
        print()

    # ---- headline + loss breakdown --------------------------------------
    keys = ["accept_len", "accept_rate", "loss", "ce_loss", "tv_loss",
            "confidence_loss", "confidence_abs_error", "confidence_pred_mean",
            "confidence_cumprod_bias"]
    rows = [[k, fmt(train.get(k)), fmt(val.get(k))] for k in keys
            if k in train or k in val]
    print("headline metrics + loss breakdown (final epoch)\n")
    emit(rows, ["metric", "train", "val"], args.markdown)

    # Calibration sanity: the confidence head should predict roughly the accept
    # rate it actually achieves. A large gap means it is over/under-confident,
    # which is what STS calibration exists to fix.
    cp, ar = val.get("confidence_pred_mean"), val.get("accept_rate")
    if cp is not None and ar is not None:
        gap = cp - ar
        verdict = "OK" if abs(gap) < 0.1 else ("OVERCONFIDENT" if gap > 0 else "UNDERCONFIDENT")
        print(f"\nval calibration : predicted {cp:.3f} vs actual {ar:.3f} "
              f"(gap {gap:+.3f}) -> {verdict}")

    # ---- validation trajectory ------------------------------------------
    if len(history) > 1:
        print("\nvalidation trajectory\n")
        rows = [[e, fmt(m.get("accept_len")), fmt(m.get("accept_rate")),
                 fmt(m.get("loss")), fmt(m.get("position_1_acc"))]
                for e, m in sorted(history.items())]
        if len(rows) > 12:  # keep it readable: first 3, last 6
            rows = rows[:3] + [["...", "...", "...", "...", "..."]] + rows[-6:]
        emit(rows, ["epoch", "accept_len", "accept_rate", "loss", "pos1_acc"],
             args.markdown)
    print()


if __name__ == "__main__":
    main()
