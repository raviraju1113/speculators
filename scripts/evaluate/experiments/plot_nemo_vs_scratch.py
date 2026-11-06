#!/usr/bin/env python3
"""Continued (ep24) vs from-scratch (ep15) on nemo782k: accept_len and tok/s
as a function of k (3/5/8), FULL benchmark sets.

Reads results/gemma4-31b-dspark-nemo-full/ and results/gemma4-31b-dspark-scratch/
(missing experiments are skipped -- rerun after both eval jobs finish).
Lines = unweighted mean of the 3 benchmarks; faint dots = per benchmark.

Usage: python plot_nemo_vs_scratch.py   -> results/nemo_vs_scratch.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent / "results"
# style: (dir, pattern, color, marker, linestyle, linewidth, label_dy)
# the two curves COINCIDE almost exactly, so: thick solid blue underneath,
# dashed orange on top, end-labels dodged vertically.
RUNS = {
    "continued ep24": ("gemma4-31b-dspark-nemo-full", "nemo_ep24_k{k}_full", "#2a78d6", "o", "-", 3.5, 10),
    "scratch ep15": ("gemma4-31b-dspark-scratch", "scratch_ep15_k{k}", "#eb6834", "s", "--", 2, -10),
}
KS = [3, 5, 8]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BASELINE_TOKS = 59.5  # backbone-only decode tok/s, same harness

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
for ax, metric, ylab in (
    (axes[0], "accept_length", "accept_len (mean of 3 benchmarks)"),
    (axes[1], "decode_tok_s", "decode tok/s (mean; right axis label = speedup)"),
):
    ax.set_facecolor(SURFACE)
    for label, (d, pat, color, mark, ls, lw, dy) in RUNS.items():
        xs, ys = [], []
        for k in KS:
            f = R / d / pat.format(k=k) / "mtp_eval_summary.json"
            if not f.exists():
                print(f"  missing, skipped: {f.parent.relative_to(R)}")
                continue
            rows = json.load(open(f))
            vals = [r[metric] for r in rows]
            xs.append(k)
            ys.append(sum(vals) / len(vals))
            ax.scatter([k] * len(vals), vals, color=color, s=14, alpha=0.35, zorder=2)
        if xs:
            ax.plot(xs, ys, color=color, marker=mark, markersize=8, linewidth=lw,
                    linestyle=ls, label=label, zorder=3)
            ax.annotate(label, (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(8, dy), color=color, fontsize=9, va="center")
    if metric == "decode_tok_s":
        sec = ax.secondary_yaxis("right", functions=(lambda v: v / BASELINE_TOKS,
                                                     lambda v: v * BASELINE_TOKS))
        sec.set_ylabel("speedup vs 59.5 tok/s baseline", color=TEXT2, fontsize=9)
        sec.tick_params(colors=TEXT2, labelsize=9)
    ax.set_xlabel("k (speculative tokens per step)", color=TEXT2, fontsize=9)
    ax.set_ylabel(ylab, color=TEXT2, fontsize=9)
    ax.set_xticks(KS)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    for s in ("top",):
        ax.spines[s].set_visible(False)
    ax.margins(x=0.18)
    ax.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=TEXT)

fig.suptitle("nemo782k: continued-from-old-corpus vs from-scratch (full benchmarks)",
             color=TEXT, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = R / "nemo_vs_scratch.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"wrote {out}")
