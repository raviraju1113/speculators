#!/usr/bin/env python3
"""Accept_len vs context length: synthetic-padding sweep + AA-LCR real-document
sweep on one chart.

Usage: python plot_ctx_sweep.py   -> results/ctx_sweep_accept_len.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent / "results"
SERIES = [
    # (label, summary json, color, marker, linestyle)
    ("synthetic padding (lcb+filler, TP=1)",
     R / "ctx_sweep_scratch_ep25_k8/ctx_sweep_summary.json", "#2a78d6", "o", "-"),
    ("AA-LCR real documents (TP=2)",
     R / "lcr_sweep_spec/ctx_sweep_summary.json", "#eb6834", "s", "--"),
]
LCR_ONLY = "--lcr-only" in sys.argv
if LCR_ONLY:
    SERIES = [s for s in SERIES if "AA-LCR" in s[0]]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRAINED_LEN = 8192

fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
all_x = []
for label, src, color, mark, ls in SERIES:
    if not src.exists():
        print(f"missing, skipped: {src}")
        continue
    rows = json.load(open(src))
    xs = [r["actual_prompt_len_mean"] for r in rows]
    ys = [r["accept_length"] for r in rows]
    all_x += xs
    ax.plot(xs, ys, color=color, marker=mark, markersize=8, linewidth=2,
            linestyle=ls, label=label, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9),
                    color=TEXT2, fontsize=8, ha="center")
    ax.annotate(label.split(" (")[0], (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(8, 0), color=color, fontsize=9, va="center")

ax.axvline(TRAINED_LEN, color=TEXT2, linewidth=1, linestyle=":", zorder=1)
ax.annotate("training seq len (8192)", (TRAINED_LEN, 0.5), color=TEXT2,
            fontsize=8.5, ha="left", xytext=(6, 0), textcoords="offset points")

ax.set_xscale("log", base=2)
ticks = sorted(set(all_x))
ax.set_xticks(ticks)
ax.set_xticklabels([f"{round(x/1024)}k" for x in ticks], fontsize=8)
ax.minorticks_off()
ax.set_ylim(0, 6.2)
ax.set_xlabel("prompt length (tokens)", color=TEXT2, fontsize=9)
ax.set_ylabel("accept_len (k=8, scratch checkpoint_best)", color=TEXT2, fontsize=9)
title = ("DSpark acceptance on AA-LCR real documents: flat to 106k tokens\n"
         "(scratch checkpoint_best, k=8, 30 questions/bucket, TP=2)") if LCR_ONLY else (
         "DSpark acceptance vs context length: flat to 106k tokens\n"
         "(real documents cost ~1.4 accept_len vs padding, but no length decay)")
ax.set_title(title, color=TEXT, fontsize=11)
ax.tick_params(colors=TEXT2, labelsize=9)
ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.legend(loc="lower left", fontsize=9, frameon=False, labelcolor=TEXT)
ax.margins(x=0.14)

fig.tight_layout()
out = R / ("lcr_accept_len.png" if LCR_ONLY else "ctx_sweep_accept_len.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"wrote {out}")
