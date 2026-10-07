#!/usr/bin/env python3
"""Compare sample_from_anchor=True vs False across training epochs.

Scans the results/ dirs for the matched-epoch evals (ep3/7/9/14) and plots
accept_len (mean over aime/gpqa/livecodebench; faint markers = individual
benchmarks) for the two runs. Two panels:

  left  : k=3           -- same draft budget, the clean quality comparison
  right : native k      -- True at k=8 vs False at k=7 (each run's own
                           serving mode; different budgets, deployment view)

Missing summaries (e.g. anchorT ep9/ep14 while that eval is still queued) are
skipped, so rerun after new evals land to fill the lines in.

Usage:  python plot_flag_comparison.py   # writes results/flag_comparison.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = HERE / "results"

# (epoch, results_subdir, experiment_name) per run and k-mode
SOURCES = {
    ("False", "k3"): [
        (3, "gemma4-31b-dspark-anchor-ablation", "main_ep3end_k3"),
        (7, "gemma4-31b-dspark-anchor-ablation-ep7", "main_ep7end_k3"),
        (9, "gemma4-31b-dspark-final", "ours_ep9_k3"),
        (14, "gemma4-31b-dspark-final", "ours_ep14_k3"),
    ],
    ("True", "k3"): [
        (3, "gemma4-31b-dspark-anchor-ablation", "anchorT_ep3_k3"),
        (7, "gemma4-31b-dspark-anchor-ablation-ep7", "anchorT_ep7_k3"),
        (9, "gemma4-31b-dspark-anchorT-final", "anchorT_ep9_k3"),
        (14, "gemma4-31b-dspark-anchorT-final", "anchorT_ep14_k3"),
    ],
    ("False", "native"): [
        (3, "gemma4-31b-dspark-anchor-ablation", "main_ep3end_k7"),
        (7, "gemma4-31b-dspark-anchor-ablation-ep7", "main_ep7end_k7"),
        (9, "gemma4-31b-dspark-final", "ours_ep9_k7"),
        (14, "gemma4-31b-dspark-final", "ours_ep14_k7"),
    ],
    ("True", "native"): [
        (3, "gemma4-31b-dspark-anchor-ablation", "anchorT_ep3_k8"),
        (7, "gemma4-31b-dspark-anchor-ablation-ep7", "anchorT_ep7_k8"),
        (9, "gemma4-31b-dspark-anchorT-final", "anchorT_ep9_k8"),
        (14, "gemma4-31b-dspark-anchorT-final", "anchorT_ep14_k8"),
    ],
}

# validated categorical pair (dataviz reference palette, light mode)
COLOR = {"True": "#2a78d6", "False": "#eb6834"}
MARKER = {"True": "o", "False": "s"}  # secondary encoding, not color-alone
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load(entries):
    pts = []  # (epoch, mean_accept_len, {bench: accept_len})
    for epoch, sub, name in entries:
        f = R / sub / name / "mtp_eval_summary.json"
        if not f.exists():
            print(f"  (missing, skipped: {sub}/{name})")
            continue
        rows = json.load(open(f))
        per = {r["benchmark"]: r["accept_length"] for r in rows}
        pts.append((epoch, sum(per.values()) / len(per), per))
    return pts


K_OF = {("True", "k3"): 3, ("False", "k3"): 3, ("True", "native"): 8, ("False", "native"): 7}

fig, grid = plt.subplots(2, 2, figsize=(11, 8.2), facecolor=SURFACE)
panels = [("k3", "k = 3 (same draft budget)"), ("native", "native k (True k=8, False k=7)")]

for row, metric in enumerate(("accept_len", "accept_rate")):
    for col, (mode, title) in enumerate(panels):
        ax = grid[row][col]
        ax.set_facecolor(SURFACE)
        for flag in ("True", "False"):
            pts = load(SOURCES[(flag, mode)])
            if not pts:
                continue
            k = K_OF[(flag, mode)]
            conv = (lambda v: v) if metric == "accept_len" else (lambda v: (v - 1) / k)
            xs = [p[0] for p in pts]
            ys = [conv(p[1]) for p in pts]
            label = f"flag={flag}" + ("" if mode == "k3" else f" (k={k})")
            ax.plot(xs, ys, color=COLOR[flag], marker=MARKER[flag], markersize=8,
                    linewidth=2, label=label, zorder=3)
            for epoch, _, per in pts:
                ax.scatter([epoch] * len(per), [conv(v) for v in per.values()],
                           color=COLOR[flag], s=14, alpha=0.35, zorder=2)
            ax.annotate(label, (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(8, 0), color=COLOR[flag], fontsize=9, va="center")
        if row == 0:
            ax.set_title(title, color=TEXT, fontsize=11)
        if row == 1:
            ax.set_xlabel("epoch (matched checkpoints)", color=TEXT2, fontsize=9)
        ax.set_xticks([3, 7, 9, 14])
        ax.tick_params(colors=TEXT2, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.margins(x=0.18)
        ax.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=TEXT)

grid[0][0].set_ylabel("accept_len (mean of 3 benchmarks;\nfaint dots = per benchmark)",
                      color=TEXT2, fontsize=9)
grid[1][0].set_ylabel("accept_rate = (accept_len − 1) / k\n(NOT comparable across different k)",
                      color=TEXT2, fontsize=9)
fig.suptitle("DSpark draft quality across training: sample_from_anchor True vs False",
             color=TEXT, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))

out = R / "flag_comparison.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"wrote {out}")
