#!/usr/bin/env python3
"""AA-LCR context-length bins: accept_len vs prompt length, per checkpoint/k.

Reads every results/lcr_bins_<tag>/mtp_eval_summary.json (Ravi's paired bins,
1k..112k) and, as a reference, the earlier 4-bucket LCR sweep of the
all-sliding scratch checkpoint (different prompt construction + n, ep25 not
ep15 -- indicative overlay, not a controlled comparison; the controlled one is
rerunning sbatch_lcr_bins.sh with the scratch checkpoint).

Usage: python plot_lcr_bins.py  -> results/lcr_bins_accept_len.png
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent / "results"
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# color follows the CHECKPOINT (entity); k is marker+linestyle.
CKPT_COLOR = {"scratch_best": "#2a78d6", "fullattn2_ep15": "#eb6834",
              "fullattn2_ep25": "#1baf7a", "redhat_dflash": "#8e5bd1"}
# k7 = RedHat DFlash's native max k (block 8, no anchor sampling) -- same
# "max-k" role as our k8, so same solid style.
K_STYLE = {"k8": ("o", "-", 2.4), "k7": ("o", "-", 2.4), "k3": ("^", "--", 1.8)}


def bin_tokens(name: str) -> int:
    return int(re.sub(r"\D", "", name)) * 1024


fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
all_x = []
endpoints = []  # (y, label, x, color) for dodged end-labels

for d in sorted(R.glob("lcr_bins_*")):
    tag = d.name.replace("lcr_bins_", "")
    ckpt, ktag = tag.rsplit("_", 1)
    if ktag not in K_STYLE or ckpt not in CKPT_COLOR:
        continue
    color = CKPT_COLOR[ckpt]
    mark, ls, lw = K_STYLE[ktag]
    # ep15 and ep25 fullattn2 coincide: draw ep15 thick underneath so both show
    if ckpt == "fullattn2_ep15":
        lw += 1.6
    rows = json.load(open(d / "mtp_eval_summary.json"))
    pts = sorted(((bin_tokens(r["benchmark"]), r["accept_length"]) for r in rows))
    xs, ys = zip(*pts)
    all_x += xs
    names = {"scratch_best": "all-sliding (scratch ep25)",
             "fullattn2_ep15": "fullattn2 (ep15)",
             "fullattn2_ep25": "fullattn2 (ep25)",
             "redhat_dflash": "RedHat DFlash (all-full-attn)"}
    label = f"{names[ckpt]} {ktag}"
    ax.plot(xs, ys, color=color, marker=mark, markersize=7, linewidth=lw,
            linestyle=ls, label=label, zorder=3 + (ckpt == "fullattn2_ep25"))
    endpoints.append((ys[-1], label, xs[-1], color))

# dodge end-labels: sort by y, push apart to >= MIN_GAP in data units
MIN_GAP = 0.17
endpoints.sort()
placed = []
for y, label, x, color in endpoints:
    if placed and y - placed[-1] < MIN_GAP:
        y = placed[-1] + MIN_GAP
    placed.append(y)
for (y0, label, x, color), y in zip(endpoints, placed):
    ax.annotate(label, (x, y), textcoords="offset points",
                xytext=(10, 0), color=color, fontsize=9, va="center")

ax.axvline(8192, color=TEXT2, linewidth=1, linestyle=":", zorder=1)
ax.annotate("training seq len (8192)", (8192, 0.4), color=TEXT2, fontsize=8.5,
            ha="left", xytext=(6, 0), textcoords="offset points")
ax.set_xscale("log", base=2)
ticks = sorted(set(all_x))
ax.set_xticks(ticks)
# 96k/112k are too close on a log axis -- stagger the last two tick labels
labs = [f"{round(x / 1024)}k" for x in ticks]
ax.set_xticklabels(labs, fontsize=8)
if len(ticks) >= 2 and ticks[-1] / ticks[-2] < 1.3:
    for t in ax.get_xticklabels()[-2:-1]:
        t.set_y(-0.035)
ax.minorticks_off()
ax.set_ylim(0, 4.6)
ax.set_xlabel("prompt length (tokens, AA-LCR paired bins)", color=TEXT2, fontsize=9)
ax.set_ylabel("accept_len", color=TEXT2, fontsize=9)
ax.set_title("AA-LCR acceptance vs context: full-attention draft layers decay "
             "with length,\nall-sliding stays flat", color=TEXT, fontsize=11)
ax.tick_params(colors=TEXT2, labelsize=9)
ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.legend(loc="lower left", fontsize=8.5, frameon=False, labelcolor=TEXT)
ax.margins(x=0.16)
fig.tight_layout()
out = R / "lcr_bins_accept_len.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"wrote {out}")
