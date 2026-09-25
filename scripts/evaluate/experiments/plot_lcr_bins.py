#!/usr/bin/env python3
"""AA-LCR context-length bins: accept_len vs prompt length, per checkpoint/k.

Reads every results/lcr_bins_<tag>/mtp_eval_summary.json (Ravi's paired bins,
1k..112k) and, as a reference, the earlier 4-bucket LCR sweep of the
all-sliding scratch checkpoint (different prompt construction + n, ep25 not
ep15 -- indicative overlay, not a controlled comparison; the controlled one is
rerunning sbatch_lcr_bins.sh with the scratch checkpoint).

Usage: python plot_lcr_bins.py              -> results/lcr_bins_accept_len.png  (k8 + k3;
                                               all-sliding, fullattn2 ep15/ep25, fullattn2 + YaRN)
       python plot_lcr_bins.py --no-yarn    -> results/lcr_bins_accept_len_noyarn.png (k8 + k3;
                                               architecture-only view: all-sliding, fullattn2
                                               ep15/ep25, fullattn1 ep25 -- all unscaled RoPE)
       python plot_lcr_bins.py --fullattn1  -> results/lcr_bins_accept_len_fullattn1.png (k8 + k3;
                                               all-sliding vs fullattn1 vs fullattn1 + YaRN on the
                                               full layer only via per-layer-type rope_parameters)
       python plot_lcr_bins.py --matched-k  -> results/lcr_bins_accept_len_k7.png (k7 + k3;
                                               k7 = RedHat DFlash's native max k)
"""

import argparse
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
              "fullattn2_ep25": "#1baf7a", "redhat_dflash": "#8e5bd1",
              "fullattn2_ep25_yarn": "#c23f92", "fullattn1_ep25": "#b8860b",
              "fullattn1_ep25_yarn_full": "#7a3fc2"}
# k7 = RedHat DFlash's native max k (block 8, no anchor sampling) -- same
# "max-k" role as our k8, so same solid style.
K_STYLE = {"k8": ("o", "-", 2.4), "k7": ("o", "-", 2.4), "k3": ("^", "--", 1.8)}

ap = argparse.ArgumentParser()
ap.add_argument("--matched-k", action="store_true",
                help="plot k7 (+k3) instead of k8 (+k3), for matched-k vs RedHat DFlash")
ap.add_argument("--no-yarn", action="store_true",
                help="architecture-only view: drop the YaRN series, add fullattn1")
ap.add_argument("--fullattn1", action="store_true",
                help="fullattn1 view: all-sliding vs fullattn1 vs fullattn1 + YaRN on "
                     "the full layer only (per-layer-type rope)")
args = ap.parse_args()
SHOW_K = {"k7", "k3"} if args.matched_k else {"k8", "k3"}
# Each view keeps a readable subset of checkpoints (all others hidden).
VIEWS = {
    "default": {"scratch_best", "fullattn2_ep15", "fullattn2_ep25",
                "fullattn2_ep25_yarn", "redhat_dflash"},
    "no_yarn": {"scratch_best", "fullattn2_ep15", "fullattn2_ep25",
                "fullattn1_ep25", "redhat_dflash"},
    "fullattn1": {"scratch_best", "fullattn1_ep25", "fullattn1_ep25_yarn_full"},
}
view = "fullattn1" if args.fullattn1 else "no_yarn" if args.no_yarn else "default"
HIDE_CKPT = set(CKPT_COLOR) - VIEWS[view]
if args.fullattn1:
    TITLE = ("AA-LCR acceptance vs context: fullattn1 (1 full + 4 sliding), "
             "unscaled vs\nserve-time YaRN on the full layer only (sliding layers "
             "keep trained RoPE)")
    OUT = "lcr_bins_accept_len_fullattn1.png"
elif args.no_yarn:
    TITLE = ("AA-LCR acceptance vs context (unscaled RoPE): two full-attention draft "
             "layers (fullattn2) decay with length;\none full layer (fullattn1) and "
             "all-sliding stay flat")
    OUT = "lcr_bins_accept_len_noyarn.png"
else:
    TITLE = ("AA-LCR acceptance vs context: full-attention draft layers decay "
             "with length (unscaled RoPE);\nserve-time YaRN recovers it to "
             "all-sliding's flat curve")
    OUT = "lcr_bins_accept_len_k7.png" if args.matched_k else "lcr_bins_accept_len.png"


def bin_tokens(name: str) -> int:
    return int(re.sub(r"\D", "", name)) * 1024


fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
all_x = []
endpoints = []  # (y, label, x, color) for dodged end-labels

for d in sorted(R.glob("lcr_bins_*")):
    tag = d.name.replace("lcr_bins_", "")
    ckpt, ktag = tag.rsplit("_", 1)
    if ktag not in SHOW_K or ckpt not in CKPT_COLOR or ckpt in HIDE_CKPT:
        continue
    if not (d / "mtp_eval_summary.json").exists():
        print(f"skip {d.name}: no summary yet (run in progress?)")
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
             "redhat_dflash": "RedHat DFlash (all-full-attn)",
             "fullattn2_ep25_yarn": "fullattn2 (ep25) + YaRN@serve",
             "fullattn1_ep25": "fullattn1 (ep25, WSD)",
             "fullattn1_ep25_yarn_full": "fullattn1 (ep25) + YaRN on full layer only"}
    label = f"{names[ckpt]} {ktag}"
    # yarn on top (it's the new result), then ep25, then the rest
    z = 3 + (ckpt == "fullattn2_ep25") + 2 * (ckpt == "fullattn2_ep25_yarn")
    ax.plot(xs, ys, color=color, marker=mark, markersize=7, linewidth=lw,
            linestyle=ls, label=label, zorder=z)
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
ax.set_title(TITLE, color=TEXT, fontsize=11)
ax.tick_params(colors=TEXT2, labelsize=9)
ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.legend(loc="lower left", fontsize=8.5, frameon=False, labelcolor=TEXT)
ax.margins(x=0.16)
fig.tight_layout()
out = R / OUT
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"wrote {out}")
