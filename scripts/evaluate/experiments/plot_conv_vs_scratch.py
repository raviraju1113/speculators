"""Conv (DSpark + DFlash2 local conv) vs scratch (plain all-sliding DSpark), both
nemo782k from scratch, 26 epochs, epoch-25 checkpoints, same harness.

Reads:
  results/gemma4-31b-dspark-{conv,scratch}-ep25/<exp>_k{8,5,3}/mtp_eval_summary.json
  results/ctx_sweep_{conv,scratch}_ep25_k8/ctx_sweep_summary.json
Writes:
  results/conv_vs_scratch_benchmarks.png   accept_len per benchmark, one panel per k
  results/conv_vs_scratch_ctx_sweep.png    accept_len vs prompt length at k=8
  results/conv_vs_scratch_tokps.png        decode tok/s per benchmark, one panel per k

Usage: python plot_conv_vs_scratch.py   (from scripts/evaluate/experiments)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parent / "results"
RUNS = {"conv": "gemma4-31b-dspark-conv-ep25", "scratch": "gemma4-31b-dspark-scratch-ep25"}
# Categorical slots 1 and 2 of the validated default palette (light surface).
COLOR = {"conv": "#2a78d6", "scratch": "#eb6834"}
LABEL = {"conv": "conv (all-sliding + DFlash2 conv)", "scratch": "scratch (all-sliding)"}
KS = [8, 5, 3]
BENCHES = [("aime", "AIME"), ("gpqa", "GPQA"), ("livecodebench", "LiveCodeBench")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load_bench():
    out = {}
    for run, d in RUNS.items():
        for k in KS:
            p = R / d / f"{run}_ep25_k{k}" / "mtp_eval_summary.json"
            for e in json.load(open(p)):
                out[(run, k, e["benchmark"])] = e
    return out


def load_ctx():
    out = {}
    for run in RUNS:
        p = R / f"ctx_sweep_{run}_ep25_k8" / "ctx_sweep_summary.json"
        d = json.load(open(p))
        d = d if isinstance(d, list) else d.get("results", d)
        out[run] = {int(e["target_len"]): e for e in d}
    return out


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(metric, ylabel, fname, fmt):
    data = load_bench()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    w, gap = 0.34, 0.02  # 2px-ish surface gap between the two bars
    for ax, k in zip(axes, KS):
        style(ax)
        xs = range(len(BENCHES))
        for i, run in enumerate(("scratch", "conv")):  # scratch first = reference left
            vals = [data[(run, k, b)][metric] for b, _ in BENCHES]
            off = (-1 if run == "scratch" else 1) * (w / 2 + gap)
            bars = ax.bar([x + off for x in xs], vals, width=w, color=COLOR[run],
                          label=LABEL[run], linewidth=0)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v, fmt % v, ha="center",
                        va="bottom", fontsize=8, color=INK)
        # delta annotation above each pair (text in ink, not series color)
        for x, (b, _) in zip(xs, BENCHES):
            dv = data[("conv", k, b)][metric] - data[("scratch", k, b)][metric]
            dv = round(dv, 2 if metric == "accept_length" else 0) + 0.0  # no "-0"
            top = max(data[(r, k, b)][metric] for r in RUNS)
            ax.text(x, top * 1.09, f"Δ {dv:+.2f}" if metric == "accept_length" else f"Δ {dv:+.0f}",
                    ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([n for _, n in BENCHES])
        ax.set_title(f"k = {k}", fontsize=11, color=INK, loc="left")
        ax.set_ylim(0, max(data[(r, k, b)][metric] for r in RUNS for b, _ in BENCHES) * 1.22)
    axes[0].set_ylabel(ylabel, color=INK2, fontsize=9)
    # Legend above the panels (never over the marks); the title row carries it.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=2,
               loc="upper left", bbox_to_anchor=(0.01, 0.93))
    fig.suptitle("Gemma-4-31B-it DSpark drafts, epoch 25, nemo782k from scratch — "
                 "conv vs scratch", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(R / fname, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def ctx_plot():
    ctx = load_ctx()
    lens = sorted(set(ctx["conv"]) & set(ctx["scratch"]))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    style(ax)
    for run in ("scratch", "conv"):
        ys = [ctx[run][L]["accept_length"] for L in lens]
        ax.plot(lens, ys, color=COLOR[run], linewidth=2, marker="o", markersize=5,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=LABEL[run])
        ax.text(lens[-1] * 1.06, ys[-1], f"{ys[-1]:.2f}", va="center", fontsize=8, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(lens)
    ax.set_xticklabels([f"{L // 1024}k" for L in lens])
    ax.set_xlim(lens[0] * 0.85, lens[-1] * 1.35)
    lo = min(ctx[r][L]["accept_length"] for r in RUNS for L in lens)
    hi = max(ctx[r][L]["accept_length"] for r in RUNS for L in lens)
    ax.set_ylim(lo - 0.3, hi + 0.3)
    ax.set_xlabel("prompt length (tokens)", color=INK2, fontsize=9)
    ax.set_ylabel("accept_len (k = 8, 24 prompts / length)", color=INK2, fontsize=9)
    ax.set_title("Context-length sweep — conv vs scratch (draft trained at 8k, window 2048)",
                 fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(R / "conv_vs_scratch_ctx_sweep.png", dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


if __name__ == "__main__":
    grouped_bars("accept_length", "accept_len (mean accepted tokens / step)",
                 "conv_vs_scratch_benchmarks.png", "%.2f")
    grouped_bars("decode_tok_s", "decode tok/s (TP=1, B200)", "conv_vs_scratch_tokps.png", "%.0f")
    ctx_plot()
    for f in ("conv_vs_scratch_benchmarks.png", "conv_vs_scratch_tokps.png", "conv_vs_scratch_ctx_sweep.png"):
        print("wrote", R / f)
