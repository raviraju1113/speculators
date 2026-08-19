#!/usr/bin/env python3
"""Plot acceptance and decode throughput vs context length from a sweep run.

Reads the ``mtp_eval_summary.json`` written by ``mtp_server_eval/run_vllm_eval.py``
for an ``aa-lcr-<N>k`` sweep (see ``prepare_aa_lcr_sweep.py``) and renders a
three-panel figure.

Why three panels and not one: acceptance (~2.1-2.3 tokens) and decode throughput
(~65-293 tok/s) are different scales, and putting them on twin y-axes would let
the arbitrary axis alignment imply a relationship that isn't in the data. Instead
the top panel indexes both to their shortest-context value, which is a real
common scale, and the bottom two panels carry the absolute numbers.

Note ``accept_rate`` is not plotted: with k speculative tokens it is exactly
``(accept_length - 1) / k``, so it carries no information the acceptance panel
doesn't already show. The script asserts that identity and warns if it breaks.

Usage::

    python scripts/evaluate/plot_ctxlen_sweep.py \\
        --summary scripts/evaluate/experiments/results/<run>/eagle3/mtp_eval_summary.json
    python scripts/evaluate/plot_ctxlen_sweep.py --summary <path> --out-dir /tmp/plots
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# dataviz reference palette, categorical slots 1 (blue) and 2 (orange).
# Validated for this 2-series pair: lightness band, chroma floor, CVD separation
# (min protan/deutan OKLab dE 24.7 light / 26.8 dark vs target 8.0), normal-vision
# separation (33.6 / 31.8 vs floor 15.0), and >=3:1 contrast on both surfaces.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True, help="mtp_eval_summary.json")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where to write the figure (default: alongside --summary)",
    )
    p.add_argument("--stem", default="ctxlen_sweep", help="output filename stem")
    p.add_argument(
        "--num-spec-tokens",
        type=int,
        default=3,
        help="k, for the accept_length ceiling annotation and the accept_rate identity",
    )
    p.add_argument("--title", default=None, help="override the figure title")
    return p.parse_args()


def load_points(summary_path: Path, k: int):
    """Return [(tokens, label, accept_len, decode_tok_s)] sorted by context length."""
    rows = json.loads(summary_path.read_text())
    pts = []
    for r in rows:
        m = re.fullmatch(r"aa-lcr-(\d+)k", str(r.get("benchmark", "")))
        if not m:
            continue
        if r.get("accept_length") is None:
            print(f"  skipping {r['benchmark']}: accept_length is null (spec off?)")
            continue
        n_k = int(m.group(1))
        pts.append(
            (
                n_k * 1024,
                f"{n_k}k",
                float(r["accept_length"]),
                float(r["decode_tok_s"]),
                r.get("accept_rate"),
            )
        )
    pts.sort(key=lambda t: t[0])

    # accept_rate must be a pure restatement of accept_length; if not, the two
    # are measuring different things and the omission below would hide something.
    for _, label, al, _, ar in pts:
        if ar is None:
            continue
        if abs((al - 1) / k - ar) > 5e-3:
            print(
                f"  WARNING {label}: accept_rate {ar} != (accept_length-1)/{k} "
                f"= {(al - 1) / k:.4f}; check --num-spec-tokens"
            )
    return [(t, lb, al, dt) for t, lb, al, dt, _ in pts]


def style_axis(ax, xs, labels):
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.minorticks_off()
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = load_points(args.summary, args.num_spec_tokens)
    if len(pts) < 2:
        raise SystemExit(f"need >=2 aa-lcr-*k rows in {args.summary}, got {len(pts)}")

    xs = [p[0] for p in pts]
    labels = [p[1] for p in pts]
    acc = [p[2] for p in pts]
    tps = [p[3] for p in pts]

    acc_idx = [100 * v / acc[0] for v in acc]
    tps_idx = [100 * v / tps[0] for v in tps]
    span = f"{labels[0]}→{labels[-1]}"
    ctx_mult = xs[-1] / xs[0]

    fig = plt.figure(figsize=(11, 8.2), facecolor=SURFACE)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.25, 1], hspace=0.40, wspace=0.22,
        left=0.07, right=0.925, top=0.83, bottom=0.085,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # ── headline: both measures indexed to the shortest context ──────────────
    style_axis(ax_a, xs, labels)
    ax_a.axhline(100, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax_a.plot(xs, acc_idx, color=BLUE, linewidth=2, marker="o", markersize=8,
              markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
              label="Acceptance length")
    ax_a.plot(xs, tps_idx, color=ORANGE, linewidth=2, marker="o", markersize=8,
              markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
              label="Decode throughput")
    ax_a.set_title(
        "Long-context slowdown is not an acceptance problem",
        color=INK, fontsize=13.5, fontweight="bold", loc="left", pad=10,
    )
    ax_a.set_ylabel(f"% of {labels[0]} value", color=INK_2, fontsize=10)
    ax_a.set_ylim(0, 118)
    ax_a.set_yticks([0, 25, 50, 75, 100])
    ax_a.set_yticklabels(["0", "25", "50", "75", "100"])

    # Direct-label the endpoints (2 series, so also a legend -- identity is
    # never carried by color alone).
    ax_a.annotate(f"{acc_idx[-1]:.0f}%", (xs[-1], acc_idx[-1]), textcoords="offset points",
                  xytext=(10, 4), color=BLUE, fontsize=10, fontweight="bold")
    ax_a.annotate(f"{tps_idx[-1]:.0f}%", (xs[-1], tps_idx[-1]), textcoords="offset points",
                  xytext=(10, -4), color=ORANGE, fontsize=10, fontweight="bold")
    leg = ax_a.legend(loc="lower left", frameon=False, fontsize=10, handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(INK_2)

    ax_a.annotate(
        f"Over a {ctx_mult:.0f}× context increase, acceptance loses only "
        f"{100 - acc_idx[-1]:.0f}% while throughput loses {100 - tps_idx[-1]:.0f}%.",
        xy=(0.5, -0.20), xycoords="axes fraction", ha="center",
        color=INK_2, fontsize=9.5,
    )

    # ── absolute acceptance ──────────────────────────────────────────────────
    style_axis(ax_b, xs, labels)
    ax_b.plot(xs, acc, color=BLUE, linewidth=2, marker="o", markersize=8,
              markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax_b.set_title("Acceptance length", color=INK, fontsize=11.5,
                   fontweight="bold", loc="left", pad=8)
    ax_b.set_xlabel("Prompt context length (tokens)", color=INK_2, fontsize=10)
    ax_b.set_ylabel("mean accepted tokens / step", color=INK_2, fontsize=10)
    lo, hi = min(acc), max(acc)
    pad = max(0.05, (hi - lo) * 0.45)
    ax_b.set_ylim(lo - pad, hi + pad * 1.25)
    # Anchor endpoint labels inward so they don't hang over the axis spines.
    for x, v, ha, dx in ((xs[0], acc[0], "left", -4), (xs[-1], acc[-1], "right", 4)):
        ax_b.annotate(f"{v:.3f}", (x, v), textcoords="offset points",
                      xytext=(dx, 11), ha=ha, color=BLUE, fontsize=9.5,
                      fontweight="bold")
    # Flag the largest rise, if any. A rebound comparable in size to the overall
    # decline is the figure's own warning that the trend is near the noise floor,
    # so it belongs on the chart rather than only in stdout.
    rises = [(acc[i] - acc[i - 1], i) for i in range(1, len(acc)) if acc[i] > acc[i - 1]]
    if rises:
        d, i = max(rises)
        ax_b.annotate(
            f"non-monotonic\n(+{d:.3f} at {labels[i]})",
            xy=(xs[i], acc[i]), xytext=(8, 24), textcoords="offset points",
            ha="left", color=INK_MUTED, fontsize=8.5, linespacing=1.4,
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.9,
                            shrinkA=2, shrinkB=6),
        )

    # Notes sit inside the axes (the lower-left corner is empty under a
    # descending curve) so they can't be clipped by the figure margin.
    ax_b.text(
        0.03, 0.05,
        f"{span}: {acc[-1] - acc[0]:+.3f} tok "
        f"({100 * (acc[-1] - acc[0]) / acc[0]:+.1f}%)\n"
        f"ceiling {1 + args.num_spec_tokens} at k={args.num_spec_tokens}; axis zoomed",
        transform=ax_b.transAxes, ha="left", va="bottom",
        color=INK_MUTED, fontsize=9, linespacing=1.45,
    )

    # ── absolute throughput (zero-based: it's a magnitude) ───────────────────
    style_axis(ax_c, xs, labels)
    ax_c.plot(xs, tps, color=ORANGE, linewidth=2, marker="o", markersize=8,
              markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax_c.set_title("Decode throughput", color=INK, fontsize=11.5,
                   fontweight="bold", loc="left", pad=8)
    ax_c.set_xlabel("Prompt context length (tokens)", color=INK_2, fontsize=10)
    ax_c.set_ylabel("decode tok/s", color=INK_2, fontsize=10)
    ax_c.set_ylim(0, max(tps) * 1.18)
    for x, v, ha, dx in ((xs[0], tps[0], "left", -4), (xs[-1], tps[-1], "right", 4)):
        ax_c.annotate(f"{v:.0f}", (x, v), textcoords="offset points",
                      xytext=(dx, 11), ha=ha, color=ORANGE, fontsize=9.5,
                      fontweight="bold")
    ax_c.text(
        0.03, 0.05, f"{span}: {tps[0] / tps[-1]:.1f}× slower",
        transform=ax_c.transAxes, ha="left", va="bottom",
        color=INK_MUTED, fontsize=9,
    )

    fig.suptitle(
        "Acceptance vs context length — AA-LCR paired sweep",
        x=0.07, y=0.955, ha="left", color=INK, fontsize=16, fontweight="bold",
    )
    fig.text(
        0.07, 0.895,
        args.title
        or (
            f"Same {int(len(pts) and 100)} multi-document questions truncated to each length; "
            f"header and question held fixed. EAGLE3 draft, k={args.num_spec_tokens}, greedy."
        ),
        ha="left", color=INK_2, fontsize=10.5,
    )

    out_dir = args.out_dir or args.summary.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{args.stem}.png"
    pdf = out_dir / f"{args.stem}.pdf"
    fig.savefig(png, dpi=200, facecolor=SURFACE)
    fig.savefig(pdf, facecolor=SURFACE)

    print(f"{'ctx':>6}{'accept_len':>12}{'idx%':>8}{'tok/s':>9}{'idx%':>8}")
    for (x, lb, al, dt), ai, ti in zip(pts, acc_idx, tps_idx):
        print(f"{lb:>6}{al:>12.3f}{ai:>8.1f}{dt:>9.1f}{ti:>8.1f}")
    mono = all(b <= a + 1e-9 for a, b in zip(acc, acc[1:]))
    if not mono:
        worst = max(range(1, len(acc)), key=lambda i: acc[i] - acc[i - 1])
        print(
            f"\nnote: acceptance is NOT monotonic (rises at {labels[worst]}), so the "
            f"{acc[-1] - acc[0]:+.3f} tok trend is within run-to-run noise of a "
            f"single-point-per-bin measurement."
        )
    print(f"\nwrote {png}\n      {pdf}")


if __name__ == "__main__":
    main()
