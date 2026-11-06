#!/usr/bin/env python
"""Markdown tables for the GLM-5.3 DFlash2 eval results in this directory.

Reads <suite>/<experiment>/mtp_eval_summary.json (copied from the eval output dirs) and
prints, per suite: a summary table (mean accept length, mean decode speedup vs the suite's
``baseline`` experiment when present) and a per-benchmark accept-length table with the
speedup in brackets.

    python examples/train/glm53/results/make_tables.py [suite ...]
"""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Column order for the experiments we ran; anything else is appended alphabetically.
ORDER = [
    "ours_ep9_k6", "ftA_ep0.5_k6", "ftA_ep1_k6", "ftA_ep2best_k6", "incoai_k7",
    "ours_ep9_k6_greedy", "incoai_k7_greedy",
    "ft_ep0.5_k6", "ft_ep0.5_k6_greedy", "ft_ep1_k6", "ft_ep1_k6_greedy",
    "ft_ep2_k6", "ft_ep2_k6_greedy", "ft_ep3_k6", "ft_ep3_k6_greedy",
]


def load(suite: Path) -> dict[str, dict[str, dict]]:
    out = {}
    for f in sorted(suite.glob("*/mtp_eval_summary.json")):
        out[f.parent.name] = {r["benchmark"]: r for r in json.loads(f.read_text())}
    return out


def table(suite: Path) -> str:
    runs = load(suite)
    base = runs.pop("baseline", None)
    names = [n for n in ORDER if n in runs] + sorted(set(runs) - set(ORDER))
    benches = list(next(iter(runs.values())))

    def speedup(n, b):
        if not base or b not in base:
            return None
        return runs[n][b]["decode_tok_s"] / base[b]["decode_tok_s"]

    lines = [f"### {suite.name}", ""]
    head = "| draft | mean accept_len |" + (" mean speedup |" if base else "")
    lines += [head, "|---|---:|" + ("---:|" if base else "")]
    for n in names:
        al = statistics.mean(r["accept_length"] for r in runs[n].values())
        row = f"| {n} | {al:.2f} |"
        if base:
            row += f" {statistics.mean(speedup(n, b) for b in runs[n]):.2f}x |"
        lines.append(row)
    if base:
        tok = statistics.mean(r["decode_tok_s"] for r in base.values())
        lines.append(f"| baseline (no spec) | -- | 1.00x ({tok:.0f} tok/s) |")

    lines += ["", "Accept length per benchmark" + (" (decode speedup vs baseline)" if base else "") + ":", ""]
    lines.append("| benchmark | n | " + " | ".join(names) + (" | baseline tok/s |" if base else " |"))
    lines.append("|---|---:|" + "---:|" * len(names) + ("---:|" if base else ""))
    for b in benches:
        cells = []
        for n in names:
            r = runs[n].get(b)
            if r is None:
                cells.append("--")
                continue
            s = speedup(n, b)
            cells.append(f"{r['accept_length']:.2f}" + (f" ({s:.2f}x)" if s else ""))
        row = f"| {b} | {runs[names[0]][b]['n']} | " + " | ".join(cells) + " |"
        if base:
            row += f" {base[b]['decode_tok_s']:.0f} |"
        lines.append(row)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    suites = [HERE / s for s in sys.argv[1:]] or sorted(p for p in HERE.iterdir() if p.is_dir())
    print("\n".join(table(s) for s in suites))
