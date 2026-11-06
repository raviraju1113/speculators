#!/usr/bin/env python3
"""Where does the per-position-decay intuition break?

For every (run, epoch, benchmark) with both a k=3 and a native-k (7/8) eval,
compare per-proposed-token acceptance rates. Intuition says rate@k3 >= rate@k_nat
(early positions are easier). Rows where that INVERTS are flagged, alongside the
mean completion length of both servings -- a large length gap means the two runs
generated different text (greedy spec decode is only deterministic in exact
arithmetic), which voids the same-distribution assumption behind the intuition.

Usage: python analyze_k_inversion.py    (from scripts/evaluate/experiments/)
"""

import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"

# (flag, epoch, k3 experiment dir, native experiment dir, native k)
PAIRS = [
    ("False", 3, "gemma4-31b-dspark-anchor-ablation/main_ep3end_k3",
     "gemma4-31b-dspark-anchor-ablation/main_ep3end_k7", 7),
    ("False", 7, "gemma4-31b-dspark-anchor-ablation-ep7/main_ep7end_k3",
     "gemma4-31b-dspark-anchor-ablation-ep7/main_ep7end_k7", 7),
    ("False", 9, "gemma4-31b-dspark-final/ours_ep9_k3",
     "gemma4-31b-dspark-final/ours_ep9_k7", 7),
    ("False", 14, "gemma4-31b-dspark-final/ours_ep14_k3",
     "gemma4-31b-dspark-final/ours_ep14_k7", 7),
    ("True", 3, "gemma4-31b-dspark-anchor-ablation/anchorT_ep3_k3",
     "gemma4-31b-dspark-anchor-ablation/anchorT_ep3_k8", 8),
    ("True", 7, "gemma4-31b-dspark-anchor-ablation-ep7/anchorT_ep7_k3",
     "gemma4-31b-dspark-anchor-ablation-ep7/anchorT_ep7_k8", 8),
    ("True", 9, "gemma4-31b-dspark-anchorT-final/anchorT_ep9_k3",
     "gemma4-31b-dspark-anchorT-final/anchorT_ep9_k8", 8),
    ("True", 14, "gemma4-31b-dspark-anchorT-final/anchorT_ep14_k3",
     "gemma4-31b-dspark-anchorT-final/anchorT_ep14_k8", 8),
]


def summaries(exp):
    return {r["benchmark"]: r for r in json.load(open(R / exp / "mtp_eval_summary.json"))}


def mean_completion(exp, bench):
    rows = [json.loads(line) for line in open(R / exp / "mtp_eval_details.jsonl")]
    toks = [r["completion_tokens"] for r in rows if r["benchmark"] == bench]
    return sum(toks) / len(toks) if toks else float("nan")


hdr = (f'{"flag":<6}{"ep":<4}{"bench":<15}{"rate@k3":>8}{"rate@nat":>9}'
       f'{"inv?":>6}{"tok@k3":>8}{"tok@nat":>8}{"len gap":>8}')
print(hdr)
print("-" * len(hdr))
inversions = 0
for flag, ep, e3, en, kn in PAIRS:
    s3, sn = summaries(e3), summaries(en)
    for b in ("aime", "gpqa", "livecodebench"):
        r3, rn = s3[b]["accept_rate"], sn[b]["accept_rate"]  # counter-based, per serving
        inv = r3 < rn
        inversions += inv
        t3, tn = mean_completion(e3, b), mean_completion(en, b)
        gap = abs(t3 - tn) / max(t3, tn) * 100
        print(f'{flag:<6}{ep:<4}{b:<15}{r3:>8.3f}{rn:>9.3f}{"  <<" if inv else "":>6}'
              f'{t3:>8.0f}{tn:>8.0f}{gap:>7.0f}%')
print(f"\ninversions (rate@k3 < rate@native): {inversions} of {len(PAIRS) * 3} rows")
print("expectation: 0 if per-position decay held across servings with identical text")
