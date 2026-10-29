#!/usr/bin/env python3
"""Paired per-prompt response-length / latency comparison vs a baseline.

Answers "does serving with this draft make the target's replies longer?" from
``mtp_eval_details.jsonl`` files: every prompt is matched by (benchmark, id)
between the baseline and each config, so the comparison is paired, not a
difference of dataset means. Repeated draws (``rep``) of one prompt are
averaged before pairing.

Per benchmark and config:
  n          paired prompts
  tok_ratio  sum(config tokens) / sum(baseline tokens)
  med_ratio  median of the per-prompt token ratios
  >1.5x/<0.67x  share of prompts whose reply is much longer / shorter than baseline
  cap        replies cut by max_tokens (baseline -> config)
  e2e_ratio  sum(config e2e_s) / sum(baseline e2e_s): wall-clock, <1 = faster
  tok/s      e2e throughput (tokens / wall-clock)
  nocap_tok_ratio  tok_ratio over prompts that hit the cap in NEITHER run

Usage:
    python compare_lengths.py baseline=results/x/baseline conv_k8=results/x/conv_k8 ...
    python compare_lengths.py --dir results/x --baseline baseline
A label's path may be the experiment dir or the details file itself.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_details(path):
    p = Path(path)
    if p.is_dir():
        p = p / "mtp_eval_details.jsonl"
    per = defaultdict(list)  # (bench, id) -> records
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            per[(r["benchmark"], r["id"])].append(r)
    out = {}
    for key, recs in per.items():
        toks = [r.get("completion_tokens") or 0 for r in recs]
        out[key] = {
            "tok": statistics.fmean(toks),
            "e2e": statistics.fmean(r["e2e_s"] for r in recs),
            "cap": sum(r.get("finish_reason") == "length" for r in recs),
            "n_rep": len(recs),
        }
    return out


def compare(base, cfg):
    rows = {}
    keys = sorted(set(base) & set(cfg))
    for bench in sorted({b for b, _ in keys}):
        bk = [k for k in keys if k[0] == bench]
        bt = sum(base[k]["tok"] for k in bk)
        ct = sum(cfg[k]["tok"] for k in bk)
        be = sum(base[k]["e2e"] for k in bk)
        ce = sum(cfg[k]["e2e"] for k in bk)
        ratios = [cfg[k]["tok"] / base[k]["tok"] for k in bk if base[k]["tok"]]
        nocap = [k for k in bk if not base[k]["cap"] and not cfg[k]["cap"]]
        nbt = sum(base[k]["tok"] for k in nocap)
        nct = sum(cfg[k]["tok"] for k in nocap)
        rows[bench] = {
            "n": len(bk),
            "base_tok": bt,
            "cfg_tok": ct,
            "base_e2e": be,
            "cfg_e2e": ce,
            "tok_ratio": ct / bt if bt else None,
            "med_ratio": statistics.median(ratios) if ratios else None,
            "gt_1_5": sum(r > 1.5 for r in ratios) / len(ratios) if ratios else None,
            "lt_0_67": sum(r < 1 / 1.5 for r in ratios) / len(ratios) if ratios else None,
            "cap_base": sum(base[k]["cap"] for k in bk),
            "cap_cfg": sum(cfg[k]["cap"] for k in bk),
            "e2e_ratio": ce / be if be else None,
            "tok_s": ct / ce if ce else None,
            "nocap_n": len(nocap),
            "nocap_tok_ratio": nct / nbt if nbt else None,
        }
    return rows


def fmt(x, spec):
    return "n/a" if x is None else format(x, spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*", help="label=path (first = baseline)")
    ap.add_argument("--dir", help="scan subdirs of DIR for mtp_eval_details.jsonl")
    ap.add_argument("--baseline", help="label/subdir of the baseline (default: first)")
    ap.add_argument("--json", type=Path, help="also write the table as JSON")
    args = ap.parse_args()

    labelled = []
    if args.dir:
        for sub in sorted(Path(args.dir).iterdir()):
            if (sub / "mtp_eval_details.jsonl").exists():
                labelled.append((sub.name, load_details(sub)))
    for spec in args.configs:
        label, _, path = spec.partition("=")
        labelled.append((label or Path(path).name, load_details(path)))
    if not labelled:
        ap.error("no details found (pass label=path args or --dir)")
    labels = [l for l, _ in labelled]
    base_label = args.baseline or labels[0]
    if base_label not in labels:
        ap.error(f"baseline {base_label!r} not among {labels}")
    base = dict(labelled)[base_label]

    table = {}
    for label, cfg in labelled:
        if label == base_label:
            continue
        table[label] = compare(base, cfg)

    benches = sorted({b for t in table.values() for b in t})
    print(f"baseline = {base_label}  (paired by (benchmark, id); repeats averaged)\n")
    hdr = (
        f"  {'config':<20}{'n':>5}{'total tok':>11}{'mean tok':>9}{'cap':>5}{'wall-clock s':>13}"
        f"{'tok_ratio':>10}{'>1.5x':>7}{'<0.67x':>7}{'e2e_ratio':>10}"
    )
    for bench in benches:
        b = [v for k, v in base.items() if k[0] == bench]
        bt, be, bcap = sum(v["tok"] for v in b), sum(v["e2e"] for v in b), sum(v["cap"] for v in b)
        print(f"=== {bench} ===")
        print(hdr)
        print(
            f"  {base_label:<20}{len(b):>5}{bt:>11,.0f}{bt / len(b):>9.0f}{bcap:>5}{be:>13,.0f}"
            f"{'1.000':>10}{'-':>7}{'-':>7}{'1.000':>10}"
        )
        for label, t in table.items():
            r = t.get(bench)
            if not r:
                continue
            print(
                f"  {label:<20}{r['n']:>5}{r['cfg_tok']:>11,.0f}{r['cfg_tok'] / r['n']:>9.0f}{r['cap_cfg']:>5}"
                f"{r['cfg_e2e']:>13,.0f}{fmt(r['tok_ratio'], '.3f'):>10}"
                f"{fmt(r['gt_1_5'], '.1%'):>7}{fmt(r['lt_0_67'], '.1%'):>7}{fmt(r['e2e_ratio'], '.3f'):>10}"
            )
        print()
    print("total/mean tok = completion tokens (repeats averaged per prompt); cap = replies cut by "
          "max_tokens; wall-clock = summed e2e seconds. tok_ratio / e2e_ratio = config / baseline, "
          "so the baseline row is 1.000; >1.5x / <0.67x = share of prompts whose reply is that much "
          "longer / shorter than the baseline's.")
    if args.json:
        args.json.write_text(json.dumps({"baseline": base_label, "table": table}, indent=2))
        print(f"json -> {args.json}")


if __name__ == "__main__":
    main()
