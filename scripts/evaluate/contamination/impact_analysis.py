#!/usr/bin/env python3
"""Decide whether contamination actually moved the numbers -- retrain or not.

Counting contaminated items does not answer the question you care about. The
question is: *would the headline acceptance change if the contaminated items
were dropped?* This splits the already-measured per-token draft-vs-target
correctness into flagged vs clean and reports both, so the decision rests on
effect size rather than on a hit count.

Inputs:
  * ``eval_item_scores.jsonl``      from detect_contamination.py
  * ``mistake_analysis/out/*.jsonl`` per-position draft-vs-target records
    (``benchmark``, ``id``, ``ttt_step``, ``correct``) -- these already exist for
    every trained draft, so no GPU time is needed to run this.

Acceptance here is the TTT-step-0 token-level agreement rate, the same quantity
``accept_rate`` aggregates. A 95% CI comes from a bootstrap resampled **over eval
items**, not over tokens: tokens within one response are strongly correlated, so
a token-level CI would be far too narrow and would make every gap look
significant.

Usage:
    python impact_analysis.py                                    # all drafts in out/
    python impact_analysis.py --mistakes ../mistake_analysis/out/stem_code_math_900k_mistakes.jsonl
    python impact_analysis.py --include-suspect                  # treat suspect as dirty
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MISTAKES_DIR = HERE.parent / "mistake_analysis/out"

# mistake_analysis writes gpqa as "gpqa_diamond"; the eval prompt files and the
# experiment summaries call it "gpqa".
BENCH_ALIASES = {"gpqa_diamond": "gpqa"}

# Decision rule: if the clean-vs-all gap is inside this band AND the CI overlaps,
# the contamination did not move the metric enough to justify a rerun.
NEGLIGIBLE_ABS_GAP = 0.01  # 1 acceptance point

# The within-draft control needs enough flagged items on BOTH sides to say
# anything. With one item per side the clustered bootstrap has no degrees of
# freedom and returns a degenerate zero-width CI, which reads as high confidence
# while being pure noise. Below this, report the benchmark as underpowered.
MIN_GROUP_ITEMS = 3


# Which training splits each scored draft actually saw. Drives the
# difference-in-differences control below; `redhat` is the published RedHat draft,
# trained on none of our corpora, so it is a pure external control.
# Mapping per mistake_analysis/README.md (--draft-checkpoint per --out file).
DRAFT_SPLITS = {
    "run1_mistakes.jsonl": {"kimi"},
    "kimi_mtp_stem_mistakes.jsonl": {"kimi", "nemotron_stem"},
    "stem_mistakes.jsonl": {"kimi", "nemotron_stem"},
    "stem_code_mistakes.jsonl": {"kimi", "nemotron_stem", "nemotron_code"},
    "gemma4_kimi_mtp_stem_code.jsonl": {"kimi", "nemotron_stem", "nemotron_code"},
    "stem_code_math_900k_mistakes.jsonl": {
        "kimi",
        "nemotron_stem",
        "nemotron_code",
        "nemotron_math",
    },
    "redhat_mistakes.jsonl": set(),
}


def norm_bench(name: str) -> str:
    return BENCH_ALIASES.get(name, name)


def load_flags(
    scores_path: Path, include_suspect: bool
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], set[str]]]:
    """Returns (verdict per item, dirty-attributed splits per item)."""
    flags: dict[tuple[str, str], str] = {}
    splits: dict[tuple[str, str], set[str]] = {}
    with scores_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            verdict = row["verdict"]
            if verdict == "suspect" and not include_suspect:
                verdict = "clean"
            elif verdict == "suspect":
                verdict = "dirty"
            key = (norm_bench(row["benchmark"]), str(row["id"]))
            flags[key] = verdict
            # `splits` is the strength-aware attribution; fall back to any-hit
            # when a suspect item is being folded in and has no dirty split.
            splits[key] = set(row.get("splits") or row.get("splits_any_hit") or [])
    return flags, splits


def apply_adjudication(
    flags: dict[tuple[str, str], str], path: Path
) -> tuple[dict[tuple[str, str], str], int]:
    """Overlay hand-checked verdicts (tier 4) onto the automatic ones.

    The detector reports candidates; a human decides. Recording those decisions
    in a file keeps the final numbers reproducible instead of resting on a
    reviewer's memory, and makes it obvious which items were actually inspected.
    Format: JSONL of {"benchmark", "id", "verdict": "dirty"|"clean", "note"}.
    """
    applied = 0
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            row = json.loads(line)
            key = (norm_bench(row["benchmark"]), str(row["id"]))
            verdict = row["verdict"]
            if verdict not in ("dirty", "clean"):
                raise SystemExit(f"bad adjudicated verdict {verdict!r} for {key}")
            if flags.get(key) != verdict:
                applied += 1
            flags[key] = verdict
    return flags, applied


def load_item_acceptance(path: Path, ttt_step: int) -> dict[tuple[str, str], tuple[int, int]]:
    """(benchmark, id) -> (n_correct, n_total) at the requested TTT depth."""
    acc: dict[tuple[str, str], list[int]] = collections.defaultdict(lambda: [0, 0])
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("ttt_step") != ttt_step:
                continue
            key = (norm_bench(row["benchmark"]), str(row["id"]))
            acc[key][1] += 1
            if row.get("correct"):
                acc[key][0] += 1
    return {k: (v[0], v[1]) for k, v in acc.items()}


def bootstrap_ci(
    items: list[tuple[int, int]], iters: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% CI for the pooled rate, resampling whole items (clustered bootstrap)."""
    if not items:
        return (float("nan"), float("nan"))
    correct = np.array([c for c, _ in items], dtype=np.float64)
    total = np.array([t for _, t in items], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(items), size=(iters, len(items)))
    rates = correct[draws].sum(axis=1) / np.maximum(total[draws].sum(axis=1), 1)
    return (float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5)))


def bootstrap_diff_ci(
    a: list[tuple[int, int]], b: list[tuple[int, int]], iters: int = 2000, seed: int = 1
) -> tuple[float, float]:
    """95% CI for rate(a) - rate(b), resampling each group's items independently."""
    if not a or not b:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)

    def draw(group: list[tuple[int, int]]) -> np.ndarray:
        correct = np.array([c for c, _ in group], dtype=np.float64)
        total = np.array([t for _, t in group], dtype=np.float64)
        picks = rng.integers(0, len(group), size=(iters, len(group)))
        return correct[picks].sum(axis=1) / np.maximum(total[picks].sum(axis=1), 1)

    diffs = draw(a) - draw(b)
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def pooled(items: list[tuple[int, int]]) -> float:
    total = sum(t for _, t in items)
    return sum(c for c, _ in items) / total if total else float("nan")


def analyze(
    mistakes_path: Path, flags: dict[tuple[str, str], str], ttt_step: int
) -> list[dict]:
    acc = load_item_acceptance(mistakes_path, ttt_step)
    by_bench: dict[str, list[tuple[str, tuple[int, int]]]] = collections.defaultdict(list)
    for (bench, item_id), counts in acc.items():
        by_bench[bench].append((item_id, counts))

    rows = []
    for bench, entries in sorted(by_bench.items()):
        allv = [c for _, c in entries]
        dirty = [c for i, c in entries if flags.get((bench, i)) == "dirty"]
        clean = [c for i, c in entries if flags.get((bench, i)) != "dirty"]
        unknown = [i for i, _ in entries if (bench, i) not in flags]

        row = {
            "benchmark": bench,
            "n_items": len(entries),
            "n_dirty": len(dirty),
            "n_clean": len(clean),
            "n_unscored": len(unknown),
            "accept_all": pooled(allv),
            "accept_clean": pooled(clean),
            "accept_dirty": pooled(dirty) if dirty else float("nan"),
        }
        row["gap_all_minus_clean"] = row["accept_all"] - row["accept_clean"]
        row["ci_clean"] = bootstrap_ci(clean)
        row["ci_all"] = bootstrap_ci(allv)
        row["ci_dirty"] = bootstrap_ci(dirty) if dirty else (float("nan"), float("nan"))

        # The pooled all-vs-clean gap is diluted by the clean majority, so it can
        # look like noise while memorization is plainly present. Test dirty vs
        # clean directly: it answers "did the draft memorize?" rather than "did
        # the headline move?", and it is the number that will bite as the
        # contaminated fraction grows.
        row["dirty_minus_clean"] = (
            row["accept_dirty"] - row["accept_clean"] if dirty else float("nan")
        )
        row["ci_dirty_minus_clean"] = (
            bootstrap_diff_ci(dirty, clean) if dirty and clean else (float("nan"), float("nan"))
        )
        rows.append(row)
    return rows


def control_report(
    paths: list[Path],
    flags: dict[tuple[str, str], str],
    item_splits: dict[tuple[str, str], set[str]],
    ttt_step: int,
) -> dict:
    """Difference-in-differences: is the dirty-vs-clean gap *caused* by training?

    A raw dirty-vs-clean gap is confounded. Flagged items are not a random
    sample -- e.g. the LiveCodeBench items that match Nemotron's code split are
    Codeforces problems, whose formulaic "Input / Output / Examples" boilerplate
    is intrinsically easier for *any* draft to predict. So a positive gap can
    appear in a draft that never saw the contaminating data.

    The fix is a control: measure the same gap on drafts that were NOT trained on
    the split the item is attributed to. If exposed and unexposed drafts show the
    same gap, the elevation is a property of the items, not memorization, and no
    retrain is warranted. Only the *excess* gap over the control is attributable
    to contamination.
    """
    per_bench: dict[str, list[dict]] = collections.defaultdict(list)

    for path in paths:
        seen = DRAFT_SPLITS.get(path.name)
        if seen is None:
            continue
        acc = load_item_acceptance(path, ttt_step)
        grouped: dict[str, list[tuple[str, tuple[int, int]]]] = collections.defaultdict(list)
        for (bench, item_id), counts in acc.items():
            grouped[bench].append((item_id, counts))

        for bench, entries in grouped.items():
            clean = [c for i, c in entries if flags.get((bench, i)) != "dirty"]
            dirty = [
                (i, c) for i, c in entries if flags.get((bench, i)) == "dirty"
            ]
            # Split the flagged items by whether THIS draft trained on the split
            # the item is attributed to. Both groups are equally "benchmark-like",
            # so the intrinsic-easiness confound cancels and the only remaining
            # difference is exposure. This holds the draft fixed, which the
            # cross-draft comparison cannot.
            exposed = [c for i, c in dirty if item_splits.get((bench, i), set()) & seen]
            unexposed = [c for i, c in dirty if not (item_splits.get((bench, i), set()) & seen)]
            if not dirty or not clean:
                continue
            per_bench[bench].append(
                {
                    "draft": path.name,
                    "saw_splits": sorted(seen),
                    "n_dirty": len(dirty),
                    "n_exposed": len(exposed),
                    "n_unexposed": len(unexposed),
                    "exposed_frac": len(exposed) / len(dirty),
                    # "Treated" = trained on the splits behind most of this
                    # benchmark's flagged items. Classifying on `any exposure`
                    # would label a draft treated on the strength of a single
                    # item and collapse the treated/control contrast.
                    "exposed": len(exposed) / len(dirty) >= 0.5,
                    "accept_clean": pooled(clean),
                    "accept_dirty": pooled([c for _, c in dirty]),
                    "accept_exposed": pooled(exposed) if exposed else float("nan"),
                    "accept_unexposed": pooled(unexposed) if unexposed else float("nan"),
                    "gap": pooled([c for _, c in dirty]) - pooled(clean),
                    "ci_gap": bootstrap_diff_ci([c for _, c in dirty], clean),
                    # The within-draft control statistic.
                    "exposed_minus_unexposed": (
                        pooled(exposed) - pooled(unexposed)
                        if exposed and unexposed
                        else float("nan")
                    ),
                    "ci_exposed_minus_unexposed": (
                        bootstrap_diff_ci(exposed, unexposed)
                        if exposed and unexposed
                        else (float("nan"), float("nan"))
                    ),
                }
            )

    out = {}
    for bench, rows in sorted(per_bench.items()):
        treated = [r["gap"] for r in rows if r["exposed"]]
        control = [r["gap"] for r in rows if not r["exposed"]]
        did = (
            (sum(treated) / len(treated)) - (sum(control) / len(control))
            if treated and control
            else float("nan")
        )
        # Only drafts with enough items on both sides carry usable signal.
        within = [
            r["exposed_minus_unexposed"]
            for r in rows
            if r["exposed_minus_unexposed"] == r["exposed_minus_unexposed"]
            and min(r["n_exposed"], r["n_unexposed"]) >= MIN_GROUP_ITEMS
        ]
        out[bench] = {
            "rows": rows,
            "mean_gap_exposed": (sum(treated) / len(treated)) if treated else float("nan"),
            "mean_gap_control": (sum(control) / len(control)) if control else float("nan"),
            "did_excess_gap": did,
            "n_exposed_drafts": len(treated),
            "n_control_drafts": len(control),
            "mean_within_draft_excess": (
                sum(within) / len(within) if within else float("nan")
            ),
            "n_within_draft": len(within),
        }
    return out


def print_control(control: dict) -> None:
    for bench, block in control.items():
        print(f"### difference-in-differences: {bench}")
        header = (
            f"{'draft':<38}{'treated':>8}{'exposure':>10}{'acc_clean':>11}"
            f"{'acc_dirty':>11}{'gap':>9}   95% CI (dirty-clean)"
        )
        print(header)
        print("-" * (len(header) + 6))
        for row in sorted(block["rows"], key=lambda r: (not r["exposed"], r["draft"])):
            lo, hi = row["ci_gap"]
            print(
                f"{row['draft']:<38}{'YES' if row['exposed'] else 'no':>8}"
                f"{row['n_exposed']:>6}/{row['n_dirty']:<4}{row['accept_clean']:>11.4f}"
                f"{row['accept_dirty']:>11.4f}{row['gap']:>+9.4f}"
                f"   [{lo:+.4f}, {hi:+.4f}]"
            )

        within = block["mean_within_draft_excess"]
        print(
            f"\n  [primary] within-draft excess, exposed-dirty vs unexposed-dirty: "
            f"{within:+.4f} (n={block['n_within_draft']} drafts)"
        )
        print(
            f"  [secondary] cross-draft DiD vs never-exposed control            : "
            f"{block['did_excess_gap']:+.4f}"
            f"  (exposed {block['mean_gap_exposed']:+.4f} / control {block['mean_gap_control']:+.4f})"
        )

        # The within-draft statistic is the one to act on: it holds the draft
        # fixed, so item difficulty cancels. The cross-draft DiD is reported for
        # context only -- it compares different models on different item sets and
        # is confounded by every other difference between them, so it must not
        # override the primary. Only a POSITIVE excess indicates inflation; a
        # negative one means the flagged items are simply harder for the exposed
        # drafts, which is evidence against memorization, not for it.
        # Prefer the within-draft statistic; fall back to the cross-draft DiD,
        # which is usable whenever there is at least one treated and one control
        # draft (weaker, since it compares different models).
        primary, source = within, "within-draft"
        if primary != primary:
            # The DiD needs enough flagged items to mean anything too: with a
            # single item the "gap" is one response's acceptance minus a pooled
            # baseline, and differences in overall draft quality dominate it.
            enough = max((r["n_dirty"] for r in block["rows"]), default=0) >= MIN_GROUP_ITEMS
            primary = block["did_excess_gap"] if enough else float("nan")
            source = "cross-draft DiD"
        if primary != primary:
            print(
                f"  -> INCONCLUSIVE (underpowered): no draft has >={MIN_GROUP_ITEMS} flagged\n"
                "     items on both sides, and there is no treated/control contrast.\n"
                "     Decide by manual adjudication of review_pairs.jsonl, or widen the\n"
                "     evaluated sample and re-score."
            )
        elif primary < NEGLIGIBLE_ABS_GAP:
            print(
                f"  -> NO RETRAIN (via {source}): exposure buys no measurable acceptance on\n"
                "     the flagged items -- drafts never trained on them show the same\n"
                "     elevation -- so it is a property of the items (formulaic benchmark\n"
                "     phrasing), not memorization of training data."
            )
        else:
            print(
                f"  -> INVESTIGATE (via {source}): exposed drafts gain beyond the control;\n"
                "     contamination is plausibly inflating this benchmark."
            )
        print()


def verdict_for(row: dict) -> str:
    if row["n_dirty"] == 0:
        return "NO ACTION (no contaminated item was evaluated)"
    gap = abs(row["gap_all_minus_clean"])
    lo_all, hi_all = row["ci_all"]
    lo_cl, hi_cl = row["ci_clean"]
    overlap = not (hi_all < lo_cl or hi_cl < lo_all)
    if gap < NEGLIGIBLE_ABS_GAP and overlap:
        return "NO ACTION (gap within noise)"
    if overlap:
        return f"MONITOR (gap {gap:.3f} but CIs overlap)"
    return f"RERUN/REPORT-CLEAN (gap {gap:.3f}, CIs disjoint)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=HERE / "out/eval_item_scores.jsonl")
    parser.add_argument(
        "--mistakes",
        type=Path,
        nargs="*",
        help="per-token mistake JSONLs (default: every *_mistakes.jsonl in mistake_analysis/out)",
    )
    parser.add_argument("--ttt-step", type=int, default=0, help="TTT depth to score (0 = first draft head)")
    parser.add_argument("--include-suspect", action="store_true", help="count `suspect` items as dirty")
    parser.add_argument(
        "--adjudicated",
        type=Path,
        default=HERE / "adjudicated.jsonl",
        help="hand-checked verdicts overriding the detector (tier 4)",
    )
    parser.add_argument("--output", type=Path, default=HERE / "out/impact_analysis.json")
    args = parser.parse_args()

    if not args.scores.exists():
        raise SystemExit(f"missing {args.scores}; run detect_contamination.py first")

    paths = args.mistakes or sorted(MISTAKES_DIR.glob("*mistakes*.jsonl"))
    if not paths:
        raise SystemExit(f"no mistake JSONLs found in {MISTAKES_DIR}")

    flags, item_splits = load_flags(args.scores, args.include_suspect)
    if args.adjudicated and args.adjudicated.exists():
        flags, applied = apply_adjudication(flags, args.adjudicated)
        print(f"adjudication: {args.adjudicated.name} overrode {applied} verdict(s)")
    n_dirty = sum(v == "dirty" for v in flags.values())
    print(
        f"flags: {len(flags)} eval items, {n_dirty} dirty"
        f"{' (suspect folded in)' if args.include_suspect else ''}\n"
    )

    report = {}
    for path in paths:
        rows = analyze(path, flags, args.ttt_step)
        report[path.name] = rows
        print(f"### {path.name}   (ttt_step={args.ttt_step})")
        header = (
            f"{'benchmark':<16}{'items':>6}{'dirty':>6}{'acc_all':>9}{'acc_clean':>11}"
            f"{'acc_dirty':>11}{'gap':>8}   verdict"
        )
        print(header)
        print("-" * (len(header) + 20))
        for row in rows:
            print(
                f"{row['benchmark']:<16}{row['n_items']:>6}{row['n_dirty']:>6}"
                f"{row['accept_all']:>9.4f}{row['accept_clean']:>11.4f}"
                f"{row['accept_dirty']:>11.4f}{row['gap_all_minus_clean']:>+8.4f}"
                f"   {verdict_for(row)}"
            )
            line = (
                f"{'':<28}"
                f"95% CI all [{row['ci_all'][0]:.4f}, {row['ci_all'][1]:.4f}]"
                f"  clean [{row['ci_clean'][0]:.4f}, {row['ci_clean'][1]:.4f}]"
            )
            if row["n_dirty"]:
                lo, hi = row["ci_dirty_minus_clean"]
                # With fewer than MIN_GROUP_ITEMS items the clustered bootstrap
                # resamples the same one or two responses every draw, so the
                # interval collapses and reads as confident when it is noise.
                if min(row["n_dirty"], row["n_clean"]) < MIN_GROUP_ITEMS:
                    sig = f"underpowered (n_dirty={row['n_dirty']})"
                else:
                    sig = "SIGNIFICANT" if lo > 0 else "n.s."
                line += (
                    f"  | dirty-clean {row['dirty_minus_clean']:+.4f} "
                    f"[{lo:+.4f}, {hi:+.4f}] {sig}"
                )
            print(line)
        print()

    control = control_report(paths, flags, item_splits, args.ttt_step)
    if control:
        print_control(control)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"per_draft": report, "control": control}, indent=2)
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
