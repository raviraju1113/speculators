"""Build a mixed-policy arm: same prompts as the OFF/ON arms, a seeded random
fraction of them answered with the on-policy reply and the rest with the original.

Models the cheap middle ground: regenerate only a slice of the corpus for a new target.
The assignment is per prompt, so the prompt set stays identical to OFF and ON and
only the response source varies.

    python make_mix.py --data-dir <abl>/data --on-frac 0.2   # -> mix20_policy.final.jsonl
"""

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--on-frac", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    off = (args.data_dir / "off_policy.final.jsonl").read_text().splitlines()
    on = (args.data_dir / "on_policy.final.jsonl").read_text().splitlines()
    assert len(off) == len(on), (len(off), len(on))

    n_on = round(args.on_frac * len(off))
    on_idx = set(random.Random(args.seed).sample(range(len(off)), n_on))

    tag = f"mix{round(100 * args.on_frac)}"
    out = args.data_dir / f"{tag}_policy.final.jsonl"
    with out.open("w") as f:
        for i, (lo, ln) in enumerate(zip(off, on)):
            ro, rn = json.loads(lo), json.loads(ln)
            assert ro["id"] == rn["id"], (i, ro["id"], rn["id"])
            f.write((ln if i in on_idx else lo) + "\n")

    stats = {"rows": len(off), "on_rows": n_on, "on_frac": args.on_frac, "seed": args.seed}
    (args.data_dir / f"{tag}_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"{out}: {stats}")


if __name__ == "__main__":
    main()
