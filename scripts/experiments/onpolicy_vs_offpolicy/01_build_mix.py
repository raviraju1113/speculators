#!/usr/bin/env python3
"""Build the balanced registry mix. v2 — fixes four defects found in review.

Output schema matches scripts/response_regeneration/prepare_aya.py (both
`messages` role/content and `conversations` from/value), so the regeneration and
prepare_data paths consume it unchanged.

Design decisions
----------------
BALANCED, NOT PROPORTIONAL. Nemotron math alone is 22.1M rows vs science's 709k;
proportional sampling would make the mix ~67% math -- the narrowness section 12
blamed for our from-scratch DSpark losing to better-balanced published drafts.

STRIDE SAMPLING over large JSONL so the sample spans the whole file without
loading 41GB. Line count is estimated from file size / mean line length: the
dataset README's per-split counts cover several files and using them as a
per-file hint made the stride 500x too large (983 rows instead of 20000).

Fixes over v1
-------------
1. MULTI-TURN PRESERVED. v1 kept only the first user turn and its reply, which
   silently discarded the multi-turn structure of ultrachat / sharegpt / magpie /
   open-perfectblend -- the structure that matters most for chat and agentic
   serving. All turns are now kept.
2. IDS ARE CONTENT HASHES. v1 used each corpus's own id field, but aya's
   `user_id` repeats across samples (one annotator contributes thousands), so
   44,531 of 400,000 rows collided -- one id appeared 2,545 times. Regeneration
   resume keys on `id`, so collisions would make it skip real work. Ids are now
   sha256 over the full conversation, with the source as a prefix so the source
   stays recoverable per row.
3. EMPTY ASSISTANT TURNS DROPPED. The loss mask covers assistant positions only,
   so a row with no assistant content carries no gradient (0.7% of v1's rows).
4. SYSTEM PROMPTS PRESERVED. Nemotron rows carry
   `system_prompt="detailed thinking on"` and their responses begin with
   `<think>`. v1 dropped it, which would have fed the target a bare problem
   statement followed by an unprompted reasoning trace -- badly off-distribution
   for off-policy training, and it would stop the target producing reasoning at
   all under on-policy regeneration.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys
from pathlib import Path

D = "/nvmedata/data/registry_datasets"
_ROLES = {"human": "user", "user": "user", "gpt": "assistant",
          "assistant": "assistant", "system": "system", "chatgpt": "assistant"}


def _seq(x):
    """pandas returns nested fields as numpy arrays, whose truthiness raises
    ('truth value of an array with more than one element is ambiguous'), so
    normalise rather than relying on `x or []`."""
    return [] if x is None else list(x)


def norm_turns(raw, system=None):
    """Normalise any role/content or from/value conversation to [{role, content}],
    keeping EVERY turn. Unknown roles are dropped rather than guessed."""
    turns = []
    if isinstance(system, str) and system.strip():
        turns.append({"role": "system", "content": system.strip()})
    for t in _seq(raw):
        if not isinstance(t, dict):
            continue
        role = _ROLES.get(str(t.get("role") or t.get("from") or "").lower())
        content = t.get("content")
        if content is None:
            content = t.get("value")
        if role and isinstance(content, str) and content.strip():
            turns.append({"role": role, "content": content})
    return turns


def rec(turns, source):
    """A mix row, or None if it carries no trainable signal.

    Requires at least one user turn and one non-empty assistant turn: the loss
    mask covers assistant positions only, so anything else is a wasted row.
    """
    if not turns:
        return None
    if not any(t["role"] == "user" for t in turns):
        return None
    if not any(t["role"] == "assistant" and t["content"].strip() for t in turns):
        return None
    # Content hash, not the corpus's own id -- see fix 2 in the module docstring.
    digest = hashlib.sha256(
        json.dumps(turns, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:20]
    return {
        "messages": turns,
        "conversations": [
            {"from": {"user": "human", "assistant": "gpt", "system": "system"}[t["role"]],
             "value": t["content"]}
            for t in turns
        ],
        "source": source,
        "id": f"{source}_{digest}",
        "num_turns": len(turns),
    }


def _estimate_lines(path, probe=2000):
    size = os.path.getsize(path)
    with open(path) as f:
        lens = [len(f.readline()) for _ in range(probe)]
    lens = [n for n in lens if n]
    return int(size / (sum(lens) / len(lens))) if lens else 0


def stream_nemotron(path, cap, source):
    """Nemotron SFT: `input` is the user turn (sometimes a message list),
    `output` the response, `system_prompt` controls reasoning -- keep all three."""
    est = _estimate_lines(path)
    stride = max(1, est // cap) if est else 1
    out, i = [], 0
    with open(path) as f:
        for line in f:
            if i % stride == 0:
                try:
                    r = json.loads(line)
                except Exception:
                    i += 1
                    continue
                inp = r.get("input")
                raw = inp if isinstance(inp, list) else [{"role": "user", "content": inp}]
                raw = list(raw) + [{"role": "assistant", "content": r.get("output")}]
                g = rec(norm_turns(raw, system=r.get("system_prompt")), source)
                if g:
                    out.append(g)
                    if len(out) >= cap:
                        break
            i += 1
    return out


def parquet_rows(pattern, cap, source, to_turns):
    import pandas as pd
    out = []
    for fp in sorted(glob.glob(pattern, recursive=True)):
        for row in pd.read_parquet(fp).to_dict(orient="records"):
            g = rec(to_turns(row), source)
            if g:
                out.append(g)
                if len(out) >= cap:
                    return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=50000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle the final mix (sources are built contiguously, "
                         "which would show training one domain at a time)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    N = args.per_source

    jobs = {
      "nemotron-math":    lambda: stream_nemotron(f"{D}/nemotron_post_training/SFT/math/math_v1.1.jsonl", N, "nemotron-math"),
      "nemotron-code":    lambda: stream_nemotron(f"{D}/nemotron_post_training/SFT/code/code_v1.1.jsonl", N, "nemotron-code"),
      "nemotron-science": lambda: stream_nemotron(f"{D}/nemotron_post_training/SFT/science/science.jsonl", N, "nemotron-science"),
      # OpenCodeInstruct is single-turn by construction (input/output columns).
      "opencodeinstruct": lambda: parquet_rows(f"{D}/OpenCodeInstruct/**/*.parquet", N, "opencodeinstruct",
                              lambda r: norm_turns([{"role": "user", "content": r.get("input")},
                                                    {"role": "assistant", "content": r.get("output")}])),
      "magpie":            lambda: parquet_rows(f"{D}/magpie_pro_300k/**/*.parquet", N, "magpie",
                              lambda r: norm_turns(r.get("conversations"))),
      "ultrachat":         lambda: parquet_rows(f"{D}/ultrachat_200k/**/*.parquet", N, "ultrachat",
                              lambda r: norm_turns(r.get("messages"))),
      "open-perfectblend": lambda: parquet_rows(f"{D}/open-perfectblend/**/*.parquet", N, "open-perfectblend",
                              lambda r: norm_turns(r.get("conversations"))),
      "aya":               lambda: parquet_rows(f"{D}/aya_dataset/**/*.parquet", N, "aya",
                              lambda r: norm_turns([{"role": "user", "content": r.get("inputs")},
                                                    {"role": "assistant", "content": r.get("targets")}])),
    }

    def sharegpt():
        out = []
        for fp in sorted(glob.glob(f"{D}/sharegpt_vicuna/**/sg_90k_part*.json", recursive=True)):
            if "html_cleaned" in fp:
                continue
            for r in json.load(open(fp)):
                g = rec(norm_turns(r.get("conversations")), "sharegpt")
                if g:
                    out.append(g)
                    if len(out) >= N:
                        return out
        return out
    jobs["sharegpt"] = sharegpt

    want = args.only.split(",") if args.only else list(jobs)
    rows, counts = [], {}
    for name in want:
        if name not in jobs:
            print(f"  !! unknown source {name}"); continue
        try:
            got = jobs[name]()
        except Exception as e:
            print(f"  !! {name}: {type(e).__name__}: {e}"); continue
        counts[name] = len(got)
        rows.extend(got)
        print(f"{name:20s} {len(got):>8,}", flush=True)

    # Global dedup: the same conversation can appear in more than one corpus
    # (open-perfectblend is itself a blend), and a duplicate prompt is wasted
    # regeneration plus a train/val leak once prepare_data splits.
    seen, uniq = set(), []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"]); uniq.append(r)
    print(f"\ndedup: {len(rows):,} -> {len(uniq):,} ({len(rows)-len(uniq):,} removed)")

    if args.shuffle:
        import random
        random.Random(args.seed).shuffle(uniq)
        print("shuffled")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(counts, open(args.out + ".counts.json", "w"), indent=2)
    print(f"TOTAL {len(uniq):,} -> {args.out}")


main()
