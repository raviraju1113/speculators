"""Build a Draft-OPD prompt pool (jsonl with {"turns": [prompt]}) from the user
turns of a prepared SFT dataset (its `messages` column), or from any jsonl with
a prompt-like field.

    python scripts/opd/make_prompt_pool.py --from-dataset <arrow dir> \
        --start 700000 --count 1000 --out data/opd_prompts/pilot_1k.jsonl

Prompts are taken in row order from --start (deterministic, documented), so a
pool can be chosen outside the SFT train split or from a different corpus.
NOTE: the paper uses a prompt pool DISJOINT from the SFT data; for the full run
take prompts from a source the drafter has not seen (decision pending, see
PLAN.md). Benchmark prompts (aime/gpqa/lcb) must never be used.
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dataset", required=True, help="load_from_disk() dir with a `messages` column")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--max-chars", type=int, default=6000, help="skip longer prompts")
    ap.add_argument("--model", default=None,
                    help="backbone dir (tokenizer.json); with --max-prompt-tokens, skip prompts "
                    "that would not fit the rollout context (4/1000 failed in the pilot)")
    ap.add_argument("--max-prompt-tokens", type=int, default=1800)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import load_from_disk

    tok = None
    if args.model:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))

    ds = load_from_disk(args.from_dataset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = skipped = 0
    seen = set()
    with open(out, "w") as f:
        i = args.start
        while n < args.count and i < len(ds):
            msgs = ds[i]["messages"]
            i += 1
            user = next((m["content"] for m in msgs if m["role"] == "user"), None)
            if not user or len(user) > args.max_chars or user in seen:
                skipped += 1
                continue
            if tok is not None and len(tok.encode(user).ids) > args.max_prompt_tokens:
                skipped += 1
                continue
            seen.add(user)
            f.write(json.dumps({"turns": [user], "src_row": i - 1}) + "\n")
            n += 1
    print(f"wrote {n} prompts to {out} (rows {args.start}..{i - 1}, skipped {skipped})")


if __name__ == "__main__":
    main()
