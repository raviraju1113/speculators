"""Build the Draft-OPD prompt pool from external sources, disjoint from the SFT data.

Why external: at temperature 0 a rollout on an SFT prompt reproduces (nearly)
the SFT response the draft already trained on, so on-policy data on seen
prompts is close to SFT data. The paper uses a pool disjoint from SFT. Our SFT
corpus nemo782k contains ALL of Nemotron-v2 stem/math/code (354,974/355,000 stem
prompts, checked 2026-10-27), so the pool comes from sources nemo782k never saw,
and every prompt is still checked against nemo782k's user turns by exact hash.

Sources (parquet shards downloaded with `hf download`, see README §8):
  --science <parquet...>   nvidia/OpenScienceReasoning-2: columns input/output
                           (+ expected_answer, ...); multiple-choice science
                           questions, decontaminated against GPQA by NVIDIA.
  --general <parquet...>   nvidia/Nemotron-Post-Training-Dataset-v2 `chat`:
                           messages list (user/assistant).

    python scripts/opd/build_opd_pool.py \
        --science datasets/opd_pool_raw/openscience/**/*.parquet \
        --general datasets/opd_pool_raw/nemotron_chat/data/chat-*.parquet \
        --exclude-dataset datasets/kimi_nemotron_782k --model <backbone> \
        --count 16000 --science-frac 0.75 --seed 0 --out data/opd_prompts/sci16k.jsonl

Output: jsonl {"turns": [prompt], "src": "science|general", "src_id": ...}, shuffled,
so a round of P prompts taken in order has the configured mix.
"""

import argparse
import glob
import hashlib
import json
import random
from pathlib import Path


def _h(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def read_science(files):
    import pyarrow.parquet as pq

    for f in files:
        t = pq.read_table(f)
        col = "input" if "input" in t.schema.names else "question"
        ids = t.column("id").to_pylist() if "id" in t.schema.names else range(t.num_rows)
        for i, x in zip(ids, t.column(col).to_pylist()):
            if isinstance(x, list):  # chat-format input: take the user turn
                x = next((m["content"] for m in x if m.get("role") == "user"), None)
            if x:
                yield str(i), x


def read_general(files):
    import pyarrow.parquet as pq

    for f in files:
        t = pq.read_table(f, columns=["uuid", "messages"])
        for u, ms in zip(t.column("uuid").to_pylist(), t.column("messages").to_pylist()):
            x = next((m["content"] for m in ms if m["role"] == "user"), None)
            if x:
                yield u, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--science", nargs="+", default=[])
    ap.add_argument("--general", nargs="+", default=[])
    ap.add_argument("--exclude-dataset", default=None, help="load_from_disk dir whose user turns are excluded")
    ap.add_argument("--model", default=None, help="backbone dir (tokenizer.json) for --max-prompt-tokens")
    ap.add_argument("--max-prompt-tokens", type=int, default=1800)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--science-frac", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    excl = set()
    if args.exclude_dataset:
        from datasets import load_from_disk

        ds = load_from_disk(args.exclude_dataset)
        for ms in ds.select_columns(["messages"])["messages"]:
            u = next((m["content"] for m in ms if m["role"] == "user"), None)
            if u:
                excl.add(_h(u))
        print(f"excluding {len(excl)} SFT user prompts")

    tok = None
    if args.model:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))

    def collect(gen, want, tag):
        rng = random.Random(args.seed + (1 if tag == "science" else 2))
        rows = list(gen)
        rng.shuffle(rows)
        out, seen, dropped = [], set(), {"sft": 0, "dup": 0, "long": 0}
        for i, x in rows:
            if len(out) >= want:
                break
            if len(x) > args.max_chars:
                dropped["long"] += 1
                continue
            h = _h(x)
            if h in excl:
                dropped["sft"] += 1
                continue
            if h in seen:
                dropped["dup"] += 1
                continue
            if tok is not None and len(tok.encode(x).ids) > args.max_prompt_tokens:
                dropped["long"] += 1
                continue
            seen.add(h)
            out.append({"turns": [x], "src": tag, "src_id": i})
        print(f"{tag}: kept {len(out)} of {len(rows)} candidates, dropped {dropped}")
        return out

    files_s = [f for p in args.science for f in sorted(glob.glob(p, recursive=True))]
    files_g = [f for p in args.general for f in sorted(glob.glob(p, recursive=True))]
    n_s = round(args.count * args.science_frac) if files_s else 0
    n_g = args.count - n_s if files_g else 0
    pool = collect(read_science(files_s), n_s, "science") + collect(read_general(files_g), n_g, "general")
    random.Random(args.seed).shuffle(pool)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(pool)} prompts to {out} (science {n_s}, general {n_g})")


if __name__ == "__main__":
    main()
