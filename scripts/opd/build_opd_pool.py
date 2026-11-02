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


def read_gsm8k(files):
    import pyarrow.parquet as pq

    for f in files:
        t = pq.read_table(f, columns=["question"])
        for i, x in enumerate(t.column("question").to_pylist()):
            yield f"{Path(f).stem}:{i}", x


def read_math(files):  # EleutherAI/hendrycks_math: problem/level/type/solution
    import pyarrow.parquet as pq

    for f in files:
        t = pq.read_table(f, columns=["problem"])
        for i, x in enumerate(t.column("problem").to_pylist()):
            yield f"{Path(f).parent.name}:{i}", x


def read_numina(files, sources):  # AI-MO/NuminaMath-CoT: source/problem/solution
    import pyarrow.parquet as pq

    for f in files:
        t = pq.read_table(f, columns=["source", "problem"])
        for i, (src, x) in enumerate(zip(t.column("source").to_pylist(), t.column("problem").to_pylist())):
            if src in sources:
                yield f"{src}:{Path(f).stem}:{i}", x


def read_codealpaca(files):  # sahil2801/CodeAlpaca-20k: json list of instruction/input/output
    for f in files:
        for i, r in enumerate(json.load(open(f))):
            x = r["instruction"].strip()
            if r.get("input", "").strip():
                x += "\n\n" + r["input"].strip()
            yield f"{Path(f).stem}:{i}", x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--science", nargs="+", default=[])
    ap.add_argument("--general", nargs="+", default=[])
    # Paper-domain sources (Draft-OPD, arXiv 2605.29343: 2k GSM8K train, 5k MATH,
    # 4k AoPS, 5k CodeAlpaca). Our evals are AIME/GPQA/LCB, so NuminaMath's
    # amc_aime source is never used.
    ap.add_argument("--gsm8k", nargs="+", default=[], help="openai/gsm8k main/train parquet")
    ap.add_argument("--math", nargs="+", default=[], help="EleutherAI/hendrycks_math */train parquet")
    ap.add_argument("--numina", nargs="+", default=[], help="AI-MO/NuminaMath-CoT train parquet")
    ap.add_argument("--numina-sources", default="aops_forum", help="comma-separated NuminaMath sources")
    ap.add_argument("--codealpaca", nargs="+", default=[], help="sahil2801/CodeAlpaca-20k json")
    ap.add_argument("--mix", default="gsm8k=2000,math=5000,aops=4000,codealpaca=5000",
                    help="prompts per paper-domain source (ignored for --science/--general)")
    ap.add_argument("--exclude-dataset", default=None, help="load_from_disk dir whose user turns are excluded")
    ap.add_argument("--model", default=None, help="backbone dir (tokenizer.json) for --max-prompt-tokens")
    ap.add_argument("--max-prompt-tokens", type=int, default=1800)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--count", type=int, default=16000, help="total for --science/--general only")
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
        rng = random.Random(args.seed + sum(map(ord, tag)))
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

    def expand(pats):
        return [f for p in pats for f in sorted(glob.glob(p, recursive=True))]

    pool = []
    paper = {"gsm8k": (expand(args.gsm8k), read_gsm8k), "math": (expand(args.math), read_math),
             "aops": (expand(args.numina), lambda fs: read_numina(fs, set(args.numina_sources.split(",")))),
             "codealpaca": (expand(args.codealpaca), read_codealpaca)}
    mix = {k: int(v) for k, v in (kv.split("=") for kv in args.mix.split(",") if kv)}
    for tag, (files, reader) in paper.items():
        if files:
            pool += collect(reader(files), mix.get(tag, 0), tag)
    files_s, files_g = expand(args.science), expand(args.general)
    n_s = round(args.count * args.science_frac) if files_s else 0
    n_g = args.count - n_s if files_g else 0
    if files_s or files_g:
        pool += collect(read_science(files_s), n_s, "science") + collect(read_general(files_g), n_g, "general")
    random.Random(args.seed).shuffle(pool)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")
    import collections

    print(f"wrote {len(pool)} prompts to {out}: {dict(collections.Counter(r['src'] for r in pool))}")


if __name__ == "__main__":
    main()
