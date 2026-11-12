"""Quality scan of the full GLM-5.3 on-policy regen + multilingual regen, per source shard.

Same checks that exposed the looping 2-node PP=2 shard B (Nov 4): truncated at the cap, no
</think>, highly repetitive tail (zlib ratio of the last 4k chars < 0.15), completion tokens.
"""
import json
import sys
import zlib
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

D = "/import/ml-sc-scratch1/mengmengj/speculators/data/glm53"
SHARDS = {"A": "prompts_nemo782k_shardA.jsonl", "B1": "prompts_nemo782k_shardB1.jsonl",
          "B2": "prompts_nemo782k_shardB2.jsonl", "B3": "prompts_nemo782k_shardB3.jsonl"}


def ids_of(path):
    out = []
    with open(path) as f:
        for line in f:
            i = line.find('"id": "') + 7
            out.append(line[i:line.index('"', i)])
    return out


def scan(line):
    try:
        d = json.loads(line)
    except Exception:
        return None
    md = d.get("metadata", {})
    reply = "".join(t["value"] for t in d["conversations"] if t["from"] == "gpt")
    tail = reply[-4000:].encode()
    zr = len(zlib.compress(tail)) / max(len(tail), 1) if len(tail) > 1000 else 1.0
    trunc = bool(md.get("truncated")) or "length" in (md.get("finish_reasons") or [])
    ct = (md.get("usage") or {}).get("completion_tokens") or 0
    nturn = sum(t["from"] == "gpt" for t in d["conversations"])
    return d["id"], trunc, "</think>" not in reply, zr < 0.15, ct, nturn


def main():
    shard_of = {}
    for s, p in SHARDS.items():
        for i in ids_of(f"{D}/{p}"):
            shard_of[i] = s
    files = sys.argv[1:]
    for fn in files:
        stats = defaultdict(lambda: [0, 0, 0, 0, [], 0])
        with open(f"{D}/{fn}") as f, Pool(32) as pool:
            for r in pool.imap_unordered(scan, f, chunksize=256):
                if r is None:
                    stats["_bad_json"][0] += 1
                    continue
                i, tr, nothink, rep, ct, nturn = r
                g = shard_of.get(i, i.split(":")[1] if i.startswith("mling") else "other")
                for key in (g, "ALL"):
                    s = stats[key]
                    s[0] += 1; s[1] += tr; s[2] += nothink; s[3] += rep; s[4].append(ct); s[5] += nturn > 1
        print(f"\n== {fn}")
        print(f"{'group':12}{'rows':>9}{'trunc%':>8}{'no</think>%':>12}{'repTail%':>9}{'ct_med':>8}{'ct_p90':>8}{'multiTurn%':>11}")
        for g in sorted(stats, key=lambda k: (k != "ALL", k)):
            n, tr, nt, rp, cts, mt = stats[g]
            if not cts:
                print(g, n); continue
            c = np.array(cts)
            print(f"{g:12}{n:9,}{100*tr/n:8.1f}{100*nt/n:12.1f}{100*rp/n:9.1f}{np.median(c):8.0f}{np.percentile(c,90):8.0f}{100*mt/n:11.1f}")


if __name__ == "__main__":
    main()
