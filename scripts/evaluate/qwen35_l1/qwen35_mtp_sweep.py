#!/usr/bin/env python3
"""Replay the l1-v2-gpt56-test set against a vLLM server and read real
acceptance from vLLM's own Prometheus counters.

AL/AR come from spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total
deltas over the run, not from anything this script infers -- the same source
the 26B suite uses.

Each row carries its OWN max_tokens (8192) and temperature (0.2). Those are
honoured rather than overridden: the point is to measure this workload as
specified. Note temperature>0 means decoding is NOT greedy, so acceptance here
is not comparable to the greedy 26-benchmark numbers.
"""
import argparse, asyncio, hashlib, json, time
import aiohttp

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True)
ap.add_argument("--input", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--concurrency", type=int, default=32)
# The dataset specifies temperature 0.2 per row. Honouring it measures the
# workload as given, but non-greedy decoding makes each arm produce a DIFFERENT
# number of tokens (a 4% spread was observed), so throughput becomes a ratio of
# two varying quantities. Forcing temperature 0 makes lossless speculative
# decoding output-IDENTICAL to the baseline, so QPS/latency compare exactly.
ap.add_argument("--force-temperature", type=float, default=None,
                help="override every row's temperature (0 = greedy)")
ap.add_argument("--no-thinking", action="store_true",
                help=("send chat_template_kwargs={'enable_thinking': False}; "
                      "required for this target, whose template otherwise "
                      "leaves a <think> block open and makes it answer "
                      "mid-string"))
a = ap.parse_args()

# Metric names move between vLLM releases -- 0.28 exposes
# `vllm:spec_decode_num_accepted_tokens_per_pos` (a vector over draft positions)
# and not the flat names an earlier version of this script assumed, which is why
# it reported null acceptance while speculation was plainly running (+51% tok/s).
# So DISCOVER every vllm:spec_decode* series rather than hardcoding names, and
# dump what was found so a future rename is visible instead of silent.
async def scrape(sess):
    """Return {series_name: summed_value} for every vllm:spec_decode* metric."""
    out = {}
    try:
        async with sess.get(f"{a.base_url}/metrics") as r:
            text = await r.text()
    except Exception:
        return out
    for line in text.splitlines():
        if not line.startswith("vllm:spec_decode"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try: val = float(line.rsplit(" ", 1)[1])
        except Exception: continue
        out[name] = out.get(name, 0.0) + val
        # num_accepted_tokens_per_pos carries a `position` label (vLLM
        # v1/spec_decode/metrics.py: pos_labelnames = labelnames +
        # ["position"]). Summing over it throws away the per-depth curve, so
        # keep each position under its own key as well.
        # prometheus_client emits a `_created` series beside every `_total`
        # counter. Its value is a fixed creation timestamp, so after-before==0,
        # and "created" sorts BEFORE "total" -- which prepended k spurious
        # zeros to every per-position curve and shifted every depth label by k.
        # Keep only the real counter.
        if "per_pos" in name and "{" in line and not name.endswith("_created"):
            labels = line.split("{", 1)[1].rsplit("}", 1)[0]
            for part in labels.split(","):
                k, _, v = part.partition("=")
                if k.strip() == "position":
                    pos = v.strip().strip('"')
                    key = f"{name}#{int(pos):03d}"
                    out[key] = out.get(key, 0.0) + val
                    break
    return out


def acceptance(before, after):
    """AL/AR from counter deltas, tolerating either naming scheme."""
    d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in set(before) | set(after)}
    def g(*cands):
        for c in cands:
            for k, v in d.items():
                if k.endswith(c):
                    return v
        return None
    drafts   = g("num_drafts", "num_drafts_total")
    draft_t  = g("num_draft_tokens", "num_draft_tokens_total")
    # per_pos is a vector; its sum over positions is total accepted
    accepted = g("num_accepted_tokens", "num_accepted_tokens_total",
                 "num_accepted_tokens_per_pos")
    res = {"raw_deltas": {k: v for k, v in sorted(d.items()) if v and "#" not in k}}
    if drafts and drafts > 0 and accepted is not None:
        res["accept_length"] = round(1 + accepted / drafts, 4)
    if draft_t and draft_t > 0 and accepted is not None:
        res["accept_rate"] = round(accepted / draft_t, 4)

    # Per-position acceptance, by vLLM's own documented formula
    # (v1/spec_decode/metrics.py:194):
    #   vllm:spec_decode_num_accepted_tokens_per_pos_total / num_drafts_total
    # The counter increments once per draft that got at least pos+1 tokens
    # accepted (metrics.py:47), so this is a survival curve: AR@1 >= AR@2 >= ...
    # AL = 1 + sum(AR@pos), which is a useful cross-check on accept_length.
    per_pos = sorted((k, v) for k, v in d.items() if "#" in k and "per_pos" in k)
    if per_pos and drafts and drafts > 0:
        res["accepted_per_pos"] = [int(v) for _, v in per_pos]
        res["ar_per_pos"] = [round(v / drafts, 4) for _, v in per_pos]
        res["al_from_per_pos"] = round(1 + sum(v / drafts for _, v in per_pos), 4)
    return res


rows = [json.loads(l) for l in open(a.input)]
if a.limit: rows = rows[:a.limit]

async def one(sess, sem, row, stats, idx):
    async with sem:
        body = {"model": a.model, "messages": row["messages"],
                "max_tokens": row.get("max_tokens", 1024),
                "temperature": (a.force_temperature
                                if a.force_temperature is not None
                                else row.get("temperature", 0.0))}
        if a.no_thinking:
            # This target's chat template leaves a <think> block open under
            # add_generation_prompt, and the model then answers mid-string:
            # 0/17 valid JSON by default, 17/17 with thinking disabled. Every
            # earlier Qwen3.5 number on this harness was measured in that
            # broken mode and is NOT comparable to runs that pass this flag.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        t0 = time.perf_counter()
        try:
            async with sess.post(f"{a.base_url}/v1/chat/completions", json=body) as r:
                status = r.status
                d = await r.json()
            if "choices" not in d:
                # record WHY -- a silent error count tells you nothing about
                # whether the run is trustworthy
                stats["errors"] += 1
                stats["error_detail"].append(f"HTTP {status}: {json.dumps(d)[:300]}")
                return
            u = d.get("usage", {})
            ct = u.get("completion_tokens", 0)
            stats["completion_tokens"] += ct
            stats["prompt_tokens"] += u.get("prompt_tokens", 0)
            fr = d["choices"][0].get("finish_reason")
            stats["finish"][fr] = stats["finish"].get(fr, 0) + 1
            stats["ok"] += 1
            lat = time.perf_counter() - t0
            stats["latency"] += lat

            # Per-request output tracking. The old harness kept only summed
            # usage and finish_reason and never looked at the text, which is
            # why a target emitting 0/2000 valid JSON went unnoticed for weeks.
            txt = (d["choices"][0].get("message") or {}).get("content") or ""
            try:
                obj = json.loads(txt)
                parses, has_interests = True, isinstance(obj, dict) and "interests" in obj
            except Exception:
                parses, has_interests = False, False
            stats["per_req"].append({
                "idx": idx,
                "tok": ct,
                "chars": len(txt),
                "lat_s": round(lat, 3),
                "finish": fr,
                # Under greedy, lossless speculative decoding must be
                # output-IDENTICAL to the no-draft baseline. The digest makes
                # that checkable after the fact instead of assumed.
                "sha1": hashlib.sha1(txt.encode("utf-8")).hexdigest(),
                # The text itself, so the FIRST point of divergence from the
                # baseline can be located. The meeting observation is that
                # speculative paths "diverge after initial tokens"; a digest
                # says THAT they diverged, not WHERE.
                "text": txt,
            })
            stats["starts_brace"] += 1 if txt.lstrip().startswith("{") else 0
            stats["parses_json"] += 1 if parses else 0
            stats["has_interests"] += 1 if has_interests else 0
        except Exception as e:
            stats["errors"] += 1
            stats["error_detail"].append(f"{type(e).__name__}: {str(e)[:200]}")

async def main():
    stats = {"ok": 0, "errors": 0, "completion_tokens": 0, "prompt_tokens": 0,
             "latency": 0.0, "error_detail": [], "finish": {},
             "starts_brace": 0, "parses_json": 0, "has_interests": 0,
             "per_req": []}
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        before = await scrape(sess)
        sem = asyncio.Semaphore(a.concurrency)
        t0 = time.perf_counter()
        await asyncio.gather(*[one(sess, sem, r, stats, i) for i, r in enumerate(rows)])
        wall = time.perf_counter() - t0
        after = await scrape(sess)
    ed = stats.pop("error_detail")
    per_req = stats.pop("per_req")
    res = {"label": a.label, "rows": len(rows), **stats,
           "error_sample": ed[:5], "error_kinds": len(set(ed)), "wall_s": round(wall, 1),
           "concurrency": a.concurrency}
    res.update(acceptance(before, after))
    res["output_tok_s"] = round(stats["completion_tokens"] / wall, 1) if wall else None

    def dist(vals):
        if not vals:
            return None
        v = sorted(vals)
        q = lambda x: v[min(len(v) - 1, int(len(v) * x))]
        return {"p10": q(.10), "p50": q(.50), "p90": q(.90), "max": v[-1],
                "mean": round(sum(v) / len(v), 1)}

    n_ok = max(1, stats["ok"])
    res["resp_tokens"] = dist([r["tok"] for r in per_req])
    res["resp_chars"] = dist([r["chars"] for r in per_req])
    res["lat_s_dist"] = dist([r["lat_s"] for r in per_req])
    res["tok_per_req"] = round(stats["completion_tokens"] / n_ok, 1)
    res["runaways"] = stats["finish"].get("length", 0)
    res["runaway_pct"] = round(100.0 * stats["finish"].get("length", 0) / n_ok, 2)
    res["valid_pct"] = round(100.0 * stats["has_interests"] / n_ok, 2)
    res["starts_brace_pct"] = round(100.0 * stats["starts_brace"] / n_ok, 2)

    # Per-row digests go to a sidecar so arms can be diffed against baseline.
    rows_path = a.out + ".rows.jsonl"
    with open(rows_path, "w") as f:
        for r in sorted(per_req, key=lambda x: x["idx"]):
            f.write(json.dumps(r) + "\n")
    res["rows_file"] = rows_path
    with open(a.out, "w") as f: json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

asyncio.run(main())
