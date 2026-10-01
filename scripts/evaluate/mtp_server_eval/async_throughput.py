#!/usr/bin/env python3
"""Concurrent output-throughput benchmark for an OpenAI-compatible server.

Unlike run_vllm_eval.py (strictly sequential, for canonical acceptance), this
drives N requests in flight so continuous batching actually engages. Reports
aggregate output tok/s, QPS and latency percentiles per concurrency level.

    python async_throughput.py --base-url http://127.0.0.1:8000 \
        --benchmark sc1_delta --concurrency 1,8,16,32,64 --num-prompts 64
"""
import argparse, asyncio, json, random, statistics, time
from pathlib import Path
import aiohttp

DATA = Path(__file__).resolve().parent / "data"


async def one(session, url, model, prompt, max_tokens, temperature):
    t0 = time.perf_counter()
    async with session.post(
        f"{url}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": max_tokens, "temperature": temperature, "stream": False},
        timeout=aiohttp.ClientTimeout(total=3600),
    ) as r:
        body = await r.json()
    dt = time.perf_counter() - t0
    usage = body.get("usage") or {}
    return usage.get("completion_tokens", 0), dt


async def run_level(url, model, prompts, conc, max_tokens, temperature):
    sem = asyncio.Semaphore(conc)
    async with aiohttp.ClientSession() as session:
        async def guarded(p):
            async with sem:
                return await one(session, url, model, p, max_tokens, temperature)
        t0 = time.perf_counter()
        out = await asyncio.gather(*[guarded(p) for p in prompts])
        wall = time.perf_counter() - t0
    toks = [t for t, _ in out]
    lats = sorted(d for _, d in out)
    return {
        "concurrency": conc, "requests": len(out), "wall_s": round(wall, 2),
        "output_tok_s": round(sum(toks) / wall, 1),
        "qps": round(len(out) / wall, 3),
        "mean_latency_s": round(statistics.mean(lats), 2),
        "p50_s": round(lats[len(lats) // 2], 2),
        "p99_s": round(lats[min(len(lats) - 1, int(len(lats) * 0.99))], 2),
        "mean_out_tokens": round(statistics.mean(toks), 1),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--benchmark", default="sc1_delta")
    ap.add_argument("--concurrency", default="1,8,16,32,64")
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(DATA / f"{a.benchmark}.jsonl") if l.strip()]
    prompts = [r["prompt"] for r in random.Random(42).sample(rows, min(a.num_prompts, len(rows)))]

    async with aiohttp.ClientSession() as s:
        async with s.get(f"{a.base_url}/v1/models") as r:
            model = (await r.json())["data"][0]["id"]

    results = []
    for conc in [int(c) for c in a.concurrency.split(",")]:
        res = await run_level(a.base_url, model, prompts, conc, a.max_tokens, a.temperature)
        res["label"] = a.label
        results.append(res)
        print(f"  conc={conc:>3}  out_tok/s={res['output_tok_s']:>8.1f}  qps={res['qps']:>6.3f}  "
              f"mean_lat={res['mean_latency_s']:>6.2f}s  p99={res['p99_s']:>6.2f}s", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
