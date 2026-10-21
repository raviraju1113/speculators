"""Drive a prompt pool through a vLLM server that runs speculative decoding with
the DSpark draft and has DSPARK_ROLLOUT_LOG set, so the engine writes one
rollout record per request (see .claude/skills/draft-opd/reference.md §2).

The server does the logging; this client only sends prompts and tags each with a
deterministic request id ("<tag>-<index>") via the X-Request-Id header, so the
rollout records can be joined back to the pool. Output text is written to
<out>/responses.jsonl for inspection; token ids come from the server log.

    python scripts/opd/rollout.py --base-url http://127.0.0.1:PORT \
        --model <backbone> --prompts <jsonl with "turns"/"prompt"/"question"> \
        --limit 200 --concurrency 8 --max-tokens 2048 --temperature 0 \
        --tag r0 --out <dir>
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp


def load_prompts(path: Path, limit: int) -> list[str]:
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "turns" in r:
            out.append(r["turns"][0])
        else:
            out.append(r.get("prompt") or r.get("question") or r.get("problem") or r["input"])
        if limit and len(out) >= limit:
            break
    return out


async def one(session, args, idx, prompt, sem, fout):
    rid = f"{args.tag}-{idx:06d}"
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": False,
    }
    async with sem:
        t0 = time.time()
        try:
            async with session.post(
                f"{args.base_url}/v1/chat/completions",
                json=body,
                headers={"X-Request-Id": rid},
                timeout=aiohttp.ClientTimeout(total=args.timeout),
            ) as resp:
                data = await resp.json()
                ok = resp.status == 200
        except Exception as e:  # noqa: BLE001
            data, ok = {"error": repr(e)}, False
        rec = {
            "request_id": rid,
            # engine-side id as the server reports it (e.g. "chatcmpl-<rid>"):
            # the rollout log is keyed by this one.
            "engine_request_id": data.get("id") if ok else None,
            "index": idx,
            "ok": ok,
            "wall_s": round(time.time() - t0, 3),
            "prompt": prompt,
            "text": (data.get("choices") or [{}])[0].get("message", {}).get("content") if ok else None,
            "usage": data.get("usage") if ok else None,
            "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason") if ok else None,
            "error": None if ok else data,
        }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        return ok


async def main_async(args):
    prompts = load_prompts(Path(args.prompts), args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    with open(out / "responses.jsonl", "w") as fout:
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                *(one(session, args, i, p, sem, fout) for i, p in enumerate(prompts))
            )
    n_ok = sum(results)
    print(f"{n_ok}/{len(prompts)} ok in {time.time() - t0:.0f}s -> {out / 'responses.jsonl'}")
    return 0 if n_ok == len(prompts) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="model name as served (backbone path)")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--tag", default="r0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
