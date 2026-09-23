#!/usr/bin/env python3
"""Context-length sweep: acceptance & decode speed vs prompt length.

No such sweep existed in the harness (GuideLLM sweeps request RATE, AgentX
sweeps CONCURRENCY); this adds the missing axis. Why it matters for DSpark:
the draft trains on sequences <= 8192 tokens with a 2048 sliding window, so
acceptance beyond ~8k context is extrapolation -- this measures how gracefully
it degrades.

Method: take real livecodebench problems as the task, prepend filler (other
problems' text, clearly delimited) trimmed so the CHAT PROMPT hits each target
token length; generate max-tokens continuation at concurrency 1; read
accept_len/accept_rate off vLLM's cumulative counters (before/after delta per
bucket) and decode tok/s from streaming timestamps. Reuses run_vllm_eval's
tested helpers. Accuracy of answers is NOT the point -- speculation efficiency
per context length is.

Server must be launched separately with --max-model-len > max(lengths)+max_tokens.

Usage:
  python sweep_context_len.py --base-url http://127.0.0.1:8400 \
      --lengths 1024,2048,4096,8192,16384,32768 \
      --requests 24 --max-tokens 512 --output-dir results/ctx_sweep
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_vllm_eval import (  # noqa: E402  (reuse tested plumbing)
    accept_stats,
    load_local,
    scrape_spec_counters,
    send_chat_stream,
)

TOKENIZER_PATH = "/sms-scratch/checkpoints/gemma-4-31B-it"


def build_prompts(target_len, n, tok):
    """n prompts, each ~target_len tokens: filler + delimiter + real task."""
    tasks = load_local("livecodebench", n, seed=13)
    filler_pool = " ".join(s["prompt"] for s in load_local("livecodebench", 200, seed=7))
    filler_ids = tok(filler_pool, add_special_tokens=False).input_ids
    prompts = []
    for s in tasks:
        task = ("\n\n=== END OF BACKGROUND MATERIAL ===\n"
                "Ignore the background above. Solve this problem:\n\n" + s["prompt"])
        budget = target_len - len(tok(task, add_special_tokens=False).input_ids) - 32
        if budget <= 0:
            prompts.append(task)
            continue
        # rotate through the pool so fillers differ per request
        start = (hash(s["prompt"]) % max(1, len(filler_ids) - budget))
        prompts.append(tok.decode(filler_ids[start:start + budget]) + task)
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default=TOKENIZER_PATH, help="model name for the API")
    ap.add_argument("--lengths", default="1024,2048,4096,8192,16384,32768")
    ap.add_argument("--prompts-files", default="",
                    help="comma-separated jsonl files (rows: {bucket, prompt, ...}); "
                         "each file = one bucket of REAL prompts (e.g. AA-LCR). "
                         "Overrides --lengths/synthetic padding.")
    ap.add_argument("--requests", type=int, default=24, help="per length bucket")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--output-dir", default="results/ctx_sweep")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.prompts_files:
        buckets = []
        for f in args.prompts_files.split(","):
            recs = [json.loads(line) for line in open(f)]
            buckets.append((recs[0]["bucket"], [r["prompt"] for r in recs[: args.requests]]))
    else:
        buckets = [(str(L), None) for L in args.lengths.split(",")]

    rows = []
    for L, pre_built in buckets:
        prompts = pre_built if pre_built is not None else build_prompts(int(L), args.requests, tok)
        before = scrape_spec_counters(args.base_url)
        toks, times, plens = 0, 0.0, []
        for p in prompts:
            plens.append(len(tok(p, add_special_tokens=False).input_ids))
            r = send_chat_stream(args.base_url, args.model, p,
                                 args.max_tokens, args.temperature)
            toks += r["completion_tokens"]
            times += r["decode_time_s"]
        after = scrape_spec_counters(args.base_url)
        alen, arate = accept_stats(before, after)
        row = {
            "target_len": L,  # bucket label (int for synthetic, str for prompt-files)
            "actual_prompt_len_mean": round(statistics.mean(plens)),
            "n": len(prompts),
            "accept_length": round(alen, 3) if alen else None,
            "accept_rate": round(arate, 4) if arate else None,
            "decode_tok_s": round(toks / times, 1) if times else None,
            "completion_tokens": toks,
        }
        rows.append(row)
        print(json.dumps(row))
    (out / "ctx_sweep_summary.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {out / 'ctx_sweep_summary.json'}")


if __name__ == "__main__":
    main()
