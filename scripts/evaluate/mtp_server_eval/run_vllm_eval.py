#!/usr/bin/env python3
"""EAGLE3 / MTP acceptance / speedup eval for a vLLM server.

The vLLM counterpart of ``run_sglang_eval.py``. Same benchmark data (the shared
``data/`` dir) and same output schema (mtp_eval_summary.json /
mtp_eval_details.jsonl), so ``compare_speedup.py`` works unmodified on either
backend's output. The one thing that genuinely differs is HOW acceptance is read:

SGLang exposes *windowed* Prometheus gauges (sglang:spec_accept_length/rate)
that reset every decode_log_interval steps, so that eval polls /metrics in a
background thread for the whole run and averages window values.

vLLM instead exposes *cumulative* Prometheus Counters:
    vllm:spec_decode_num_drafts_total
    vllm:spec_decode_num_draft_tokens_total
    vllm:spec_decode_num_accepted_tokens_total
(see vllm/v1/spec_decode/metrics.py). Because they only grow, a single
before/after read brackets exactly this benchmark's contribution -- no
polling thread or windowing/deduping needed. Per vLLM's own documented
formulas:
    accept_length (incl. bonus) = 1 + accepted_tokens / num_drafts
    accept_rate                 = accepted_tokens / draft_tokens
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent / "data"

DATA_FILES = {
    "aime": "aime.jsonl",
    "gpqa": "gpqa_diamond.jsonl",
    "livecodebench": "livecodebench.jsonl",
    "gsm8k": "gsm8k.jsonl",
    "math500": "math500.jsonl",
    "humaneval": "humaneval.jsonl",
    "mbpp": "mbpp.jsonl",
    # Inferact/Kimi-K3-DSpark acceptance-suite additions
    "mt-bench": "mt-bench.jsonl",
    "aime26": "aime26.jsonl",
    "swe-bench-pro": "swe-bench-pro.jsonl",
    "swe-rebench": "swe-rebench.jsonl",
    "aa-lcr": "aa-lcr.jsonl",
    # AA-LCR context-length sweep: same 100 questions truncated to each length,
    # so acceptance differences isolate context length (../prepare_aa_lcr_sweep.py).
    # Needs max_model_len >= bin + max_tokens.
    "aa-lcr-1k": "aa-lcr-1k.jsonl",
    "aa-lcr-2k": "aa-lcr-2k.jsonl",
    "aa-lcr-4k": "aa-lcr-4k.jsonl",
    "aa-lcr-8k": "aa-lcr-8k.jsonl",
    "aa-lcr-16k": "aa-lcr-16k.jsonl",
    "aa-lcr-32k": "aa-lcr-32k.jsonl",
    "aa-lcr-64k": "aa-lcr-64k.jsonl",
    "aa-lcr-96k": "aa-lcr-96k.jsonl",
    "aa-lcr-112k": "aa-lcr-112k.jsonl",
    # SPEED-Bench qualitative: all 11 categories x 80 prompts (see ../README.md)
    "speed-coding": "speed-coding.jsonl",
    "speed-humanities": "speed-humanities.jsonl",
    "speed-math": "speed-math.jsonl",
    "speed-multilingual": "speed-multilingual.jsonl",
    "speed-qa": "speed-qa.jsonl",
    "speed-rag": "speed-rag.jsonl",
    "speed-reasoning": "speed-reasoning.jsonl",
    "speed-roleplay": "speed-roleplay.jsonl",
    "speed-stem": "speed-stem.jsonl",
    "speed-summarization": "speed-summarization.jsonl",
    "speed-writing": "speed-writing.jsonl",
    # SPEED-Bench throughput slice
    "speed-low-entropy": "speed-low-entropy.jsonl",
    # RedHatAI/speculator_benchmarks (former evaluate.py defaults)
    "HumanEval": "HumanEval.jsonl",
    "math_reasoning": "math_reasoning.jsonl",
    "qa": "qa.jsonl",
    "question": "question.jsonl",
    "rag": "rag.jsonl",
    "summarization": "summarization.jsonl",
    "tool_call": "tool_call.jsonl",
    "translation": "translation.jsonl",
    "writing": "writing.jsonl",
}


def load_local(bench, n, seed=42):
    path = DATA_DIR / DATA_FILES[bench]
    if not path.exists():
        print(f"[{bench}] {path} not found; run prepare_data.py first. skipping")
        return []
    with path.open() as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if n and n < len(recs):
        recs = random.Random(seed).sample(recs, n)
    return recs


# --------------------------------------------------------------------------- #
# Metrics (cumulative counters -- see module docstring)
# --------------------------------------------------------------------------- #
_COUNTER_NAMES = {
    "drafts": "vllm:spec_decode_num_drafts",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens",
}


def scrape_spec_counters(base_url):
    """Return {drafts, draft_tokens, accepted_tokens} summed across any
    label combinations (e.g. multiple engines), or None per key if the
    metric isn't present (spec decoding off)."""
    try:
        r = requests.get(f"{base_url}/metrics", timeout=10)
        r.raise_for_status()
    except Exception:
        return dict.fromkeys(_COUNTER_NAMES)
    totals = dict.fromkeys(_COUNTER_NAMES)
    for line in r.text.splitlines():
        if line.startswith("#"):
            continue
        for key, name in _COUNTER_NAMES.items():
            # prometheus_client renders Counters with a `_total` suffix.
            if line.startswith(name) and (
                line[len(name) : len(name) + 1] in ("", " ", "{")
                or line[len(name) : len(name) + 6] == "_total"
            ):
                m = re.search(r"\s([0-9.eE+-]+)\s*$", line)
                if m:
                    try:
                        v = float(m.group(1))
                    except ValueError:
                        continue
                    totals[key] = (totals[key] or 0.0) + v
    return totals


def accept_stats(before, after):
    """(accept_length, accept_rate) from before/after cumulative counters, or
    (None, None) if spec decoding wasn't active / no drafts were made."""
    d_drafts = (after["drafts"] or 0) - (before["drafts"] or 0)
    d_draft_tokens = (after["draft_tokens"] or 0) - (before["draft_tokens"] or 0)
    d_accepted = (after["accepted_tokens"] or 0) - (before["accepted_tokens"] or 0)
    if after["drafts"] is None or d_drafts <= 0:
        return None, None
    accept_length = 1 + d_accepted / d_drafts
    accept_rate = d_accepted / d_draft_tokens if d_draft_tokens else None
    return accept_length, accept_rate


# --------------------------------------------------------------------------- #
# Server interaction (streaming, to separate TTFT from decode time) --
# identical to the SGLang eval's approach; vLLM's OpenAI-compatible streaming
# API shape (delta.content / usage in the final chunk) matches.
# --------------------------------------------------------------------------- #
def send_chat_stream(base_url, model, prompt, max_tokens, temperature, sampling=None):
    """``sampling``: optional extra sampling fields (top_p, top_k, seed, ...)
    merged into the request. Keys with value None are dropped, so a knob the
    caller did not set falls back to the server's default (vLLM applies the
    model's generation_config.json for fields the request omits)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    payload.update({k: v for k, v in (sampling or {}).items() if v is not None})
    t0 = time.perf_counter()
    first_tok_t = None
    usage = None
    finish_reason = None
    with requests.post(
        f"{base_url}/v1/chat/completions", json=payload, stream=True, timeout=3600
    ) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:]
            if data == b"[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
                delta = choices[0].get("delta") or {}
                # First token = first chunk carrying any non-empty textual field
                # (content, reasoning_content, or any parser-specific field), except
                # the leading role-only chunk. Checking just content/reasoning_content
                # misses reasoning tokens under some reasoning parsers (e.g. glm45),
                # which left long reasoning outputs with ttft=None -> bogus decode rate.
                if first_tok_t is None and any(
                    k != "role" and isinstance(v, str) and v for k, v in delta.items()
                ):
                    first_tok_t = time.perf_counter()
    t_end = time.perf_counter()
    usage = usage or {}
    completion = usage.get("completion_tokens")
    ttft = (first_tok_t - t0) if first_tok_t else None
    decode_time = (t_end - first_tok_t) if first_tok_t else None
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion,
        "ttft_s": ttft,
        "decode_time_s": decode_time,
        "e2e_s": t_end - t0,
        # "length" = the reply was cut by max_tokens (a runaway / cap hit);
        # "stop" = the model ended its turn.
        "finish_reason": finish_reason,
    }


def get_model_name(base_url):
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=10)
        r.raise_for_status()
        return r.json()["data"][0]["id"]
    except Exception:
        return "default"


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-url", default=os.getenv("BASE_URL", "http://127.0.0.1:8082")
    )
    ap.add_argument(
        "--benchmarks",
        default="aime,gpqa,livecodebench",
        help=(
            "comma-separated subset of: aime,gpqa,livecodebench,gsm8k,math500,"
            "humaneval,mbpp,mt-bench,aime26,swe-bench-pro,swe-rebench,aa-lcr,"
            "speed-coding,speed-humanities,speed-math,speed-multilingual,"
            "speed-qa,speed-rag,speed-reasoning,speed-roleplay,speed-stem,"
            "speed-summarization,speed-writing,"
            "speed-low-entropy,HumanEval,math_reasoning,qa,question,rag,"
            "summarization,tool_call,translation,writing"
        ),
    )
    ap.add_argument(
        "--num-samples", type=int, default=20, help="per benchmark (0 = all)"
    )
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None, help="unset = server default")
    ap.add_argument("--top-k", type=int, default=None, help="unset = server default")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="per-request sampling seed (seed + repeat index); unset = nondeterministic",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="send every prompt this many times (sampled runs: averages out the "
        "per-draw length noise); each pass is tagged `rep` in the details",
    )
    ap.add_argument(
        "--output-dir",
        default=os.getenv("RESULT_DIR", str(DATA_DIR.parent / "results")),
    )
    args = ap.parse_args()

    model = get_model_name(args.base_url)
    baseline_counters = scrape_spec_counters(args.base_url)
    spec_on = baseline_counters["drafts"] is not None
    print(
        f"server={args.base_url}  model={model}  spec_decoding={'on' if spec_on else 'off/unknown'}  "
        f"max_tokens={args.max_tokens}  temperature={args.temperature}  "
        f"top_p={args.top_p}  top_k={args.top_k}  seed={args.seed}  repeats={args.repeats}"
    )
    if not spec_on:
        print(
            "NOTE: vllm:spec_decode_num_drafts not on /metrics -- either speculative "
            "decoding is off (baseline) or this vLLM build doesn't expose it. "
            "accept_length/accept_rate will be n/a; throughput still reported."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "mtp_eval_details.jsonl"
    detail_f = detail_path.open("w")

    summary = []
    for bench in [b.strip() for b in args.benchmarks.split(",") if b.strip()]:
        if bench not in DATA_FILES:
            print(f"[{bench}] unknown benchmark; skipping")
            continue
        samples = load_local(bench, args.num_samples)
        if not samples:
            continue
        print(f"\n=== {bench}: {len(samples)} samples ===")

        counters_before = scrape_spec_counters(args.base_url)

        n_ok = 0
        tot_completion = 0
        tot_decode_tokens = 0
        tot_decode_time = 0.0
        tot_e2e = 0.0
        cap_hits = 0
        ttfts = []
        for rep in range(args.repeats):
            for s in samples:
                sampling = {"top_p": args.top_p, "top_k": args.top_k}
                if args.seed is not None:
                    sampling["seed"] = args.seed + rep
                try:
                    res = send_chat_stream(
                        args.base_url,
                        model,
                        s["prompt"],
                        args.max_tokens,
                        args.temperature,
                        sampling,
                    )
                except Exception as e:
                    print(f"  [{bench}:{s['id']}] request failed: {e}")
                    continue
                ct = res["completion_tokens"] or 0
                n_ok += 1
                tot_completion += ct
                tot_e2e += res["e2e_s"]
                cap_hit = res["finish_reason"] == "length"
                cap_hits += cap_hit
                if res["decode_time_s"] and ct > 1:
                    tot_decode_tokens += ct - 1  # 1st token is TTFT, not decode
                    tot_decode_time += res["decode_time_s"]
                if res["ttft_s"]:
                    ttfts.append(res["ttft_s"])
                rec = {**{k: s[k] for k in ("benchmark", "id")}, "rep": rep, **res}
                detail_f.write(json.dumps(rec) + "\n")
                detail_f.flush()
                dtoks = (ct - 1) if ct > 1 else 0
                dts = res["decode_time_s"] or 0.0
                print(
                    f"  [{bench}:{s['id']}{f'#{rep}' if args.repeats > 1 else ''}] {ct} tok"
                    f"{' (CAP)' if cap_hit else ''}, ttft={res['ttft_s'] or 0:.2f}s, "
                    f"decode={dtoks / dts if dts else 0:.1f} tok/s, e2e={res['e2e_s']:.1f}s"
                )

        counters_after = scrape_spec_counters(args.base_url)
        accept_len, accept_rate = accept_stats(counters_before, counters_after)
        decode_tok_s = tot_decode_tokens / tot_decode_time if tot_decode_time else 0.0
        e2e_tok_s = tot_completion / tot_e2e if tot_e2e else 0.0
        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else None
        row = {
            "benchmark": bench,
            "n": n_ok,
            "num_steps": None,
            "total_completion_tokens": tot_completion,
            "decode_tok_s": round(decode_tok_s, 1),
            "e2e_tok_s": round(e2e_tok_s, 1),
            "mean_ttft_s": round(mean_ttft, 3) if mean_ttft else None,
            "accept_length": round(accept_len, 3) if accept_len else None,
            "accept_rate": round(accept_rate, 4) if accept_rate else None,
            "metric_samples": None,
            # Length / latency add-ons (does a draft change WHAT the target
            # says, not just how fast?): mean reply length, how many replies
            # were cut by max_tokens, and wall-clock per request.
            "repeats": args.repeats,
            "mean_completion_tokens": round(tot_completion / n_ok, 1) if n_ok else None,
            "cap_hits": cap_hits,
            "cap_hit_rate": round(cap_hits / n_ok, 4) if n_ok else None,
            "total_e2e_s": round(tot_e2e, 1),
            "mean_e2e_s": round(tot_e2e / n_ok, 2) if n_ok else None,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        summary.append(row)
        print(
            f"--- {bench}: n={n_ok}  decode_tok/s={decode_tok_s:.1f}  "
            f"accept_length={row['accept_length']}  "
            f"accept_rate={row['accept_rate']}  "
            f"mean_tok={row['mean_completion_tokens']}  cap_hits={cap_hits}  "
            f"e2e={tot_e2e:.0f}s"
        )

    detail_f.close()
    summary_path = out_dir / "mtp_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n============================= SUMMARY =============================")
    print(
        f"{'benchmark':<16}{'n':>5}{'decode_tok/s':>14}{'accept_len':>12}{'accept_rate':>13}"
        f"{'mean_tok':>10}{'cap_hits':>10}{'e2e_s':>9}"
    )
    for r in summary:
        al = "n/a" if r["accept_length"] is None else f"{r['accept_length']:.3f}"
        ar = "n/a" if r["accept_rate"] is None else f"{r['accept_rate']:.4f}"
        print(
            f"{r['benchmark']:<16}{r['n']:>5}{r['decode_tok_s']:>14.1f}{al:>12}{ar:>13}"
            f"{r['mean_completion_tokens'] or 0:>10.1f}{r['cap_hits']:>10}{r['total_e2e_s']:>9.0f}"
        )
    print(
        "\naccept_rate = accepted_tokens / draft_tokens (vLLM cumulative counters, delta over this run)"
    )
    print(f"\ndetails : {detail_path}\nsummary : {summary_path}")


if __name__ == "__main__":
    main()
