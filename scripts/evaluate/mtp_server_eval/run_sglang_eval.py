#!/usr/bin/env python3
"""MTP acceptance / speedup eval for an SGLang server.

Loads samples from LiveCodeBench, AIME, and GPQA-Diamond, sends them to the
running OpenAI-compatible server (streaming), and reports per benchmark:

  * accept_length  — avg tokens committed per target forward pass, in [1, N+1]
                     for speculative_num_steps=N (1.0 = no speculation benefit).
  * accept_rate    — SGLang's sglang:spec_accept_rate, in [0, 1]
                     (= accept_length / num_draft_tokens).
  * decode_tok/s   — output tokens per second EXCLUDING the first token (TTFT),
                     i.e. the decode-phase speed that speculative decoding
                     actually accelerates. This is the number to compare against
                     a SPEC=off baseline for the end-to-end speedup.
  * e2e_tok/s, ttft — end-to-end rate and time-to-first-token, for reference.

Acceptance is read from the server's Prometheus gauges (sglang:spec_accept_*),
which are *windowed* (reset every decode_log_interval) — so we POLL /metrics in
a background thread for the whole benchmark and average the samples, rather than
reading once at the end (which would only catch the last window).

Run greedy (temperature=0): the canonical acceptance setting — the target
accepts a draft token iff it exactly matches argmax.
"""

import argparse
import json
import os
import random
import re
import threading
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent / "data"

# Benchmark -> prepared jsonl file (produced by prepare_data.py).
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
    "speed-coding": "speed-coding.jsonl",
    "speed-multilingual": "speed-multilingual.jsonl",
    "speed-rag": "speed-rag.jsonl",
    "speed-qa": "speed-qa.jsonl",
    "speed-writing": "speed-writing.jsonl",
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
    """Read prepared {benchmark,id,prompt} records; sample n (0 = all)."""
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
# Metrics
# --------------------------------------------------------------------------- #
def scrape_spec_metrics(base_url):
    """Return (accept_length, accept_rate) from /metrics, each float or None."""
    try:
        r = requests.get(f"{base_url}/metrics", timeout=10)
        r.raise_for_status()
    except Exception:
        return None, None
    out = {"sglang:spec_accept_length": None, "sglang:spec_accept_rate": None}
    for line in r.text.splitlines():
        if line.startswith("#"):
            continue
        for key in out:
            # match the metric name as a token (optionally followed by {labels})
            if line.startswith(key) and (
                len(line) == len(key) or line[len(key)] in " {"
            ):
                m = re.search(r"\s([0-9.eE+-]+)\s*$", line)
                if m:
                    try:
                        out[key] = float(m.group(1))
                    except ValueError:
                        pass
    return out["sglang:spec_accept_length"], out["sglang:spec_accept_rate"]


class MetricSampler(threading.Thread):
    """Poll /metrics on an interval and collect non-zero spec gauge samples.

    The gauges are reset by the server every decode_log_interval (a *fixed*
    number of decode forwards), so each gauge value summarizes one window. We
    poll faster than windows update and would see the same value repeatedly, so
    we record each window's value ONCE (dedupe consecutive identical reads).
    With concurrency=1 every window covers the same #forwards, so the plain mean
    over windows equals the token-weighted pooled rate — not a biased
    mean-of-ratios. (Exact pooled counts aren't exposed over HTTP in v0.5.9.)
    """

    def __init__(self, base_url, interval=0.25):
        super().__init__(daemon=True)
        self.base_url = base_url
        self.interval = interval
        # NB: not `_stop` — that shadows threading.Thread's internal _stop().
        self._stop_event = threading.Event()
        self.lengths = []
        self.rates = []
        self._last = (None, None)

    def run(self):
        while not self._stop_event.is_set():
            al, ar = scrape_spec_metrics(self.base_url)
            # Only record a new window (value changed since last read). >0 means
            # a decode window with speculation actually ran.
            if (al, ar) != self._last:
                if al is not None and al > 0:
                    self.lengths.append(al)
                if ar is not None and ar > 0:
                    self.rates.append(ar)
                self._last = (al, ar)
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5)

    @staticmethod
    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    def summary(self):
        return self._mean(self.lengths), self._mean(self.rates), len(self.lengths)


# --------------------------------------------------------------------------- #
# Server interaction (streaming, to separate TTFT from decode time)
# --------------------------------------------------------------------------- #
def send_chat_stream(base_url, model, prompt, max_tokens, temperature):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    first_tok_t = None
    usage = None
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
                delta = choices[0].get("delta") or {}
                # First token = first chunk carrying any non-empty textual field
                # (content, reasoning_content, or any parser-specific field), except
                # the leading role-only chunk. Checking just content/reasoning_content
                # misses reasoning tokens under some reasoning parsers, leaving long
                # reasoning outputs with ttft=None -> bogus decode rate.
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
    }


def get_model_name(base_url):
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=10)
        r.raise_for_status()
        return r.json()["data"][0]["id"]
    except Exception:
        return "default"


def get_num_steps(base_url):
    """speculative_num_steps from /server_info (None if unavailable / spec off)."""
    for ep in ("/server_info", "/get_server_info"):
        try:
            r = requests.get(f"{base_url}{ep}", timeout=10)
            r.raise_for_status()
            info = r.json()
        except Exception:
            continue

        def find(d):
            if isinstance(d, dict):
                if d.get("speculative_num_steps") is not None:
                    return d["speculative_num_steps"]
                for v in d.values():
                    f = find(v)
                    if f is not None:
                        return f
            elif isinstance(d, list):
                for v in d:
                    f = find(v)
                    if f is not None:
                        return f
            return None

        n = find(info)
        if n:
            return int(n)
    return None


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-url", default=os.getenv("BASE_URL", "http://127.0.0.1:8080")
    )
    ap.add_argument(
        "--benchmarks",
        default="aime,gpqa,livecodebench",
        help=(
            "comma-separated subset of: aime,gpqa,livecodebench,gsm8k,math500,"
            "humaneval,mbpp,mt-bench,aime26,swe-bench-pro,swe-rebench,aa-lcr,"
            "speed-coding,speed-multilingual,speed-rag,speed-qa,speed-writing,"
            "speed-low-entropy"
        ),
    )
    ap.add_argument(
        "--num-samples", type=int, default=20, help="per benchmark (0 = all)"
    )
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--poll-interval",
        type=float,
        default=0.25,
        help="seconds between /metrics polls (should be < a decode "
        "window so windows aren't skipped)",
    )
    ap.add_argument(
        "--output-dir",
        default=os.getenv("RESULT_DIR", str(DATA_DIR.parent / "results")),
    )
    args = ap.parse_args()

    model = get_model_name(args.base_url)
    num_steps = get_num_steps(args.base_url)
    print(
        f"server={args.base_url}  model={model}  num_steps={num_steps}  "
        f"max_tokens={args.max_tokens}  temperature={args.temperature}"
    )
    al0, _ = scrape_spec_metrics(args.base_url)
    spec_on = al0 is not None
    if not spec_on:
        print(
            "NOTE: sglang:spec_accept_length not on /metrics — either the server "
            "wasn't launched with --enable-metrics, or speculative decoding is off "
            "(baseline). accept_length/accept_rate will be n/a; throughput still reported."
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

        sampler = MetricSampler(args.base_url, args.poll_interval)
        sampler.start()

        n_ok = 0
        tot_completion = 0
        tot_decode_tokens = 0
        tot_decode_time = 0.0
        tot_e2e = 0.0
        ttfts = []
        for s in samples:
            try:
                res = send_chat_stream(
                    args.base_url, model, s["prompt"], args.max_tokens, args.temperature
                )
            except Exception as e:
                print(f"  [{bench}:{s['id']}] request failed: {e}")
                continue
            ct = res["completion_tokens"] or 0
            n_ok += 1
            tot_completion += ct
            tot_e2e += res["e2e_s"]
            if res["decode_time_s"] and ct > 1:
                tot_decode_tokens += ct - 1  # 1st token is TTFT, not decode
                tot_decode_time += res["decode_time_s"]
            if res["ttft_s"]:
                ttfts.append(res["ttft_s"])
            rec = {**{k: s[k] for k in ("benchmark", "id")}, **res}
            detail_f.write(json.dumps(rec) + "\n")
            detail_f.flush()
            dtoks = (ct - 1) if ct > 1 else 0
            dts = res["decode_time_s"] or 0.0
            print(
                f"  [{bench}:{s['id']}] {ct} tok, ttft={res['ttft_s'] or 0:.2f}s, "
                f"decode={dtoks / dts if dts else 0:.1f} tok/s"
            )

        sampler.stop()
        accept_len, accept_rate, n_samp = sampler.summary()
        decode_tok_s = tot_decode_tokens / tot_decode_time if tot_decode_time else 0.0
        e2e_tok_s = tot_completion / tot_e2e if tot_e2e else 0.0
        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else None
        # accept_rate = SGLang's spec_accept_rate (= accept_length/num_draft_tokens).
        row = {
            "benchmark": bench,
            "n": n_ok,
            "num_steps": num_steps,
            "total_completion_tokens": tot_completion,
            "decode_tok_s": round(decode_tok_s, 1),
            "e2e_tok_s": round(e2e_tok_s, 1),
            "mean_ttft_s": round(mean_ttft, 3) if mean_ttft else None,
            "accept_length": round(accept_len, 3) if accept_len else None,
            "accept_rate": round(accept_rate, 4) if accept_rate else None,
            "metric_samples": n_samp,
        }
        summary.append(row)
        print(
            f"--- {bench}: n={n_ok}  decode_tok/s={decode_tok_s:.1f}  "
            f"accept_length={row['accept_length']}  "
            f"accept_rate={row['accept_rate']}  (samples={n_samp})"
        )

    detail_f.close()
    summary_path = out_dir / "mtp_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n============================= SUMMARY =============================")
    print(
        f"{'benchmark':<16}{'n':>4}{'decode_tok/s':>14}{'accept_len':>12}{'accept_rate':>13}"
    )
    for r in summary:
        al = "n/a" if r["accept_length"] is None else f"{r['accept_length']:.3f}"
        ar = "n/a" if r["accept_rate"] is None else f"{r['accept_rate']:.4f}"
        print(
            f"{r['benchmark']:<16}{r['n']:>4}{r['decode_tok_s']:>14.1f}{al:>12}{ar:>13}"
        )
    print("\naccept_rate = SGLang spec_accept_rate = accept_length / num_draft_tokens")
    print(f"\ndetails : {detail_path}\nsummary : {summary_path}")


if __name__ == "__main__":
    main()
