"""vLLM-matching accept_length / accept_rate aggregation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from _suite import GPU_K5_SUMMARY


def vllm_accept_stats(n_drafts: int, n_accepted: int, k: int):
    """Same formulas as run_vllm_eval.accept_stats."""
    if n_drafts <= 0:
        return None, None
    accept_length = 1.0 + n_accepted / n_drafts
    accept_rate = n_accepted / (n_drafts * k) if k else None
    return accept_length, accept_rate


def summarize_rows(rows: list[dict], bench: str) -> dict:
    n_ok = n_skip = n_err = 0
    tot_completion = tot_decode_tokens = 0
    tot_decode_time = tot_e2e = 0.0
    ttfts = []
    tot_drafts = tot_accepted = 0
    k = 5
    for rec in rows:
        if rec.get("skipped"):
            n_skip += 1
            continue
        if rec.get("error"):
            n_err += 1
            continue
        ct = rec.get("completion_tokens") or 0
        n_ok += 1
        tot_completion += ct
        tot_e2e += rec.get("e2e_s") or 0.0
        if rec.get("decode_time_s") and ct > 1:
            tot_decode_tokens += ct - 1
            tot_decode_time += rec["decode_time_s"]
        if rec.get("ttft_s"):
            ttfts.append(rec["ttft_s"])
        k = rec.get("k") or k
        tot_drafts += rec.get("num_drafts") or 0
        tot_accepted += rec.get("num_accepted_tokens") or 0

    decode_tok_s = tot_decode_tokens / tot_decode_time if tot_decode_time else 0.0
    e2e_tok_s = tot_completion / tot_e2e if tot_e2e else 0.0
    mean_ttft = sum(ttfts) / len(ttfts) if ttfts else None
    accept_len, accept_rate = vllm_accept_stats(tot_drafts, tot_accepted, k)
    return {
        "benchmark": bench,
        "n": n_ok,
        "n_skipped": n_skip,
        "n_error": n_err,
        "num_steps": None,
        "total_completion_tokens": tot_completion,
        "decode_tok_s": round(decode_tok_s, 1),
        "e2e_tok_s": round(e2e_tok_s, 1),
        "mean_ttft_s": round(mean_ttft, 3) if mean_ttft is not None else None,
        "accept_length": round(accept_len, 3) if accept_len is not None else None,
        "accept_rate": round(accept_rate, 4) if accept_rate is not None else None,
        "metric_samples": None,
        "num_drafts": tot_drafts,
        "num_accepted_tokens": tot_accepted,
    }


def rewrite_summary(detail_path: Path, summary_path: Path, benches: list[str]) -> list[dict]:
    by_bench = defaultdict(list)
    if detail_path.exists():
        with detail_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                by_bench[str(rec.get("benchmark"))].append(rec)
    summary = []
    order = list(benches)
    for b in by_bench:
        if b not in order:
            order.append(b)
    for bench in order:
        if bench in by_bench:
            summary.append(summarize_rows(by_bench[bench], bench))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def print_accept_compare(summary: list[dict]) -> None:
    gpu = {}
    if GPU_K5_SUMMARY.exists():
        gpu = {r["benchmark"]: r for r in json.loads(GPU_K5_SUMMARY.read_text())}
    print("\n========== ACCEPT vs GPU assistant_k5 ==========", flush=True)
    print(
        f"{'benchmark':<18}{'n':>4}{'gpu_alen':>10}{'rdu_alen':>10}{'d_alen':>8}"
        f"{'gpu_arate':>11}{'rdu_arate':>11}{'d_arate':>9}",
        flush=True,
    )
    for r in summary:
        g = gpu.get(r["benchmark"], {})
        ga, ra = g.get("accept_length"), r.get("accept_length")
        gr, rr = g.get("accept_rate"), r.get("accept_rate")
        d_al = f"{ra - ga:+.3f}" if ga is not None and ra is not None else "n/a"
        d_ar = f"{rr - gr:+.4f}" if gr is not None and rr is not None else "n/a"
        print(
            f"{r['benchmark']:<18}{r['n']:>4}"
            f"{(f'{ga:.3f}' if ga is not None else 'n/a'):>10}"
            f"{(f'{ra:.3f}' if ra is not None else 'n/a'):>10}{d_al:>8}"
            f"{(f'{gr:.4f}' if gr is not None else 'n/a'):>11}"
            f"{(f'{rr:.4f}' if rr is not None else 'n/a'):>11}{d_ar:>9}",
            flush=True,
        )
    print(
        "\naccept_length = 1 + accepted_draft_tokens / num_drafts  (vLLM formula, K=5)",
        flush=True,
    )
    print("decode tok/s omitted: GPU vs RDU is not comparable.", flush=True)
