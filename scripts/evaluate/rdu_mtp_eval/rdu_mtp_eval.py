#!/usr/bin/env python3
"""RDU MTP k=5 eval on every benchmark in gemma4-31b-full.yaml.

Same JSONL, seed=42, n=50, greedy, max_tokens=4096 as the GPU assistant_k5
run. Comparison target is accept_length / accept_rate, not decode tok/s.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

from _metrics import print_accept_compare, rewrite_summary
from _pipeline import DEFAULT_ASSISTANT, DEFAULT_CKPT, DEFAULT_PEF, build_pipeline, disable_rdu_profiler, run_one
from _suite import DATA_FILES, DEFAULT_OUT, FULL_BENCHMARKS, load_local


def load_done(detail_path: Path) -> set[tuple[str, str]]:
    done = set()
    if not detail_path.exists():
        return done
    with detail_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error") or rec.get("skipped"):
                continue
            done.add((str(rec.get("benchmark")), str(rec.get("id"))))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pef", default=os.getenv("PEF", DEFAULT_PEF))
    ap.add_argument("--ckpt", default=os.getenv("CKPT", DEFAULT_CKPT))
    ap.add_argument("--assistant", default=os.getenv("ASSISTANT", DEFAULT_ASSISTANT))
    ap.add_argument("--benchmarks", default=",".join(FULL_BENCHMARKS))
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=os.getenv("RESULT_DIR", str(DEFAULT_OUT)))
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("sambanova_modelzoo.pipelines.text_generation").setLevel(logging.WARNING)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "mtp_eval_details.jsonl"
    summary_path = out_dir / "mtp_eval_summary.json"
    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    print(
        f"pef={args.pef}\nckpt={args.ckpt}\nassistant={args.assistant}\n"
        f"benchmarks={benches}\nnum_samples={args.num_samples} seed={args.seed} "
        f"max_tokens={args.max_tokens}\noutput={out_dir}",
        flush=True,
    )

    disable_rdu_profiler()
    profiler_dir = os.path.join(os.environ.get("OUTPUT_FOLDER", "/tmp"), "profiler")
    os.makedirs(profiler_dir, exist_ok=True)
    t_load = time.perf_counter()
    pipeline, k = build_pipeline(args.pef, args.ckpt, args.assistant, profiler_dir)
    print(f"pipeline ready in {time.perf_counter() - t_load:.1f}s  k={k}", flush=True)
    if k != 5:
        print(f"WARNING: GPU comparison target is k=5; this PEF has k={k}", flush=True)

    done = load_done(detail_path)
    print(f"resume: {len(done)} completed (benchmark, id) pairs", flush=True)

    with detail_path.open("a") as detail_f:
        for bench in benches:
            if bench not in DATA_FILES:
                print(f"[{bench}] unknown; skipping", flush=True)
                continue
            samples = load_local(bench, args.num_samples, seed=args.seed)
            if not samples:
                continue
            print(f"\n=== {bench}: {len(samples)} samples ===", flush=True)
            for s in samples:
                key = (bench, str(s["id"]))
                if key in done:
                    print(f"  [{bench}:{s['id']}] resume-skip", flush=True)
                    continue
                try:
                    res = run_one(pipeline, s["prompt"], args.max_tokens, k)
                except Exception as exc:  # noqa: BLE001
                    res = {
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                    print(f"  [{bench}:{s['id']}] FAILED: {res['error']}", flush=True)
                rec = {"benchmark": bench, "id": s["id"], **res}
                detail_f.write(json.dumps(rec) + "\n")
                detail_f.flush()
                if not rec.get("error") and not rec.get("skipped"):
                    done.add(key)
                if rec.get("skipped"):
                    print(f"  [{bench}:{s['id']}] SKIP {rec.get('skip_reason')}", flush=True)
                    continue
                if rec.get("error"):
                    continue
                al, ar = rec.get("accept_length"), rec.get("accept_rate")
                print(
                    f"  [{bench}:{s['id']}] {rec.get('completion_tokens')} tok, "
                    f"accept_len={al if al is not None else 'n/a'}, "
                    f"accept_rate={ar if ar is not None else 'n/a'}",
                    flush=True,
                )
            summary = rewrite_summary(detail_path, summary_path, benches)
            row = next((r for r in summary if r["benchmark"] == bench), None)
            if row:
                print(
                    f"--- {bench}: n={row['n']}  accept_length={row['accept_length']}  "
                    f"accept_rate={row['accept_rate']}",
                    flush=True,
                )

    summary = rewrite_summary(detail_path, summary_path, benches)
    print_accept_compare(summary)
    print(f"\ndetails : {detail_path}\nsummary : {summary_path}", flush=True)
    (out_dir / "EVAL_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
