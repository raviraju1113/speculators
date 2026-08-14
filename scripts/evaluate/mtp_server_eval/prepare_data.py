#!/usr/bin/env python3
"""Prepare eval datasets for the server MTP/EAGLE acceptance eval.

Writes normalized, ready-to-send prompt files to this dir's ``data/``:
each line = {"benchmark", "id", "prompt"}.

Core shipped sets (refresh as needed):
  * AIME 2024     — local parquet (set AIME_PARQUET, or use the shipped file)
  * LiveCodeBench — livecodebench/code_generation_lite (plain test*.jsonl shards)
  * GPQA-Diamond  — Idavidrein/gpqa (GATED: ``hf auth login`` after accepting terms)

Also derives Inferact/Kimi-K3-DSpark suite prompts from sibling
``eval_datasets/`` turns files and from SPEED-Bench / AA-LCR preparers
(``../prepare_speedbench.py``, ``../prepare_aa_lcr.py``).

Run once on a machine with internet:
    python prepare_data.py                      # aime,gpqa,livecodebench
    python prepare_data.py --only gsm8k,aime26,speed-coding,aa-lcr
"""

import argparse
import csv
import gzip
import json
import os
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
# AIME source parquet is environment-specific; override via AIME_PARQUET. If it
# is missing, prep_aime() skips and the shipped data/aime.jsonl is used as-is.
AIME_PARQUET = Path(os.getenv("AIME_PARQUET", DATA_DIR / "aime_2024.parquet"))

# Static benchmarks derived offline from the sibling eval_datasets/ turns files
# ({"turns": [...]} -> {"benchmark","id","prompt"}, first user turn as prompt).
# No network/gated access needed -- the prompts are already fully formatted.
EVAL_DATASETS_DIR = Path(__file__).resolve().parent.parent / "eval_datasets"
TURNS_BENCHMARKS = {
    "gsm8k": "gsm8k.jsonl",
    "math500": "math500.jsonl",
    "humaneval": "humaneval.jsonl",
    "mbpp": "mbpp.jsonl",
    "mt-bench": "mt-bench.jsonl",
    "aime26": "aime26.jsonl",
    "swe-bench-pro": "swe-bench-pro.jsonl",
    "swe-rebench": "swe-rebench.jsonl",
    "aa-lcr": "aa-lcr.jsonl",
}

# SPEED-Bench materialised splits (see ../prepare_speedbench.py). Env override
# points at the directory that contains qualitative_*.jsonl /
# throughput_16k_*.jsonl files.
SPEEDBENCH_DIR = Path(
    os.getenv(
        "SPEEDBENCH_DIR",
        str(Path(__file__).resolve().parent.parent / "speedbench_data"),
    )
)
# RedHatAI/speculator_benchmarks subsets (former evaluate.py defaults).
SPECULATOR_BENCHMARKS = (
    "HumanEval",
    "math_reasoning",
    "qa",
    "question",
    "rag",
    "summarization",
    "tool_call",
    "translation",
    "writing",
)
SPECULATOR_REPO = "RedHatAI/speculator_benchmarks"
SPEEDBENCH_BENCHMARKS = {
    "speed-coding": "qualitative_coding.jsonl",
    "speed-multilingual": "qualitative_multilingual.jsonl",
    "speed-rag": "qualitative_rag.jsonl",
    "speed-qa": "qualitative_qa.jsonl",
    "speed-writing": "qualitative_writing.jsonl",
    # Card lists "low-entropy, 10k input" (512 prompts); closest public split is
    # throughput_16k low_entropy (512). Override via SPEEDBENCH_DIR contents.
    "speed-low-entropy": "throughput_16k_low_entropy.jsonl",
}

# Prompt templates mirror benchmark/math_reason conventions.
AIME_TMPL = "Problem: {problem}\n\nMark your solution with \\boxed Answer:"
GPQA_TMPL = (
    "Return your final response within \\boxed{{}} and only include the letter "
    "choice (A, B, C, or D) as your final response.\n"
    "Problem: {problem}\nOptions: {options}\nAnswer:"
)
# For mirrors whose `question` already contains the options inline.
GPQA_TMPL_INLINE = (
    "Return your final response within \\boxed{{}} and only include the letter "
    "choice (A, B, C, or D) as your final response.\n"
    "Problem: {problem}\nAnswer:"
)
LCB_TMPL = (
    "You are an expert Python programmer. Solve the following competitive "
    "programming problem. Provide a complete, correct solution.\n\n"
    "### Problem\n{problem}\n\n{starter}### Solution\n"
)

# LiveCodeBench cumulative shards per version tag.
LCB_SHARDS = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
    ],
    "release_v6": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
}


def _write(name, records):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(records):>5} -> {path}")


def prep_aime():
    import pandas as pd

    if not AIME_PARQUET.exists():
        print(f"[aime] parquet missing at {AIME_PARQUET}; skipping")
        return
    df = pd.read_parquet(AIME_PARQUET)
    recs = [
        {
            "benchmark": "aime",
            "id": str(r["id"]),
            "prompt": AIME_TMPL.format(problem=r["problem"]),
        }
        for _, r in df.iterrows()
    ]
    _write("aime.jsonl", recs)


def prep_gpqa(seed=42):
    """Prefer the official (gated) repo; fall back to an ungated mirror."""
    from huggingface_hub import hf_hub_download

    # 1) Official Idavidrein/gpqa csv — needs `hf auth login` (gated).
    try:
        path = hf_hub_download(
            "Idavidrein/gpqa", "gpqa_diamond.csv", repo_type="dataset"
        )
        rng = random.Random(seed)
        recs = []
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        for i, ex in enumerate(rows):
            answers = [
                ex["Correct Answer"],
                ex["Incorrect Answer 1"],
                ex["Incorrect Answer 2"],
                ex["Incorrect Answer 3"],
            ]
            rng.shuffle(answers)
            options = ", ".join(f"{l}) {a}" for l, a in zip("ABCD", answers))
            recs.append(
                {
                    "benchmark": "gpqa",
                    "id": str(ex.get("Record ID", i)),
                    "prompt": GPQA_TMPL.format(problem=ex["Question"], options=options),
                }
            )
        _write("gpqa_diamond.jsonl", recs)
        return
    except Exception as e:
        print(
            f"[gpqa] official repo unavailable ({type(e).__name__}); "
            "trying ungated mirror fingertap/GPQA-diamond ..."
        )

    # 2) Ungated mirror — `question` already contains options inline (a/b/c/d).
    try:
        import pandas as pd

        p = hf_hub_download(
            "fingertap/GPQA-diamond", "test/gpqa_diamond.parquet", repo_type="dataset"
        )
        df = pd.read_parquet(p)
        recs = [
            {
                "benchmark": "gpqa",
                "id": str(i),
                "prompt": GPQA_TMPL_INLINE.format(problem=row["question"]),
            }
            for i, row in df.iterrows()
        ]
        _write("gpqa_diamond.jsonl", recs)
    except Exception as e:
        print(
            f"[gpqa] mirror also failed ({type(e).__name__}); skipping. "
            "Accept terms at https://huggingface.co/datasets/Idavidrein/gpqa "
            "then `hf auth login`."
        )


def prep_livecodebench(version="release_v6"):
    from huggingface_hub import hf_hub_download

    shards = LCB_SHARDS.get(version, LCB_SHARDS["release_v6"])
    recs = []
    for shard in shards:
        try:
            p = hf_hub_download(
                "livecodebench/code_generation_lite", shard, repo_type="dataset"
            )
        except Exception as e:
            print(
                f"[livecodebench] {shard} failed ({type(e).__name__}); skipping shard"
            )
            continue
        with open(p, "rb") as f:
            gz = f.read(2) == b"\x1f\x8b"
        opener = gzip.open if gz else open
        with opener(p, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                starter = ex.get("starter_code") or ""
                starter = (
                    f"### Starter code\n```python\n{starter}\n```\n\n"
                    if starter.strip()
                    else ""
                )
                recs.append(
                    {
                        "benchmark": "livecodebench",
                        "id": str(ex.get("question_id", len(recs))),
                        "prompt": LCB_TMPL.format(
                            problem=ex["question_content"], starter=starter
                        ),
                    }
                )
    # de-dup by id (shards are cumulative supersets)
    seen, uniq = set(), []
    for r in recs:
        if r["id"] not in seen:
            seen.add(r["id"])
            uniq.append(r)
    _write("livecodebench.jsonl", uniq)


def prep_from_turns(name):
    """Convert eval_datasets/<name>.jsonl ({turns}) to data/<name>.jsonl."""
    src = EVAL_DATASETS_DIR / TURNS_BENCHMARKS[name]
    if not src.exists():
        print(f"[{name}] source missing at {src}; skipping")
        return
    recs = []
    with src.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            turns = row.get("turns")
            prompt = (
                turns[0]
                if isinstance(turns, list) and turns and isinstance(turns[0], str)
                else row.get("prompt")
            )
            if not isinstance(prompt, str) or not prompt:
                continue
            recs.append(
                {
                    "benchmark": name,
                    "id": str(row.get("id", i)),
                    "prompt": prompt,
                }
            )
    _write(f"{name}.jsonl", recs)


def prep_from_speedbench(name):
    """Convert a SPEED-Bench split file into mtp_server_eval data/<name>.jsonl."""
    src = SPEEDBENCH_DIR / SPEEDBENCH_BENCHMARKS[name]
    if not src.exists():
        print(
            f"[{name}] source missing at {src}; run "
            f"`python ../prepare_speedbench.py --data-dir {SPEEDBENCH_DIR} "
            f"--download --configs qualitative,throughput_16k` first"
        )
        return
    recs = []
    with src.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            turns = row.get("turns")
            if isinstance(turns, list) and turns and isinstance(turns[0], str):
                prompt = turns[0]
            elif isinstance(turns, str) and turns:
                prompt = turns
            else:
                prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                continue
            recs.append({"benchmark": name, "id": str(row.get("id", i)), "prompt": prompt})
    _write(f"{name}.jsonl", recs)


def prep_speculator_benchmark(name):
    """Download one RedHatAI/speculator_benchmarks subset → data/<name>.jsonl."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            SPECULATOR_REPO, f"{name}.jsonl", repo_type="dataset"
        )
    except Exception as e:
        print(
            f"[{name}] download from {SPECULATOR_REPO} failed ({type(e).__name__}); "
            "skipping"
        )
        return
    recs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt") or row.get("text") or row.get("question")
            if not isinstance(prompt, str) or not prompt:
                turns = row.get("turns")
                if isinstance(turns, list) and turns and isinstance(turns[0], str):
                    prompt = turns[0]
            if not isinstance(prompt, str) or not prompt:
                continue
            recs.append(
                {
                    "benchmark": name,
                    "id": str(row.get("id", row.get("question_id", i))),
                    "prompt": prompt,
                }
            )
    _write(f"{name}.jsonl", recs)


def main():
    ap = argparse.ArgumentParser()
    all_named = (
        ["aime", "gpqa", "livecodebench"]
        + list(TURNS_BENCHMARKS)
        + list(SPEEDBENCH_BENCHMARKS)
        + list(SPECULATOR_BENCHMARKS)
    )
    ap.add_argument(
        "--only",
        default="aime,gpqa,livecodebench",
        help="comma-separated subset to prepare (also: " + ",".join(all_named) + ")",
    )
    ap.add_argument("--lcb-version", default="release_v6")
    args = ap.parse_args()
    todo = {x.strip() for x in args.only.split(",") if x.strip()}

    print(f"Preparing into {DATA_DIR}")
    if "aime" in todo:
        print("[aime]")
        prep_aime()
    if "gpqa" in todo:
        print("[gpqa]")
        prep_gpqa()
    if "livecodebench" in todo:
        print("[livecodebench]")
        prep_livecodebench(args.lcb_version)
    for name in TURNS_BENCHMARKS:
        if name in todo:
            print(f"[{name}]")
            prep_from_turns(name)
    for name in SPEEDBENCH_BENCHMARKS:
        if name in todo:
            print(f"[{name}]")
            prep_from_speedbench(name)
    for name in SPECULATOR_BENCHMARKS:
        if name in todo:
            print(f"[{name}]")
            prep_speculator_benchmark(name)
    print("done.")


if __name__ == "__main__":
    main()
