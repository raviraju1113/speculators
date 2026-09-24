"""GPU full-suite names, JSONL paths, and sampling (seed=42).

Must stay in lockstep with ``mtp_server_eval/run_vllm_eval.py`` DATA_FILES and
``experiments/gemma4-31b-full.yaml`` eval.benchmarks.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EVAL_ROOT / "mtp_server_eval" / "data"
DEFAULT_OUT = EVAL_ROOT / "experiments" / "results" / "gemma4-31b-rdu-k5"
GPU_K5_SUMMARY = (
    EVAL_ROOT / "experiments" / "results" / "gemma4-31b-full" / "assistant_k5" / "mtp_eval_summary.json"
)

DATA_FILES = {
    "aime": "aime.jsonl",
    "gpqa": "gpqa_diamond.jsonl",
    "livecodebench": "livecodebench.jsonl",
    "gsm8k": "gsm8k.jsonl",
    "math500": "math500.jsonl",
    "humaneval": "humaneval.jsonl",
    "mbpp": "mbpp.jsonl",
    "mt-bench": "mt-bench.jsonl",
    "aime26": "aime26.jsonl",
    "swe-bench-pro": "swe-bench-pro.jsonl",
    "swe-rebench": "swe-rebench.jsonl",
    "aa-lcr": "aa-lcr.jsonl",
    "bfcl": "bfcl.jsonl",
    "speed-coding": "speed-coding.jsonl",
    "speed-multilingual": "speed-multilingual.jsonl",
    "speed-rag": "speed-rag.jsonl",
    "speed-qa": "speed-qa.jsonl",
    "speed-writing": "speed-writing.jsonl",
    "speed-low-entropy": "speed-low-entropy.jsonl",
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

# Exact gemma4-31b-full.yaml list (aa-lcr / speed-low-entropy omitted there too).
FULL_BENCHMARKS = [
    "aime",
    "gpqa",
    "livecodebench",
    "gsm8k",
    "humaneval",
    "mbpp",
    "math500",
    "mt-bench",
    "aime26",
    "bfcl",
    "swe-bench-pro",
    "speed-coding",
    "speed-multilingual",
    "speed-rag",
    "speed-qa",
    "speed-writing",
    "HumanEval",
    "math_reasoning",
    "qa",
    "question",
    "rag",
    "summarization",
    "tool_call",
    "translation",
    "writing",
]


def load_local(bench: str, n: int, seed: int = 42):
    path = DATA_DIR / DATA_FILES[bench]
    if not path.exists():
        print(f"[{bench}] {path} not found; skipping", flush=True)
        return []
    with path.open() as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if n and n < len(recs):
        recs = random.Random(seed).sample(recs, n)
    return recs
