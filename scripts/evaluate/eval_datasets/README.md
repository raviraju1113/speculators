# Benchmark eval datasets

Prompt sources for the Gemma-4-31B 25-benchmark suite in
[`gemma4_31b_full_spec_decode_results.md`](../../../docs/user_guide/tutorials/gemma4_31b_full_spec_decode_results.md)
(`gemma4-31b-full.yaml`, `mode: acceptance`).

The eval reads `mtp_server_eval/data/<name>.jsonl`. Turns files in **this**
folder (`{"turns": [...]}`) are converted by
[`../mtp_server_eval/prepare_data.py`](../mtp_server_eval/prepare_data.py)
(first user turn → `prompt`). Other benches are shipped, SPEED-Bench, or
`RedHatAI/speculator_benchmarks` — they are not stored here.

Skipped in that run (no JSONL on the box): `aa-lcr`, `speed-low-entropy`.

## The 25 benches

| Eval name | n | Source | This folder |
|-----------|--:|--------|-------------|
| `aime` | 30 | shipped `mtp_server_eval/data/aime.jsonl` (AIME 2024 parquet / `prepare_data.py`) | no — not `aime24.jsonl` |
| `gpqa` | 50 | shipped `gpqa_diamond.jsonl` (`Idavidrein/gpqa`) | no |
| `livecodebench` | 50 | `prepare_data.py` from `livecodebench/code_generation_lite` | no |
| `gsm8k` | 50 | `openai/gsm8k` | `gsm8k.jsonl` |
| `humaneval` | 50 | `openai/openai_humaneval` | `humaneval.jsonl` |
| `mbpp` | 50 | `google-research-datasets/mbpp` | `mbpp.jsonl` |
| `math500` | 50 | `HuggingFaceH4/MATH-500` | `math500.jsonl` |
| `mt-bench` | 50 | `HuggingFaceH4/mt_bench_prompts` | `mt-bench.jsonl` |
| `aime26` | 30 | `MathArena/aime_2026` | `aime26.jsonl` |
| `bfcl` | 50 | BFCL v3 AST core (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`) | `bfcl.jsonl` |
| `swe-bench-pro` | 50 | `ScaleAI/SWE-bench_Pro` | `swe-bench-pro.jsonl` (often scratch) |
| `speed-coding` | 50 | NVIDIA SPEED-Bench qualitative / coding | no |
| `speed-multilingual` | 47 | qualitative / multilingual | no |
| `speed-rag` | 50 | qualitative / RAG | no |
| `speed-qa` | 50 | qualitative / QA | no |
| `speed-writing` | 50 | qualitative / writing | no |
| `HumanEval` | 50 | `RedHatAI/speculator_benchmarks` | no |
| `math_reasoning` | 50 | same | no |
| `qa` | 50 | same | no |
| `question` | 50 | same | no |
| `rag` | 50 | same | no |
| `summarization` | 50 | same | no |
| `tool_call` | 50 | same | no |
| `translation` | 50 | same | no |
| `writing` | 50 | same | no |

`humaneval` and `HumanEval` are different prompt sets.

SPEED-Bench: [`../prepare_speedbench.py`](../prepare_speedbench.py) then `prepare_data.py`.
RedHat subsets: `python ../mtp_server_eval/prepare_data.py --only HumanEval,...`.

## Regenerating turns files in this folder

```bash
python scripts/evaluate/eval_datasets/convert_eval_datasets_to_jsonl.py openai/gsm8k
python scripts/evaluate/eval_datasets/convert_eval_datasets_to_jsonl.py MathArena/aime_2026
python scripts/evaluate/eval_datasets/convert_eval_datasets_to_jsonl.py gorilla-llm/Berkeley-Function-Calling-Leaderboard
python scripts/evaluate/eval_datasets/convert_eval_datasets_to_jsonl.py ScaleAI/SWE-bench_Pro
```

By default the converter refuses to overwrite an existing JSONL. Extra converter
targets (`aime24`, `aime25`, `lbpp`, `alpaca`, `swe-bench` Lite, …) are **not**
in the 25-bench results table.

## Provenance

Turns converters are adapted from [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec)
(commit `005e03b`), MIT. Upstream datasets keep their own licenses.
