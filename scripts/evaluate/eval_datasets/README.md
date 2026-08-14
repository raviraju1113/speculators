# Benchmark eval datasets

Prompt sets for speculative-decoding acceptance-rate / throughput evaluation with
[`../mtp_server_eval/run_eval.sh`](../mtp_server_eval/run_eval.sh) (`MODE=acceptance`
or GuideLLM `MODE=throughput`/`sweep`).

Each `*.jsonl` file stores one benchmark, one row per example, in the format:

```json
{"turns": ["<user turn 1>", "<user turn 2>", ...]}
```

## Included datasets

**DeepSpec-style (shipped or convertible):**

| File | Hugging Face source |
|------|---------------------|
| `gsm8k.jsonl` | `openai/gsm8k` |
| `math500.jsonl` | `HuggingFaceH4/MATH-500` |
| `aime24.jsonl` | `HuggingFaceH4/aime_2024` |
| `aime25.jsonl` | `MathArena/aime_2025` |
| `aime26.jsonl` | `MathArena/aime_2026` |
| `humaneval.jsonl` | `openai/openai_humaneval` |
| `mbpp.jsonl` | `google-research-datasets/mbpp` |
| `lbpp.jsonl` | `CohereLabs/lbpp` |
| `livecodebench.jsonl` | `livecodebench/code_generation_lite` |
| `mt-bench.jsonl` | `HuggingFaceH4/mt_bench_prompts` |
| `alpaca.jsonl` | `tatsu-lab/alpaca` |
| `arena-hard-v2.jsonl` | (shipped) |
| `swe-bench.jsonl` | `princeton-nlp/SWE-bench_Lite` |
| `swe-bench-pro.jsonl` | `ScaleAI/SWE-bench_Pro` (large — prefer scratch `turns/`; `mtp_server_eval/data/` copy may be shipped) |
| `swe-rebench.jsonl` | `nebius/SWE-rebench` (large — scratch only) |

**SPEED-Bench** (NVIDIA; map into `mtp_server_eval` as `speed-*`):

| Eval name | How to build | Source split |
|-----------|----------------|--------------|
| `speed-coding` | [`../prepare_speedbench.py`](../prepare_speedbench.py) then `prepare_data.py` | qualitative / coding |
| `speed-multilingual` | same | qualitative / multilingual |
| `speed-rag` | same | qualitative / RAG |
| `speed-qa` | same | qualitative / QA |
| `speed-writing` | same | qualitative / writing |
| `speed-low-entropy` | same | `throughput_16k` / `low_entropy` |

Large dumps (`aa-lcr`, `swe-rebench`, `speed-low-entropy`) may live under
`/import/ml-sc-scratch5/chenw/datasets/eval/` (gitignored symlinks). AA-LCR:
[`../prepare_aa_lcr.py`](../prepare_aa_lcr.py). See the parent
[`README.md`](../README.md) benchmark table.

## Using them with GuideLLM (`run_eval.sh MODE=throughput`)

GuideLLM reads a single text column (`prompt`). Convert the `turns` files first:

```bash
python scripts/evaluate/eval_datasets/to_guidellm.py
MODE=throughput BASE_URL=http://localhost:8000 \
  DATASET=scripts/evaluate/eval_datasets/guidellm \
  SUBSETS=gsm8k,humaneval,math500 \
  ./scripts/evaluate/mtp_server_eval/run_eval.sh
```

## Regenerating from source

[`convert_eval_datasets_to_jsonl.py`](./convert_eval_datasets_to_jsonl.py) rebuilds
the `turns` files from their upstream Hugging Face datasets:

```bash
python scripts/evaluate/eval_datasets/convert_eval_datasets_to_jsonl.py --list-supported
python scripts/evaluate/eval_datasets/convert_eval_datasets_to_jsonl.py openai/gsm8k
python scripts/evaluate/eval_datasets/convert_eval_datasets_to_jsonl.py MathArena/aime_2026
python scripts/evaluate/eval_datasets/convert_eval_datasets_to_jsonl.py ScaleAI/SWE-bench_Pro
python scripts/evaluate/eval_datasets/convert_eval_datasets_to_jsonl.py nebius/SWE-rebench
```
By default the converter refuses to overwrite an existing JSONL.

## Provenance & license

The DeepSpec-adapted `*.jsonl` files and `convert_eval_datasets_to_jsonl.py` are
adapted from [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec)
(commit `005e03b`), **MIT-licensed** (Copyright (c) 2026 The DeepSpec Authors).
Upstream benchmarks retain their own licenses — see the source table in
`convert_eval_datasets_to_jsonl.py`. SPEED-Bench and AA-LCR have separate
upstream terms; do not redistribute materialised prompt dumps without checking
those licenses.
