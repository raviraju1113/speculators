# Benchmark eval datasets

Prompt sets for speculative-decoding acceptance-rate / throughput evaluation with
[`evaluate.py`](../evaluate.py) and (after `prepare_data.py`) with
[`../mtp_server_eval`](../mtp_server_eval).

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

**Kimi-K3-DSpark suite extras** (large — **not committed**; live under
`/import/ml-sc-scratch5/chenw/datasets/eval/`):

| File | How to build | Scratch path |
|------|----------------|--------------|
| `aa-lcr.jsonl` | [`../prepare_aa_lcr.py`](../prepare_aa_lcr.py) ← `ArtificialAnalysis/AA-LCR` | `…/eval/turns/aa-lcr.jsonl` |
| `swe-bench-pro.jsonl` / `swe-rebench.jsonl` | converter below | `…/eval/turns/` |
| SPEED-Bench slices | [`../prepare_speedbench.py`](../prepare_speedbench.py) ← `nvidia/SPEED-Bench` (then map into `mtp_server_eval` as `speed-*`) | `…/eval/turns/throughput_16k_low_entropy.jsonl` (+ small slices may stay in-repo) |

`mtp_server_eval` prompt-format copies live in `…/eval/mtp/`. Symlink into
`eval_datasets/` / `mtp_server_eval/data/` as needed (paths are gitignored).

See the parent [`README.md`](../README.md#kimi-k3-dspark-acceptance-suite) for the
full card → eval-name mapping.

## Using them with `evaluate.py`

`evaluate.py` drives GuideLLM, which reads a single text column (`prompt`). Convert
the `turns` files to GuideLLM-ready `prompt` files first:

```bash
python scripts/evaluate/eval_datasets/to_guidellm.py
python scripts/evaluate/evaluate.py \
    --target http://localhost:8000/v1 \
    --dataset scripts/evaluate/eval_datasets/guidellm \
    throughput --subsets "gsm8k,humaneval,math500"
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
