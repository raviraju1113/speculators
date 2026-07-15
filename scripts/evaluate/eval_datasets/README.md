# Benchmark eval datasets

Prompt sets for speculative-decoding acceptance-rate / throughput evaluation with
[`evaluate.py`](../evaluate.py).

Each `*.jsonl` file stores one benchmark, one row per example, in the format:

```json
{"turns": ["<user turn 1>", "<user turn 2>", ...]}
```

Datasets included: `gsm8k`, `math500`, `aime25`, `humaneval`, `mbpp`,
`livecodebench`, `mt-bench`, `alpaca`, `arena-hard-v2`.

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
python scripts/evaluate/eval_datasets/convert_eval_datasets_to_jsonl.py openai/gsm8k
python scripts/evaluate/eval_datasets/convert_eval_datasets_to_jsonl.py --list-supported
```

## Provenance & license

The `*.jsonl` benchmark files and `convert_eval_datasets_to_jsonl.py` are adapted
from [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec) (commit
`005e03b`), which is **MIT-licensed** (Copyright (c) 2026 The DeepSpec Authors).
The underlying benchmarks retain their own upstream licenses — see the source
table in `convert_eval_datasets_to_jsonl.py`.
