# Benchmark eval datasets

Prompt sources for the Gemma-4-31B 25-benchmark suite in
[`gemma4_31b_full_spec_decode_results.md`](../../../docs/user_guide/tutorials/gemma4_31b_full_spec_decode_results.md)
(`gemma4-31b-full.yaml`, `mode: acceptance`). Same JSONLs work for any
backbone you `vllm serve` (e.g. Kimi K3).

The eval reads `mtp_server_eval/data/<name>.jsonl`. Turns files in **this**
folder (`{"turns": [...]}`) are converted by
[`../mtp_server_eval/prepare_data.py`](../mtp_server_eval/prepare_data.py)
(first user turn → `prompt`). Everything else is fetched with the commands
below — it is not stored here.

Skipped in that Gemma run (no JSONL on the box): `aa-lcr`, `speed-low-entropy`.

## The 25 benches

| Eval name | This folder | If not here, get it from |
|-----------|-------------|--------------------------|
| `aime` | no | already in `mtp_server_eval/data/aime.jsonl`. Rebuild: `cd mtp_server_eval && python prepare_data.py --only aime` (optional `AIME_PARQUET=...`) |
| `gpqa` | no | `mtp_server_eval/data/gpqa_diamond.jsonl`. Rebuild (gated): `hf auth login` then `python prepare_data.py --only gpqa` from [`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa) |
| `livecodebench` | `livecodebench.jsonl`\* | eval uses `mtp_server_eval/data/livecodebench.jsonl`. Rebuild: `python prepare_data.py --only livecodebench` from [`livecodebench/code_generation_lite`](https://huggingface.co/datasets/livecodebench/code_generation_lite) |
| `gsm8k` | `gsm8k.jsonl` | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) via converter below |
| `humaneval` | `humaneval.jsonl` | [`openai/openai_humaneval`](https://huggingface.co/datasets/openai/openai_humaneval) |
| `mbpp` | `mbpp.jsonl` | [`google-research-datasets/mbpp`](https://huggingface.co/datasets/google-research-datasets/mbpp) |
| `math500` | `math500.jsonl` | [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) |
| `mt-bench` | `mt-bench.jsonl` | [`HuggingFaceH4/mt_bench_prompts`](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts) |
| `aime26` | `aime26.jsonl` | [`MathArena/aime_2026`](https://huggingface.co/datasets/MathArena/aime_2026) |
| `bfcl` | `bfcl.jsonl` | [`gorilla-llm/Berkeley-Function-Calling-Leaderboard`](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) (v3 AST core) |
| `swe-bench-pro` | often gitignored | [`ScaleAI/SWE-bench_Pro`](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro); eval copy may already be `mtp_server_eval/data/swe-bench-pro.jsonl` |
| `speed-coding` | no | NVIDIA [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) qualitative / coding — see SPEED-Bench below |
| `speed-multilingual` | no | SPEED-Bench qualitative / multilingual |
| `speed-rag` | no | SPEED-Bench qualitative / RAG |
| `speed-qa` | no | SPEED-Bench qualitative / QA |
| `speed-writing` | no | SPEED-Bench qualitative / writing |
| `HumanEval` | no | [`RedHatAI/speculator_benchmarks`](https://huggingface.co/datasets/RedHatAI/speculator_benchmarks) — already in `mtp_server_eval/data/HumanEval.jsonl` |
| `math_reasoning` | no | same HF dataset → `mtp_server_eval/data/math_reasoning.jsonl` |
| `qa` | no | same |
| `question` | no | same |
| `rag` | no | same |
| `summarization` | no | same |
| `tool_call` | no | same |
| `translation` | no | same |
| `writing` | no | same |

\* DeepSpec dump in this folder; the eval name `livecodebench` uses `prepare_data.py`’s file under `mtp_server_eval/data/`.

`humaneval` (DeepSpec) and `HumanEval` (RedHat) are different prompt sets.

### Fetch commands (from `scripts/evaluate/`)

Shipped / HF into `mtp_server_eval/data/`:

```bash
cd scripts/evaluate/mtp_server_eval
python prepare_data.py --only aime,livecodebench
python prepare_data.py --only gpqa          # gated: hf auth login first
python prepare_data.py --only gsm8k,humaneval,mbpp,math500,mt-bench,aime26,bfcl,swe-bench-pro
python prepare_data.py --only HumanEval,math_reasoning,qa,question,rag,summarization,tool_call,translation,writing
```

Turns files in this folder (then re-run `prepare_data.py --only …`):

```bash
cd scripts/evaluate
python eval_datasets/convert_eval_datasets_to_jsonl.py openai/gsm8k
python eval_datasets/convert_eval_datasets_to_jsonl.py openai/openai_humaneval
python eval_datasets/convert_eval_datasets_to_jsonl.py google-research-datasets/mbpp
python eval_datasets/convert_eval_datasets_to_jsonl.py HuggingFaceH4/MATH-500
python eval_datasets/convert_eval_datasets_to_jsonl.py HuggingFaceH4/mt_bench_prompts
python eval_datasets/convert_eval_datasets_to_jsonl.py MathArena/aime_2026
python eval_datasets/convert_eval_datasets_to_jsonl.py gorilla-llm/Berkeley-Function-Calling-Leaderboard
python eval_datasets/convert_eval_datasets_to_jsonl.py ScaleAI/SWE-bench_Pro
```

SPEED-Bench (not in this folder):

```bash
cd scripts/evaluate
python prepare_speedbench.py --data-dir ./speedbench_data \
    --download --configs qualitative,throughput_16k
cd mtp_server_eval
SPEEDBENCH_DIR=../speedbench_data python prepare_data.py --only \
  speed-coding,speed-multilingual,speed-rag,speed-qa,speed-writing,speed-low-entropy
```

Long-context / extra (skipped in the Gemma 8k run; from repo root):

```bash
python scripts/evaluate/prepare_aa_lcr.py          # ArtificialAnalysis/AA-LCR
python scripts/evaluate/eval_datasets/convert_eval_datasets_to_jsonl.py nebius/SWE-rebench
```

By default the converter refuses to overwrite an existing JSONL. Extra files in
this folder (`aime24`, `aime25`, `lbpp`, `alpaca`, `arena-hard-v2`, `swe-bench`
Lite, …) are **not** in the 25-bench results table.

## Provenance

Turns converters are adapted from [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec)
(commit `005e03b`), MIT. Upstream datasets keep their own licenses.
