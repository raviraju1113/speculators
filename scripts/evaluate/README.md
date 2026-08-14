# Evaluation

Tools for evaluating a speculative-decoding **draft model** against its
**backbone** (target / verifier): acceptance rate, throughput, and the end-to-end
speedup over running the backbone alone.

## Recent changes

History of notable updates to this eval tree (newest first). Use this so
contributors can see what landed without digging through git alone.

| When | Change |
|------|--------|
| 2026-08-14 | **Removed `evaluate.py`.** GuideLLM throughput/sweep lives in `mtp_server_eval/run_guidellm_eval.py` and is reached only via `run_eval.sh` (`MODE=throughput`/`sweep`) or YAML `eval.mode`. |
| 2026-08-14 | **YAML full-eval entrypoint** — [`experiments/full-eval.yaml`](./experiments/full-eval.yaml) + [`run_full_eval.sh`](./experiments/run_full_eval.sh); guide: [How to run a full evaluation](#how-to-run-a-full-evaluation). |
| 2026-08-14 | **Docs: SPEED-Bench is in the suite** — six slices documented and listed in `full-eval.yaml` (`speed-coding`, `speed-multilingual`, `speed-rag`, `speed-qa`, `speed-writing`, `speed-low-entropy`). GuideLLM can also use `DATASET=speedbench/…`. |
| 2026-08-14 | **`RedHatAI/speculator_benchmarks`** — nine subsets (`HumanEval`, `math_reasoning`, `qa`, `question`, `rag`, `summarization`, `tool_call`, `translation`, `writing`) in acceptance mode via `prepare_data.py` and in `full-eval.yaml`. |
| 2026-08-14 | Merged **upstream `vllm-project/speculators` main** into this eval branch (D-PACE defaults, Inkling, fused losses, Mooncake, NaN hidden-state skip). |
| 2026-08 | **Large generated JSONLs off-git** — `aa-lcr`, `swe-rebench`, `speed-low-entropy` / `throughput_16k_low_entropy` (and turns-format `swe-bench-pro`) live under scratch (`…/datasets/eval/{turns,mtp}/`); repo paths are gitignored symlinks. See [`eval_datasets/README.md`](./eval_datasets/README.md). |
| 2026-08 | **New / extended benchmarks** — `aime26`, `swe-bench-pro`, `swe-rebench`, `aa-lcr`, SPEED-Bench slices wired through converters, `prepare_data.py`, and `run_eval.sh`. |
| 2026-08 | **Preparers** — [`prepare_aa_lcr.py`](./prepare_aa_lcr.py); [`prepare_speedbench.py`](./prepare_speedbench.py) gains `throughput_16k` / list-shaped `turns`. |
| 2026-08 | **Docs / TODO** — `mtp_server_eval` §H points at YAML full eval; [`TODO.md`](./TODO.md) tracks remaining harness gaps (quality check, position-wise accept, YAML resume, etc.). |

## How to run a full evaluation

**Yes — the recommended full eval is YAML-driven.** One config describes the
backbone, drafts, GPUs, and benchmarks; [`experiments/run_experiments.py`](./experiments/run_experiments.py)
does **serve → eval → compare** for you.

### 1. One-time setup

```bash
cd /path/to/speculators
python -m venv .venv-eval && source .venv-eval/bin/activate   # or conda
pip install "vllm>=0.12.0"
pip install -r scripts/evaluate/requirements.txt
export HF_TOKEN=hf_xxx   # if you need gated models / GPQA / full SPEED-Bench
```

### 2. Edit the YAML

Copy or edit [`experiments/full-eval.yaml`](./experiments/full-eval.yaml):

| Field | What to set |
|-------|-------------|
| `backbone` | Target / verifier model (local path or HF id) |
| `experiments[].draft` | Speculator / draft checkpoint |
| `gpus` / `server.tensor_parallel_size` | Your GPU layout |
| `eval.benchmarks` | Which workloads to run (see list below) |
| `eval.num_samples` | `0` = all prompts; `20–50` for a smoke run |
| `server.max_model_len` | Raise for long-context benches (e.g. `aa-lcr`); `8192` is fine without them |

First experiment must be the **baseline** (no draft). Later entries attach drafts.

### 3. Prepare datasets (once)

Small shipped sets already live under `mtp_server_eval/data/`. For extras:

```bash
cd scripts/evaluate

# turns JSONL (math/code/SWE/…):
python eval_datasets/convert_eval_datasets_to_jsonl.py --list-supported
python eval_datasets/convert_eval_datasets_to_jsonl.py openai/gsm8k
# …or whatever you listed under eval.benchmarks

# large / not-in-git sets:
python prepare_aa_lcr.py
python prepare_speedbench.py --download --configs qualitative,throughput_16k

# convert turns → mtp prompt files (includes SPEED-Bench + speculator_benchmarks):
cd mtp_server_eval
python prepare_data.py --only gsm8k,humaneval,mbpp,math500,mt-bench,aime26,swe-bench-pro,aa-lcr,speed-coding,speed-multilingual,speed-rag,speed-qa,speed-writing,speed-low-entropy,HumanEval,math_reasoning,qa,question,rag,summarization,tool_call,translation,writing
```

Details: [`eval_datasets/README.md`](./eval_datasets/README.md),
[`mtp_server_eval/README.md`](./mtp_server_eval/README.md#g-regenerating-datasets).

### 4. Run

```bash
cd scripts/evaluate/experiments
./run_full_eval.sh --dry-run     # prints serve + eval commands; no GPUs needed
./run_full_eval.sh               # full serve → eval → speedup table
./run_full_eval.sh --only baseline,draft_k5
```

Same as `python run_experiments.py --config full-eval.yaml …`. Use `tmux`/`screen`
for long sweeps. Results: `results/full-eval/<name>/mtp_eval_summary.json`.

Optional Markdown/CSV table:

```bash
python tabulate_results.py --dir ./results/full-eval --baseline baseline \
    --out-dir ./results/full-eval
```

### 5. Fair-comparison checklist

- Same GPUs/config, benchmarks, and sample budget for baseline and draft runs
- **Greedy** (`temperature: 0.0`) for canonical acceptance
- Compare **decode tok/s** (and accept length/rate), not end-to-end latency alone
- `acceptance_length ≈ 1` → the draft is not helping on that workload

---

More detail on the YAML schema: [`experiments/README.md`](./experiments/README.md).

If a server is **already running**, hit it with
[`mtp_server_eval/run_eval.sh`](./mtp_server_eval/run_eval.sh):

```bash
cd mtp_server_eval
# sequential per-benchmark acceptance (default)
BACKEND=vllm BASE_URL=http://localhost:8000 ./run_eval.sh
# GuideLLM max-rate / SLA sweep
MODE=throughput BASE_URL=http://localhost:8000 SUBSETS=HumanEval ./run_eval.sh
MODE=sweep BASE_URL=http://localhost:8000 ./run_eval.sh
```

## What gets measured

| Metric | Meaning |
|--------|---------|
| **acceptance_length** | avg tokens committed per backbone forward pass (`1.0` = speculation gave no benefit; higher is better, up to `num_spec_tokens + 1`) |
| **acceptance_rate** | accepted / drafted tokens, in `[0, 1]` |
| **decode tok/s** (a.k.a. `output_tokens_per_second`) | decode-phase output speed — the number speculative decoding actually accelerates |
| **speedup** | decode tok/s with the draft ÷ decode tok/s of the backbone alone |

## The two evaluators

[`mtp_server_eval/run_eval.sh`](./mtp_server_eval/run_eval.sh) is the single
entrypoint. `MODE` selects the load driver:

| | `MODE=acceptance` (default) | `MODE=throughput` / `MODE=sweep` |
|---|---|---|
| Engine | vLLM **or** SGLang | vLLM (GuideLLM) |
| Load driver | direct streaming requests | GuideLLM rate/sweep control |
| Deps | `requests` | `guidellm`, `vllm` (see [requirements.txt](./requirements.txt)) |
| Best for | acceptance + decode-tok/s per benchmark; used by the YAML runner | SLA-style rate sweeps, HF `RedHatAI/speculator_benchmarks`, SPEED-Bench |
| Output | `mtp_eval_summary.json` | `acceptance.csv` / `perf_results.csv` (for [`plot.py`](./plot.py)) |
| Python | `run_vllm_eval.py` / `run_sglang_eval.py` | `run_guidellm_eval.py` |

The **YAML full eval** defaults to `MODE=acceptance` (`mtp_server_eval`). Set
`eval.mode: throughput` or `sweep` in the YAML to use GuideLLM instead.

## Datasets / benchmark names

Names below are valid in `eval.benchmarks` / `BENCHMARKS=` (`MODE=acceptance`)
and ship (or prepare) as `mtp_server_eval/data/<name>.jsonl`.

| Eval name | Notes |
|-----------|--------|
| `aime`, `gpqa`, `livecodebench` | Default smoke trio in `run_eval.sh` |
| `gsm8k`, `math500`, `humaneval`, `mbpp`, `mt-bench`, `aime26` | From `eval_datasets/` |
| `swe-bench-pro`, `swe-rebench` | SWE-style; large — often kept off-git |
| `aa-lcr` | Long-context (~tens of k tokens) |

### SPEED-Bench (included)

NVIDIA [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) is in
[`full-eval.yaml`](./experiments/full-eval.yaml). Prepare once with
[`prepare_speedbench.py`](./prepare_speedbench.py), then `prepare_data.py`.

| Eval name | SPEED-Bench split |
|-----------|-------------------|
| `speed-coding` | qualitative / coding |
| `speed-multilingual` | qualitative / multilingual |
| `speed-rag` | qualitative / RAG |
| `speed-qa` | qualitative / QA |
| `speed-writing` | qualitative / writing |
| `speed-low-entropy` | `throughput_16k` / `low_entropy` (card wording: “10k input”) |

GuideLLM (`MODE=throughput`/`sweep`) can also run the full NVIDIA tree:

```bash
MODE=throughput DATASET=speedbench/qualitative ./run_eval.sh
MODE=throughput DATASET=speedbench/throughput_16k/low_entropy ./run_eval.sh
```

(`SPEEDBENCH_DATA_DIR` defaults to `scripts/evaluate/speedbench_data`.)

### `RedHatAI/speculator_benchmarks`

Nine subsets (also the GuideLLM default `SUBSETS`): `HumanEval`,
`math_reasoning`, `qa`, `question`, `rag`, `summarization`, `tool_call`,
`translation`, `writing`. Distinct from DeepSpec `humaneval`. Prepare with
`python prepare_data.py --only HumanEval,math_reasoning,qa,...`.

The template list is [`experiments/full-eval.yaml`](./experiments/full-eval.yaml).
Follow-ups: [`TODO.md`](./TODO.md).

## Layout

```
README.md              ← you are here (start with “How to run a full evaluation”)
perf_utils.py          metric parsing + GuideLLM helpers
plot.py                plots from sweep output
requirements.txt
TODO.md
eval_datasets/         turns JSONL + converter + GuideLLM bridge
prepare_speedbench.py  NVIDIA SPEED-Bench → turns JSONL
prepare_aa_lcr.py      AA-LCR → turns JSONL
mtp_server_eval/       run_eval.sh (acceptance + GuideLLM) + AgentX
experiments/           YAML runner (serve → eval → compare)
  full-eval.yaml       recommended full-suite template
  run_full_eval.sh     thin wrapper around run_experiments.py
  example.yaml         smaller example config
```
