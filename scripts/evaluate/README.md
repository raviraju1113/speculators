# Evaluation

Tools for evaluating a speculative-decoding **draft model** against its
**backbone** (target / verifier): acceptance rate, throughput, and the end-to-end
speedup over running the backbone alone.

## Recent changes

History of notable updates to this eval tree (newest first). Use this so
contributors can see what landed without digging through git alone.

| When | Change |
|------|--------|
| 2026-08-19 | **Context-length sweep for acceptance** — [`prepare_aa_lcr_sweep.py`](./prepare_aa_lcr_sweep.py) builds `aa-lcr-{1k,2k,4k,8k,16k,32k}`: the same 100 AA-LCR questions truncated to each length with header + question held fixed, so acceptance vs context length isn't confounded by domain or entropy (the old `speed-low-entropy` comparison was). Runs via [`experiments/gemma4-kimi-mtp-stem-code-math-900k-ctxlen-sweep.yaml`](./experiments/gemma4-kimi-mtp-stem-code-math-900k-ctxlen-sweep.yaml). |
| 2026-08-19 | **Fixed silently-skipped benchmarks.** `DATA_FILES` in `run_vllm_eval.py` / `run_sglang_eval.py` still held only the 5-category partial SPEED-Bench prep, so the six categories added on 2026-08-14 hit `unknown benchmark; skipping` and runs quietly evaluated one slice. Registered all 11 plus the sweep bins, in both runners and `run_eval.sh`. |
| 2026-08-18 | **AgentX repaired and wired into the YAML runner.** Its pinned client (InferenceX branch `chore/agentx-integration`, `utils/trace-replay/trace_replay_tester.py`) no longer exists upstream; `run_agentx.sh` now drives [aiperf](https://github.com/SemiAnalysisAI/aiperf)'s `--scenario inferencex-agentx-mvp` from a dedicated venv. New `eval.mode: agentx` does serve → replay → compare for baseline vs draft ([`experiments/agentx-gemma4.yaml`](./experiments/agentx-gemma4.yaml)), with `compare_agentx.py` + `agentx_metrics.py`. |
| 2026-08-14 | **SPEED-Bench qualitative is now complete — all 11 categories (880 prompts)**, not the 5 that a partial prep had produced. Added `speed-humanities`, `speed-math`, `speed-reasoning`, `speed-roleplay`, `speed-stem`, `speed-summarization`; `prepare_speedbench.py` now fails loudly when external sources don't materialise instead of dropping rows silently. |
| 2026-08-14 | **Removed `evaluate.py`.** GuideLLM throughput/sweep lives in `mtp_server_eval/run_guidellm_eval.py` and is reached only via `run_eval.sh` (`MODE=throughput`/`sweep`) or YAML `eval.mode`. |
| 2026-08-14 | **YAML full-eval entrypoint** — [`experiments/full-eval.yaml`](./experiments/full-eval.yaml) + [`run_full_eval.sh`](./experiments/run_full_eval.sh); guide: [How to run a full evaluation](#how-to-run-a-full-evaluation). |
| 2026-08-14 | **Docs: SPEED-Bench is in the suite** — slices documented and listed in `full-eval.yaml`. GuideLLM can also use `DATASET=speedbench/…`. |
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

> **Activate the env — don't just call its python.** The runner launches the
> server via a bare `vllm serve`, so `vllm` must be on `PATH`. Invoking
> `/path/to/env/bin/python run_experiments.py` without activating dies with
> `FileNotFoundError: 'vllm'` right after printing the serve command.

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

## The evaluators

[`mtp_server_eval/run_eval.sh`](./mtp_server_eval/run_eval.sh) is the entrypoint
for the two prompt-file drivers; AgentX has its own script because it replays
sessions rather than sending prompts from a file.

| | `MODE=acceptance` (default) | `MODE=throughput` / `MODE=sweep` | AgentX |
|---|---|---|---|
| Engine | vLLM **or** SGLang | vLLM (GuideLLM) | vLLM **or** SGLang |
| Load driver | direct streaming requests | GuideLLM rate/sweep control | aiperf agentic trace replay |
| Deps | `requests` | `guidellm`, `vllm` (see [requirements.txt](./requirements.txt)) | `aiperf` (own venv) |
| Best for | acceptance + decode-tok/s per benchmark; used by the YAML runner | SLA-style rate sweeps, HF `RedHatAI/speculator_benchmarks`, SPEED-Bench | long-context multi-user agentic load; concurrency scaling |
| Output | `mtp_eval_summary.json` | `acceptance.csv` / `perf_results.csv` (for [`plot.py`](./plot.py)) | `matrix.tsv` |
| Compare | `compare_speedup.py` | — (plot the CSVs) | `compare_agentx.py` |
| Entrypoint | `run_vllm_eval.py` / `run_sglang_eval.py` | `run_guidellm_eval.py` | `run_agentx.sh` |

The **YAML full eval** defaults to `MODE=acceptance` (`mtp_server_eval`). Set
`eval.mode` to `throughput`/`sweep` for GuideLLM, or `agentx` for AgentX.

**AgentX** replays real Claude-Code agentic traces at fixed concurrency, so it
measures speculative decoding where the static prompt sets can't: ~110k-token
median input, heavy prefix-cache reuse, several concurrent sessions. Note its
corpus carries no prompt *text* (only token counts and KV block hashes, from
which aiperf synthesizes prompts), so read its `decode_tok_s` scaling as the
headline and cross-check absolute acceptance against the real-text benchmarks.
Setup and caveats: [AgentX section](./mtp_server_eval/README.md#agentx-agentic-trace-replay-load-test).

## Datasets / benchmark names

Names below are valid in `eval.benchmarks` / `BENCHMARKS=` (`MODE=acceptance`)
and ship (or prepare) as `mtp_server_eval/data/<name>.jsonl`.

| Eval name | Notes |
|-----------|--------|
| `aime`, `gpqa`, `livecodebench` | Default smoke trio in `run_eval.sh` |
| `gsm8k`, `math500`, `humaneval`, `mbpp`, `mt-bench`, `aime26` | From `eval_datasets/` |
| `swe-bench-pro`, `swe-rebench` | SWE-style; large — often kept off-git |
| `aa-lcr` | Long-context (~tens of k tokens) |
| `aa-lcr-1k` … `aa-lcr-32k` | Context-length sweep — see [Context-length sweep](#context-length-sweep) |

### Context-length sweep

To measure **acceptance vs context length**, use the AA-LCR sweep rather than
contrasting the short qualitative slices against `speed-low-entropy`: that
contrast confounds length with prompt entropy, since the low-entropy slice is
repetitive code boilerplate that drafts unusually well at any length.

[`prepare_aa_lcr_sweep.py`](./prepare_aa_lcr_sweep.py) takes the 100
[AA-LCR](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR)
multi-document questions (all ≥71k tokens) and truncates **only the document
block** to each target, holding the instruction header and the question fixed at
the end. Every bin therefore contains the same 100 items at a different context
length — a paired design, so a difference across bins is attributable to length.

```bash
python scripts/evaluate/prepare_aa_lcr_sweep.py          # ~25 MB, gitignored
cd scripts/evaluate/experiments
python run_experiments.py --config gemma4-kimi-mtp-stem-code-math-900k-ctxlen-sweep.yaml
```

| Eval name | Prompt tokens | n |
|-----------|---------------|---|
| `aa-lcr-1k` | 1024–1025 | 100 |
| `aa-lcr-2k` | 2048–2049 | 100 |
| `aa-lcr-4k` | 4096–4097 | 100 |
| `aa-lcr-8k` | 8192–8193 | 100 |
| `aa-lcr-16k` | 16384–16385 | 100 |
| `aa-lcr-32k` | 32768–32769 | 100 |

Each bin is its own benchmark, so the existing per-benchmark counter deltas give
one `accept_length` / `accept_rate` per context length — read the curve directly
off the `SUMMARY` table. `server.max_model_len` must cover the largest bin plus
`eval.max_tokens` (the shipped config uses 36864 for the 32k bin + 1024).

Caveat: acceptance is aggregated per benchmark from cumulative vLLM counters, so
each bin yields a single point with no within-bin variance. For error bars, the
counters would need scraping per request — see [`TODO.md`](./TODO.md).

### SPEED-Bench (included)

NVIDIA [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) is in
[`full-eval.yaml`](./experiments/full-eval.yaml). Prepare once with
[`prepare_speedbench.py`](./prepare_speedbench.py), then `prepare_data.py`.

The qualitative split is **11 categories x 80 prompts = 880** (SPEED-Bench paper,
[arxiv 2604.09557](https://arxiv.org/abs/2604.09557) Table 1). All 11 are wired up:

| Eval name | SPEED-Bench split |
|-----------|-------------------|
| `speed-coding` | qualitative / coding |
| `speed-humanities` | qualitative / humanities |
| `speed-math` | qualitative / math |
| `speed-multilingual` | qualitative / multilingual |
| `speed-qa` | qualitative / QA |
| `speed-rag` | qualitative / RAG |
| `speed-reasoning` | qualitative / reasoning |
| `speed-roleplay` | qualitative / roleplay |
| `speed-stem` | qualitative / STEM |
| `speed-summarization` | qualitative / summarization |
| `speed-writing` | qualitative / writing |
| `speed-low-entropy` | `throughput_16k` / `low_entropy` (512 prompts; 15.8k–21.4k tokens each, so it needs `max_model_len` ~32768 — the card's “10k input” understates it) |

**`export HF_TOKEN` before preparing.** Several categories are built from gated
or auth-only sources (notably `cais/hle`, which feeds STEM / humanities / math).
Without it those rows materialise empty; `prepare_speedbench.py` now fails loudly
listing the affected categories instead of silently writing a partial split.

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
mtp_server_eval/       run_eval.sh (acceptance + GuideLLM)
  run_agentx.sh        AgentX agentic trace replay (aiperf)
  compare_agentx.py    AgentX per-concurrency speedup table
  agentx_metrics.py    one aiperf artifact dir → one matrix cell
experiments/           YAML runner (serve → eval → compare)
  full-eval.yaml       recommended full-suite template
  agentx-gemma4.yaml   AgentX baseline vs draft (eval.mode: agentx)
  run_full_eval.sh     thin wrapper around run_experiments.py
  example.yaml         smaller example config
```
