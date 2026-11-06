# Gemma-4-31B + Assistant Draft: BFCL Evaluation Results

Speculative-decoding evaluation of `google/gemma-4-31B-it` with the official
`google/gemma-4-31B-it-assistant` draft model on **BFCL** (Berkeley Function
Calling Leaderboard, v3 AST core: simple / multiple / parallel /
parallel-multiple). BFCL prompts carry long structured JSON function schemas
and expect short, low-entropy function-call completions — a near-ideal workload
for speculative decoding.

## Setup

| Parameter | Value |
|---|---|
| Backbone | `google/gemma-4-31B-it` (local, bf16) |
| Draft | `google/gemma-4-31B-it-assistant` |
| Hardware | 4× NVIDIA A100 80GB |
| `tensor_parallel_size` | 4 |
| `max_model_len` | 8,192 |
| Benchmark | `bfcl` (100 of 1,000 prompts, seed 42) |
| `max_tokens` | 2,048 |
| Temperature | 0.0 (greedy) |
| vLLM | 0.24.0 |
| Eval tool | `scripts/evaluate/experiments/run_experiments.py` |
| Config | `scripts/evaluate/experiments/gemma4-31b-bfcl.yaml` |

## Main Results

| config | n | decode tok/s | e2e tok/s | TTFT (s) | accept_len | accept_rate | speedup (decode) |
|---|---|---|---|---|---|---|---|
| baseline | 100 | 55.6 | 48.5 | 0.140 | — | — | 1.00× |
| assistant k=3 | 100 | 145.4 | 94.8 | 0.178 | 3.87 | 0.955 | **2.62×** |
| assistant k=5 | 100 | 193.7 | 119.9 | 0.153 | 5.51 | 0.902 | **3.48×** |

`accept_rate` = accepted_tokens / draft_tokens (vLLM cumulative counters,
delta over the run). `accept_len` is out of a maximum of k+1 (the k speculated
tokens plus the target's always-accepted bonus token).

### Key Observations

- **BFCL yields the highest speedups we've measured with this pairing** —
  3.48× decode throughput at k=5. Function-call output is highly predictable
  (structured syntax, argument names copied from the schema in the prompt), so
  the draft almost always agrees with the target.
- **Acceptance is near the ceiling**: 95.5% of drafted tokens accepted at k=3
  (accept_len 3.87 / 4) and still 90.2% at k=5 (accept_len 5.51 / 6). Deeper
  speculation keeps paying off — k=5 clearly beats k=3 here, whereas on
  harder free-form workloads acceptance usually decays faster with depth.
- **Completions are short** (~47 tokens on average), so end-to-end tok/s is
  diluted by per-request overhead and prefill; the decode-throughput speedup
  is the cleaner signal for this workload.

## Running the Evaluation

```bash
# One-time data prep (BFCL v3 AST core → turns → prompt files):
cd scripts/evaluate
python eval_datasets/convert_eval_datasets_to_jsonl.py \
    gorilla-llm/Berkeley-Function-Calling-Leaderboard
python mtp_server_eval/prepare_data.py --only bfcl

# Automated sweep (launches server, runs eval, compares):
cd experiments
python run_experiments.py --config gemma4-31b-bfcl.yaml

# Dry run (preview commands without launching):
python run_experiments.py --config gemma4-31b-bfcl.yaml --dry-run
```

Raw outputs land in `scripts/evaluate/experiments/results/gemma4-31b-bfcl/`
(`mtp_eval_summary.json` / `mtp_eval_details.jsonl` per experiment).

> **Environment note:** if vLLM startup fails with
> `nvcc fatal: Unknown option '-generate-dependencies-with-compile'`, the
> FlashInfer JIT picked up an old system `nvcc`. Point it at the toolkit
> matching your torch build, e.g.
> `export CUDA_HOME=/usr/local/cuda-12.9 && export PATH=$CUDA_HOME/bin:$PATH`.

## Caveats

- This measures **acceptance and throughput only** — it does not score
  function-call correctness (no BFCL AST checking). Speculative decoding with
  greedy sampling is output-lossless, so leaderboard-style scoring is not
  needed to validate the speedup.
- The 100-prompt sample spans all four AST core categories, but per-category
  breakdown is not reported (prompt ids are sample indices, not BFCL ids).
