# Gemma-4-31B Google Assistant (MTP) k=5: RDU vs GPU throughput (internal only)

Internal companion to
[Gemma-4-31B Assistant k=5: RDU vs GPU Acceptance](../user_guide/tutorials/gemma4_31b_assistant_rdu_vs_gpu.md).
That customer-facing doc is intentionally acceptance-only
(`accept_len`/`accept_rate`). This doc holds the raw throughput numbers pulled
from the same two runs, and why they should **not** be shared externally or
read as a hardware throughput comparison.

Same hardware as the acceptance doc: GPU 4x A100 80GB, TP=4, vLLM
0.24.0+cu129; RDU 16x SN40 (`sc3-s240`), CoE PEF, `snruntime=1.38.0`.
Source: `scripts/evaluate/experiments/results/gemma4-31b-full/assistant_k5/mtp_eval_summary.json`
(GPU) and `scripts/evaluate/experiments/results/gemma4-31b-rdu-k5-corrected/mtp_eval_summary.json`
(RDU corrected run).

## Table

| benchmark | n GPU/RDU | GPU decode tok/s | RDU decode tok/s | GPU e2e tok/s | RDU e2e tok/s | GPU TTFT (s) | RDU TTFT (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| HumanEval | 50 / 50 | 186.6 | 123.8 | 178.0 | 49.1 | 0.086 | 3.901 |
| aime | 30 / 30 | 168.8 | 224.5 | 166.8 | 144.5 | 0.096 | 3.844 |
| aime26 | 30 / 30 | 158.7 | 230.2 | 157.5 | 149.6 | 0.095 | 3.977 |
| bfcl | 50 / 50 | 200.5 | 304.8 | 121.2 | 10.1 | 0.141 | 3.975 |
| gpqa | 50 / 50 | 155.4 | 232.3 | 152.0 | 109.1 | 0.129 | 3.845 |
| gsm8k | 50 / 50 | 199.2 | 189.4 | 190.2 | 56.4 | 0.077 | 3.845 |
| humaneval | 50 / 50 | 197.1 | 95.2 | 185.3 | 40.8 | 0.093 | 3.835 |
| livecodebench | 50 / 50 | 152.0 | 211.4 | 147.3 | 105.9 | 0.167 | 3.845 |
| math500 | 50 / 50 | 185.4 | 185.9 | 181.4 | 90.1 | 0.085 | 3.858 |
| math_reasoning | 50 / 50 | 201.7 | 178.7 | 187.2 | 34.8 | 0.071 | 3.970 |
| mbpp | 50 / 50 | 171.7 | 208.1 | 169.1 | 96.0 | 0.066 | 3.849 |
| mt-bench | 50 / 50 | 128.5 | 176.8 | 126.3 | 74.3 | 0.073 | 3.891 |
| qa | 50 / 50 | 124.8 | 164.3 | 121.1 | 43.4 | 0.062 | 4.001 |
| question | 50 / 50 | 128.3 | 177.1 | 126.2 | 75.3 | 0.072 | 3.813 |
| rag | 50 / 50 | 141.1 | 91.0 | 106.8 | 13.8 | 0.151 | 4.006 |
| speed-coding | 50 / 50 | 174.7 | 178.0 | 168.8 | 84.5 | 0.100 | 3.970 |
| speed-multilingual | 47 / 50 | 169.9 | 191.7 | 164.5 | 57.1 | 0.082 | 3.967 |
| speed-qa | 50 / 50 | 124.8 | 164.2 | 121.2 | 43.4 | 0.063 | 3.997 |
| speed-rag | 50 / 50 | 150.6 | 97.0 | 104.6 | 15.9 | 0.204 | 3.978 |
| speed-writing | 50 / 50 | 103.1 | 137.6 | 101.6 | 98.6 | 0.196 | 3.905 |
| summarization | 50 / 50 | 111.2 | 106.1 | 95.1 | 26.2 | 0.219 | 3.992 |
| swe-bench-pro | 50 / 50 | 147.0 | 110.2 | 140.0 | 66.4 | 0.234 | 3.981 |
| tool_call | 50 / 50 | 148.5 | 147.1 | 116.5 | 22.5 | 0.192 | 3.912 |
| translation | 50 / 50 | 160.7 | 232.4 | 153.7 | 43.2 | 0.066 | 3.998 |
| writing | 50 / 50 | 128.4 | 176.7 | 126.3 | 75.3 | 0.070 | 3.808 |

Macro average: `decode_tok_s` GPU 156.7 / RDU 173.4; `e2e_tok_s` GPU 144.3 /
RDU 65.1; `mean_ttft_s` GPU 0.116 / RDU 3.919.

## Why this is not a hardware throughput comparison

- **RDU's `decode_tok_s`** is competitive with, and on many benchmarks higher
  than, GPU's. But RDU's mean TTFT is a near-constant ~3.8-4.0s on *every*
  benchmark (vs GPU's ~0.06-0.23s), which drags `e2e_tok_s` down to roughly
  half of GPU's. A flat, benchmark-independent TTFT is the signature of fixed
  per-request overhead, not a hardware decode-speed difference. Verified
  directly against per-sample records in `mtp_eval_details.jsonl`: RDU TTFT
  is ~3.82-3.87s on every single `aime` sample regardless of prompt length
  (85-416 tokens) — it does not scale with prefill work, so it isn't prefill
  compute.
- **Root cause**: the RDU harness (`scripts/evaluate/rdu_mtp_eval/rdu_mtp_eval.py`)
  says outright in its own docstring that the comparison target is
  `accept_length`/`accept_rate`, not decode tok/s. The compile log for the
  corrected run (`scripts/evaluate/rdu_mtp_eval/logs-corrected/snrdu.log`)
  confirms the CoE PEF was compiled with `is_continuous_batching: False` and
  `batch_size: 'BS:2:8:2:8'`.
- **Batch size, precisely**: neither backend is actually running batched
  requests in this eval. The client-side loop on both sides
  (`scripts/evaluate/mtp_server_eval/run_vllm_eval.py` and
  `scripts/evaluate/rdu_mtp_eval/rdu_mtp_eval.py`) sends one request at a
  time, sequentially, and waits for the full response before sending the
  next — there is never more than one real logical request in flight on
  either backend.
  - GPU: vLLM server is launched with no `--max-num-seqs` override
    (`scripts/evaluate/experiments/run_experiments.py::build_serve_command`),
    so it defaults to 256 — real serving capacity that this eval never
    exercises.
  - RDU: the CoE PEF has a **hard, statically compiled** batch size of 2.
    `_pipeline.py::run_one()` fills that compiled shape by duplicating the
    single real prompt into both batch slots (see its docstring: "Duplicate
    prompt to BS=2") — it is not two independent concurrent requests.
  - Net effect: both backends' numbers are single-request (effective batch
    size 1) latency measurements. GPU's stack has low fixed cost per request
    at batch 1 (mature continuous-batching engine, unused here); RDU's stack
    has a large, unavoidable fixed cost per request at batch 1 (dispatch
    into a statically compiled graph). The size of the resulting gap is
    largest on short-completion benchmarks (`bfcl`, `rag`: ~42-65 mean
    completion tokens, where the ~3.9s fixed overhead is 83-95% of total
    wall time) and smallest on long-completion benchmarks (`aime`,
    `swe-bench-pro`: ~660-1560 mean completion tokens, where it's only
    36-39%) — consistent with a fixed additive overhead, not a
    multiplicative decode-speed penalty.
- A meaningful RDU throughput number would require a continuous-batching RDU
  serving stack, which this evaluation does not use. Do not extrapolate a
  production RDU-vs-GPU throughput claim from this table.

## Do not share externally

These numbers reflect an unbatched correctness harness, not a fair
serving-stack throughput test. Sharing them without this context would
misrepresent RDU serving performance. Keep this doc internal; the
customer-facing doc stays acceptance-only.
