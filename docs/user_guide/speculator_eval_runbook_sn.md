# Speculator Evaluation Runbook (SambaNova)

How to evaluate a speculator (speculative-decoding draft model) against any
backbone on SambaNova's 25-benchmark acceptance suite, with the same prompts,
sample draw, decoding settings and metric definitions we use, so results from
different teams, backbones and speculator architectures can be placed side by
side.

The evaluation is black-box and engine-level. A vLLM server hosts the backbone
with the speculator attached; a client sends the prompts one at a time and
reads vLLM's own speculative-decoding counters. There is no grading and no
per-benchmark logic: a benchmark is a file of prompts. Any backbone vLLM can
serve and any speculator vLLM can attach with `--speculative-config` works.

Throughout, "speculator" and "draft" both mean the draft model; "target" or
"backbone" means the model it speculates for. Gemma-4-31B-it and Kimi K3 appear
only as worked examples (§7).

**What you receive with this document:** a `data/` directory with the 25
prompt files, and `reference_manifest.md5` (§2). Everything else is public
software (vLLM) plus the protocol in §3, which is small enough to implement in
any HTTP client.

Contents

1. The benchmark suite
2. The data
3. Running the evaluation
4. Metrics to collect
5. Reporting and what to send back
6. Gotchas
7. Worked examples: Gemma-4-31B-it (dense) and Kimi K3 (MoE)

---

## 1. The benchmark suite

25 named benchmarks, each a JSONL file of prompts. Every request is a single
user turn, formatted by the backbone's own chat template, answered greedily
with up to 4096 new tokens. **50 prompts per benchmark** are drawn with a fixed
seed (§3.2); benchmarks with 30 rows use all of them.

| Group | Benchmark | Rows | Prompt tokens (mean / max)¹ | Typical output² | Source |
|---|---|---:|---|---|---|
| Math | `aime` | 30 | 116 / 403 | long CoT, ~1.4k | AIME 2024 |
| | `aime26` | 30 | 139 / 355 | long CoT, ~1.7k; 4–5 of 30 hit the 4096 cap | [MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026) |
| | `math500` | 500 | 107 / 866 | ~650 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) |
| | `gsm8k` | 1319 | 75 / 175 | ~300 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| | `math_reasoning` | 80 | 57 / 115 | ~170 | [RedHatAI/speculator_benchmarks](https://huggingface.co/datasets/RedHatAI/speculator_benchmarks) |
| Code | `humaneval` | 164 | 150 / 441 | ~280 | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval), DeepSpec prompt format |
| | `HumanEval` | 164 | 130 / 421 | ~310 | speculator_benchmarks, RedHat prompt format (different prompts from `humaneval`) |
| | `mbpp` | 257 | 18 / 50 | ~670 | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) |
| | `livecodebench` | 1055 | 479 / 970 | ~740 | [livecodebench/code_generation_lite](https://huggingface.co/datasets/livecodebench/code_generation_lite) |
| | `swe-bench-pro` | 731 | 766 / 2178 | ~670 | [ScaleAI/SWE-bench_Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) |
| | `speed-coding` | 80 | 282 / 1459 | ~460 | [nvidia/SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) qualitative / coding |
| Reasoning / QA | `gpqa` | 198 | 295 / 2402 | ~760 | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) Diamond (gated) |
| | `qa` | 80 | 11 / 17 | ~230 | speculator_benchmarks (short trivia) |
| | `speed-qa` | 80 | 11 / 17 | ~230 | SPEED-Bench / QA (same prompts as `qa`) |
| | `question` | 80 | 58 / 361 | ~490 | speculator_benchmarks (MT-Bench first turns) |
| | `mt-bench` | 80 | 58 / 361 | ~490 | [HuggingFaceH4/mt_bench_prompts](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts) |
| Tool use | `bfcl` | 1000 | 419 / 975 | ~40 (a function call) | [gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) v3 AST core |
| | `tool_call` | 200 | 527 / 3088 | ~100 | speculator_benchmarks |
| Long input | `rag` | 80 | 703 / 821 | ~60 | speculator_benchmarks |
| | `speed-rag` | 80 | 875 / 4629 | ~65 | SPEED-Bench / RAG |
| | `summarization` | 80 | 709 / 1461 | ~140 | speculator_benchmarks |
| Language | `translation` | 80 | 36 / 69 | ~210 | speculator_benchmarks (WMT de→en) |
| | `speed-multilingual` | 80 | 170 / 784 | ~400 | SPEED-Bench / multilingual |
| | `speed-writing` | 80 | 584 / 2554 | ~1.3k | SPEED-Bench / writing |
| | `writing` | 80 | 58 / 361 | ~490 | speculator_benchmarks (same prompts as `question`) |

¹ Over the 50 sampled prompts, Gemma-4 tokenizer, before chat template. Other
tokenizers differ by ±20%. The longest prompt in the suite is under 5k tokens;
there is no long-context benchmark in this suite.
² Mean completion tokens from Gemma-4-31B-it greedy; other backbones differ.

### 1.1 Duplicate prompt sets

The 25 names are **22 distinct prompt sets**. Keep all 25 for per-benchmark
tables (that is how every historical run was done), but compute means over the
distinct sets (§5).

| Kept | Duplicate (drop from means) | Overlap |
|---|---|---|
| `qa` | `speed-qa` | 80 / 80 identical |
| `question` | `writing`, `mt-bench` | 80 / 80 and 79 / 80 identical |
| `rag` | `speed-rag` (kept; partial overlap) | 55 / 80 shared |

### 1.2 What each group tells you

Math and code have the highest acceptance for every speculator we have
measured; open-ended chat (`mt-bench`, `qa`, `speed-writing`, `summarization`)
the lowest. `bfcl`, `rag`, `speed-rag` and `tool_call` produce 40 to 100 output
tokens per request, so their acceptance rests on only 2k to 5k decode tokens
and is noisy at the ±0.1 level. `speed-multilingual` separates speculators
trained on English-only data from the rest.

---

## 2. The data

### 2.1 Format

`data/<benchmark>.jsonl`, one JSON object per line, three string fields:

```json
{"benchmark": "gsm8k", "id": "17", "prompt": "Janet's ducks lay 16 eggs per day. ..."}
```

`prompt` is the complete user message. It is sent as-is: no system prompt, no
few-shot prefix, no instruction appended. The backbone's chat template is the
only formatting applied.

### 2.2 Use the files as delivered

Every file derives from the public dataset in the §1 table, but the exact rows
kept, their order, and therefore which 50 the fixed seed selects, come from our
conversion. A file you rebuild yourself yields a different sample. Verify the
delivered files:

```bash
cd data
md5sum *.jsonl > my_manifest.md5
diff <(sort my_manifest.md5) <(sort ../reference_manifest.md5) && echo "data OK"
```

`reference_manifest.md5`:

```
4efac42e10a60df434d4335857039cfd  aime.jsonl
2c2f2ccbb37db40950ca8aa9959d90b3  aime26.jsonl
70b2621d03f8bba4dc9f40359a0752d8  bfcl.jsonl
6e6e7b6ffd7a0e864478465d12cf0113  gpqa.jsonl
0ac068705bcab8f1eb6e76ecd34e7357  gsm8k.jsonl
6b21bda72fef34e67d919c7154edd769  HumanEval.jsonl
e6bfb1c34b32fbbebaef1a76e10f97ad  humaneval.jsonl
dc3239e00ca6221191f60498b3aaf036  livecodebench.jsonl
e2de4fb12186839446101c4acbf5179e  math500.jsonl
3a88be0d8ae521e94552e5342223b91a  math_reasoning.jsonl
2f72c2789469992beb1bb05373f40eb0  mbpp.jsonl
43c813681257833f4a70f00850c50fd9  mt-bench.jsonl
a74550d479599d6220b39ba197afaade  qa.jsonl
db3a49843ea05d49d9f9ab3dad864549  question.jsonl
c0635b3054ee13735de73a8ccb8fa15f  rag.jsonl
3aeef8449ad706158029a0416fbfce8a  speed-coding.jsonl
fbcdb9fc8294c1146ea437d9ad1029f8  speed-multilingual.jsonl
ca3b99968988324dc92fff0888d11c0f  speed-qa.jsonl
c6907938f6ad412675c359b71dfc44f8  speed-rag.jsonl
2047ccc26b88a148fea6d3e50c2658b8  speed-writing.jsonl
4b759c853cbb6ff8cd38496d0216452e  summarization.jsonl
403b5f7279daa9be25ac65c25b59e8a7  swe-bench-pro.jsonl
6a03e258bcfd4e26fc3459a66d39916d  tool_call.jsonl
ed619153b41c8b321021a933887ab33a  translation.jsonl
1f1a97293ac9af2d87af3538b285763e  writing.jsonl
```

GPQA is gated on Hugging Face. Accept its terms before using `gpqa.jsonl`, and
do not redistribute it further. All other sources are open; their licenses
apply.

### 2.3 How the files were derived (for reference, not for rebuilding)

- Multi-turn sources (GSM8K, HumanEval, MBPP, MATH-500, MT-Bench, AIME 2026,
  BFCL, SWE-bench Pro): the first user turn of each conversation, in the
  upstream test-split order, with the prompt wording of the
  [DeepSpec](https://github.com/deepseek-ai/DeepSpec) evaluation set.
- `speed-*`: the 80 prompts of the corresponding SPEED-Bench qualitative
  category, verbatim.
- `HumanEval`, `math_reasoning`, `qa`, `question`, `rag`, `summarization`,
  `tool_call`, `translation`, `writing`: the corresponding
  RedHatAI/speculator_benchmarks subset, verbatim.
- `aime`: the 30 AIME 2024 problems (I and II). `gpqa`: the 198 GPQA-Diamond
  questions with shuffled multiple-choice options. `livecodebench`:
  code_generation_lite problem statements with the starter code appended.

### 2.4 Adding a benchmark

Any `<name>.jsonl` with the same three fields can be run with the same
protocol. Report it in a separate table; it is not part of the 25 and has no
reference numbers.

---

## 3. Running the evaluation

### 3.1 Software

- **vLLM 0.28.0** is what we run. Any version that serves your speculator
  works, provided the baseline runs on the **same build**. If the speculator
  needs a fork or plugin, the baseline runs on that fork too. Record the version
  or commit.
- A client that speaks the OpenAI chat-completions API with streaming and can
  fetch `/metrics`. §3.4 specifies exactly what it must do.

### 3.2 Settings

Fixed for every backbone and speculator:

| Setting | Value |
|---|---|
| Decoding | `temperature = 0.0` (greedy), `max_tokens = 4096` |
| Samples | 50 per benchmark, drawn as `random.Random(42).sample(rows, 50)` over the file's rows in order (Python's `random` module). Files with ≤ 50 rows use all rows. Do not change the seed or the count. |
| Request | one user message = the `prompt` field; no system prompt; streaming with `stream_options.include_usage = true`; the backbone's own chat template |
| Concurrency | 1: requests are sent one after another |
| Server extras | none: no reasoning parser, tool parser, chat-template override, or quantization unless the backbone ships that way. If you must add one, add it to the baseline too and report it. |
| Benchmarks | all 25, in the order of the §1 table's first column groups as listed in §3.3 |

Chosen per backbone, identical between its baseline and every speculator run:

| Setting | Guidance |
|---|---|
| Backbone | HF id or local path, bf16 as published. |
| GPUs, tensor parallelism | Whatever fits the backbone. Acceptance is TP-independent; tok/s is not. |
| `--max-model-len` | `>= 5000 + 4096`. Use **16384** unless memory forbids it. The longest prompt is ~4.6k tokens; at 8192 it is rejected and that benchmark quietly ends with 49 requests. |
| `--gpu-memory-utilization` | 0.9 |
| `num_speculative_tokens` (`k`) | The speculator's native depth. Report it. |
| Other `--speculative-config` keys | Whatever your proposer needs (`method`, …). Report them. |

### 3.3 Serve

Baseline (target alone):

```bash
vllm serve <backbone> --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size <TP> --gpu-memory-utilization 0.9 --max-model-len 16384
```

With the speculator:

```bash
vllm serve <backbone> --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size <TP> --gpu-memory-utilization 0.9 --max-model-len 16384 \
  --speculative-config '{"model": "<speculator-dir>", "num_speculative_tokens": <k>, "method": "<method>"}'
```

Wait for `GET /health` to return 200 before sending anything. Weight load plus
CUDA-graph capture takes 5 to 10 minutes for a 30B-class model.

Benchmark order:

```
aime, gpqa, livecodebench, gsm8k, humaneval, mbpp, math500, mt-bench, aime26, bfcl,
swe-bench-pro, speed-coding, speed-multilingual, speed-rag, speed-qa, speed-writing,
HumanEval, math_reasoning, qa, question, rag, summarization, tool_call, translation, writing
```

### 3.4 Client protocol

For each benchmark, in order:

1. **Load** `data/<name>.jsonl` into a list, one record per non-empty line, in
   file order. If it has more than 50 rows, `rows = random.Random(42).sample(rows, 50)`.
2. **Read counters** `GET /metrics` and sum, over all label sets, the Prometheus
   counters
   `vllm:spec_decode_num_drafts`, `vllm:spec_decode_num_draft_tokens`,
   `vllm:spec_decode_num_accepted_tokens` (they carry a `_total` suffix in the
   exposition). Absent counters mean speculative decoding is off (baseline).
3. **For each row**, `POST /v1/chat/completions`:

   ```json
   {"model": "<served model id>",
    "messages": [{"role": "user", "content": "<prompt>"}],
    "max_tokens": 4096, "temperature": 0.0,
    "stream": true, "stream_options": {"include_usage": true}}
   ```

   Record, per request: `prompt_tokens` and `completion_tokens` from the final
   usage chunk; `t0` when the request is sent; `t_first` when the first chunk
   arrives whose delta carries any non-empty text field other than `role`
   (`content`, `reasoning_content`, …); `t_end` when the stream closes. A
   request that errors is logged and skipped; it does not count toward `n`.
4. **Read counters again**; take deltas Δdrafts, Δdraft_tokens, Δaccepted.
5. **Per-benchmark row:**

   | field | formula |
   |---|---|
   | `n` | requests that completed |
   | `total_completion_tokens` | Σ completion_tokens |
   | `decode_tok_s` | Σ (completion_tokens − 1) / Σ (t_end − t_first), over requests with completion_tokens > 1 |
   | `e2e_tok_s` | Σ completion_tokens / Σ (t_end − t0) |
   | `mean_ttft_s` | mean (t_first − t0) |
   | `accept_length` | 1 + Δaccepted / Δdrafts (null for the baseline) |
   | `accept_rate` | Δaccepted / Δdraft_tokens (null for the baseline) |

Write the 25 rows as `summary.json` and the per-request records as
`details.jsonl`. Do not send any other traffic to the server between steps 2
and 4.

Budget: the baseline generates about 565k completion tokens in total, so it
takes roughly 565k ÷ (single-stream decode tok/s) seconds, about 3 hours for a
30B dense model on four A100s. Each speculator run is a third to a half of that.

Smoke test first with one benchmark and 5 samples, then run the full list.

### 3.5 What the speculator must provide

1. Servable through `--speculative-config`: an upstream vLLM `method`
   (`eagle3`, `mtp`, `dspark`, `draft_model`, …) or your own proposer in a fork
   or plugin.
2. The three counters in §3.4 step 2 must advance. Upstream vLLM emits them for
   every method; a new proposer must feed the same `SpecDecodingStats`, or
   acceptance comes back null.
3. Exact greedy verification. Relaxed acceptance (`accept_tolerance`,
   typical-acceptance, etc.) changes the output and must be reported as a
   separate configuration.

---

## 4. Metrics to collect

| Field | Meaning | Compare across backbones / sites? |
|---|---|---|
| **`accept_length`** | Tokens committed per target forward pass, bonus token included. Upper bound `k + 1`. | **Yes.** The primary metric. Hardware- and TP-independent: the same Gemma-4-31B-it + assistant run on 4× A100 and on 16× SambaNova SN40 RDU differed by at most 0.17 on any benchmark (median 0.05). |
| `accept_rate` | Share of drafted tokens accepted. | Only between speculators at the same `k`. |
| `decode_tok_s` | Decode-phase output rate, prefill excluded. | Only against the same box's baseline. |
| **speedup** | `decode_tok_s(speculator) / decode_tok_s(baseline)`, same box, same build, same `max_model_len`. | Yes, as a ratio, with the hardware stated. |
| `e2e_tok_s`, `mean_ttft_s` | End-to-end rate incl. prefill; time to first token. | Reference only. |
| `n` | Must be 50 (30 for aime / aime26). | Sanity. Fewer = a rejected request. |
| `total_completion_tokens` | Baseline and speculator must agree within a few percent under greedy. | Sanity. A larger gap means the speculator changed the output. |

Also record, once per run: backbone id and revision; speculator checkpoint,
its `config.json` and `k`; every `--speculative-config` key; vLLM version or
fork commit; GPU model and count; TP; `max_model_len`; `my_manifest.md5`.

---

## 5. Reporting and what to send back

Per backbone, one table with a row per benchmark and, per speculator, columns
`accept_len`, `accept_rate`, `decode tok/s`, `speedup`.

**Means:** over the 22 distinct sets, i.e. all 25 minus `speed-qa`, `writing`
and `mt-bench`. State that next to the number. Also report the per-group means
from §1 (math, code, reasoning/QA, tool use, long input, language); speculators
differ far more by group than in the overall mean.

Send:

- `summary.json` and `details.jsonl` for the baseline and for each speculator
  configuration, unedited, plus the vLLM server logs.
- The exact `vllm serve` command lines.
- `pip freeze`, `nvidia-smi`, vLLM version or fork commit, and any patch or
  plugin needed to serve the speculator.
- The speculator's `config.json` and every `k` evaluated.
- `my_manifest.md5`.

---

## 6. Gotchas

- **`accept_length` null** with a speculator attached: the counters did not
  move. The proposer is not reporting stats, or vLLM fell back to another
  method. Check the loaded `speculative_config` in the server log.
- **`n` below 50** on a benchmark: a request was rejected, almost always
  prompt + `max_tokens` > `max_model_len`. Use 16384.
- **`total_completion_tokens` differs by more than a few percent** between
  baseline and speculator: the output changed. Exact greedy verification must
  reproduce the baseline. Check for relaxed-acceptance settings.
- **Reasoning / thinking backbones**: thinking tokens count as completion
  tokens and TTFT is the first streamed token of any kind. Serve the backbone
  with its default thinking behaviour, the same for baseline and speculator,
  and say which it was.
- **Counters are cumulative** since server start. Always use deltas, and do not
  share the server with other traffic during a run.
- **Noise floor**: ±0.1 `accept_len` on the short-output benchmarks (§1.2),
  ±0.03 elsewhere; `aime26` hits the 4096 cap on 4 or 5 of 30 prompts.

---

## 7. Worked examples

### 7.1 Gemma-4-31B-it with the Google assistant speculator (setup check)

Use this to check a new setup: run the public backbone with its public
speculator, then compare `accept_len` to the table. Agreement within about
±0.15 on every benchmark means the server, data and client match ours, and any
later difference is the speculator under test.

```bash
export HF_TOKEN=hf_...     # Gemma weights are gated
hf download google/gemma-4-31B-it           --local-dir /models/gemma-4-31B-it
hf download google/gemma-4-31B-it-assistant --local-dir /models/gemma-4-31B-it-assistant

vllm serve /models/gemma-4-31B-it --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --max-model-len 16384          # baseline
vllm serve /models/gemma-4-31B-it --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --max-model-len 16384 \
  --speculative-config '{"model": "/models/gemma-4-31B-it-assistant", "num_speculative_tokens": 5}'
```

Our numbers. 4× A100 80GB, TP=4, vLLM 0.24.0, `max_model_len` 8192, greedy,
50 samples (30 for aime / aime26). Baseline decode ≈ 55 tok/s on this box;
compare `accept_len`, not tok/s.

| benchmark | n | accept_len (k=5) | accept_rate | speedup on our box |
|---|---:|---:|---:|---:|
| aime | 30 | 4.85 | 0.770 | 3.11× |
| gpqa | 50 | 4.48 | 0.696 | 2.85× |
| livecodebench | 50 | 4.54 | 0.708 | 2.80× |
| gsm8k | 50 | 5.07 | 0.814 | 3.60× |
| humaneval | 50 | 5.18 | 0.837 | 3.58× |
| mbpp | 50 | 4.52 | 0.705 | 3.12× |
| math500 | 50 | 5.04 | 0.807 | 3.38× |
| mt-bench | 50 | 3.41 | 0.482 | 2.33× |
| aime26 | 30 | 4.77 | 0.754 | 2.94× |
| bfcl | 50 | 5.59 | 0.917 | 3.63× |
| swe-bench-pro | 50 | 4.47 | 0.693 | 2.72× |
| speed-coding † | 50 | 4.73 | 0.747 | 3.18× |
| speed-multilingual † | 47 | 4.43 | 0.686 | 3.08× |
| speed-rag † | 50 | 4.39 | 0.677 | 2.75× |
| speed-qa | 50 | 3.16 | 0.431 | 2.25× |
| speed-writing | 50 | 3.12 | 0.423 | 1.91× |
| HumanEval | 50 | 4.90 | 0.780 | 3.38× |
| math_reasoning | 50 | 5.07 | 0.813 | 3.64× |
| qa | 50 | 3.16 | 0.431 | 2.25× |
| question | 50 | 3.41 | 0.482 | 2.33× |
| rag | 50 | 4.15 | 0.631 | 2.58× |
| summarization | 50 | 3.29 | 0.458 | 2.04× |
| tool_call | 50 | 4.01 | 0.602 | 2.69× |
| translation | 50 | 4.02 | 0.605 | 2.90× |
| writing | 50 | 3.41 | 0.482 | 2.33× |
| **mean, 22 distinct sets** | | **4.42** | | |

† These three files were completed after this run (63→80, 47→80 and 70→80
rows), so the delivered files give a different 50-sample draw. Expect those rows
to be close but not matching. All other rows use identical prompts.

### 7.2 Kimi K3 with public DSpark speculators (large MoE backbone)

Same suite and metric definitions on a ~1T-parameter MoE backbone, to show the
range of the suite on a very different target. Kimi K3 served with vLLM 0.29.0
on 8× NVIDIA B300, TP=8; speculator `Inferact/Kimi-K3-DSpark` attached with
`method: dspark`, `k = 7`, greedy. No-draft baseline decode ≈ 108 tok/s.
`accept_len` is read from the same vLLM counters as §3.4.

This run predates the protocol in §3 and differs from it in three ways, so
treat the numbers as indicative, not as reference values: 15 to 25 prompts
per benchmark instead of 50; generation capped at each prompt's earlier
greedy output length (≤ 512 tokens) instead of 4096; and the set list has
`aa-lcr-1k`, `aa-lcr-4k` and `swe-rebench` in place of `humaneval`,
`HumanEval`, `math_reasoning` and `question`.

| benchmark | n | accept_len (k=7) | accept_rate | decode tok/s | speedup |
|---|---:|---:|---:|---:|---:|
| gsm8k | 25 | 5.55 | 0.650 | 360 | 3.33× |
| bfcl | 15 | 4.86 | 0.552 | 255 | 2.36× |
| speed-coding | 15 | 4.72 | 0.531 | 326 | 3.02× |
| translation | 15 | 4.48 | 0.497 | 303 | 2.81× |
| mbpp | 15 | 4.26 | 0.465 | 292 | 2.70× |
| speed-multilingual | 15 | 4.20 | 0.458 | 299 | 2.77× |
| speed-rag | 15 | 4.09 | 0.441 | 278 | 2.57× |
| rag | 15 | 4.08 | 0.440 | 271 | 2.51× |
| tool_call | 15 | 3.89 | 0.413 | 252 | 2.33× |
| math500 | 15 | 3.82 | 0.402 | 274 | 2.54× |
| swe-bench-pro | 15 | 3.80 | 0.400 | 256 | 2.37× |
| livecodebench | 15 | 3.67 | 0.382 | 274 | 2.54× |
| summarization | 15 | 3.66 | 0.380 | 245 | 2.27× |
| speed-writing | 15 | 3.20 | 0.314 | 218 | 2.02× |
| aa-lcr-4k | 15 | 3.17 | 0.310 | 188 | 1.74× |
| swe-rebench | 15 | 3.17 | 0.310 | 229 | 2.12× |
| qa | 15 | 3.14 | 0.305 | 217 | 2.01× |
| speed-qa | 15 | 3.14 | 0.305 | 217 | 2.01× |
| aa-lcr-1k | 15 | 3.08 | 0.297 | 189 | 1.75× |
| gpqa | 15 | 3.03 | 0.289 | 216 | 2.00× |
| mt-bench | 25 | 2.82 | 0.260 | 202 | 1.87× |
| aime | 15 | 2.77 | 0.253 | 201 | 1.86× |
| aime26 | 15 | 2.66 | 0.238 | 190 | 1.76× |
| writing | 15 | 2.63 | 0.233 | 187 | 1.73× |
| **mean over sets** | | **3.66** | | **247** | **2.29×** |

Two things this example shows about reading the suite. The ordering of
domains is backbone-specific: on Kimi K3 the math competition sets (`aime`,
`aime26`) are near the bottom while they are near the top on Gemma-4-31B-it,
because K3's long chain-of-thought is harder to draft than Gemma's. And
`speedup` tracks `accept_len` only loosely across benchmarks (gsm8k's short
formulaic answers reach 360 tok/s; long-context sets sit at 190 tok/s with a
similar `accept_len`), which is why `accept_len` is the metric to compare
across sites and `speedup` is reported per box.
