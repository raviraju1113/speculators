# Speculator Evaluation Runbook (SambaNova)

How to evaluate a speculator (speculative-decoding draft model) against any
backbone on SambaNova's 25-benchmark acceptance suite, with the same prompts,
sample draw, decoding settings and metric definitions we use, so results from
different teams, backbones and speculator architectures can be placed side by
side.

The harness is black-box. It sends prompts to a vLLM OpenAI-compatible server
and reads vLLM's Prometheus spec-decode counters. Any backbone vLLM can serve
and any speculator vLLM can attach with `--speculative-config` works. Nothing
in the eval depends on how the speculator works internally.

Throughout, "speculator" and "draft" both mean the draft model; "target" or
"backbone" means the model it speculates for. `google/gemma-4-31B-it` appears
only as the worked example (§7).

Contents

1. The benchmark suite
2. Getting the data
3. Running the evaluation
4. Metrics to collect
5. Reporting and what to send back
6. Gotchas
7. Worked example: Gemma-4-31B-it with the Google assistant speculator

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
tokenizers differ by ±20%. The longest prompt in the suite is under 5k tokens.
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

## 2. Getting the data

### 2.1 Use the shipped files (preferred)

All 25 files are in the `speculators` repo under
`scripts/evaluate/mtp_server_eval/data/<name>.jsonl` (`gpqa` is
`gpqa_diamond.jsonl`). One JSON object per line:

```json
{"benchmark": "gsm8k", "id": "17", "prompt": "Janet's ducks lay 16 eggs per day. ..."}
```

Use them byte-for-byte. Several were re-fetched during 2026 and the 50-sample
draw depends on row order, so a regenerated file gives a different sample.
Verify:

```bash
cd scripts/evaluate/mtp_server_eval/data
md5sum aime.jsonl gpqa_diamond.jsonl livecodebench.jsonl gsm8k.jsonl humaneval.jsonl \
  mbpp.jsonl math500.jsonl mt-bench.jsonl aime26.jsonl bfcl.jsonl swe-bench-pro.jsonl \
  speed-coding.jsonl speed-multilingual.jsonl speed-rag.jsonl speed-qa.jsonl speed-writing.jsonl \
  HumanEval.jsonl math_reasoning.jsonl qa.jsonl question.jsonl rag.jsonl summarization.jsonl \
  tool_call.jsonl translation.jsonl writing.jsonl > my_manifest.md5
diff my_manifest.md5 reference_manifest.md5 && echo "data OK"
```

`reference_manifest.md5` (save this block verbatim):

```
4efac42e10a60df434d4335857039cfd  aime.jsonl
6e6e7b6ffd7a0e864478465d12cf0113  gpqa_diamond.jsonl
dc3239e00ca6221191f60498b3aaf036  livecodebench.jsonl
0ac068705bcab8f1eb6e76ecd34e7357  gsm8k.jsonl
e6bfb1c34b32fbbebaef1a76e10f97ad  humaneval.jsonl
2f72c2789469992beb1bb05373f40eb0  mbpp.jsonl
e2de4fb12186839446101c4acbf5179e  math500.jsonl
43c813681257833f4a70f00850c50fd9  mt-bench.jsonl
2c2f2ccbb37db40950ca8aa9959d90b3  aime26.jsonl
70b2621d03f8bba4dc9f40359a0752d8  bfcl.jsonl
403b5f7279daa9be25ac65c25b59e8a7  swe-bench-pro.jsonl
3aeef8449ad706158029a0416fbfce8a  speed-coding.jsonl
fbcdb9fc8294c1146ea437d9ad1029f8  speed-multilingual.jsonl
c6907938f6ad412675c359b71dfc44f8  speed-rag.jsonl
ca3b99968988324dc92fff0888d11c0f  speed-qa.jsonl
2047ccc26b88a148fea6d3e50c2658b8  speed-writing.jsonl
6b21bda72fef34e67d919c7154edd769  HumanEval.jsonl
3a88be0d8ae521e94552e5342223b91a  math_reasoning.jsonl
a74550d479599d6220b39ba197afaade  qa.jsonl
db3a49843ea05d49d9f9ab3dad864549  question.jsonl
c0635b3054ee13735de73a8ccb8fa15f  rag.jsonl
4b759c853cbb6ff8cd38496d0216452e  summarization.jsonl
6a03e258bcfd4e26fc3459a66d39916d  tool_call.jsonl
ed619153b41c8b321021a933887ab33a  translation.jsonl
1f1a97293ac9af2d87af3538b285763e  writing.jsonl
```

GPQA is gated on Hugging Face. Accept its terms before using
`gpqa_diamond.jsonl`, and do not redistribute it further.

### 2.2 Rebuilding from source (only if you must)

The converters live in `scripts/evaluate/`. Row order follows the upstream
dataset, so a rebuild matches the shipped file only if upstream has not changed.
Check against the manifest afterwards.

```bash
cd scripts/evaluate
export HF_TOKEN=hf_...     # GPQA and SPEED-Bench externals

# HF datasets -> turns files in eval_datasets/
for d in openai/gsm8k openai/openai_humaneval google-research-datasets/mbpp \
         HuggingFaceH4/MATH-500 HuggingFaceH4/mt_bench_prompts MathArena/aime_2026 \
         gorilla-llm/Berkeley-Function-Calling-Leaderboard ScaleAI/SWE-bench_Pro; do
  python eval_datasets/convert_eval_datasets_to_jsonl.py "$d"
done

# SPEED-Bench (all 11 qualitative categories; only 5 are in this suite)
python prepare_speedbench.py --data-dir ./speedbench_data --download --configs qualitative

# turns files + HF pulls -> mtp_server_eval/data/<name>.jsonl
cd mtp_server_eval
python prepare_data.py --only aime,gpqa,livecodebench
python prepare_data.py --only gsm8k,humaneval,mbpp,math500,mt-bench,aime26,bfcl,swe-bench-pro
SPEEDBENCH_DIR=../speedbench_data python prepare_data.py --only \
  speed-coding,speed-multilingual,speed-rag,speed-qa,speed-writing
python prepare_data.py --only HumanEval,math_reasoning,qa,question,rag,summarization,tool_call,translation,writing
```

### 2.3 Adding a benchmark

Drop `<name>.jsonl` with the same three fields into `mtp_server_eval/data/`,
register it in `DATA_FILES` in `run_vllm_eval.py`, list it under
`eval.benchmarks`. Report it separately; it is not part of the 25.

---

## 3. Running the evaluation

### 3.1 Environment

```bash
python3.12 -m venv ~/envs/spec-eval && source ~/envs/spec-eval/bin/activate
pip install --upgrade pip
pip install "vllm==0.28.0"          # what we run; >=0.25 for method=dspark, >=0.12 minimum
pip install requests pyyaml         # the acceptance eval needs only these
git clone <the speculators repo> && cd speculators
```

If your speculator needs a vLLM fork or plugin, install that instead and run
the **baseline on the same build**. Record the version or commit.

**Activate the venv; do not call its python by path.** The runner spawns a
bare `vllm serve`, so `vllm` must be on `PATH`.

### 3.2 Settings

Fixed for every backbone and speculator:

| Setting | Value | Where |
|---|---|---|
| Decoding | `temperature 0.0` (greedy), `max_tokens 4096` | `eval:` block |
| Samples | `num_samples: 50`, drawn as `random.Random(42).sample(rows, 50)` | fixed in `run_vllm_eval.py`; do not change the seed or the count |
| Request | one user turn, no system prompt, streaming with usage; the backbone's own chat template | `run_vllm_eval.py` |
| Concurrency | 1, sequential | |
| Server extras | none: no reasoning parser, tool parser, chat-template override, or quantization unless the backbone ships that way. If you must add one, add it to the baseline too and report it. | `server.extra_args` |
| Benchmarks | all 25, in the order listed in §3.3 | `eval.benchmarks` |

Chosen per backbone, identical between its baseline and every speculator run:

| Setting | Guidance |
|---|---|
| `backbone` | HF id or local path. bf16 as published. |
| `gpus`, `tensor_parallel_size` | Whatever fits the backbone. Acceptance is TP-independent; tok/s is not. |
| `max_model_len` | `>= 5000 + 4096`. Use **16384** unless memory forbids it. The longest prompt is ~4.6k tokens; at 8192 it is rejected and that benchmark quietly reports `n = 49`. |
| `gpu_memory_utilization` | 0.9 |
| `num_speculative_tokens` (`k`) | The speculator's native depth. Report it. |
| `speculative_config` extras | Passed through verbatim to vLLM (`method`, and any keys your proposer reads). |

### 3.3 The YAML

Save as `scripts/evaluate/experiments/<backbone>-speculator-eval.yaml`. Fill
the `<...>` fields.

```yaml
engine: vllm
backbone: <HF id or /path/to/backbone>
gpus: "<0,1,...>"

server:
  host: 127.0.0.1
  port: 8000
  tensor_parallel_size: <TP>
  gpu_memory_utilization: 0.9
  max_model_len: 16384
  health_timeout: 1800
  extra_args: []

eval:
  backend: vllm
  mode: acceptance
  benchmarks: [aime, gpqa, livecodebench, gsm8k, humaneval, mbpp, math500, mt-bench,
               aime26, bfcl, swe-bench-pro, speed-coding, speed-multilingual, speed-rag,
               speed-qa, speed-writing, HumanEval, math_reasoning, qa, question, rag,
               summarization, tool_call, translation, writing]
  num_samples: 50
  max_tokens: 4096
  temperature: 0.0

output_dir: ./results/<backbone>-speculator-eval

experiments:
  - name: baseline                       # target alone; must be first

  - name: <speculator>_k<k>
    draft: </path/to/speculator>
    num_speculative_tokens: <k>
    # speculative_config:                # optional passthrough, e.g.
    #   method: <method>
```

One `experiments` entry per speculator and per `k`. Keep one YAML per
backbone: the runner reuses a baseline only within one `output_dir`.

### 3.4 Run

```bash
cd scripts/evaluate/experiments
python run_experiments.py --config <yaml> --dry-run       # prints serve + eval commands; launches nothing
tmux new -s eval
python run_experiments.py --config <yaml>                 # baseline, then each speculator
python run_experiments.py --config <yaml> --only <name>   # later: add a speculator against the stored baseline
```

For each experiment the runner sets `CUDA_VISIBLE_DEVICES`, launches
`vllm serve <backbone> [--speculative-config '{...}']`, waits for `/health`,
runs the 25 benchmarks sequentially, stops the server, and finally prints a
decode-speedup table against `baseline`.

Budget: baseline ≈ 565k completion tokens at the backbone's single-stream
decode rate (about 3 h for a 31B dense model on 4× A100); each speculator run
is 1 to 1.5 h.

Smoke test first: `benchmarks: [gsm8k]`, `num_samples: 5`, `--only baseline`.
Then restore the full list.

### 3.5 What the speculator must provide

1. Servable through `--speculative-config`: an upstream `method` (`eagle3`,
   `mtp`, `dspark`, `draft_model`, ...) or your own proposer in a fork or
   plugin.
2. The three standard counters on `/metrics`, which the eval reads as deltas
   around each benchmark:
   `vllm:spec_decode_num_drafts`, `vllm:spec_decode_num_draft_tokens`,
   `vllm:spec_decode_num_accepted_tokens`. A new proposer must feed the same
   `SpecDecodingStats`, or acceptance comes back `null`.
3. Exact greedy verification. Relaxed acceptance (`accept_tolerance`,
   typical-acceptance, etc.) changes the output and must be reported as a
   separate configuration.

---

## 4. Metrics to collect

All from `results/<output_dir>/<experiment>/mtp_eval_summary.json` (one row per
benchmark) and `mtp_eval_details.jsonl` (one row per request). Definitions are
the code in `scripts/evaluate/mtp_server_eval/run_vllm_eval.py`.

| Field | Definition | Compare across backbones / sites? |
|---|---|---|
| **`accept_length`** | `1 + Δaccepted_tokens / Δdrafts`. Tokens committed per target forward pass, bonus token included. Upper bound `k + 1`. | **Yes.** The primary metric. Hardware- and TP-independent. |
| `accept_rate` | `Δaccepted_tokens / Δdraft_tokens`. Share of drafted tokens accepted. | Only between speculators at the same `k`. |
| `decode_tok_s` | `Σ(completion_tokens − 1) / Σ decode_time`; decode_time = stream end minus first-token time. Excludes prefill. | Only against the same box's baseline. |
| **speedup** | `decode_tok_s(speculator) / decode_tok_s(baseline)`, same box, same build, same `max_model_len`. | Yes, as a ratio, with the hardware stated. |
| `e2e_tok_s`, `mean_ttft_s` | End-to-end rate incl. prefill; mean time to first token. | Reference only. |
| `n` | Requests that completed. Must be 50 (30 for aime / aime26). | Sanity. Fewer = a rejected request; see `run.log`. |
| `total_completion_tokens` | Generated tokens over the benchmark. Baseline and speculator must agree within a few percent under greedy. | Sanity. A larger gap means the speculator changed the output. |

Counters are read before and after each benchmark, so do not send requests to
the server by hand during a run.

Also record, once per run: backbone id and revision, speculator checkpoint and
its `config.json`, `k`, vLLM version or fork commit, GPU model and count, TP,
`max_model_len`, and the `my_manifest.md5` from §2.1.

---

## 5. Reporting and what to send back

Per backbone, one table with a row per benchmark and, per speculator, columns
`accept_len`, `accept_rate`, `decode tok/s`, `speedup`. The tabulator writes it:

```bash
python tabulate_results.py --dir ./results/<output_dir> --baseline baseline \
    --out-dir ./results/<output_dir>          # results_table.md + .csv
```

**Means:** over the 22 distinct sets, i.e. all 25 minus `speed-qa`, `writing`
and `mt-bench`. State that next to the number. The tabulator averages all rows
it is given, so compute the 22-set mean yourself or drop those three columns
first. Also report the per-group means from §1 (math, code, reasoning/QA,
tool use, long input, language); speculators differ far more by group than in
the overall mean.

Send:

- The tarball of `results/<output_dir>/` (every experiment dir has
  `mtp_eval_summary.json`, `mtp_eval_details.jsonl`, `server.log`; unedited).
- The YAML(s) and the exact command lines.
- `pip freeze`, `nvidia-smi`, vLLM version or fork commit, and any patch or
  plugin needed to serve the speculator.
- The speculator's `config.json` and every `k` evaluated.
- `my_manifest.md5`.

---

## 6. Gotchas

- **`FileNotFoundError: 'vllm'`** right after the serve command prints: venv not
  activated.
- **`accept_length: null`** with a speculator attached: the §3.5 counters did
  not move. The proposer is not reporting stats, or vLLM fell back to another
  method. Check the loaded `speculative_config` in `server.log`.
- **`n` below 50** on a benchmark: a request was rejected, almost always
  prompt + `max_tokens` > `max_model_len`. Use 16384.
- **`total_completion_tokens` differs by more than a few percent** between
  baseline and speculator: the output changed. Exact greedy verification must
  reproduce the baseline. Check for relaxed-acceptance settings.
- **Reasoning / thinking backbones**: the eval counts thinking tokens as
  completion tokens and TTFT as the first streamed token of any kind. Serve the
  backbone with its default thinking behaviour, the same for baseline and
  speculator, and say which it was.
- **Server start is 5 to 10 min** (weights + CUDA-graph capture);
  `health_timeout: 1800` covers it.
- **Noise floor**: ±0.1 `accept_len` on the short-output benchmarks (§1.2),
  ±0.03 elsewhere; `aime26` hits the 4096 cap on 4 or 5 of 30 prompts.

---

## 7. Worked example: Gemma-4-31B-it with the Google assistant speculator

Use this to check a new setup: run the public backbone with its public
speculator, then compare `accept_len` to the table. Agreement within about
±0.15 on every benchmark means the server, data and harness match ours, and
any later difference is the speculator under test.

```bash
export HF_TOKEN=hf_...     # Gemma weights are gated
hf download google/gemma-4-31B-it           --local-dir /models/gemma-4-31B-it
hf download google/gemma-4-31B-it-assistant --local-dir /models/gemma-4-31B-it-assistant
```

YAML fields for §3.3: `backbone: /models/gemma-4-31B-it`, `gpus: "0,1,2,3"`,
`tensor_parallel_size: 4`, and

```yaml
  - name: google_assistant_k5
    draft: /models/gemma-4-31B-it-assistant
    num_speculative_tokens: 5
```

Our numbers. 4× A100 80GB PCIe, TP=4, vLLM 0.24.0, `max_model_len` 8192,
greedy, 50 samples (30 for aime / aime26). Baseline decode ≈ 55 tok/s on this
box; compare `accept_len`, not tok/s.

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
rows), so the shipped files give a different 50-sample draw. Expect those rows
to be close but not matching. All other rows use identical prompts.
