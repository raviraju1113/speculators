# Config-driven evaluation experiments

Describe a speculative-decoding evaluation sweep in one YAML and let
[`run_experiments.py`](./run_experiments.py) drive it. For each experiment it
launches a vLLM server (backbone, optionally with a draft attached via
`--speculative-config`), waits for `/health`, runs the acceptance/throughput
eval, stops the server, then prints a speedup table (first experiment =
baseline). Ideal for a single multi-GPU box (e.g. 8xA100).

See the [parent eval README](../README.md) for how the metrics are defined and
how this fits alongside the other evaluators.

## Running it yourself (GPU machine)

### 1. Environment (one-time setup)

```bash
# from the repo root
cd /path/to/speculators

# isolated env (venv or conda — either is fine)
python -m venv .venv-eval && source .venv-eval/bin/activate
# or:  conda create -n spec-eval python=3.11 -y && conda activate spec-eval

# vLLM (needs >=0.12 for speculators / --speculative-config) + the eval deps
pip install "vllm>=0.12.0"
pip install -r scripts/evaluate/requirements.txt   # requests, pyyaml, matplotlib, guidellm, ...
```

### 2. Environment variables

```bash
export HF_TOKEN=hf_xxx                      # for gated backbones (Llama) + gated GPQA dataset
export HF_HOME=$HOME/.cache/huggingface     # optional: where weights cache
# optional: cap which GPUs the whole job may touch (else the YAML's `gpus:` selects)
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 3. Run a full evaluation

```bash
cd scripts/evaluate/experiments

# Edit full-eval.yaml: backbone, draft, gpus, tensor_parallel_size, benchmarks.
./run_full_eval.sh --dry-run
./run_full_eval.sh
./run_full_eval.sh --only baseline,draft_k5
```

Same as `python run_experiments.py --config full-eval.yaml`. For a smaller
smoke config, use `example.yaml` instead.

Step-by-step (env, data prep, fair-comparison tips):
[parent README → How to run a full evaluation](../README.md#how-to-run-a-full-evaluation).

Per experiment the runner sets `CUDA_VISIBLE_DEVICES` from `gpus:`, launches
`vllm serve <backbone> [--speculative-config ...]`, waits for `/health`, runs the
eval into `results/<output_dir>/<name>/`, stops the server, then prints the
speedup table.

### Practical notes

- **Run inside `tmux`/`screen`** — a sweep is several multi-minute server
  launches; it should survive a disconnect.
- **Results** land at `results/experiments/<name>/` (`mtp_eval_summary.json` +
  `server.log`); the final speedup table prints to stdout. Set `output_dir:` in
  the YAML for a different location.
- **Readable tables:** the runner prints the comparison to stdout via
  `compare_speedup.py`. For a Markdown/CSV table (adds `n`, `e2e tok/s`, `ttft`,
  bolded speedups, files you can paste into a doc or spreadsheet) run
  [`tabulate_results.py`](./tabulate_results.py) over the output dir:
  ```bash
  python tabulate_results.py --dir ./results/experiments --baseline baseline \
      --out-dir ./results/experiments        # writes results_table.md + .csv
  ```
- **First run downloads weights** into `$HF_HOME` (once; cached after).
- **Quick smoke test:** set `eval.benchmarks: [aime]` and `num_samples: 5` in the
  YAML and use `--only baseline,<one-draft>` before the full run.

## Configs in this directory

| Config | What it runs | Results |
|---|---|---|
| [`example.yaml`](./example.yaml) | Small smoke-test template | — |
| [`full-eval.yaml`](./full-eval.yaml) | Full multi-domain suite template (edit backbone/draft first) | — |
| [`glm52-eval.yaml`](./glm52-eval.yaml) | GLM-5.2 native MTP vs baseline | [GLM-5.2 MTP results](../../../docs/user_guide/tutorials/glm52_mtp_results.md) |
| [`glm52-kvcache-ablation.yaml`](./glm52-kvcache-ablation.yaml) | GLM-5.2 KV-cache dtype ablation | same doc |
| [`gemma4-31b.yaml`](./gemma4-31b.yaml) | Gemma-4-31B-it + assistant draft, k=3/5, math+code trio | — |
| [`gemma4-31b-full.yaml`](./gemma4-31b-full.yaml) | Full 25-bench suite: baseline + Google Assistant (MTP) k=3/5 | [full-suite results](../../../docs/user_guide/tutorials/gemma4_31b_full_spec_decode_results.md) |
| [`gemma4-31b-full-eagle3.yaml`](./gemma4-31b-full-eagle3.yaml) | Same suite, Eagle-3 Qwen (Ravi) k=3/5 | same doc |
| [`gemma4-31b-full-redhat-ft.yaml`](./gemma4-31b-full-redhat-ft.yaml) | Same suite, Eagle-3 Llama (John) k=3/5 | same doc |
| [`gemma4-31b-full-dspark-nemo782k.yaml`](./gemma4-31b-full-dspark-nemo782k.yaml) | Same suite, DSpark Qwen (Mengmeng) k=8 | same doc |
| [`gemma4-31b-agentx.yaml`](./gemma4-31b-agentx.yaml) | AgentX concurrency 1/8/16: baseline + Assistant / Eagle-3 Qwen / Eagle-3 Llama k=5 | — |
| [`gemma4-31b-agentx-dspark.yaml`](./gemma4-31b-agentx-dspark.yaml) | Same AgentX sweep, DSpark Qwen k=8 (vLLM 0.28) | — |
| [`gemma4-31b-compare-1gpu.yaml`](./gemma4-31b-compare-1gpu.yaml) | Draft head-to-head (ours vs Google assistant vs RedHat eagle-3) on 1 GPU | — |
| [`gemma4-31b-bfcl.yaml`](./gemma4-31b-bfcl.yaml) | Gemma-4-31B-it + assistant draft on BFCL function calling, k=3/5 | [BFCL results](../../../docs/user_guide/tutorials/gemma4_31b_assistant_bfcl_results.md) — 2.62× (k=3) / 3.48× (k=5) decode speedup, ~90–96% acceptance |

## Config schema (`example.yaml`)

```yaml
engine: vllm
backbone: <target/verifier model>          # required
gpus: "0,1,2,3,4,5,6,7"                     # CUDA_VISIBLE_DEVICES for the server

server:
  host: 127.0.0.1
  port: 8000
  tensor_parallel_size: 8                   # 8xA100
  gpu_memory_utilization: 0.9
  max_model_len: 8192
  health_timeout: 1800                      # s to wait for /health (weight load + graph capture)
  extra_args: []                            # extra `vllm serve` flags

eval:
  backend: vllm                             # mtp_server_eval evaluator: vllm | sglang
  mode: acceptance                          # acceptance | throughput | sweep (GuideLLM) | agentx
  # AgentX-only (when mode is agentx): concurrency sweep via run_agentx.sh
  # users_list: [1, 8, 16]
  # duration: 600                             # seconds per concurrency level
  # max_context: 32768                        # default: cap to server.max_model_len
  # Any name supported by mtp_server_eval (aime, gpqa, livecodebench, gsm8k,
  # math500, humaneval, mbpp, mt-bench, aime26, swe-bench-pro, swe-rebench, aa-lcr,
  # bfcl, speed-coding, speed-multilingual, speed-rag, speed-qa, speed-writing,
  # speed-low-entropy, HumanEval, math_reasoning, qa, question, rag,
  # summarization, tool_call, translation, writing). Generate extras via
  # prepare_data.py / parent README.
  benchmarks: [aime, gpqa, livecodebench]  # also GuideLLM subsets when `subsets` is unset
  num_samples: 50                           # per benchmark (0 = all)
  max_tokens: 4096
  temperature: 0.0                          # greedy = canonical acceptance
  # GuideLLM-only (when mode is throughput or sweep):
  # dataset: RedHatAI/speculator_benchmarks   # default: mtp_server_eval/data
  # subsets: [HumanEval, qa]                  # default: eval.benchmarks
  # max_concurrency: 128
  # max_requests: 200
  # sweep_rate: 10
  # speedbench_data_dir: ../speedbench_data   # for DATASET=speedbench/...

output_dir: ./results/experiments

experiments:
  - name: baseline                          # first entry = baseline (no draft)
  - name: eagle3_k5
    draft: <draft/speculator model>
    num_speculative_tokens: 5
  - name: eagle3_k3
    draft: <draft/speculator model>
    num_speculative_tokens: 3
    eval: { num_samples: 100 }              # per-experiment overrides (merged over defaults)
```

Defaults merge: `server`/`eval` come from the top level, and each experiment's
own `server`/`eval`/`draft`/`num_speculative_tokens`/`speculative_config` block
overrides them.

## Serving form

The runner serves as:

```
vllm serve <backbone> --speculative-config '{"model": <draft>, "num_speculative_tokens": N}'
```

with baseline = `vllm serve <backbone>` (no draft). If your drafts are
speculators-format checkpoints you'd rather serve directly (`vllm serve <draft>`,
which pulls the verifier itself), or your vLLM tag expects a different
`--speculative-config` shape, adjust `build_serve_command` in
[`run_experiments.py`](./run_experiments.py) (or open an issue to add a
`serve_mode` switch).
