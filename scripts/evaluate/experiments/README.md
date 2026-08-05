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
  # Any name supported by mtp_server_eval (aime, gpqa, livecodebench, gsm8k,
  # math500, humaneval, mbpp, mt-bench, aime26, swe-bench-pro, swe-rebench, aa-lcr,
  # speed-coding, speed-multilingual, speed-rag, speed-qa, speed-writing,
  # speed-low-entropy). Generate extras via prepare_data.py / parent README.
  benchmarks: [aime, gpqa, livecodebench]
  num_samples: 50                           # per benchmark (0 = all)
  max_tokens: 4096
  temperature: 0.0                          # greedy = canonical acceptance

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
