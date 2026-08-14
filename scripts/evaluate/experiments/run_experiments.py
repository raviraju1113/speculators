#!/usr/bin/env python3
"""Config-driven speculative-decoding evaluation experiments.

Reads a YAML that describes a backbone (target/verifier) model, one or more draft
configs, serving params, and eval settings. For each experiment it:

  1. launches a vLLM server for the backbone (optionally with a speculative
     draft attached),
  2. waits for /health,
  3. runs the acceptance / throughput eval (mtp_server_eval evaluator),
  4. stops the server,

then prints a speedup comparison across all experiments (baseline first).

Designed for a single multi-GPU box (e.g. 8xA100): set `server.tensor_parallel_size`
and `gpus`. Nothing here needs a GPU to *parse* — use `--dry-run` to print the
exact serve + eval commands without launching anything.

Usage:
    python run_experiments.py --config example.yaml
    python run_experiments.py --config example.yaml --dry-run
    python run_experiments.py --config example.yaml --only baseline,eagle3_k5
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from copy import deepcopy
from http import HTTPStatus
from pathlib import Path

import requests
import yaml

HERE = Path(__file__).resolve().parent
MTP_EVAL_DIR = HERE.parent / "mtp_server_eval"
GUIDELLM_EVAL = MTP_EVAL_DIR / "run_guidellm_eval.py"

# Need at least a baseline + one draft config to compute a speedup.
MIN_EXPERIMENTS_FOR_COMPARISON = 2

DEFAULT_SERVER = {
    "host": "127.0.0.1",
    "port": 8000,
    "tensor_parallel_size": 8,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 8192,
    "extra_args": [],
    "health_timeout": 1800,  # seconds; large models + graph capture are slow
}
DEFAULT_EVAL = {
    "backend": "vllm",  # which mtp_server_eval evaluator: vllm | sglang
    # acceptance = sequential mtp_server_eval; throughput/sweep = GuideLLM
    "mode": "acceptance",
    "benchmarks": ["aime", "gpqa", "livecodebench"],
    "num_samples": 50,
    "max_tokens": 4096,
    "temperature": 0.0,
}
EVAL_MODES = ("acceptance", "throughput", "sweep")


def deep_merge(base: dict, override: dict | None) -> dict:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if not cfg.get("backbone"):
        sys.exit("config error: `backbone` (target model) is required")
    if not cfg.get("experiments"):
        sys.exit("config error: `experiments` list is required")
    return cfg


def build_serve_command(cfg: dict, exp: dict, server: dict) -> list[str]:
    """vLLM serve command for the backbone, with an optional speculative draft."""
    backbone = cfg["backbone"]
    cmd = [
        "vllm",
        "serve",
        backbone,
        "--host",
        str(server["host"]),
        "--port",
        str(server["port"]),
        "--tensor-parallel-size",
        str(server["tensor_parallel_size"]),
        "--gpu-memory-utilization",
        str(server["gpu_memory_utilization"]),
        "--max-model-len",
        str(server["max_model_len"]),
    ]
    draft = exp.get("draft")
    speculative_config = exp.get("speculative_config")
    if draft or speculative_config:
        spec: dict = {}
        if draft:
            spec["model"] = draft
        if exp.get("num_speculative_tokens") is not None:
            spec["num_speculative_tokens"] = exp["num_speculative_tokens"]
        # allow arbitrary extra speculative-config keys from the experiment
        spec.update(speculative_config or {})
        cmd += ["--speculative-config", json.dumps(spec)]
    cmd += list(server.get("extra_args", []))
    return cmd


def _as_csv(value) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    return str(value)


def build_eval_command(evalcfg: dict, base_url: str, out_dir: Path) -> list[str]:
    mode = str(evalcfg.get("mode", "acceptance")).lower()
    if mode not in EVAL_MODES:
        sys.exit(
            f"eval.mode must be acceptance|throughput|sweep (got {evalcfg.get('mode')!r})"
        )
    if mode in ("throughput", "sweep"):
        target = base_url.rstrip("/")
        if not target.endswith("/v1"):
            target = f"{target}/v1"
        cmd = [
            sys.executable,
            str(GUIDELLM_EVAL),
            mode,
            "--target",
            target,
            "--output-dir",
            str(out_dir),
        ]
        if evalcfg.get("dataset"):
            cmd += ["--dataset", str(evalcfg["dataset"])]
        if evalcfg.get("subsets"):
            cmd += ["--subsets", _as_csv(evalcfg["subsets"])]
        if evalcfg.get("max_concurrency") is not None:
            cmd += ["--max-concurrency", str(evalcfg["max_concurrency"])]
        if evalcfg.get("max_requests") is not None:
            cmd += ["--max-requests", str(evalcfg["max_requests"])]
        if evalcfg.get("max_tokens") is not None:
            cmd += ["--max-tokens", str(evalcfg["max_tokens"])]
        if evalcfg.get("gen_len_rate") is not None:
            cmd += ["--gen-len-rate", str(evalcfg["gen_len_rate"])]
        if evalcfg.get("sweep_rate") is not None:
            cmd += ["--sweep-rate", str(evalcfg["sweep_rate"])]
        gen_kwargs = evalcfg.get("gen_kwargs")
        if gen_kwargs is None and evalcfg.get("temperature") is not None:
            gen_kwargs = {"temperature": evalcfg["temperature"]}
        if gen_kwargs is not None:
            if not isinstance(gen_kwargs, str):
                gen_kwargs = json.dumps(gen_kwargs)
            cmd += ["--gen-kwargs", gen_kwargs]
        if evalcfg.get("data_column_mapper"):
            cmd += ["--data-column-mapper", str(evalcfg["data_column_mapper"])]
        if evalcfg.get("speedbench_data_dir"):
            cmd += ["--speedbench-data-dir", str(evalcfg["speedbench_data_dir"])]
        elif str(evalcfg.get("dataset", "")).startswith("speedbench/"):
            default_sb = HERE.parent / "speedbench_data"
            cmd += ["--speedbench-data-dir", str(default_sb)]
        return cmd

    script = {
        "vllm": "run_vllm_eval.py",
        "sglang": "run_sglang_eval.py",
    }.get(evalcfg["backend"])
    if script is None:
        sys.exit(f"eval.backend must be vllm|sglang (got {evalcfg['backend']!r})")
    return [
        sys.executable,
        str(MTP_EVAL_DIR / script),
        "--base-url",
        base_url,
        "--benchmarks",
        ",".join(evalcfg["benchmarks"]),
        "--num-samples",
        str(evalcfg["num_samples"]),
        "--max-tokens",
        str(evalcfg["max_tokens"]),
        "--temperature",
        str(evalcfg["temperature"]),
        "--output-dir",
        str(out_dir),
    ]


def wait_for_health(base_url: str, proc: subprocess.Popen, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"!! server exited early (rc={proc.returncode})", flush=True)
            return False
        try:
            if (
                requests.get(f"{base_url}/health", timeout=5).status_code
                == HTTPStatus.OK
            ):
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    print(f"!! server not healthy within {timeout}s", flush=True)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    # kill the whole process group (vLLM spawns TP workers)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=30)


def run_experiment(cfg: dict, exp: dict, out_root: Path, dry_run: bool) -> Path | None:
    server = deep_merge(DEFAULT_SERVER, cfg.get("server"))
    server = deep_merge(server, exp.get("server"))
    evalcfg = deep_merge(DEFAULT_EVAL, cfg.get("eval"))
    evalcfg = deep_merge(evalcfg, exp.get("eval"))

    base_url = f"http://{server['host']}:{server['port']}"
    out_dir = out_root / exp["name"]
    serve_cmd = build_serve_command(cfg, exp, server)
    eval_cmd = build_eval_command(evalcfg, base_url, out_dir)
    gpus = str(cfg.get("gpus", "")) if cfg.get("gpus") is not None else ""

    print(f"\n{'=' * 64}\n=== experiment: {exp['name']} ===")
    if exp.get("draft"):
        print(f"  draft: {exp['draft']}")
    elif exp.get("speculative_config"):
        print(f"  spec : {json.dumps(exp['speculative_config'])}")
    else:
        print("  draft: (none — baseline)")
    print(f"  CUDA_VISIBLE_DEVICES={gpus or '(inherit)'}")
    print(f"  serve: {' '.join(serve_cmd)}")
    print(f"  eval : {' '.join(eval_cmd)}")
    if dry_run:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    server_log = open(out_dir / "server.log", "w")  # noqa: SIM115
    proc = subprocess.Popen(
        serve_cmd,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group, so stop_server can kill workers
    )
    try:
        if not wait_for_health(base_url, proc, server["health_timeout"]):
            print(f"   see {out_dir / 'server.log'}", flush=True)
            return None
        print(f"  server healthy at {base_url}; running eval ...", flush=True)
        subprocess.run(eval_cmd, check=False)
    finally:
        stop_server(proc)
        server_log.close()
        # let GPU memory settle before the next launch
        time.sleep(10)
    summary = out_dir / "mtp_eval_summary.json"
    return summary if summary.exists() else None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument(
        "--output-dir", type=Path, default=None, help="override the config's output_dir"
    )
    ap.add_argument("--only", help="comma-separated experiment names to run")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print serve + eval commands without launching anything",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_root = args.output_dir or Path(cfg.get("output_dir", "./results/experiments"))

    experiments = cfg["experiments"]
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        experiments = [e for e in experiments if e["name"] in want]
        if not experiments:
            sys.exit(
                f"--only matched no experiments; have: "
                f"{[e['name'] for e in cfg['experiments']]}"
            )

    print(f"backbone: {cfg['backbone']}")
    print(f"experiments: {[e['name'] for e in experiments]}")
    print(f"output: {out_root}")

    summaries: list[tuple[str, Path]] = []
    for exp in experiments:
        summary = run_experiment(cfg, exp, out_root, args.dry_run)
        if summary is not None:
            summaries.append((exp["name"], summary))

    if args.dry_run or len(summaries) < MIN_EXPERIMENTS_FOR_COMPARISON:
        return

    # Speedup comparison (first experiment = baseline).
    compare = [
        sys.executable,
        str(MTP_EVAL_DIR / "compare_speedup.py"),
        *[f"{name}={path}" for name, path in summaries],
    ]
    print(f"\n{'=' * 64}\n=== speedup comparison ===")
    subprocess.run(compare, check=False)


if __name__ == "__main__":
    main()
