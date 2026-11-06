# Evaluation follow-ups

Tracked improvements for `scripts/evaluate/`.

**How to run a full evaluation (YAML):** see
[`README.md`](./README.md#how-to-run-a-full-evaluation) —
[`experiments/full-eval.yaml`](./experiments/full-eval.yaml) +
[`experiments/run_full_eval.sh`](./experiments/run_full_eval.sh).

Dataset sources: [`eval_datasets/README.md`](./eval_datasets/README.md).
Lower-level server recipes: [`mtp_server_eval/README.md`](./mtp_server_eval/README.md).

## Suite / data gaps

- [x] SPEED-Bench slices in `full-eval.yaml` / `run_eval.sh` (`speed-coding`, `speed-multilingual`, `speed-rag`, `speed-qa`, `speed-writing`, `speed-low-entropy`)
- [ ] Confirm SPEED-Bench low-entropy ISL matches published “10k input” cards (currently `throughput_16k`)
- [ ] Full SPEED-Bench fills after `huggingface-cli login` (NVIDIA prepare needs gated sources like `cais/hle`)
- [x] `RedHatAI/speculator_benchmarks` nine subsets in acceptance mode + YAML
- [x] SWE-Rebench (`swe-rebench` ← `nebius/SWE-rebench`) — wired; generate JSONL then `prepare_data.py`
- [x] YAML full-eval template (`experiments/full-eval.yaml` + `run_full_eval.sh`) + guide / [Recent changes](./README.md#recent-changes) in [`README.md`](./README.md)
- [ ] Optional: temperature=1.0 column (`draft_sample_method=probabilistic` as on some published cards)

## Eval harness

- [ ] **Output-quality check** — optional exact-match / pass@k (or hash equality vs greedy backbone) so serve/config bugs aren’t hidden by “lossless by assumption”
- [ ] **Position-wise acceptance** — per-draft-step conditional accept curves (esp. useful for EAGLE / DFlash / DSpark), not only server aggregate `accept_length` / `accept_rate`
- [ ] **Unify dataset trees** — reduce drift between `eval_datasets/` and `mtp_server_eval/data/` (single source + generated derivatives)
- [ ] **Sample-size guidance** — defaults are small (20–50); document noise on hard sets (e.g. AIME) and/or raise recommended `num_samples`
- [ ] **Unit tests** — cover metric scraping (`accept_stats`, Prometheus parsing), compare/tabulate helpers, and YAML→command building
- [ ] **Research diagnostics** (DSpark / confidence) — confidence AUC/ECE and calibrated STS checks when those heads exist

## AgentX (`run_agentx.sh`)

- [x] Repaired onto aiperf `--scenario inferencex-agentx-mvp` — the pinned InferenceX branch/`trace_replay_tester.py` no longer exist upstream
- [x] Wired into the YAML runner as `eval.mode: agentx` (+ `compare_agentx.py`, `agentx_metrics.py`)
- [ ] **Per-request acceptance from aiperf** — its `SpecDecodeAcceptanceRecord` carries an `acceptance_histogram` and optional per-step accepted/drafted arrays, which is strictly richer than the aggregate `1 + Δaccepted/Δdrafts` this script scrapes off `/metrics`; would also close “Position-wise acceptance” below
- [ ] Cross-check AgentX acceptance against a real-text long-context bench (`aa-lcr`, `swe-bench-pro`) — AgentX prompts are synthesized from token counts + KV block hashes, so its absolute acceptance is regime-specific
- [ ] Optional: sweep `max_context` (128k vs the model's native max) as a second axis

## YAML experiment runner (`experiments/`)

- [ ] **Resume / skip** — don’t relaunch experiments that already have a valid `mtp_eval_summary.json` unless `--force`
- [ ] **`serve_mode` switch** — support `vllm serve <draft>` (speculators-format pulls verifier) as well as `vllm serve <backbone> --speculative-config ...`
- [ ] **SGLang serve path** — `engine: sglang` should launch SGLang, not only select the eval metric reader
- [ ] **Matrix sugar** — optional cartesian product over drafts × `k` × benchmarks instead of hand-written list items
- [ ] **Parallel runs** — optional multi-port / multi-GPU concurrent experiments when the box allows
