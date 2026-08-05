# Evaluation follow-ups

Tracked improvements for `scripts/evaluate/`.

**How to run a full evaluation (YAML):** see
[`README.md`](./README.md#how-to-run-a-full-evaluation) —
[`experiments/full-eval.yaml`](./experiments/full-eval.yaml) +
[`experiments/run_full_eval.sh`](./experiments/run_full_eval.sh).

Dataset sources: [`eval_datasets/README.md`](./eval_datasets/README.md).
Lower-level server recipes: [`mtp_server_eval/README.md`](./mtp_server_eval/README.md).

## Suite / data gaps

- [ ] Confirm SPEED-Bench low-entropy ISL matches published “10k input” cards (currently `throughput_16k`)
- [ ] Full SPEED-Bench fills after `huggingface-cli login` (NVIDIA prepare needs gated sources like `cais/hle`)
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

## YAML experiment runner (`experiments/`)

- [ ] **Resume / skip** — don’t relaunch experiments that already have a valid `mtp_eval_summary.json` unless `--force`
- [ ] **`serve_mode` switch** — support `vllm serve <draft>` (speculators-format pulls verifier) as well as `vllm serve <backbone> --speculative-config ...`
- [ ] **SGLang serve path** — `engine: sglang` should launch SGLang, not only select the eval metric reader
- [ ] **Matrix sugar** — optional cartesian product over drafts × `k` × benchmarks instead of hand-written list items
- [ ] **Parallel runs** — optional multi-port / multi-GPU concurrent experiments when the box allows
