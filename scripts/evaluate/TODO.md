# Evaluation follow-ups

Tracked improvements for `scripts/evaluate/`. How-to for the
[Kimi-K3-DSpark acceptance suite](https://huggingface.co/Inferact/Kimi-K3-DSpark)
lives in the docs (not here):

- Mapping + overview: [`README.md`](./README.md#kimi-k3-dspark-acceptance-suite)
- Generate + run: [`mtp_server_eval/README.md`](./mtp_server_eval/README.md#h-kimi-k3-dspark-acceptance-suite)
- Dataset sources: [`eval_datasets/README.md`](./eval_datasets/README.md)

## Kimi suite gaps

- [ ] Confirm SPEED-Bench low-entropy ISL matches the card’s “10k input” (currently `throughput_16k`)
- [ ] Full SPEED-Bench fills after `huggingface-cli login` (NVIDIA prepare needs gated sources like `cais/hle`)
- [x] SWE-Rebench (`swe-rebench` ← `nebius/SWE-rebench`) for [RadixArk/Kimi-K3-DSpark](https://huggingface.co/RadixArk/Kimi-K3-DSpark) — wired; generate JSONL then `prepare_data.py` (card uses 50 prompts via `NUM_SAMPLES=50`)
- [ ] Optional: YAML experiment preset for the full suite
- [ ] Optional: temperature=1.0 column (`draft_sample_method=probabilistic` as on the Inferact card)

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
