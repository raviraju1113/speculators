# Gemma-4-31B-it speculative-decoding experiments

Results for the **dense** `gemma-4-31b-it` backbone. The MoE sibling
(`gemma-4-26B-A4B-it`) has its own doc:
[gemma4_26b_moe_results.md](gemma4_26b_moe_results.md).

All evals: single-stream (batch=1), greedy (`temperature=0`), via
`scripts/evaluate/mtp_server_eval/run_vllm_eval.py`. Metrics:

- **accept_len** — avg tokens committed per target forward pass (max = `k+1`).
- **accept_rate** — accepted / drafted tokens.
- **decode tok/s** — decode-phase output speed (what spec-decoding accelerates).
- **e2e tok/s / ttft** — end-to-end rate / time-to-first-token (reference).
- **speedup** — decode tok/s ÷ the target-alone baseline.

---

## 1. MTP assistant, k sweep (4xA100, tp=4)

Target: `gemma-4-31b-it`, draft: `gemma-4-31B-it-assistant`. Baseline = backbone alone.

| benchmark | config | accept_len | accept_rate | decode tok/s | speedup |
|---|---|--:|--:|--:|--:|
| aime | baseline | — | — | 54.4 | 1.00× |
| | assistant k=3 | 3.549 | 85.0% | 133.3 | 2.45× |
| | assistant k=5 | 4.788 | 75.8% | 165.3 | **3.04×** |
| gpqa | assistant k=5 | 4.465 | 69.3% | 155.6 | 2.84× |
| gsm8k | assistant k=5 | 5.074 | 81.5% | 199.3 | **3.58×** |
| humaneval | assistant k=5 | 5.155 | 83.1% | 195.4 | 3.53× |
| livecodebench | assistant k=5 | 4.523 | 70.5% | 151.1 | 2.78× |
| math500 | assistant k=5 | 5.045 | 80.9% | 185.7 | 3.36× |
| mbpp | assistant k=5 | 4.510 | 70.2% | 170.7 | 3.08× |

(baseline ≈ 54–56 tok/s across benchmarks; k=3 rows omitted for brevity — see
`scripts/evaluate/experiments/results/gemma4-31b/results_table.md`.)

**Takeaways:** k=5 > k=3 everywhere (longer accepted runs beat higher per-token
accept rate); **2.8–3.6× speedup**, best on short-output math/code (gsm8k 3.58×).

---

## Related 31B material elsewhere in the repo

- Eval configs: `scripts/evaluate/experiments/gemma4-31b*.yaml` (full suite,
  AgentX concurrency sweeps, BFCL, DSpark, EAGLE3, RedHat fine-tune).
- Raw results: `scripts/evaluate/experiments/results/gemma4-31b*/`.
- Training recipe: `examples/train/eagle3_online_gemma4_31b.sh` and the
  tutorial `docs/user_guide/tutorials/train_eagle3_online_gemma4_31b.md`.
- BFCL results doc: `docs/user_guide/tutorials/gemma4_31b_assistant_bfcl_results.md`.
