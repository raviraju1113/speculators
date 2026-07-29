# Gemma-4 MTP / speculative-decoding experiments

Consolidated results for Gemma-4 speculative-decoding drafts (MTP assistant &
EAGLE3), the acceptance/throughput evals, and the training-time **shared-KV
attention leak** discovered and fixed here.

All evals: single-stream (batch=1), greedy (`temperature=0`), vLLM 0.24.0+cu129,
via `scripts/evaluate/mtp_server_eval/run_vllm_eval.py`. Metrics:

- **accept_len** — avg tokens committed per target forward pass (max = `k+1`).
- **accept_rate** — accepted / drafted tokens.
- **decode tok/s** — decode-phase output speed (what spec-decoding accelerates).
- **e2e tok/s / ttft** — end-to-end rate / time-to-first-token (reference).
- **speedup** — decode tok/s ÷ the target-alone baseline.

---

## 1. Gemma-4-26B-A4B (MoE) — draft comparison + k-depth sweep (1×A100)

Target: `gemma-4-26B-A4B-it`. Drafts (vanilla MTP assistant, EAGLE3, DFlash)
compared against the same target, sweeping speculative depth k.

| benchmark | config | accept_len | accept_rate | decode tok/s | e2e tok/s | speedup |
|---|---|--:|--:|--:|--:|--:|
| **aime** | baseline (no draft) | — | — | 127.1 | 126.4 | 1.00× |
| | vanilla MTP, k=3 | 3.571 | 85.7% | 198.9 | 195.9 | 1.56× |
| | vanilla MTP, **k=7** | 5.733 | 67.6% | 269.2 | 267.3 | **2.12×** |
| | eagle3, k=3 | 3.007 | 66.9% | 182.7 | 180.2 | 1.44× |
| | eagle3, **k=7** | 3.697 | 38.5% | 208.8 | 207.7 | 1.64× |
| | DFlash, k=7 | 4.789 | 54.1% | 273.9 | 271.8 | **2.15×** |
| | DFlash, **k=15** | 5.531 | 30.2% | 283.3 | 278.0 | **2.23×** |
| **gpqa** | baseline | — | — | 127.2 | 126.4 | 1.00× |
| | vanilla MTP, k=3 | 3.338 | 77.9% | 190.8 | 188.6 | 1.50× |
| | vanilla MTP, **k=7** | 4.955 | 56.5% | 235.8 | 232.6 | **1.85×** |
| | eagle3, k=3 | 2.472 | 49.1% | 152.2 | 150.9 | 1.20× |
| | eagle3, **k=7** | 2.747 | 25.0% | 160.1 | 158.9 | 1.26× |
| | DFlash, k=7 | 3.806 | 40.1% | 221.0 | 218.6 | **1.74×** |
| | DFlash, **k=15** | 4.163 | 21.1% | 218.0 | 214.4 | 1.71× |
| **livecodebench** | baseline | — | — | 126.2 | 125.4 | 1.00× |
| | vanilla MTP, k=3 | 3.418 | 80.6% | 186.7 | 184.9 | 1.48× |
| | vanilla MTP, **k=7** | 5.157 | 59.4% | 232.9 | 229.7 | **1.85×** |
| | eagle3, k=3 | 2.583 | 52.8% | 153.1 | 152.0 | 1.21× |
| | eagle3, **k=7** | 2.974 | 28.2% | 163.2 | 161.8 | 1.29× |
| | DFlash, k=7 | 3.987 | 42.7% | 219.7 | 217.1 | **1.74×** |
| | DFlash, **k=15** | 4.142 | 20.9% | 201.9 | 199.6 | 1.60× |

**Takeaways**
- **Higher k helps a lot, and the vanilla MTP assistant scales best:** k=3→k=7
  lifts it from 1.48–1.56× to **1.85–2.12×**, at k=7 **matching/beating DFlash
  k=15** on gpqa/livecodebench and near it on aime.
- **DFlash** (block-diffusion, block_size=16) is strong (**1.60–2.23×**) with a
  distinct profile: *low* accept_rate but *high* accept_len — it drafts a block
  in parallel, so even low per-token acceptance commits many tokens/pass. Notably
  **k=7 ≈ k=15** (and k=7 slightly *wins* on gpqa/livecodebench: 1.74 vs
  1.71/1.60×) because fewer speculative tokens raise the accept_rate — so k≈7 is
  a sweet spot, not the max block. Served `method: dflash`, `--attention-backend
  triton_attn` (no flash_attn).
- **EAGLE3** gains least from depth (k=3→k=7: only accept_len 2.5→3.0), trailing
  the others here.
- k is the method's speculative depth; each method's natural config differs, so
  compare achieved speedup across the k-sweep, not raw k.

---

## 2. Gemma-4-31B-it — MTP assistant, k sweep (4×A100, tp=4)

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

## 3. Training-time shared-KV attention leak (root-caused & fixed)

**Symptom.** Fine-tuning the MTP assistant with the in-repo online trainer
(`scripts/gemma4_mtp/train_online.py`) *destroyed* it: accept_len collapsed from
~3.5 (stock) to ~1.1 (§1, trained-50k), i.e. ~3× *slower* than the stock draft
and 2× slower than no draft.

**Root cause.** The assistant's `create_attention_masks`
(`transformers/models/gemma4_assistant/modeling_gemma4_assistant.py`) builds an
**all-ones bidirectional** mask over the target's full teacher-forced KV. During
TTT training (`_assistant_step`, `attention_mask=None`, q_len = L = T−k−1 > 1),
query row `t` — which predicts token `k+t+1` — can attend the target's
`KV[k+t+1]`, i.e. the **KV derived from the token it must predict**. The draft
learns to read this leak (train loss → ~0 fast), then collapses at inference
where future KV does not exist.

**Why it's invisible at inference.** vLLM drafts one token at a time
(q_len == 1) against a KV cache that holds only the past. With a single query
row, "bidirectional" ≡ "causal", and there is no future KV to leak — so the
stock/EAGLE3 drafts (trained with correct masking) serve correctly and the bug
never surfaces during decoding. It only bites the multi-position TTT training.

**Fix.** Make the shared-KV attention **block-causal**: query row `t` (absolute
position `k+t`) may attend target KV `j` only for `j ≤ k+t` (full-attn) or
`k+t−window < j ≤ k+t` (sliding). The offset `k` is derived inside the mask
builder as `k = kv_len − q_len − 1` (in the TTT loop kv_len = T, q_len = T−k−1),
so nothing extra needs threading through. Applied as a monkeypatch
(`patch_causal_shared_kv_masks` in `train_online.py`), verified with a
mask-print test (lower-triangular w.r.t. the k-offset; zero future leak).

---

## 4. Training runs

| run | init | data | GPUs (backbone / draft split) | status |
|---|---|---|---|---|
| pretrained-50k | stock assistant | 50k (first-N) | 0 / 1 | done (leaky objective) |
| random-init 20k | from scratch | 20k | 2 / 3 | done (leaky objective) |
| random-init full | from scratch | 338k (all) | 2 / 3 | superseded by mask-fix rerun |

Online trainer places the frozen 26B target on one GPU and the trained assistant
(+ optimizer) on another (`train_online.py` device split); target `shared_kv_states`
come from the in-process HF forward (`return_shared_kv_states=True`) — **not** the
vLLM `Gemma4SharedKVStatesConnector` (that connector is the decoupled
vLLM-server export path, unused here).

> **Note:** runs above predate the §3 fix and produce non-inference-valid drafts.
> Post-fix reruns supersede them.

_Raw results: `scripts/evaluate/mtp_server_eval/results/26b_compare/results_table.{md,csv}`
and `scripts/evaluate/experiments/results/gemma4-31b/results_table.{md,csv}`._
