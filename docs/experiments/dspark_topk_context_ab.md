# DSpark top-k context selection: first A/B on Gemma-4-26B-A4B (2026-10-05)

Does a DSA-style lightning indexer that restricts the draft's full-attention
layer to its top-k context positions cost acceptance? First controlled test of
the implementation in commit `227c3a6` (`src/speculators/models/dspark/topk.py`,
design in [`docs/user_guide/algorithms/dspark.md`](../user_guide/algorithms/dspark.md#top-k-context-selection-dsa-style-experimental)).

**Result: at 32% average density on the one full-attention layer, top-k costs
1.2% accept_len (3.853 vs 3.898) on the held-out split, with the selection
capturing only half of the dense attention mass.** The draft is nearly
indifferent to which two thirds of its context KV it loses, which is the same
conclusion the sliding-window ablation reached on Kimi-K3 DSpark. The indexer
trained (KL 2.5 → 1.5) but its recall over an untrained selection is not yet
isolated; see follow-ups.

## Setup

| item | value |
|---|---|
| Base checkpoint | `output/gemma4_26b_dspark_from_dflash/dspark/checkpoints/checkpoint_best` — the warm-start-from-DFlash DSpark (2.03× best result): 5 layers, layers 0–3 sliding (2048), **layer 4 full attention**, block 16, full 262k vocab, aux taps `[2, 7, 12, 18, 23, 28]` |
| Data | the same run's `data_prep` (400k regen rows, seq 4096). Token length percentiles 10/25/50/75/90: 232 / 366 / 644 / 1515 / 3846 |
| Both arms | continued training, **8000 steps, 1 GPU each**, lr 1e-4 constant, max_anchors 512, loss `{"ce": 0.1, "tv": 0.9}`, confidence alpha 1.0, `--train-data-ratio 0.99` (shared held-out split), hidden states generated online by one 26B server per arm |
| A (dense) | nothing else |
| B (top-k) | `--topk-context 128 --topk-local-window 32 --topk-warmup-steps 1000 --indexer-loss-weight 1.0`; top-k layers default to `[4]`; indexer 4 heads × 64 |
| Launchers | `examples/train/dspark_topk_ab_local.sh` (A) and `output/gemma4_26b_dspark_topk_ab/run_B_k128.sh` (B, relaunched after the first B attempt OOMed — see "bugs found") |
| Hardware | 4× A100 80GB: GPU0/2 servers, GPU1 A, GPU3 B. A: 8000 steps in ~62 min; B: ~68 min (indexer target adds ~10%) |

K=512 / window 128 was the first choice and was dropped: with a 644-token
median sample it would have covered 87% of candidate positions on average, so
the arm would have measured nothing. K=128 / window 32 gives ~0.5 expected
density over the length distribution and ~0.1 on 4k-token samples.

## Validation (same held-out split, ~4k samples)

| metric | A dense | B top-k 128+32 | Δ |
|---|---:|---:|---:|
| **accept_len** (block 16) | **3.898** | **3.853** | −1.2% |
| accept_rate | 0.462 | 0.458 | −0.9% |
| full_acc | 0.494 | 0.489 | −1.0% |
| position_1 acc | 0.729 | 0.727 | |
| position_8 acc | 0.466 | 0.461 | |
| position_15 acc | 0.370 | 0.365 | |
| ce / tv loss | 0.684 / 0.124 | 0.697 / 0.125 | |
| topk_density | — | 0.316 | selected / candidate positions |
| topk_recall | — | 0.496 | dense-attention mass inside the selection |
| indexer_kl | — | 12.6 (val mean), 1.5 (train, last 500 steps) | see note |

Training-side accept_len over the last 500 steps: A 3.934, B 3.888 (−1.2%,
consistent with validation). B's indexer KL fell from 2.53 (first 100 steps,
dense warm-up) to 1.48 (last 500, sparse); recall rose from 0.45 to 0.50 at
constant density.

**The val-mean KL of 12.6 is an outlier effect, not a different regime.** The
per-step train KL has a max of 801 with a median of 1.4: on some samples the
indexer's unnormalised ReLU scores (summed over 16 block queries × 4 heads)
make `softmax(I)` extremely peaked, and KL(p‖q) explodes wherever the dense
attention spreads mass outside that peak. It does not affect the draft (the
KL only reaches the indexer) but it does dominate `val/loss`, so **do not use
`val/loss` or `--save-best` to pick checkpoints for top-k runs**; use
`val/accept_len`. Fix candidates: scale the indexer logits (divide by
block_size·√d or learn a temperature) before the softmax, or clip.

## Reading

- The one full-attention layer tolerates losing ~68% of its context KV for a
  1.2% accept_len cost, while the selection keeps only ~50% of the attention
  mass. The layer's output is not very sensitive to the pruned positions; most
  of what the draft needs is in the block itself and the sliding layers.
- This matches the Kimi-K3 sliding-window ablation (≤0.5% loss at a 2048
  window) and the Gemma-4 context sweep (accept_len flat from 1k to 32k). On
  this suite the draft's useful context is local and small.
- **It is not yet shown that the indexer matters.** Recall 0.50 at density
  0.32 is above the 0.32 a random selection would get, but the always-on
  32-token local window plausibly supplies most of that. A window-only control
  (same density, no ranking) is the missing arm.
- Nothing here is measured at long context: 90% of samples are under 3.9k
  tokens and the eval split has the same distribution. The efficiency case for
  top-k (cutting the draft's context attention at 32k–128k) is untested, and
  it needs vLLM-side support for the indexer before it can be served at all.

## Bugs found while running this

1. **Flex attention must be compiled on its own.** Arm B's first attempt
   OOMed (12 GiB alloc in `sdpa_dense_backward`): building the per-layer top-k
   mask mid-loop broke the compiled graph, the following `flex_attention` ran
   eagerly, and eager flex falls back to a dense `[H, Q, KV]` path. Fixed in
   `src/speculators/models/attention.py` by calling a `torch.compile`d
   `flex_attention` (PyTorch's recommended usage). Memory went from 78 GB to
   38 GB, the same as the dense arm.
2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` must not be exported to
   the vLLM hidden-state server; its KV connector refuses it. Set it on the
   trainer only (the launcher does).
3. Passing `--loss-fn '{"ce": 0.1, "tv": 0.9}'` through a `$(...)` echo splits
   the JSON; the launcher now builds the argument list as a bash array.

## Follow-ups, in order of value

1. **Window-only control** at the same density (e.g. `--topk-context 0
   --topk-local-window 160`, or random scores) to isolate the indexer's
   contribution to recall and accept_len.
2. **Indexer score normalisation** (temperature / scale) so the KL is
   well-conditioned; then re-check recall. If recall climbs well above the
   window-only control at equal density, the indexer is doing real work.
3. **Density sweep** (K = 32, 64, 128, 256 at window 32) to find where
   accept_len starts to drop.
4. **Long-context arm**: train and validate on ≥16k sequences (AA-LCR style
   data) where the full layer's context is 4–30× larger than K; this is the
   only setting where the mechanism can pay for itself.
5. vLLM proposer support for the indexer, so a top-k checkpoint can be served
   and measured for real decode throughput.

Raw logs and checkpoints: `output/gemma4_26b_dspark_topk_ab/`
(`logs/{A_dense,B_topk}.log`, `{A_dense,B_topk}/checkpoints/checkpoint_best`;
`logs/B_topk_k512_oom.log` and `logs_failed_attempt{1,2}` are the aborted
attempts).
