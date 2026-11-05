# GLM-5.3 draft training plan — many drafts, one verifier stream

Status: planning, 2026-11-05. Branch `mengmengj/glm-dflash2`. Owner: mengmengj.
Companion to README §9 (data generation). Nothing here has run yet.

## 0. Summary

- The verifier forward (GLM-5.3, 744B) is ~8× the cost of a draft's forward+backward
  per row. One verifier stream can therefore feed several drafts for free; what we
  pay for extra drafts is trainer GPUs, not verifier time.
- One B300 node hosts the bf16 verifier with TP8 and no pipeline stage. That removes
  the two-node TP8×PP2 layout and the `vllm_pp_aux` plugin ravir needed on B200s.
- One B200 node (8 × 180 GB) trains **8 drafts at one per GPU** with margin, likely
  **16 at two per GPU** (to be measured, §3). A draft rank keeps up with the stream
  at ~3× margin, so one GPU per draft is enough.
- Ravir's trainer, Mooncake transport, draft config and recipe are reused as-is;
  the only new code is a shared-stream mode in the trainer's data path (§5).

## 1. What ravir already has (reuse, do not rebuild)

Worktree `/import/ml-sc-scratch1/ravir/spec_decoding_work/speculators/.claude/worktrees/glm53-dflash2-kimi-recipe`,
recipe doc `examples/train/glm53/README_dflash2_kimi_recipe.md`, runs under
`/import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2/`.

| piece | what it is |
|---|---|
| verifier | vLLM 0.26 **TP8 × PP2 on two B200 nodes** (bf16 does not fit one B200 node: 1.51 TB vs 1.44 TB), `vllm_pp_aux` plugin so stage-0 aux hidden states reach the connector |
| transport | Mooncake store, tcp over the nodes' 100 GbE (`mooncake-transfer-engine 0.3.13`, env `/import/ml-sc-scratch1/ravir/envs/glm53_dflash2`) |
| trainer | third B200 node, `torchrun --nproc_per_node 8`, DDP (no FSDP), one packed 16k sequence per rank per step |
| draft | `glm53_dflash2_draft_config.json`: 5 Qwen3-style layers, hidden 6144, 64 q / 16 kv heads × 64, MLP 12288, all-sliding 2048, full 154,880 vocab; aux layers 7/20/44/57/71 (+78). ≈1.7B trainable + 1.9B frozen embed/lm_head ≈ 3.6B |
| recipe (Kimi K3) | AdamW 5e-5, cosine, warmup 0.04, 10 epochs, block 7, anchors 512, `kl_div` loss with exp decay γ 4, grad clip 1 |
| measured | their on-policy fine-tune on our 20% snapshot: 2,905 steps/epoch in 5h39m. Verifier delivered mean 18.0k / p90 20.2k / max 24.7k prompt tok/s with ~11 requests running and 0 waiting — the verifier had headroom, the trainer side was pacing. KV 504,768 tokens. |
| results | off-policy (gemma782k) draft: LCB accept_len 4.33 at fp8/v033 (≈ the incoai release). Their 3-epoch on-policy fine-tune is being evaluated. |

TP8 = tensor parallel over 8 GPUs: every weight matrix is split eight ways and all
eight GPUs compute each token together. PP2 = the 78 layers are cut into two
stages on two nodes, with one ~24 KB activation hop per token between them.

## 2. GPU layout

| layout | verifier | drafts | status | when |
|---|---|---|---|---|
| **A (plan)** | 1 B300, vLLM 0.33 TP8 bf16, stock aux-hidden-state path | 1 B200 node, 8–16 drafts | components exist; untested as a combination | first B300 free of regen |
| B | 2 B300, one engine each | 1 B200 node | trainer must pick its endpoint by rank (small change) | both B300 free; halves epoch time |
| C | 1 B300 shared: vLLM at ~0.7 memory utilisation + 8 single-GPU drafts on the same GPUs | same node | untested; ~65 GB/GPU for a draft means anchors 256 or activation checkpointing; trainer steals ~20–30 % of verifier time | no B200 node available |

Numbers behind the layout:

| item | value |
|---|---|
| verifier weights | 1.4 TB bf16 → 190 GB per B300 GPU (275 GB), ~60 GB headroom at util 0.9; prefill-only requests need little KV |
| hidden-state payload | 6 layers × 6144 × bf16 = 74 KB/token (ravir's 5 aux + last); 86 KB with 6 aux + last |
| stream rate | 18–25k tok/s measured on 2×B200 PP2; one B300 TP8 should land in the same band (not measured: the probe was cancelled for regen priority) |
| transport | 20k tok/s × 74 KB = 1.5 GB/s → Mooncake over 100 GbE is fine; NFS (~400 MB/s) is not; same-node `/dev/shm` is fine |
| data per epoch | ~600–650k usable rows after the §9.v cleanup; 5.3k tokens/row mean with truncated rows kept (14 % at the 16k cap), ~3.5k if dropped → 2–3.4B tokens |
| epoch time | ~1.2–2 days per verifier node; 10-epoch recipe = 2–3 weeks on layout A, half on B; a 3-epoch warm-start fine-tune = 4–6 days |
| draft cost per row | ≈ 6 × 1.7B × 3,584 positions (512 anchors × block 7) ≈ 37 TFLOP vs verifier ≈ 80 GFLOP/token × 5.3k ≈ 420 TFLOP → 1:11; one B200 draft rank handles ~10 rows/s against a stream of ~3.8 rows/s |

Decode (on-policy rollouts, OPD-style) is a different budget: ~3.4k tok/s per B300,
so a 16k-prompt round is 6–7 h of verifier time. The Gemma OPD line was negative on a
converged seed (opd/README.md §6); plan at most one pilot, late.

## 3. How many drafts fit on one B200 node

Per-draft footprint, ravir's config, one rank, no sharding (estimate):

| component | GB |
|---|---|
| trainable 1.7B: fp32 params + grads + AdamW m, v (16 B/param) | 27 |
| frozen embed + lm_head, bf16 | 4 |
| hidden-state input for one packed 16k sequence (6 × 6144 × bf16 × 16k) | 1.2 |
| draft activations, 5 layers × 3,584 positions | ~5 |
| loss: draft logits + verifier logits (kl_div) 3,584 × 154,880 fp32, with softmax/grad copies (fused implementation chunks this) | 10–20 |
| CUDA/NCCL/allocator overhead | ~5 |
| **total** | **~55–65** |

Reference point: the Gemma DSpark draft (2.4B, 32k vocab, FSDP over 4 ranks) sits at
82 GB per B200 rank in the running OPD job; its logits are 5× smaller but it holds a
full-vocab verifier distribution too.

Plan:
- **8 drafts, one per GPU — safe.** 183 GB per GPU vs ~65 GB.
- **16 drafts, two per GPU — likely.** Two processes per GPU with `CUDA_VISIBLE_DEVICES`
  pinned; needs the measured footprint ≤ ~85 GB. Compute margin drops from 3× to
  1.5× per draft, still enough.
- 24 (three per GPU) only with anchors 256 and `--loss-implementation fused` verified
  under 55 GB; not planned.

**Measure before deciding (no verifier needed):** run one trainer rank on the H200
sc3-c127 against cached hidden states (`--on-missing skip`, a few hundred rows cached
once from any verifier run, or ravir's `probe_hidden_states.py` output), with ravir's
exact flags, and read `torch.cuda.max_memory_allocated`. Repeat with anchors 1024 and
block 16 (the largest variants in §4). Half a day.

## 4. Variations: fixed architecture, one shared stream

Everything downstream of the verifier hidden states can differ per draft. The
architecture (layers, hidden, heads, window) stays ravir's so checkpoints are
comparable and the serving path is already proven.

| knob | flag | shareable? | candidates |
|---|---|---|---|
| block size | `--block-size` | yes (blocks are built from the same tokens + hidden states) | 7 (Kimi/incoai k=6), 12, 16 — Gemma showed k=12 +7…16 % tok/s over k=8 |
| loss | `--loss-fn` | yes | `kl_div` (ravir), `ce` (DFlash default), `{"ce":0.1,"tv":0.9}` (our Gemma recipe), `jsd`, `rkl` |
| position weighting | `--per-position-loss-weight`, `--dflash-decay-gamma`, `--dpace-alpha` | yes | γ 4 vs 12 (γ 12 was used for block 12 on Gemma); `dpace` vs `fixed-exp-decay` |
| selector | `--selector-loss-alpha`, `--selector-top-k` | yes | α 1 vs 0.5; top-k 16 vs 32 |
| conv | `--conv-kernel-size`, `--conv-group-size` | yes | kernel 2 vs 4 |
| anchors per row | `--max-anchors` | yes (changes draft cost, not the stream) | 512 vs 1024 |
| optimiser / LR / schedule | `--optimizer`, `--lr`, `--scheduler-*` | yes | AdamW 5e-5 (Kimi) vs 3e-4 from scratch; Muon (branch default) vs AdamW |
| init | checkpoint path | yes | from scratch vs warm start from ravir's off-policy checkpoint |
| aux layer set | `--target-layer-ids` | **only as a union**: the verifier dumps one fixed set, each draft selects its subset (needs a per-draft index-select in the dataset transform; file size grows) | 7/20/44/57/71 vs a 6-layer set like incoai's 5/19/33/47/61/75 |
| `sample_from_anchor` | fixed False | — | DFlash2 default and what vLLM decodes; do not vary |
| data, row order, seq length, truncated-row policy | — | **no** | one choice per stream; run a second stream for a data ablation |

First sweep, 8 slots (one per GPU), all warm-started from ravir's off-policy
checkpoint so differences show within 1–2 epochs:

| slot | block | loss | γ | other | question it answers |
|---|---|---|---|---|---|
| 1 | 7 | kl_div | 4 | ravir's recipe, LR 5e-5 | baseline / parity with ravir |
| 2 | 12 | kl_div | 4 | | block size (Gemma's main win) |
| 3 | 16 | kl_div | 12 | | block size, longer |
| 4 | 7 | ce 0.1 + tv 0.9 | 4 | | Gemma loss on the DFlash2 class |
| 5 | 7 | kl_div | 4 | anchors 1024 | more supervised positions per row |
| 6 | 7 | kl_div | 4 | selector α 0.5, top-k 32 | selector strength |
| 7 | 7 | kl_div | 4 | LR 2e-4 | LR headroom on warm start |
| 8 | 12 | ce 0.1 + tv 0.9 | 12 | | best-guess combination |

With 16 slots: add from-scratch twins of 1, 2 and 8, γ 12 on block 7, conv kernel 4,
dpace weighting, jsd, and the 6-aux-layer set (needs the union dump).

Evaluation of every slot: `glm53-full-v033-b300.yaml` entries with `ONLY=<name>`
(~3.5 h each on a B300; short set first). Report greedy and sampled (T 1.0 / top-p
0.95) accept_len, as in README §9.w. Bar: beat the incoai release (full-suite mean
3.90 sampled / DFlash2 k7) and ravir's baseline slot.

## 5. How to train them at the same time

### 5.1 The problem in the current code

`ArrowDataset._get_raw_data` asks the transfer backend for a cached row, and on a
miss (`--on-missing generate`) sends the row to vLLM, loads the hidden states by
handle, then **deletes** them (`--on-generate delete`). Every trainer process is its
own consumer, so N independent trainers = N verifier passes per row. Mooncake's
transfer has no cache at all (`get_cached` returns None) — the handle comes back in
the vLLM response, so only the requester knows it.

### 5.2 Option A — N processes, one shared stream (plan)

One trainer process per draft (`torchrun --nproc_per_node 1`, `CUDA_VISIBLE_DEVICES`
pinned), all reading the same dataset with the same seed, so the multipack sampler
(seeded by `seed + epoch`, lengths only) gives every process the same row order.
Changes, all in `src/speculators/train/data.py` and `hs_connectors`:

1. **Deterministic key per row.** The vLLM connector already honours a client-supplied
   `hidden_states_path` (file backend) / key (Mooncake) in `kv_transfer_params`.
   Use `hs_<epoch>_<file_idx>` so every process can compute the key without talking
   to the requester.
2. **One requester.** `--hs-role leader|follower`. The leader generates on a miss as
   today but keeps the result (`on_generate cache`). Followers get a new
   `--on-missing wait`: poll the key (file exists and is complete / Mooncake `exists`)
   with a timeout, never request.
3. **Refcounted delete.** Each reader drops a marker (`<key>.done.<slot>` on the file
   backend; a counter key in Mooncake). A small sweeper in the leader deletes a row
   once `N` markers exist, and refuses to run more than `CACHE_ROWS` rows ahead of
   the slowest follower. `/dev/shm` on a B200 node is 1.2 TB ≈ 14M tokens ≈ 2,600
   rows of buffer; Mooncake's segment is sized the same way.
4. **Union of aux layers** (only if §4's aux-layer variant is in the sweep): the
   verifier dumps the union; a per-process index-select in the transform.

Properties: independent wandb runs, independent crashes (a follower can restart and
resume at its own step; only the leader's death stops the stream), different batch
sizes and LRs are free, and the verifier sees exactly one request per row. The
whole group runs at the pace of its slowest member, which is fine because every
member has ≥1.5× compute margin over the stream.

Estimated effort: 1–2 days, including a two-process test on the H200 with the file
backend (no verifier: pre-generated rows + a fake leader that copies files in).

### 5.3 Option B — one process, N drafts

One trainer holds N draft modules, runs one batch through all of them, N optimisers.
Simplest data path, exact apples-to-apples batches. Rejected for the sweep: the N
drafts time-share one GPU set (throughput ÷ N unless each is sharded over all 8
GPUs), one crash kills all N, and every variant must share batch size and anchors.
Worth revisiting for pairs that must see identical batches (loss ablations).

### 5.4 Option C — zero code, two drafts

Run two independent trainers against the same verifier today. The verifier does
2× the work, which ravir's numbers say it can absorb (11 running, 0 waiting at 18k
tok/s). This is the pilot: it validates layout A end to end (B300 verifier, vLLM
0.33, Mooncake, trainer env) before §5.2 exists. Beyond two, build §5.2.

### 5.5 Launch shape (layout A)

```
B300 node   : vllm serve GLM-5.3-bf16, TP8, --kv-transfer-config {Mooncake producer,
              target-layer-ids = union}, max-num-seqs 64, max-model-len 32768
B200 node   : mooncake_master; for slot in 0..7:
                CUDA_VISIBLE_DEVICES=$slot torchrun --nproc_per_node 1 scripts/train.py
                  --hidden-states-backend mooncake --hs-role {leader|follower} \
                  --seed 0 <slot flags from §4> --run-name glm53_multi_s$slot
```

Two sbatch jobs (verifier, trainer) with a rendezvous file, like
`sbatch_regen_glm53_b200pair.sh`; the verifier job is the "worker", the trainer job
depends on it. Reservation name rotates (`sn50-hdi-1` today) — check with
`scontrol show res` before submitting.

## 6. Order of work

1. **Footprint measurement** on sc3-c127 (§3): decides 8 vs 16 slots. No verifier.
2. **Verifier on one B300 with the stock 0.33 aux path + Mooncake producer**, gated by
   ravir's `probe_hidden_states.py --verifier` (lm_head top-1 agreement, NaN scan).
   Needs a B300 free of regen (shard A ends ~Nov 9; c131 is on shard B2).
3. **Option C pilot**: two drafts (slots 1 and 2) for one epoch on the full cleaned
   data, evaluated on the short set. Validates layout A and gives the first
   block-size answer.
4. **Shared-stream mode** (§5.2), tested on the H200, then the 8-slot sweep.
5. Full-suite evals of the top slots; pick the recipe; 10-epoch run from scratch of
   the winner (or continue the warm start, depending on slot 1 vs its scratch twin).

## 7. Open questions

- Truncated rows (14 % at the 16k cap, 30–40 % of tokens): keep or drop is a
  per-stream decision (§4). README §9.x recommends keeping; one stream can't test both.
- Whether ravir's on-policy fine-tune improved over the off-policy draft — their
  eval is in flight; if it did not, warm-start vs scratch (slot 1 vs twin) becomes
  the first question, not block size.
- vLLM 0.33 on the verifier with the 0.26-era trainer env on the trainer node:
  they only share HTTP and Mooncake, so it should be fine; confirm in step 2.
