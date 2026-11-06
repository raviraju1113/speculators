# GLM-5.3 draft training plan — many drafts, one verifier stream

Status: planning, 2026-11-05. Branch `mengmengj/glm-dflash2`. Owner: mengmengj.
Companion to README §9 (data generation). Nothing here has run yet.

## 0. Summary

**Scope decision (2026-11-05):** no sweep and no Draft-OPD until a working recipe
exists. Phase 1 is the from-scratch on-policy recipe (§2a: incoai shape, LR from a one-day
pilot), run as the leader of an 8-slot group if group mode passes its acceptance test
in time, otherwise alone (§6); validation every half epoch. §3–§5 describe the
capacity and the multi-draft mechanism for phase 2, once one of the two beats the
incoai release. "On-policy" here means SFT on GLM-5.3's own regenerated answers;
Draft-OPD (the rollout objective, negative on Gemma) is a separate, later question.

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
| results | their `examples/train/glm53/results/README.md`: off-policy epoch 9 = 2.83 mean sampled accept_len (22 benchmarks), clean on-policy fine-tune ≈ 3.0, incoai 3.89; position-0 accuracy 0.68 vs 0.80; multilingual and tool_call worst |

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
| epoch time | ~1.2–2 days per verifier node; the full 10-epoch recipe = 2–3 weeks on layout A, half on B. Our rows are ~5× ravir's gemma782k rows in tokens (5.3k vs 991 mean), so one of our epochs ≈ several of theirs; checkpoint every half epoch and stop when the short set plateaus — first evaluable draft after 1–2 days |
| draft cost per row | ≈ 6 × 1.7B × 3,584 positions (512 anchors × block 7) ≈ 37 TFLOP vs verifier ≈ 80 GFLOP/token × 5.3k ≈ 420 TFLOP → 1:11; one B200 draft rank handles ~10 rows/s against a stream of ~3.8 rows/s |

Decode (on-policy rollouts, OPD-style) is a different budget: ~3.4k tok/s per B300,
so a 16k-prompt round is 6–7 h of verifier time. The Gemma OPD line was negative on a
converged seed (opd/README.md §6); plan at most one pilot, late.

## 2a. Phase-1 hyperparameters — from scratch, on-policy

Anchor = ravir's mapping of the Kimi K3 recipe, changed only where our data differs.

| | value | why |
|---|---|---|
| data | full cleaned regen (`nemo782k_regen_glm53_all.jsonl` after the §9.v `!{32,}` filter), `prepare_glm53_data.sh` with `SEQ_LENGTH=16384`; keep rows truncated at the 16k cap (they are cut at 16k, README §9.x); loss mask = `<think>…</think>answer` of every assistant turn, non-final-turn thinking blanked | the served distribution, whole |
| validation | `train_ratio 0.998` (~1.3k held-out rows) → val accept_len every half epoch | Gemma practice |
| draft | **incoai shape** (2026-11-06): 6 layers, hidden 6144, 64 q / 8 kv heads × head_dim 128, MLP 12288, SWA 2048 all layers, full vocab, conv kernel 2 / group 16, selector rank 256 / top-16, `sample_from_anchor=False` (≈2.3B trainable + 1.9B frozen) | the shape that reaches 3.89 on this target; ravir's results put the gap at position 0 (0.68 vs 0.80), i.e. draft capacity — 5 layers × head_dim 64 is half the attention width |
| aux layers | `--target-layer-ids 5 19 33 47 61 75` (+78 appended by launch_vllm) | incoai's set (0-based 5…75, matching its config); 6 layers incl. a deep one |
| block / anchors | `--block-size 8 --max-anchors 512` (k=7 at serve) | incoai; same k as the bar |
| loss | `--loss-fn kl_div --per-position-loss-weight fixed-exp-decay --dflash-decay-gamma 4`, selector α 1, `--loss-implementation fused` | Kimi |
| optimiser | `--optimizer adamw --lr <from the §6 step-1b pilot; prior 2e-4> --weight-decay 0 --scheduler-type cosine --scheduler-warmup-ratio 0.04`, grad clip 1.0 (hard-coded) | from-scratch LR; Kimi's 5e-5 produced 2.83 off-policy and its 2e-5 fine-tune saturated in half an epoch (ravir's results README), so 5e-5 becomes a contrast slot, not the default. Gemma drafts of this size trained stably at 3e-4–6e-4 |
| batch | 8 ranks × 1 packed 16k sequence = 131k tokens/step, no accumulation (~25 conversations/step at 5.3k tok/row vs ~120 on gemma782k) | same tokens/step as ravir |
| epochs | **3** (≈7.5B tokens ≈ Kimi's 10 × 776M budget), cosine over 3 epochs, checkpoint every 0.5 epoch; extend if val accept_len is still rising at the end | token budget, not epoch count |
| precision | bf16 hidden states, fp32 master weights, DDP | ravir |
| launch | ravir's `run_glm53_dflash2_3node.sh` trainer role unchanged; verifier role = one B300, vLLM 0.33 TP8, same Mooncake producer flags | reuse |

Resolved 2026-11-06 (ravir): the validation curve of
`glm53_dflash2_gemma782k_kimirecipe_v2` had converged by epoch 9. So the 2.83 is what
the Kimi shape reaches on off-policy data at 5e-5 given unlimited epochs: the gap to
incoai is shape and data, not training length. Consequences: the incoai shape stays
the phase-1 default; 3 epochs of on-policy data is a budget, not a convergence
promise — watch val accept_len at every half epoch and extend only while it rises;
5e-5 remains a contrast slot (it converged to a plateau, which does not say the
plateau is the best reachable one).

**What ravir's results (their `examples/train/glm53/results/README.md`, 2026-11-06)
fix as the floor:** Kimi-shape off-policy epoch 9 = 2.83 mean sampled accept_len on the
22-benchmark set (1.90x); + clean on-policy fine-tune on shard A (141k rows, LR 2e-5) =
~3.0 (2.01x), saturating within half an epoch; incoai = 3.89 (2.53x). Largest gaps:
speed-multilingual (0.61 of incoai) and tool_call (0.67) — a prompt-pool problem.

## 2b. Contrast drafts (planned after the recipe is launched, run in parallel per §5)

| contrast | shape | data | init | what it isolates | shares the stream? |
|---|---|---|---|---|---|
| recipe (§2a) | incoai | on-policy | scratch | the candidate | leader |
| Kimi shape, scratch | Kimi (5L × hd64, block 7) | on-policy | scratch | shape vs data: direct comparison with ravir's 2.83 off-policy | yes |
| Kimi shape, warm start | Kimi | on-policy | ravir's epoch-9 checkpoint | floor = their ftA ~3.0; does a real LR move it further | yes |
| LR 5e-5 | incoai | on-policy | scratch | the Kimi LR on the new shape | yes |
| block 12 | incoai | on-policy | scratch | the Gemma block-size win on this class | yes |
| off-policy from scratch | Kimi | gemma782k | scratch | = ravir's existing run; **not rerun** | n/a |

Three parallel drafts at first (recipe, Kimi-shape scratch, Kimi-shape warm start);
the rest once the recipe clears the bar.

**Data action now (in the regen lane, before training starts):** add multilingual and
function-calling prompts to the pool (own `--tag`, merged with
`scripts/merge_dedup_jsonl.py`, regenerated with `--resume`; README §9 "Adding data
later"). These are the two categories where every draft we have is furthest from
incoai, and nemo782k barely covers them.

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
- **Design point: 8 drafts, one per GPU.** 183 GB per GPU vs ~65 GB, and each GPU
  carries a full-recipe draft (131k tokens per optimiser step via accumulation, §5.2)
  with ~2.5× compute margin over the stream.
- 16 (two per GPU) is possible for light variants (anchors 256) if the measured
  footprint is ≤ ~85 GB, at ~1.3× margin; not the plan.

**Measure before deciding (no verifier needed):** run one trainer rank on the H200
sc3-c127 against cached hidden states (`--on-missing skip`, a few hundred rows cached
once from any verifier run, or ravir's `probe_hidden_states.py` output), with ravir's
exact flags, and read `torch.cuda.max_memory_allocated`. Repeat with anchors 1024 and
block 16 (the largest variants in §4). Half a day.

## 4. Variations: fixed architecture, one shared stream

*Phase 2 material — deferred until phase 1 (§6) has a working recipe.*

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

### 5.1 The problem in the current code, and how much chenw's pipeline already solves

`ArrowDataset._get_raw_data` asks the transfer backend for a cached row, and on a miss
(`--on-missing generate`) sends the row to vLLM, loads the hidden states by handle,
then either deletes them (`--on-generate delete`, our Gemma runs) or keeps them
(`--on-generate cache`). Mooncake's transfer has no cache lookup at all (`get_cached`
returns None): the handle comes back in the vLLM response, so only the requester can
read the row.

chenw's penta-draft pipeline (§5.2.5) uses the `cache` path with one shared
`--hidden-states-path` and staggered launches. Against the three sub-problems:

| sub-problem | chenw's pipeline | what remains for us |
|---|---|---|
| N trainers = N verifier passes | **solved** on the file backend: the first trainer to reach a row generates it, every later trainer finds it in the cache | only while the later trainers stay behind the first; a follower that overtakes regenerates. Needs `--on-missing wait` |
| rows are never freed | not addressed: the cache keeps everything (fine at 26 MB/sample on Gemma-26B, 30k samples ≈ 780 GB) | at ~450 MB/row for GLM the full data is ~300 TB/epoch; needs the refcounted delete + bounded buffer |
| Mooncake (cross-node) has no shared read | not addressed: his verifier and trainers share one node and one filesystem | layout A puts the verifier on a B300 and trainers on a B200, so rows must cross nodes. NFS cannot carry it (verifier writes ~1.5 GB/s, 8 readers pull 8× that); Mooncake needs a deterministic key so followers can `get` what the leader requested |

So his pipeline is the skeleton and solves the first point; the other two are the
day of work in §5.2.2. For a same-node pilot with ≤ ~20k rows on node-local NVMe
(§5.2.4 acceptance test, layout C) his script runs unchanged.

### 5.2 The solution: one stream, one leader, followers in lockstep (plan)

Goal: a draft trained inside a group gets **the same batches, the same hidden-state
bytes and the same optimiser steps** it would get alone. Then sharing cannot change
the result, only the wall-clock, and that is checked by the test in 5.2.4.

**5.2.1 Group rules.** Every draft in a group uses the same dataset, seed,
`--total-seq-len` and sampler world size. The multipack sampler is seeded by
`seed + epoch` and depends only on row lengths, so every process then walks the
identical row order. Each draft is **one process on one GPU** (`torchrun
--nproc_per_node 1`, `CUDA_VISIBLE_DEVICES` pinned). The recipe's global batch is
kept by gradient accumulation: 1 rank × accum 8 is the same mean gradient as 8
ranks × accum 1. This costs no wall-clock, because the stream (~3.8 rows/s), not the
GPU (~10 rows/s per draft), is the bottleneck. Things that may differ per draft:
block size, loss, γ, anchors, LR, schedule, init, optimiser, selector/conv settings
(§4). Things that may not: data, seed, sequence length, world size, truncated-row policy.

**5.2.2 Leader and followers** (changes in `src/speculators/train/data.py` and
`hs_connectors`):

1. *Deterministic key per row.* The vLLM connector already honours a
   client-supplied `hidden_states_path` / key in `kv_transfer_params`; use
   `hs_<epoch>_<file_idx>` so every process computes the key itself.
2. *One requester.* `--hs-role leader|follower`. The leader generates on a miss as
   today but keeps the result. Followers get `--on-missing wait`: poll the key
   (complete file on the file backend, `exists` on Mooncake) with a timeout; if the
   key is gone (they fell behind the buffer, see 4), they fall back to
   `generate` for that row — still the same row, just one extra verifier pass.
3. *Refcounted delete.* Each reader drops a marker (`<key>.done.<slot>` on the file
   backend, a counter key on Mooncake); the leader's sweeper deletes a row once all
   `N` markers exist.
4. *Bounded buffer.* The leader does not run more than `CACHE_ROWS` rows ahead of
   the slowest live follower; a follower silent for `T` seconds is declared dead
   and no longer holds the buffer (it catches up via the fallback in 2 when it
   restarts). `/dev/shm` on a B200 node is 1.2 TB ≈ 14M tokens ≈ 2,600 rows; a
   Mooncake segment is sized the same way.
5. *Stream rate is the leader's in-flight request count.* Ravir's run had ~11
   requests running and the verifier idle; set the leader's `num_workers ×
   prefetch_factor` so ≥32 requests are in flight and watch `Running:` / `Waiting:`
   in the vLLM log.
6. *Union of aux layers* only if an aux-layer variant is in the group: the verifier
   dumps the union, each process index-selects in the transform.
7. *Validation:* `check_hidden_states` (shape/token check) already runs per row in
   every process; add a content checksum written by the leader and verified by
   followers so a torn read can never train on partial tensors.

Isolation: each draft has its own process, wandb run, checkpoint dir and log. A
follower crash stalls nobody (rule 4) and resumes at its own step; only the leader's
death stops the stream, and restarting the leader restarts it.

**5.2.3 Launch shape** — see 5.5.

**5.2.4 Acceptance test, before any run that matters uses group mode.** Train the
§2a recipe for 300 optimiser steps three ways from the same seed:

| run | launch |
|---|---|
| A | alone, 8 ranks × accum 1 (ravir's launch, the reference) |
| B | alone, 1 rank × accum 8 |
| C | as a follower in a 3-draft group, 1 rank × accum 8 |

Pass = the loss, `train/accept_len` curves and the step-300 val accept_len of B and C
agree with A within the noise measured by two A runs with different seeds, and the
hidden-state checksums read by C match the leader's. Fail = group mode is not used
for anything that feeds the SOTA candidate; contrasts then run as independent
trainers (§5.4) at the verifier's expense.

Estimated effort: about a day on top of chenw's launcher pattern (§5.2.5), plus the
test (it needs the verifier, so it runs on the phase-1 setup during a checkpoint gap
or on the second B300).

### 5.2.5 Prior art we build on (reviewed 2026-11-06)

**chenw's multi-draft pipeline on `origin/main`** (`examples/train/gemma4_26b_penta_draft_online.sh`,
`docs/TRAINING.md` "Simultaneous multi-draft training", commits da62bd76 / 88e3a3eb):
N independent `scripts/train.py` processes, one per draft, on one vLLM hidden-states
server, all with `--on-generate cache --on-missing generate` and one shared
`--hidden-states-path`. Each sample's features are generated once and kept on disk for
every trainer and epoch; launches are staggered 90 s so later trainers hit the cache.
Validated on Gemma4-26B (4 GPUs, five draft types, trainer GPUs at 100 %, fetch wait
2–5 %). It is exactly rules 1–2 of §5.2.2 using code that is already in our `data.py`
(the `cache` path is upstream), so **the launcher pattern is reused as-is**. What it
lacks for our scale: it keeps everything (26 MB/sample at Gemma-26B seq 4096; for GLM
at 16k it is ~1.4 GB/sample, ~450 MB/row at the 5.3k mean → ~300 TB per epoch), and
nothing stops a follower from overtaking the leader and regenerating. So for ≤ ~20k
rows on node-local NVMe (11 TB) his script works unchanged — that is the §5.2.4
acceptance test and any pilot — and the full-data run needs only the two additions:
`--on-missing wait` for followers and the refcounted delete / bounded buffer (rules
2–4). Effort drops to about a day. (`train_online.py`'s YAML `drafts:` mode is the
Gemma4-MTP trainer and runs drafts sequentially; not relevant.)

**ASpec** (`github.com/inclusionAI/ASpec`, Ant/Ling, built on TorchSpec): a full
training framework with the same shared-Target design already implemented —
`aspec/multi_draft.py` runs independent drafts over one bounded Target stream (layer
*union* captured once, each consumer reads its ordered view, bounded backpressure,
fenced readers), `aspec/transfer/mooncake/deferred_delete.py` is the lease-based
refcounted delete, plus draft-side kernels (DSpark QK-norm/RoPE CuTeDSL, FA4 sparse,
cuDNN THD packing, context parallelism) giving ~2× draft-only throughput vs TorchSpec.
Not directly usable for GLM-5.3 today:

| blocker | detail |
|---|---|
| target support | capture runs inside **SGLang 0.5.21** with per-target contracts; recipes cover Qwen3 family and Ling only. GLM-5.3 (`glm_moe_dsa`, 744B) would need a new target contract |
| topology | shared-Target mode is **single-node** (Target + Drafts on one NVLink node); cross-node Target TP and multi-node multi-draft are explicitly out of scope / roadmap. A bf16 GLM-5.3 target fills all 8 GPUs of a B300, leaving no draft GPU. Only an fp8 TP4 target (4 + 4 on one B300) would fit their contract, and ravir rejected fp8 target numerics |
| algorithm | DFlash2 is marked **experimental**, "no claim to reproduce an official recipe" |
| export / eval | native and SGLang assets; our eval and serving are vLLM (`export_dflash2_vllm.py`) |
| bottleneck | their speedups are draft-side; our epoch time is set by the 744B verifier, where the draft already has ~2.5× margin |

What we take from it: the design confirmation (union capture + ordered views + bounded
backpressure + deferred delete is what a shipped framework does), and
`deferred_delete.py` as the template for the refcounted delete in `hs_connectors`.
Revisit if ASpec adds a GLM/K3 target contract (a `kimi_k3` data renderer already
exists) and multi-node shared mode.

### 5.2.6 What adopting ASpec's ideas would cost us

| idea | in our repo | effort | verdict |
|---|---|---|---|
| shared-Target stream (union capture, per-draft views, backpressure, refcounted delete) | the two additions in §5.2.2 on top of chenw's launcher; `deferred_delete.py` (MIT) as template | ~1 day + acceptance test | **do it** |
| overlap feature generation with training; GPU-to-GPU transport | ravir's verifier sat idle at ~11 in-flight requests, so the fetch path paced the run: raise in-flight requests (`num_workers × prefetch_factor`) first; Mooncake already has an `rdma` protocol flag if the NICs allow it | hours (config); days if RDMA needs debugging | try the knob |
| draft-side kernels (CuTeDSL QK-norm/RoPE, FA4 sparse block attention, cuDNN THD packing, CP) | tied to ASpec's model classes and attention layouts; a rewrite of our draft's attention path | weeks | **no** — the verifier bounds epoch time, the draft has ~2.5× margin |
| sequence packing, selective head/loss computation | already have: multipack sampler at 16k, fused loss, logits only on anchor positions | 0 | nothing to do |
| Plan/Evidence, exact recovery, strict export | checkpoint resume and ravir's vLLM export already cover our needs | days | not now |

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

**Phase 1 — a working recipe (two drafts, zero new code).**

1. **Verifier on one B300** with the stock vLLM 0.33 aux-hidden-state path and the
   Mooncake producer, gated by ravir's `probe_hidden_states.py --verifier` (lm_head
   top-1 agreement, NaN scan). Needs a B300 free of regen (shard A ends ~Nov 9;
   c131 is on shard B2). Data: the full cleaned regen set, tokenized with
   `prepare_glm53_data.sh` (truncated-row policy decided once, README §9.x).
1b. **LR pilot, three independent trainers, no new code (~1 day).** LR 5e-5 / 2e-4 /
   6e-4 on the first 40k cleaned rows, same seed, warmup 4 % then constant (cosine
   only in the full run), three trainers on one B200 node with 2 ranks × accum 4
   each (131k tokens/step kept), ~1,500 optimiser steps each; the verifier serves
   each row three times (~120k row-passes ≈ 9 h). Read val accept_len at steps
   500/1000/1500 and take the highest LR that still improves without instability.
   This is also the end-to-end check of layout A (B300 verifier, Mooncake, B200
   trainers) before the long run. Group mode (§5.2) is not needed for this.
2. **The full run: 8 slots if group mode has passed its test, else 1.** Group mode
   (§5.2) is ~1 day of code on top of chenw's launcher and needs no GPU to write, so it
   is built while the B300s are still on regen; its acceptance test (§5.2.4) runs on
   the verifier right after the LR pilot. Then:
   - *test passed:* one B200 node, 8 single-GPU trainers (1 rank × accum 8 each,
     131k tokens/step), the candidate (§2a, pilot LR) as **leader** and 7 contrasts
     as followers — Kimi shape scratch, Kimi warm start, block 12, the pilot's
     second-best LR, `ce 0.1 + tv 0.9`, γ 12, anchors 1024. One verifier pass feeds
     all 8; the contrasts cost nothing on the verifier. If the candidate ever looks
     wrong it restarts standalone from its own checkpoint.
   - *test failed or not ready:* the candidate alone, ravir's 8-rank launch
     unchanged; the contrasts follow on the second B300 verifier (layout B) when
     it is free.
   Evaluation budget: half-epoch checks are the in-trainer val accept_len for all
   8; the short set runs only at epoch ends for the top 3 (≈1 h each on a B300).
3. **Evaluate every half epoch** on the short set (`glm53-short-v033-b300.yaml`,
   greedy + sampled, README §9.w). Stop a run when it plateaus. Bar: beat the incoai
   release (short set greedy 5.03 / 3.66 / 4.27, sampled 4.44 / 2.73 / 3.98) and
   ravir's off-policy draft.

**Phase 2 — only after phase 1 produces a draft above the bar.**

4. Footprint measurement on sc3-c127 (§3): 8 vs 16 slots.
5. Shared-stream mode (§5.2), tested on the H200.
6. The §4 sweep around the working recipe; full-suite evals of the top slots.
7. Draft-OPD: at most one pilot on an early (non-converged) checkpoint, last.

## 7. Open questions

- Truncated rows (14 % at the 16k cap, 30–40 % of tokens): keep or drop is a
  per-stream decision (§4). README §9.x recommends keeping; one stream can't test both.
- Whether ravir's on-policy fine-tune improved over the off-policy draft — their
  eval is in flight; if it did not, warm-start vs scratch (slot 1 vs twin) becomes
  the first question, not block size.
- vLLM 0.33 on the verifier with the 0.26-era trainer env on the trainer node:
  they only share HTTP and Mooncake, so it should be fine; confirm in step 2.
