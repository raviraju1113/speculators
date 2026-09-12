# Mamba2 EAGLE-3 Draft Head — Progress

Tracking doc for the [Mamba2 vs Qwen3 draft-head study](https://docs.google.com/document/d/1ZTKBJi51_wfUMIzLN_AKRiAy4_Aa-YUuc6a5BJ0Z32Q/edit)
for **Gemma 4 31B**. Branch `ravir/mamba2_draft`.

**The question:** replace the EAGLE-3 draft's transformer block with a Mamba2 (SSM)
block. The draft's KV cache grows linearly with context; a Mamba2 recurrent state is
constant at 5.6 MB. Does the draft stay within **10% of the transformer's acceptance
length**? If yes it wins on long-context serving capacity.

| Phase | Status |
|---|---|
| 0 — Kernel & cost validation | **Complete — gate passed** |
| 1 — Model implementation | **Complete** |
| 2 — Training | **Smoke run passed**; full run not yet launched |
| 3 — Evaluation | Not started |
| 4 — Ablations | Not started |

**Decisions taken:** all arms use the **262144 full-vocab head** (matching the shipped
Arm A) for consistency; Arm A is reused as-is rather than retrained; both B and B2 are
in scope; vLLM serving support deferred pending Phase 0 results (now available).

---

## Phase 0 — kernel & cost validation: GATE PASSED

### Kernels build and run on B200

Two blockers, both solved without modifying the working conda env.

**(a) The env's CUDA wheels are mutually inconsistent.** `nvidia-cuda-nvcc` 13.2.86 vs
`nvidia-cuda-runtime` headers 13.0.96 vs `nvidia-cuda-cccl` 13.3.3.4.1. cccl's
`cuda_toolkit.h` hard-errors when `__CUDACC_VER__ != CUDART_VERSION`, so *any* CUDA
extension build fails out of the box. Fixed with an isolated CUDA 13.0 toolkit built in
scratch from matched wheels (`nvcc/nvvm/cuda-crt==13.0.88`, `cccl==13.0.85`,
`runtime==13.0.96`) used as `CUDA_HOME` for builds only, plus an unversioned
`libcudart.so` alias the linker requires.

**(b) `causal-conv1d` and `mamba-ssm` hardcode 8 `-gencode` archs** and ignore
`TORCH_CUDA_ARCH_LIST`. Patched both sdists to emit only `compute_100` (~8x faster build).

Installed and verified: `causal_conv1d 1.7.0`, `mamba_ssm 2.3.2.post1`,
`is_fast_path_available == True`.

### Arm B is faster at every context and flat in context length

Single draft block + LM head, batch 1, single GPU, bf16, 32K draft vocab:

| ctx | A block | B block | ratio | c_A | c_B |
|---|---|---|---|---|---|
| 1K | 0.382 ms | 0.281 ms | 1.36x | 0.098 | 0.063 |
| 64K | 0.920 ms | 0.273 ms | 3.38x | 0.220 | 0.066 |
| 256K | 2.585 ms | 0.279 ms | **9.27x** | 0.597 | 0.065 |

Arm B is context-independent as predicted; Arm A grows 6.8x. **Arm B lands inside the
proposal's assumed `c_draft` band (0.04–0.07); Arm A does not** — beyond 64K it is 2–6x
worse than assumed, so the proposal *understates* Arm A's long-context penalty.
`c_draft` uses the recorded target-only baseline (~225 tok/s = 4.44 ms/step); note the
draft was timed single-GPU while that baseline is a TP>1 server run, so a
deployment-accurate figure needs the draft timed under the same TP.

### P3 — state rollback: exact

The proposal's highest-risk silent failure (§8.2). `transformers` 5.12.1 removed
`Mamba2Cache`; the replacement `LinearAttentionLayer.crop()` is an explicit **no-op**, so
rejection cannot be undone through the public API. Manual snapshot/restore of
`conv_states` + `recurrent_states` (via `copy_`, preserving cudagraph-safe addresses) is
**bit-exact** (0.000e+00 on output, SSM state and conv state; control without restore
differs by 1.289). `cost_replay(tau)` is linear at ~0.18 ms/token — at tau=3 that is
~12% of a target step, far below Arm A's 2.2 ms attention penalty at 256K.

### P1 — document packing: `seq_idx` is correct, with an alignment caveat

Decisive test — change document 0's content, measure document 1's output:

| doc0 length | chunk-aligned? | bf16 leak | fp32 leak |
|---|---|---|---|
| 256, 512, 768 | yes | **0.000e+00** | **0.000e+00** |
| 100, 300, 700 | no | 6.25e-02 | ~5–8e-03 |
| *(no `seq_idx` at all)* | — | **2.46e+01** | 2.44e+01 |

Chunk-aligned boundaries isolate **exactly**. The residual on unaligned boundaries is a
numerical artifact of the one chunk that straddles a boundary, not semantic leakage.
Mitigation: pad document boundaries to multiples of `chunk_size`.

### The proposal's Arm A spec does not match the shipped artifact

Checked against the checkpoint that produced the recorded baseline curve
(`/sms-scratch/ravira/checkpoints/gemma4_draft_model_900k_eagle3_kimi_mtp_stem_code_math/2`):

| | proposal §4 | shipped Arm A |
|---|---|---|
| heads / head_dim / KV heads | 42 / 128 / 8 | 32 / 256 / 16 |
| valid GQA? | **no** (42 % 8 != 0) | yes |
| KV bytes/token | 4,096 | **16,384** |
| KV @ 256K | 1.07 GB | **4.29 GB** |
| draft vocab | 32,768 | **262,144** |
| trainable | 736 M (2.40%) | **2,063 M (6.72%)** |

1. **The Mamba2 memory win is 4x larger than claimed** — 4.29 GB -> 5.6 MB is ~766x, not
   ~190x. This strengthens the business case.
2. **The draft vocab is the biggest single lever on draft-step cost**, not the mixer:
   262144 -> 32768 halves the draft step and is available to *both* arms. Deferred to a
   Phase 4 ablation to keep the headline comparison consistent.
3. Arm A's §4 config is not buildable as written and should be restated.

---

## Phase 1 — model implementation: COMPLETE

Both arms build, train-step, checkpoint and round-trip at full Gemma-4 scale.

### Parameter counts (262144 vocab) — B and B2 bracket A

| | block (incl. fused W_FC) | trainable | % of 30.7B target |
|---|---|---|---|
| A (shipped qwen3) | 567.02 M | 2,063.01 M | 6.72% |
| **B (mamba2)** | 312.93 M | **1,808.93 M** | 5.89% |
| **B2 (mamba2+mlp)** | 659.75 M | **2,155.75 M** | 7.02% |

B sits below A and B2 just above, so an acceptance difference is attributable to
architecture rather than capacity — which is the point of running B2.

> Blocks are larger than the proposal's §5 table (185.40 / 532.22 M) because EAGLE-3
> fuses `W_FC` into the block by widening its input projection. For Mamba2 that widening
> costs `5376 x 23720 = 127.5 M`, more than a standalone `W_FC` (57.8 M), because
> `in_proj` is wider than `hidden_size`. Both arms use Arm A's convention, so the
> comparison stays fair; only the absolute figures differ from the proposal.

### What was built

| file | change |
|---|---|
| `src/speculators/models/eagle3/mamba2.py` | **new** — `Eagle3Mamba2Mixer`, `Mamba2DecoderEagle3FirstLayer` (B), `Mamba2MLPDecoderEagle3FirstLayer` (B2), `build_mamba2_draft_config`, `seq_idx_from_document_ids` |
| `models/base_components.py` | registered `mamba2`; `ModelComponents` gained `rotary_emb_class: None` and `uses_attention` |
| `models/eagle3/model_definitions.py` | registered the eagle3 mamba2 arm |
| `models/eagle3/core.py` | rotary optional, norm-eps name-agnostic, mask machinery attention-only, `seq_idx` threaded into the layer call |
| `models/utils.py` | `resolve_norm_eps` (`rms_norm_eps` vs `layer_norm_epsilon`) |
| `scripts/train.py` | `mamba2` in `DRAFT_ARCH_CONFIGS`; `_create_mamba2_layer_config` branch |
| `train/config/schema.py` | `Mamba2Args` group (`--mamba-*`), `mamba2` draft arch, eagle3-only guard |
| `train/config/resolution.py` | associated the `mamba2` group with eagle3 |
| `pyproject.toml` | `[mamba]` extra |
| `tests/unit/models/test_eagle3_mamba2.py` | **new** — 16 tests |

### Design decisions

**Only the packed-training scan is overridden.** `transformers` hardcodes `seq_idx=None`
in every scan call (its own source reads `seq_idx=None,  # was seq_idx`), so document
isolation is impossible with the stock mixer — subclassing was mandatory, not stylistic.
The single-token decode path is left entirely to upstream so serving behaviour stays
identical to `transformers`.

**Both arms share `model_type: "mamba2"`;** B2 is selected by the presence of
`intermediate_size` on the config, so the arm is recoverable from the serialized config
alone and a checkpoint always reconstructs the arm it was trained as (verified by
round-trip).

**`seq_idx_from_document_ids` stays on-device** — an `.item()` there caused a dynamo
graph break inside the compiled Eagle3 forward.

### Verification

- **714 passed, 5 skipped** across `tests/unit/`; the 16 new Mamba2 tests all pass.
- `ruff`: 49 errors with these changes vs **50 on a clean tree** — no new lint, one
  removed. Every error in a touched file is pre-existing (`PLR0917` signatures and the
  dead `parse_args` `F821`s). (`tests/unit/scripts/test_response_regeneration.py`
  excluded: a pre-existing `NameError: DatasetConfig` at collection, confirmed to fail
  identically with these changes stashed.)
- 16 new tests, including the three plan-mandated equivalence tests **each with a
  control** proving the test is not vacuous: packed-vs-independent scans (and
  doc0-cannot-affect-doc1); prefill-then-decode == one whole scan (catches the
  `dt[:, 0, :]` multi-token trap and the `update_conv_state` off-by-one); bit-exact
  rollback after rejection.
- Full-scale build + `save_pretrained` + config round-trip for both arms.

### How to build an arm

```bash
# Arm B
python scripts/train.py --speculator-type eagle3 --draft-arch mamba2 --num-layers 1 \
    --verifier-name-or-path /sms-scratch/checkpoints/gemma-4-31B-it \
    --data-path <data> --save-path <out> --dry-run
# Arm B2: add --mamba-mlp
```
Shape knobs: `--mamba-expand --mamba-head-dim --mamba-state-size --mamba-n-groups
--mamba-conv-kernel --mamba-chunk-size --mamba-mlp`.

---

## `scripts/train.py` startup bug (pre-existing) — FIXED

`scripts/train.py` could not start **on any branch, for any architecture**:

```
line 588: local_rank, world_size, rank, is_distributed = maybe_setup_distributed()
TypeError: cannot unpack non-iterable NoneType object
```

**How it happened.** Upstream PR #656 (`e65222a`, 2026-06-29) refactored
`maybe_setup_distributed` to return `None` and set module-level topology state,
updating the call site to a bare `maybe_setup_distributed()`. Commit `d20d469` ("add
constant LR, metric_logger", 2026-07-29) then reintroduced the pre-refactor unpacking
line — a rebase/merge that resurrected the old form. It sat on `main` for 31 commits and
was present on all four local branches.

**Why models still got trained.** The 900k draft is dated 2026-08-12, after the
breakage, so that run did not use this checkout — most likely the Docker image's
baked-in copy of the code, which `TRAINING.md` §6 already lists as a known gotcha
("*Stale code: when iterating, make sure you're running the mounted checkout, not the
image's baked-in copy*"). The bug was real but masked by the container workflow.

**The fix** restores PR #656's intended call site — a bare `maybe_setup_distributed()`,
reading topology back through the getters. `local_rank`, `world_size` and `rank` were
never used after that line, and `is_distributed` was already being called as the
imported function, so nothing else changed. This also cleared the `F811` shadowing and
`F401` unused-import that the stale line caused.

Verified by `--dry-run` on **all three** draft architectures against the real Gemma-4
31B verifier — `mamba2`, `mamba2 --mamba-mlp`, `qwen3` and `llama` all build, load
verifier weights and write a checkpoint. So the fix unblocks existing work, not just
this project.

Remaining `ruff` `F821`s in the same file's unreachable `parse_args`
(`DEFAULT_REQUEST_TIMEOUT`, `DEFAULT_MAX_RETRIES`, `explicitly_provided_dests`,
`resolve_loss_fn`) are dead code from the same unfinished refactor — left alone as
out of scope, but worth a separate cleanup.

## Phase 2 — training: smoke run PASSED

The online path works end to end and **the Mamba2 draft learns**.

Setup: vLLM serving `gemma-4-31B-it` on one B200 with `extract_hidden_states` (aux
layers `(2, 30, 57, 60)` confirmed active in the engine log), streaming hidden states on
demand to a training process on a second GPU. 600-row subset of Arm A's own mix, 1 epoch,
177 optimizer steps, `total_seq_len=4096`, `ttt_steps=3`, `lr=1e-4`, full 262144 vocab.

Identical data and step count for both arms:

| arm | first-10% avg | last-10% avg | drop | val/loss_epoch |
|---|---|---|---|---|
| **B (mamba2)** | 28.55 | 13.33 | 53.3% | **14.449** |
| A (qwen3) | 35.09 | 14.36 | 59.1% | 15.383 |

Loss descends smoothly and all three TTT steps track together, so the `seq_idx` packing
and the fresh-state-per-TTT-step decision (P2) both behave under real training.

> **Do not read the val-loss ordering as a result.** 177 steps on 600 samples, one seed,
> and the arms differ in parameter count — this establishes *"Arm B trains correctly and
> comparably"*, which is what the plan asked of this step. It says nothing yet about
> acceptance length, which is the actual decision metric.

### Two things to fix before the full run

1. **Raise the server's `--max-model-len`.** 4 of 600 samples failed hidden-state
   generation: a prompt of exactly 8192 tokens plus 1 output token exceeds an 8192
   context. Use `--max-model-len 8448` (or cap `--total-seq-len` below it).
2. The verifier server is slow to start (~2 min, 58 GiB over NFS). Keep it warm across
   the B and B2 runs rather than relaunching.

### Reproducing

`scratchpad/smoke_train.sh` (`ARM=B|B2|A`, `TRAIN_GPU`, `PORT`). Server:

```bash
python scripts/launch_vllm.py /sms-scratch/checkpoints/gemma-4-31B-it \
    --hidden-states-backend file --hidden-states-path <hs> \
    -- --port 8000 --tensor-parallel-size 1 --max-model-len 8448 \
    --gpu-memory-utilization 0.85
```

---

## Hidden-states lock timeout (pre-existing) — FIXED

Full-scale online training failed on most samples with:

```
Failed to load/cache hidden states for sample N:
  Timed out waiting for lock: .../cmpl-....safetensors.lock
```

**Not a deadlock — the producer was still writing.** One hidden-states sample is
`num_tokens x num_layers x hidden_size x 2` bytes: ~350 MB at 8192 tokens with 4 tapped
layers on a 5376-wide verifier (observed files up to 167 MB). `hs_connectors.wait_for_lock`
polled with a **hardcoded 10 s** deadline, which is shorter than a legitimate write of
that size to NFS with several vLLM workers writing concurrently. vLLM's own reference
reader (`ExampleHiddenStatesConnector.load_hidden_states`) blocks indefinitely on
`LOCK_SH` for exactly this reason.

The timeout is now `DEFAULT_LOCK_TIMEOUT` — 300 s, overridable via
`SPECULATORS_HS_LOCK_TIMEOUT` — and exists only to avoid hanging forever on a crashed
producer, which is the semantics vLLM assumes.

**Second-order effect: a disk leak.** A sample that times out is never loaded, so
`--on-generate delete` never fires for it and the file is orphaned — 121 files / 6.2 GB
accumulated in about two minutes. Worth watching even with the longer timeout.

**Also: keep the hidden-states path on local disk.** vLLM and the trainer share a node,
so a shared filesystem buys nothing and NFS costs latency and lock semantics.
`/sms-scratch` is NFS4 and 92% full; `/home` is local xfs with ~553 GB free.

## Phase 2 — matched Arm B run vs Aug-12 baseline: Arm B clearly ahead

The first full Arm B run was confounded: current code defaults `norm_before_fc=True` /
`norm_output=True` for eagle3-type speculators and `--optimizer muon`, whereas the
Aug-12 baseline had all three off (the optimizer was recovered from
`optimizer_state_dict.pt` — a single AdamW param group over 18 tensors). Relaunched with
`--optimizer adamw --no-norm-before-fc --no-norm-output`. Every hyperparameter logged to
W&B now matches except `draft_arch`.

### Validation (held-out) — mamba2 at 77% of training vs eagle3 **final**

| metric | mamba2 (step 53.9k/70.3k) | eagle3 (final) |
|---|---|---|
| `val/loss_epoch` | **3.105** | 6.494 |
| `val/cond_acc_0` | **0.790** | 0.722 |
| `val/cond_acc_1` | **0.771** | 0.684 |
| `val/cond_acc_2` | **0.768** | 0.699 |
| `val/full_acc_2` (3 correct) | **0.468** | 0.345 |
| tau proxy `1 + sum_k full_acc_k` | **2.866** | 2.561 |

Arm B beats the baseline's *finished* run by ~12% on the acceptance proxy while still
mid-epoch-3 and still improving; the baseline's training curve flattened after ~30k
(loss 6.39 -> 6.38 -> 6.41) while Arm B's is still descending (3.19 -> 3.07 -> 2.84).

### The early `cond_acc_2` erosion did not hold

At step 1.7k the deepest-TTT gap looked like it was collapsing (+0.190 -> +0.036), which
suggested the P2 asymmetry (no cross-step diagonal connection for a recurrent mixer) was
costing the chain. It reversed: the gap bottomed at +0.029 around step 11k and has grown
steadily since, to **+0.078**. That early narrowing was a transient, not a trend, and the
chain advantage is intact.

| step window | 0–2k | 10–12k | 20–22k | 30–32k | 40–42k | 52–54k |
|---|---|---|---|---|---|---|
| `cond_acc_2` delta | +0.071 | +0.029 | +0.053 | +0.062 | +0.067 | **+0.078** |

### What this does and does not establish

It establishes that the Mamba2 mixer is a *better distillation student* at 8192 training
length, from a smaller block (313 M vs 567 M) and fewer trainable params (1.81 B vs
2.06 B). It does **not** yet establish the decision criterion. Two gaps remain:

1. **Measured acceptance, not proxy.** The baseline's val proxy (2.561) overstates its
   measured eval accept length (2.315 at 1k) by ~10%, so the proxy is optimistic in
   absolute terms even if the ordering holds.
2. **Long context is entirely untested.** Training ran at 8192; the business case lives
   at 64k–256k. That is where a constant-size recurrent state should win on memory but
   could also lose on fidelity, since a fixed state must compress more as context grows.
   Nothing measured so far speaks to it.

## Phase 2 — FINAL RESULT: Arm B wins decisively on the matched run

Both runs finished at step 70,280 on identical data with matched scaffolding
(`--optimizer adamw --no-norm-before-fc --no-norm-output`; every W&B-logged
hyperparameter matches except `draft_arch`).

| validation metric | **mamba2 (Arm B)** | eagle3 (Arm A) |
|---|---|---|
| `loss_epoch` | **2.950** | 6.494 |
| `cond_acc_0` | **0.796** | 0.722 |
| `cond_acc_1` | **0.779** | 0.684 |
| `cond_acc_2` | **0.778** | 0.699 |
| `full_acc_2` (3 correct) | **0.482** | 0.345 |
| tau proxy `1 + sum_k full_acc_k` | **2.899** | 2.561 |

+13.2% on the acceptance proxy, from a smaller block (313 M vs 567 M) and fewer
trainable params (1.81 B vs 2.06 B). Checkpoint weights match the proposal's arithmetic
exactly (`in_proj` 23720x10752, `conv_dim` 12800, 168 heads).

Caveat unchanged: this is a distillation proxy at 8192 training length, not measured
acceptance, and says nothing yet about long context.

---

## Phase 3 — vLLM serving: WORKING end to end

A Mamba2 Eagle-3 draft now serves on vLLM at TP=4 against Gemma-4-31B, producing
coherent deterministic output and a healthy acceptance profile.

### Measured (k=3, TP=4), eager vs CUDA graphs vs target-only

Six-prompt mixed set, per-request counter deltas (not bulk totals):

| prompt | eager | CUDA graphs |
|---|---|---|
| fibonacci | 2.817 | 2.806 |
| sky-blue | 2.040 | 2.040 |
| 17*23 | 2.871 | 2.792 |
| paperclip | 2.261 | 2.261 |
| translate FR/ES | 1.900 | 1.900 |
| sql duplicates | 2.519 | 2.519 |
| **aggregate** | **2.530** (rate 0.510) | **2.512** (rate 0.504) |

**There is no eager-vs-CUDA-graph discrepancy.** The earlier "2.484 eager vs 2.857
graphs" came from my own bad arithmetic on a *bulk* metrics read (I divided totals by
the wrong request count; the residual 60 drafts / 18 accepted in that reading is
unreconciled and the figure should simply be discarded). Measured per request the two
paths agree to within 0.02 accept_len, and agree exactly on four of six prompts. No
path updates the recurrent state incorrectly.

Acceptance is strongly prompt-dependent (1.90 on translation, 2.87 on arithmetic),
matching the published eagle3 pattern where `speed-multilingual` was its weakest bench.
A single-prompt figure is not a headline; **2.51-2.53 aggregate** on this mixed set is
the honest number. Still not comparable to the eagle3 2.315, which was measured on
AA-LCR 1k prompts -- that needs the eval harness.

### Losslessness: confirmed, with tie-breaking as the only source of difference

Compared speculative output against target-only (no draft, otherwise identical config),
token-for-token:

* **5 of 6 prompts byte-identical**, including three that ran to the full 200 tokens.
* The one mismatch (fibonacci, graphs path) first diverges at **token 7**, where the
  target's own top-2 logprobs are **exactly tied**:

  ```
  position 7  ' different'  logprob=-0.752081
              ' several'     logprob=-0.752081
              gap(top1-top2) = 0.000000 nats
  ```

  So the target is genuinely indifferent between the two continuations; which one wins
  is decided by floating-point reduction order in the argmax, not by the draft.
* Neither path is systematically favoured -- on this prompt eager matched the target and
  graphs did not; on `17*23` graphs matched and eager did not. A real state bug would
  show a consistent loser.
* Within one server instance the output is bit-stable: four identical requests gave
  identical SHA-256, draft counts and acceptance.

Losslessness therefore holds in the meaningful sense. Worth noting for the eval harness:
exact-output comparison across server configurations will show occasional differences
purely from tied-logit tie-breaks, so an exactness check should compare against the
*same* config or treat tied positions as equivalent.

### Four vLLM gaps found; one needed a real code change

Fixed from outside vLLM (plugin, `speculators/integrations/vllm/`):

* Model class + `ModelRegistry` entry for `Eagle3Mamba2ForCausalLM`.
* Eagle3 `model_type -> architecture` map extended for `mamba2`.
* `init_model_state` override, so a pure-attention target still gets the mamba-aware
  `ModelState` when the *draft* owns the Mamba layers (`is_hybrid` describes the target
  and is a read-only derived property).

Needed a vLLM change -- `vllm_patches/0001-eagle3-mamba2-draft-metadata.patch`
(3 files, ~75 added lines):

* `_build_draft_attn_metadata` builds its own `CommonAttentionMetadata` and passed
  neither `model_specific_attn_metadata` (the only route supplying `is_prefilling`) nor
  `seq_lens_cpu_upper_bound`; the Mamba builder asserts on both. Inside the multi-step
  draft loop every request advances by exactly one token, so `is_prefilling` is
  all-False there -- which is what makes the fix small and safe. `seq_lens_cpu` is only
  consulted as `seq_lens_cpu > 1` for *prefilling* rows, so the target's upper bound is
  a sound source.
* Mamba's full-CUDAGraph capture asserted one decode shape,
  `max_query_len == 1 + num_spec_tokens` (the target's verify pass). A *draft* capturing
  its own loop has uniform `query_len == 1`, equally decode-only. Relaxed to accept
  either; validated by the output being byte-identical to the eager run.

### Required serving flags, and the upstream issue each represents

```
--speculative-config '{"method": "eagle3", ...}'   # method is NOT auto-detected
--mamba-block-size 8192                            # must equal max_model_len
--no-enable-prefix-caching
--gpu-memory-utilization 0.65
--max-num-batched-tokens 4096                      # >= 2496 for the Gemma-4 VLM
```

CUDA graphs need no flag -- they work with the patch applied.

1. **`method` is inferred from the checkpoint path string** --
   `elif "eagle3" in draft_model_config.model.lower()`. Our directory is
   `..._e3_mamba2_...`, so it silently fell back to `method='draft_model'` and called
   the draft as a plain LM. **The Aug-12 eagle3 draft only worked by accident of its
   directory name.** This should read `speculators_model_type`.
2. **`mamba_block_size` is only derived for hybrid targets**, yet
   `MambaBase.get_kv_cache_spec` asserts it is set.
3. **Prefix caching is incompatible** as configured: `validate_mamba_block_size` forces
   `mamba_block_size == max_model_len` without it, and with it the block-hash
   divisibility assert fires. A draft-owned state need not participate in target prefix
   caching at all.
4. **Draft chunked-scan intermediates are not in vLLM's memory profile**: at 0.9
   utilization, warmup OOM'd wanting 32 GiB inside `_bmm_chunk_fwd`.

### A self-correction worth recording

The first generation looked corrupted (`"the capital of France of France of France..."`).
It was not: a target-only control on the same prompt produced the byte-identical string
-- it is the instruct model on a raw, non-templated completion prompt. Speculative output
matched target output exactly, which is the losslessness property. The initial
"positions 1 and 2 accept 0%" reading came from that same pathological generation; with
a chat-templated prompt the profile is a normal decay.

### Still open

Prefix caching (passed on deliberately -- see above) and the full 25-benchmark run for
throughput. The eager/graph question and the losslessness check are both resolved.

## RESULT — full 25-benchmark suite: Arm B matches and slightly beats Arm A

Both arms on the same server, same run, same flags. 1,209 requests each, 50 samples per
benchmark (30 for AIME), `max_tokens` 4096, greedy, TP=4, `max_model_len` 8192.

| config | accept_len | decode tok/s |
|---|---|---|
| eagle3 qwen (Arm A) k=5 | 2.835 | 311.9 |
| **mamba2 (Arm B) k=5** | **2.937** | 308.5 |
| mamba2 (Arm B) k=3 | 2.637 | 300.4 |

*token-weighted over 25 benchmarks, ~565k completion tokens each*

**Arm B vs Arm A: accept_len 1.036x, decode tok/s 0.989x.**

**Decision criterion `tau_B >= 0.90 * tau_A`: 2.937 >= 2.551 — PASS**, and not marginally.
Arm B *exceeds* Arm A on acceptance while running a smaller mixer block (313 M vs
567 M) and fewer trainable params (1.81 B vs 2.06 B). Arm B leads on **17 of 25**
benchmarks.

### The split is systematic, not noise

Arm B wins on prose-shaped work and loses slightly on math/code:

| Arm B stronger | Arm B weaker |
|---|---|
| speed-qa / qa 1.12x | bfcl 0.96x |
| mt-bench, writing, question 1.10x | speed-multilingual 0.97x |
| summarization 1.09x | aime, aime26, math500 0.98x |
| tool_call, translation 1.08x | math_reasoning, humaneval 0.99x |

Throughput tracks acceptance closely (B is faster exactly where it accepts more), so the
two drafts' per-step costs are near enough to equal that acceptance is what moves
wall-clock. The ~1% aggregate tok/s deficit comes from the math/code benches where Arm A
accepts more, not from the SSM being slower per step.

### What this does and does not settle

Settles the project's core question: **a recurrent/linear-attention drafter works.** It
is competitive with a matched transformer draft on a real 25-benchmark suite at equal
throughput — not a degraded fallback accepted for its memory profile.

Does **not** settle the business case. The proposal's argument was long context: a
constant 5.6 MB recurrent state against a KV cache reaching 4.29 GB per sequence at
256K. This suite ran at `max_model_len` 8192, where that advantage is worth almost
nothing. The long-context sweep (`aa-lcr` 1k->128k bins, already built) is the test that
values the memory story, and it has not been run for Arm B.

Absolute tok/s here is not comparable to the published Gemma-4 table: those were A100s
with prefix caching on at 0.9 utilization; this is B200s at 0.65 with prefix caching off
(required by the draft — see `vllm_patches/README.md`). Both arms share those flags, so
the head-to-head is clean; only the absolute numbers shift.

Raw data: `scripts/evaluate/experiments/results/gemma4-31b-full-mamba2/`.

## LONG-CONTEXT RESULT — the hypothesis holds, decisively

AA-LCR 1k->128k, 100 samples/bin, TP=8, `max_model_len` 131072, baseline as denominator.
Full table in
[gemma4_31b_mamba2_draft_results.md](docs/user_guide/tutorials/gemma4_31b_mamba2_draft_results.md).

| bin | A accept_len | **B accept_len** | B/A | A speedup | **B speedup** |
|---|---:|---:|---:|---:|---:|
| 1k | 2.464 | **2.550** | 1.03x | 1.34x | 1.33x |
| 8k | 2.435 | **2.564** | 1.05x | 1.14x | **1.20x** |
| 16k | 2.294 | **2.542** | 1.11x | 0.86x | 0.97x |
| 32k | 2.196 | **2.589** | 1.18x | 0.66x | 0.81x |
| 64k | 1.949 | **2.609** | 1.34x | 0.41x | 0.58x |
| 128k | 1.755 | **2.646** | **1.51x** | 0.30x | 0.47x |

**Acceptance: transformer draft decays -28.8% from 1k to 128k; the Mamba2 draft is flat
(+3.8%).** The B/A ratio grows monotonically to **1.51x at 128k**. The going-in concern
was the opposite -- that a fixed-size state would compress harder and lose fidelity with
length. It does not. What degrades is the *transformer* draft.

**Throughput: both arms fall below no-draft past ~16k, and this is pre-existing.** Arm B
is uniformly better (0.47x vs 0.30x at 128k, ~1.6x more throughput) but still under 1.0x.
The independently recorded 2026-08-26 eagle3 run reproduces here almost exactly
(baseline 64k/128k 78.5/62.7 vs our 79.2/63.6; eagle3 accept_len 1.911/1.741 vs our
1.949/1.755), which both validates the harness and shows the collapse predates this
work. Arm B breaks even around 16k (0.97x) where Arm A is already 0.86x.

**What is and is not settled.** Draft quality: settled, a recurrent drafter is strictly
better at long context and the margin widens with length. Serving: not settled -- neither
draft pays for itself past ~8k here. That is a throughput-engineering problem (flags
deliberately conservative, draft path unoptimized), not a draft-quality one.

## Environment changes made

- `causal_conv1d 1.7.0` and `mamba_ssm 2.3.2.post1` built for sm_100 and installed into
  the `speculators` conda env (`--no-deps`, so the pinned set is untouched).
- `hs_connectors` installed editable (`pip install --no-deps -e ./hs_connectors`); it was
  missing, which broke `import speculators.train.data`.
- Neither `requirements-lock.txt` nor `environment.yml` has been regenerated yet — do
  that before treating this env as reproducible elsewhere.

**Disk is tight.** `/` was 99% full and had to be cleared of dry-run checkpoints (each
full-scale draft checkpoint is ~6–7 GB); `/sms-scratch` is at 99% with ~137 GB free.
A full Phase 2 run writing per-epoch checkpoints for two arms needs headroom.

---

## Next

1. **Phase 2 full run** — Arm B and B2 on the full 782,413-row
   `kimi-mtp-nemotron-stem-code-math` mix (Arm A's own mix), holding tap ids
   `[2, 30, 57]`, 262144 vocab, `ttt_steps`, `kl_div` loss, LR schedule and seed
   constant. No cached hidden states exist (`/sms-scratch/ravira/hidden_states` is
   empty), so this runs online — which also avoids an offline dump the disk cannot hold.
2. **Phase 3** — add `gemma4-31b-mamba2-ctxlen-sweep.yaml` beside the existing sweeps and
   compare against the recorded Arm A curve (1k 2.315 -> 32k 2.137 accept length).
   **Blocker to decide:** vLLM has no EAGLE-3 Mamba2 draft runner (`train.py` warns on
   this at config time), so end-to-end tok/s needs either a vLLM implementation built on
   `mamba_mixer2.py` or an explicit statement that it was not measured on a serving
   stack. Phase 0 showed the speed upside is real (9.3x block-level at 256K), which
   argues for building it.
3. **Phase 4** — B vs B2; 32K vs 262144 draft vocab (the largest single lever on draft
   step cost, and it applies to both arms); `d_state`; `expand`; tap-layer choice.

## Reproducing Phase 0

Scripts live in the session scratchpad (not committed): `build_env.sh` (isolated CUDA
toolkit), `phase0_bench.py` (A/B/B2 latency + replay), `phase0_rollback_exact.py`,
`phase0_seqidx_equiv.py`, `seqidx_causality_probe.py`, `seqidx_alignment_probe.py`.
The permanent equivalents of the correctness checks are in
`tests/unit/models/test_eagle3_mamba2.py`.
