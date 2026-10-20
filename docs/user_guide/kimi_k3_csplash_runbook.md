# Kimi K3 + CSplash Serving Runbook

How to serve Kimi K3 with the CSplash speculative-decoding draft on vLLM,
and what's involved in porting CSplash to a vLLM fork that doesn't have it
yet. Written so another team can reproduce the deployment or the port
without needing the discovery history behind it (that history — the full
chronological investigation, bug hunts, and additional configs tried — is
in `docs/experiments/kimi_k3_2node_b200_launch.md`, "CSplash" sections, if
you want it).

Contents

1. What CSplash is
2. Porting CSplash to vLLM (if your fork doesn't have it)
3. Launching: single-node (B300) vs. two-node (B200)
4. Verifying a deployment
5. Validation results
6. Gotchas

---

## 1. What CSplash is

CSplash is a **natively-trained** (not retrofit) block-parallel
speculative-decoding drafter for Kimi K3, served at "7+1": one forward pass
proposes 7 draft tokens, the target verifies them in one pass.

- 5-layer GQA (64 query heads / 16 kv heads / head_dim 64), YaRN RoPE,
  per-head q/k norm.
- 4 sliding-attention layers (4096-token window) + 1 full-attention layer.
  Always bidirectional *within* the block regardless of layer type — only
  context visibility (not in-block positions) is windowed.
- 5-tap target hidden-state capture (K3 decoder layers 7/23/51/67/83),
  projected via `fc` + `hidden_norm` into one context vector per verified
  token.
- Does not own an embedding or LM head — borrows K3's (untied).
- Two components beyond a standard DFlash-family drafter:
  - **Grouped causal convolution**: an input-dependent short causal
    convolution (2 taps, 448 groups of 16 channels) wrapping both the
    attention and the MLP sublayer in every decoder layer.
  - **CandidateSelector**: a low-rank (rank=256) predecessor/successor
    codebook that rescores the top-16 base-logit candidates of each draft
    slot, conditioned on the *chosen* token of the previous slot (not its
    argmax) — walked sequentially, left to right, across the block. This
    replaces the Markov head DSpark-family drafters use for the same job.

Offline reference acceptance (vendor-measured, 7+1 layout, greedy, prefix
caching off): **22-set mean AL = 4.181, all-25-set mean AL = 4.068** —
ahead of a RadixArk DSpark draft on the same target (3.635). Package:
sha256-verified checkpoint + plain-PyTorch reference implementation +
golden parity vectors, under `CSplashKimi-Kimi-K3-<date>/`.

## 2. Porting CSplash to vLLM (if your fork doesn't have it)

CSplash's block layout — `[anchor, MASK × 7]`, bidirectional attention
within the block, context K/V pre-inserted into the KV cache before the
block forward — is architecturally identical to vLLM's existing **DFlash**
drafter family's non-causal "1+N fill-in" layout
(`vllm/model_executor/models/qwen3_dflash.py`). The port **subclasses**
DFlash's machinery rather than reimplementing it: `DFlashQwen3Attention`
(GQA, RoPE, per-layer sliding window, paged KV cache), `DFlashQwen3Model`
(fused context-KV precompute, target embed/lm_head sharing), and
`DFlashQwen3ForCausalLM` (weight loading, `AutoWeightsLoader`-based, all
reused unchanged). New code is limited to the two components above and the
sequential candidate-selector sampling loop.

**Files to add** (paths relative to your vLLM source tree):

| file | purpose |
|---|---|
| `vllm/model_executor/models/csplash_kimi.py` | `CSplashKimiGroupedConv`, `CandidateSelector`, `CSplashKimiDecoderLayer(DFlashQwen3DecoderLayer)`, `CSplashKimiModel(DFlashQwen3Model)`, `CSplashKimiForCausalLM(DFlashQwen3ForCausalLM)` |
| `vllm/v1/worker/gpu/spec_decode/csplash/speculator.py` | `CSplashSpeculator(DFlashSpeculator)` — sequential candidate-selector walk, adapted from `DSparkSpeculator._sample_sequential_topk`'s loop structure (rescore → sparse scatter → dense sampler), swapping its Markov embed/bias mechanism for `CandidateSelector.apply_step_logits` |
| `vllm/v1/worker/gpu/spec_decode/csplash/__init__.py` | empty |
| `vllm/transformers_utils/configs/csplash_kimi.py` | `CSplashKimiConfig(Qwen3Config)` — HF config registration (§2.3) |

Plus two small edits to existing files:

- `vllm/model_executor/models/registry.py`: register
  `"DFlashCSplashKimiDraftModel"` → `("csplash_kimi", "CSplashKimiForCausalLM")`
  (and `"CSplashKimiDraftModel"` as a defensive alias — see §2.2 for why
  the rewritten name is the one that actually matters).
- `vllm/v1/worker/gpu/spec_decode/__init__.py`: inside the
  `method == "dflash"` branch, dispatch to `CSplashSpeculator` when
  `"CSplashKimiDraftModel"` or `"DFlashCSplashKimiDraftModel"` is in
  `speculative_config.draft_model_config.architectures` — same pattern
  already used there for `DFlash2DraftModel`/`LiLiCorrDraftModel`.
- `vllm/transformers_utils/config.py`: add `csplash_kimi="CSplashKimiConfig"`
  to `_CONFIG_REGISTRY`, and the class to
  `vllm/transformers_utils/configs/__init__.py`'s `_CLASS_TO_MODULE`/`__all__`.

### 2.1 Key design decisions worth knowing before you read the code

- **Attention causality is hardcoded, not inherited.** DFlash resolves
  per-layer `causal` from a checkpoint-family heuristic
  (`_dflash_layer_causal`, keyed on `layer_types`) meant for *other*
  DFlash checkpoints with different per-layer causality conventions. That
  heuristic gives the wrong answer for CSplash (which is always
  bidirectional in-block, confirmed by the reference implementation's own
  mask construction). `CSplashKimiDecoderLayer.__init__` overrides
  `self.self_attn.causal = False` after calling
  `super().__init__(...)`, rather than fighting the heuristic.
- **`num_reqs` for the block-reshape is derived dynamically**
  (`hidden_states.shape[0] // self.block_size`), not threaded through
  every conv module's constructor — keeps `CSplashKimiDecoderLayer`'s
  `__init__` signature identical to `DFlashQwen3DecoderLayer`'s, so it
  can be used as `decoder_layer_cls` with zero changes to
  `DFlashQwen3Model.__init__`'s existing layer-construction loop.
- **The sequential selector walk reuses DSpark's buffer-reuse trick**:
  compute dense `base_logits` once, `topk` for candidates, then
  `fill_(-inf)` that *same* tensor and `scatter_` each step's corrected
  candidate scores into it in place — avoids allocating a fresh
  `[num_reqs, vocab]` tensor per sequential step (7 per block), and keeps
  the output shape exactly what the standard dense sampler/rejection path
  expects.
- **CSplash's block layout matches DFlash's own default**
  (`sample_from_anchor=False`, `num_query_per_req = 1 + num_speculative_steps`),
  **not** DSpark's (`sample_from_anchor=True`,
  `num_query_per_req = num_speculative_steps`) — only the *sampling*
  step is DSpark-shaped (sequential), not the block layout. Don't subclass
  `DSparkSpeculator`; subclass `DFlashSpeculator` and borrow DSpark's
  sampling *pattern*.

### 2.2 Integration pitfalls (none are numerics bugs — all are "this
checkpoint doesn't match a convention vLLM's shared infra assumes")

1. **Unregistered HF `model_type`.** CSplash's `config.json` declares
   `"model_type": "csplash_kimi"`, which transformers' `AutoConfig` won't
   recognize on its own. Fix: register `CSplashKimiConfig(Qwen3Config)`
   (§2.3) into vLLM's own `_CONFIG_REGISTRY` — the same mechanism
   DSpark's `K3DSparkConfig` uses.
2. **Architecture name gets rewritten under you.** For
   `method: "dflash"`, `EAGLEConfig` rewrites
   `draft_model_config.architectures` to `DFlash{arch}` unless the name
   already starts/ends with `"DFlash"`
   (`transformers_utils/configs/eagle.py`). `CSplashKimiDraftModel`
   doesn't, so at runtime the registry is actually asked for
   `DFlashCSplashKimiDraftModel` — register under *that* name (same
   pattern the existing `DFlashMuseGlimmerAssistantModel` alias uses).
   The same rewritten name is what you need to check for in the
   speculator-dispatch `architectures` lookup.
3. **`mask_token_id`/`target_layer_ids` location.** Shared DFlash-family
   infra (`get_parallel_drafting_token_id`,
   `get_eagle3_aux_layers_from_config`) looks for these under a
   top-level `dflash_config` dict, not wherever your checkpoint happens
   to nest its own config (CSplash's is `csplash_config`). Mirror the
   needed keys into a `dflash_config` dict in your config class's
   `__post_init__`.

### 2.3 Minimal `CSplashKimiConfig`

```python
from transformers import Qwen3Config
from huggingface_hub.dataclasses import strict

@strict
class CSplashKimiConfig(Qwen3Config):
    model_type = "csplash_kimi"
    csplash_config: dict | None = None
    rope_theta: float | None = None       # checkpoint ships a redundant top-level copy
    dflash_config: dict | None = None     # see pitfall 3 above

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.dflash_config is None and self.csplash_config:
            self.dflash_config = {
                "mask_token_id": self.csplash_config.get("mask_token_id"),
                "target_layer_ids": self.csplash_config.get("target_layer_ids"),
            }
```
(`transformers` 5.x's `@strict` dataclass configs reject undeclared
fields — if your transformers version predates `@strict`, a plain
subclass with a normal `__init__(self, **kwargs)` forwarding to
`super().__init__(**kwargs)` is simpler and doesn't need explicit field
declarations.)

## 3. Launching

Two deployment shapes, picked by how much memory your GPUs have per
shard. K3's weights need **~194 GiB/GPU at TP=8**.

### 3.1 Single node (TP=8, no PP, no DCP) — e.g. 8×B300 (275 GiB/GPU)

```bash
scripts/evaluate/mtp_server_eval/kimi_k3/run_k3_csplash_vllm.sh
```
Env-var overrides: `MODEL` (target checkpoint path), `DRAFT_DIR` (CSplash
checkpoint, defaults to the known package path), `PORT`, `TP`, `NUM_SP`
(speculative depth, default 7), `CTX` (`max_model_len`, default 262144 —
untuned for hardware with more headroom than B300's baseline, raise and
watch for KV-cache OOM at startup), `GPU_MEM`, `VENV_DIR`. `DRY_RUN=1`
prints the resolved `vllm serve` command without launching. See the
script's own header comment for the full list and examples.

### 3.2 Two nodes (TP=8 × PP=2, cross-node) — e.g. 2×8×B200 (~183-194 GiB/GPU)

No margin for weights alone on one B200 node, so split pipeline-parallel
across two. Key flags beyond the single-node case:
`--pipeline-parallel-size 2 --nnodes 2 --node-rank {0,1} --master-addr
<rank0-fabric-ip> --decode-context-parallel-size 1` (DCP=1 — some drafters
are incompatible with DCP>1 in this vLLM build; untested either way for
CSplash, kept at the safe default), plus `--headless` on the non-API-server
rank. `--language-model-only` skips a vision-tower code path unrelated to
topology. Cross-node NCCL/Gloo uses `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` on the
inter-node fabric and `NCCL_P2P_LEVEL=NVL`. The original sc3-cp81/sc3-cp82
server ran with `NCCL_CUMEM_ENABLE=1` and NVLS left on; that allocation
ended on 2026-10-16. That NVLS setup stalls on other B200 pairs
(`ncclNvlsSetup` → `initNvlsChannel` on local GPU 1). Later pairs serve
with the same script after adding the exports below.

```bash
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export NCCL_MAX_NCHANNELS=4
```

Pairs that finished a real serve, the `brcm0` address rule, and the full
`salloc`/`srun` sequence are in
`docs/experiments/kimi_k3_2node_b200_launch.md` ("Pairs that served Kimi
K3"). Proven pairs: sc3-cp81+sc3-cp82 (script-default NCCL),
sc3-cp107+sc3-cp108, sc3-cp87+sc3-cp97, sc3-cp103+sc3-cp105, and
sc3-cp100+sc3-cp102. Skip sc3-cp78,
sc3-cp79, sc3-cp80, sc3-cp83 (GPU 2 unavailable), sc3-cp84 (NCCL hang),
sc3-cp88, sc3-cp98+sc3-cp104 (graph capture timed out), and sc3-cp106
(slurmd).

## 4. Verifying a deployment

`/health` returns 200 as soon as the HTTP server is up — well before the
model has actually finished loading/compiling. Confirm with a real
completion:
```bash
curl -s http://<host>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"moonshotai/Kimi-K3","messages":[{"role":"user","content":"What is 17*24? Answer with just the number."}],"max_tokens":50,"temperature":0}'
```
Then check `SpecDecoding metrics` in the server log for a sane
acceptance length — a working deployment shows mean acceptance length in
the 5-8 range (out of a max of 8, the "7+1" block) on simple prompts,
with per-position acceptance decaying *smoothly* toward later slots
rather than collapsing abruptly to zero. A near-zero, bimodal, or
abruptly-collapsing pattern signals a real bug (this project separately
hit — and has not yet root-caused — exactly this symptom with a
*different*, retrofit sliding-window drafter; CSplash's own native
sliding-window attention has not shown it).

## 5. Validation results

Measured on the 2-node B200 deployment (§3.2 shape), vLLM
`0.30.1rc1.dev493+gac9126e58`.

**Live acceptance length vs. the vendor's offline reference**, using this
repo's own real-text 25-benchmark harness
(`scripts/evaluate/mtp_server_eval/run_eval.sh` — sequential, one request
at a time, `temperature=0`, `max_tokens=4096`, same methodology as the
eval runbook in this directory): **all-25 mean AL 4.087 vs. reference
4.068** (Δ +0.019), **22-distinct-set mean 4.202 vs. reference 4.181**
(Δ +0.021). Every individual dataset within ~±0.18 of its reference
value, deltas bidirectional (no systematic drift). One known methodology
difference: this run had prefix caching on; the vendor's reference had it
off.

**Live-serving comparison against a RadixArk DSpark draft** (same target,
same vLLM build, AgentX agentic-trace replay — see
`docs/experiments/kimi_k3_2node_b200_launch.md` for what AgentX actually
measures), 1800s/cell, `users=1`, `max_context=256000`:

| draft | attention | decode_tok_s | accept_len |
|---|---|---:|---:|
| CSplash (block=7) | sliding 4096 (4L) + full (1L) | 222.6 | 2.977 |
| RadixArk DSpark (block=3, best RadixArk config) | full | 179.2 | 2.349 |
| RadixArk DSpark sliding2048 (block=7, known-broken) | sliding 2048 | 82.2 | 1.295 |

CSplash beats every RadixArk configuration tested, and its own
sliding-window attention does not reproduce the collapse the
sliding-window retrofit shows.

## 6. Gotchas

- **SLURM/scheduler wall-clock limits.** A default `--time=24:00:00`
  (or your scheduler's default) has silently killed a live server
  mid-benchmark twice in this project — no warning in the server log, it
  just stops. Symptom: the server log stops abruptly mid-request with no
  traceback, both nodes (if multi-node) at the identical timestamp.
  Confirm via `scontrol show job` → `JobState=TIMEOUT Reason=TimeLimit`.
  Use a generous limit (≥48h) for anything that might span multiple eval
  sessions.
- **`pgrep -f "<pattern>"` matching its own invocation.** If you script
  "is the server running" checks, `pgrep -fa "vllm serve"` run via
  `bash -c 'pgrep -fa "vllm serve" ...'` will match the `pgrep` command
  line itself (which contains the search string inside its own quoted
  argument). Pipe through `grep -v pgrep`, or anchor the pattern
  (`^vllm serve|bin/vllm serve`).
