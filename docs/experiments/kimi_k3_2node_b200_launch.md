# Launching Kimi K3 + DSpark across 2x B200 nodes (SLURM `gpuonly`)

## RESOLVED (2026-10-12): working 2-node launch found

After 13 independent attempts (documented below) progressively diagnosed and
fixed three real bugs — (1) PyTorch/NCCL auto-attempting cross-node
"symmetric memory" setup, fixed via `NCCL_CUMEM_ENABLE=0`/`NCCL_NVLS_ENABLE=0`/
`ENABLE_INTRA_NODE_COMM=0`; (2) Kimi K3's own latent-MoE-tail kernel requiring
genuine GPU peer memory across the *entire* TP group, meaning **TP can never
span nodes** for this model; (3) a reproducible NCCL P2P/IPC stall between a
specific local GPU pair — a colleague (ruobingw)'s independently-developed,
already-validated recipe arrived with the actual fixes for issues we were
still chasing: `NCCL_P2P_LEVEL=NVL` (not full P2P disable) for the stall, and
`--language-model-only` for a vision-tower crash we'd just hit. Their recipe
also uses a parallelism dimension we hadn't configured at all — **DCP
(decode-context-parallel-size), sized to match TP** — for sharding KDA's
recurrent linear-attention state, on top of TP=8 (intra-node) × PP=2
(cross-node).

**Working launch (2026-10-12, job `10368001`=cp82/head,
`10368011`=cp81/second node)**: vLLM `0.30.1rc1.dev493+gac9126e58` (wheels.vllm.ai,
NOT our conda env's vLLM 0.24.0 fork — this build has native Kimi K3 support
plus flags like `--decode-context-parallel-size`, `--language-model-only`,
`TOKENSPEED_MLA` attention backend that our older install lacks), torch
2.13.0+cu130, NCCL 2.29.7 (2.30.7 is known to hang on the second
communicator). `TP=8 × PP=2 × DCP=8`, `--language-model-only`,
`NCCL_P2P_LEVEL=NVL`, `NCCL_CUMEM_ENABLE=1` (fine once TP is correctly
intra-node), raw checkpoint directly off NFS (no local copy needed — same
source as our existing `Kimi-K3-patched`, used unpatched since this vLLM
build has its own native modeling code), `--load-format fastsafetensors`,
no speculative decoding yet (`NO_SPEC=1`), no Mooncake KV connector (not
needed for basic serving). Confirmed via a real `/v1/chat/completions`
request. Full scripts: `.scratch_kimi_launch/k3-b200-exp/scripts/`
(adapted from `/import/snvm-sc-podscratch4/ruobingw/k3-b200-exp/scripts/`
for our own paths/job IDs — ruobingw's originals are hardcoded to their
own home/scratch directories).

**Not yet done**: the Mooncake KV-cache tier (needed only for large
prefix-cache-sharing workloads, per ruobingw's `RECIPE.md`) — not required
for basic serving. DSpark speculative decoding **is** now working, see
below.

## Runbook: launching the deployment (current, as of 2026-10-13)

Copy-pasteable steps for standing up the 2-node server from a cold start.
Everything below targets `sc3-cp81`/`sc3-cp82` by name — the exact
commands won't work on other node pairs without the IP-resolution change
noted in step 3 (the launch script currently hardcodes these two nodes'
fabric IPs).

**1. Allocate the two nodes** (use a generous `--time` — a 10-hour
default silently killed a server mid-benchmark once, see the AgentX
section below):
```bash
salloc -p gpuonly -w sc3-cp81,sc3-cp82 -N2 --ntasks-per-node=1 \
  --gres=gpu:8 --time=24:00:00 --no-shell
# note the JOBID it reports — needed for every srun below
```

**2. Pick a venv.** Two options, same vLLM build
(`0.30.1rc1.dev493+gac9126e58`), same patches applied to both:
- `/scratch/chenw/vllm-venv` — node-local, only exists on `cp81`/`cp82`,
  fastest Python-import time.
- `/import/ml-sc-scratch1/chenw/vllm-venv-b200` — shared NFS copy (added
  2026-10-13 specifically so other node pairs can use it without a
  rebuild), works from any node, modest import-latency cost per boot.
  `serve_k3_host.sh` always sources the node-local path
  (`source /scratch/chenw/vllm-venv/bin/activate`) — point it at the
  shared copy instead if launching on a node that doesn't have the
  node-local build.

**3. Launch the server** — `cp81` first (rank=1, headless), then `cp82`
~3s later (rank=0, API server), both via
`.scratch_kimi_launch/k3-b200-exp/scripts/serve_k3_host.sh`:
```bash
export HEAD_IP=10.17.16.54   # cp82's fabric (brcm0) IP — the API/rank0 node
JOBID=<from step 1>

# cp81 (rank 1, headless)
srun --jobid=$JOBID -w sc3-cp81 -N1 -n1 --gres=gpu:8 --overlap \
  env HEAD_IP=$HEAD_IP PP_PARTITION=48,45 DCP_SIZE=1 \
      DRAFT_DIR=<draft checkpoint dir> NUM_SPEC_TOKENS=<block size> \
  bash .scratch_kimi_launch/k3-b200-exp/scripts/serve_k3_host.sh \
  control-noconn 72 <phase-dir-name> &

sleep 3

# cp82 (rank 0, API server)
srun --jobid=$JOBID -w sc3-cp82 -N1 -n1 --gres=gpu:8 --overlap \
  env HEAD_IP=$HEAD_IP PP_PARTITION=48,45 DCP_SIZE=1 \
      DRAFT_DIR=<same draft checkpoint dir> NUM_SPEC_TOKENS=<same block size> \
  bash .scratch_kimi_launch/k3-b200-exp/scripts/serve_k3_host.sh \
  control-noconn 72 <phase-dir-name> &
```
`DRAFT_DIR`/`NUM_SPEC_TOKENS` are env-var overrides added 2026-10-13 —
previously hardcoded to RadixArk's full-attention checkpoint at block=7.
Omit both for that default. `DCP_SIZE=1` is required for any
DSpark-speculative-decoding config (see the DCP-incompatibility finding
below); omit `DRAFT_DIR`/`NUM_SPEC_TOKENS`/`DCP_SIZE` entirely (and drop
the `control-noconn` spec logic) for a no-speculative-decoding smoke
test.

**4. Verify — don't trust `/health` alone.** It returns 200 the moment
the HTTP server is up, often well before the model has actually finished
loading/compiling. Always confirm with a real completion:
```bash
curl -s http://$HEAD_IP:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"moonshotai/Kimi-K3","messages":[{"role":"user","content":"What is 7*8?"}],"max_tokens":10,"temperature":0}'
```

**5. Run AgentX** (`scripts/evaluate/mtp_server_eval/run_agentx.sh`):
```bash
BACKEND=vllm BASE_URL=http://$HEAD_IP:8000 MODEL=moonshotai/Kimi-K3 \
TOKENIZER_TRUST_REMOTE_CODE=1 USERS_LIST="1" DURATION=1800 MAX_CONTEXT=128000 \
RESULT_DIR=<output dir> \
AIPERF_BIN=/import/ml-sc-scratch1/chenw/speculators/.scratch_kimi_launch/.venv-aiperf/bin/aiperf \
bash scripts/evaluate/mtp_server_eval/run_agentx.sh
```
`AIPERF_BIN` **must be an absolute path** — the script `cd`s into its own
directory before resolving it, so a relative path silently fails to
find the binary. `DURATION` below 900s stamps `submission_valid=false`
(scenario minimum).

**6. Tear down cleanly before relaunching a different config:**
```bash
kill <the two srun PIDs from step 3>
sleep 5
srun --jobid=$JOBID -w sc3-cp81 -N1 -n1 --gres=gpu:0 --overlap pkill -9 -f "vllm.entrypoints"
srun --jobid=$JOBID -w sc3-cp82 -N1 -n1 --gres=gpu:0 --overlap pkill -9 -f "vllm.entrypoints"
sleep 5
curl -s -m 3 -o /dev/null -w "%{http_code}\n" http://$HEAD_IP:8000/health   # expect connection refused / 000
```
Check `nvidia-smi --query-gpu=memory.used` on both nodes reads 0 before
relaunching — a `kill` on the wrapping `srun` alone doesn't always
reliably tear down the `vllm.entrypoints` worker processes underneath.

**Known gotchas** (all covered in more depth elsewhere in this doc):
`TP` must stay intra-node (never split across nodes for this model);
`DCP` is required for the no-speculative-decoding and Inferact-draft
cases but **breaks** RadixArk's DSpark draft unless `DCP_SIZE=1`; the
sliding-window DSpark checkpoints are currently broken in live serving
at *any* block size (see "Isolation test" above) — don't serve them;
watch the `salloc --time` limit on any run that'll outlive a few hours.

## DSpark speculative decoding: two checkpoints, one real bug found (2026-10-12)

Two different DSpark draft checkpoints were tried on top of the working
base deployment:

- **Inferact/Kimi-K3-DSpark** (MLA-architecture, matches the target's own
  attention family — `q_lora_rank`/`kv_lora_rank`/`qk_nope_head_dim` etc.
  in its config). This is the checkpoint ruobingw's own recipe validates
  (confirmed by reading his `get_dspark.sh` and `errors-and-changes.md` —
  he downloads this exact 7.1GB HF file, never touches RadixArk's).
- **RadixArk/Kimi-K3-DSpark** (Qwen3/GQA architecture — `class
  DSparkConfig(Qwen3Config)`, plain multi-head GQA attention with a custom
  "dual-source" KV mechanism that concatenates target-model context and
  the draft's own embeddings into one attended-over sequence). This is the
  checkpoint with real measured accept-length numbers from this project's
  own September single-node (8×B300) evaluation work — never validated by
  ruobingw, and never tested on this newer vLLM build before now.

### Architecture misrouting (real bug, fixed, but not load-bearing for the final fix)

RadixArk's checkpoint declares `architectures: ["DSparkDraftModel"]`.
vLLM's model registry (`model_executor/models/registry.py`) maps that
exact string to `vllm.models.deepseek_v4.DSparkDeepseekV4ForCausalLM` —
a **DeepSeek-V4-specific** implementation, regardless of the checkpoint's
actual (Qwen3/GQA) architecture. This is the same misrouting bug this
project discovered in September on the older vLLM fork (see
`kimi-k3-setup.md` memory: "generic `DSparkDraftModel` gets misrouted...
inherits mxfp4 → crash") — it is still present in this newer
`0.30.1rc1.dev493` build. Fix: edit the checkpoint's local `config.json`
to declare `architectures: ["Qwen3DSparkModel"]` instead, which correctly
routes to `vllm.model_executor.models.qwen3_dspark.Qwen3DSparkForCausalLM`
(confirmed: `class Qwen3DSparkForCausalLM(DFlashQwen3ForCausalLM)`, the
genuine Qwen3/GQA-based native implementation). This fix is real and
worth keeping/reporting upstream, but turned out **not** to be what fixed
the actual launch failure below — the byte-level KV-cache error was
identical before and after, since it's computed from raw numeric config
fields (`num_key_value_heads`, `head_dim`, `block_size`) that don't differ
between the two `architectures` values.

### The real blocker and root cause: DCP is incompatible with this draft architecture

With RadixArk's checkpoint attached to the working TP=8×PP=2×**DCP=8**
base deployment, the engine crashed during KV-cache allocation:

```
ValueError: The resolved KV cache layout (BLNHC) does not store blocks as
dense, unpadded pages (block stride 6193152 != page 516096), so a manager
block cannot be split into 14 kernel blocks of 64 tokens. ...
```

This reproduced **identically** (same byte numbers) across every
externally-controllable variable tried: default vs `TOKENSPEED_MLA` vs
`FLASH_ATTN` attention backend for the draft, `--block-size 64`, both
valid `VLLM_KV_CACHE_LAYOUT` values (`BLNHC`/`BLHNC`), and the
architecture-routing fix above. Two source patches to
`vllm/v1/core/kv_cache_utils.py`'s `_annotate_eagle_groups` (adding
draft-group detection by layer-name prefix, then by `(num_kv_heads,
head_size)` shape-matching) also made no difference — debug instrumentation
confirmed this function's "no KV cache group could be identified as the
draft model's" warning is a **red herring**: it's checking Mamba/KDA
recurrent-state group annotation (irrelevant to RadixArk's plain-GQA,
no-Mamba-component draft), not the actual KV-cache byte-layout computation
that was crashing.

**Root cause, found by isolating against the known-working single-B300
config**: the working single-node deployment never used
`--decode-context-parallel-size` at all. Testing the exact same multi-node
topology (TP=8×PP=2) with `DCP_SIZE=1` (i.e. DCP disabled, decode-context-
parallelism off) instead of `DCP_SIZE=8` **fixed the crash immediately** —
first attempt, no other changes. DCP specifically (not PP, not multi-node
generally, not the draft's architecture family) is incompatible with
RadixArk's dual-source-KV draft's KV-cache sizing in this vLLM build.

**Working config**: TP=8 (intra-node) × PP=2 (cross-node), **no DCP**,
RadixArk's checkpoint with the `Qwen3DSparkModel` architecture fix,
`num_speculative_tokens=7` (matching the checkpoint's own trained
`block_size: 7`), no forced attention-backend override (let it
auto-select). Confirmed via a real `/v1/chat/completions` request (correct
arithmetic answer) and genuine `SpecDecoding metrics` log output:

```
Mean acceptance length: 5.53, Accepted throughput: 1.90 tokens/s,
Drafted throughput: 2.94 tokens/s, Accepted: 77 tokens, Drafted: 119 tokens,
Per-position acceptance rate: 0.882, 0.765, 0.765, 0.647, 0.529, 0.529, 0.412,
Avg Draft acceptance rate: 64.7%
```

The per-position curve is a smooth monotonic decay (expected, matches the
pattern documented throughout `kimi_k3_draft_eval.md`), and AL=5.53 lines
up closely with this project's own September single-node finding for
GSM8K-style structured reasoning (AL=5.58 at block=7) — this test prompt
(arithmetic) is in the same category. Real evidence of correct behavior,
not just absence of a crash.

**Tradeoff**: without DCP, this config loses DCP's long-context KV-sharding
benefit (lower effective concurrency/context headroom than the DCP=8 base
deployment). **Inferact's checkpoint was never tested against this DCP
incompatibility** — since it's MLA-shaped (matching the target's own
attention family and ruobingw's validated recipe, which does use DCP=8
successfully), it's plausible Inferact's draft is DCP-compatible and could
be run with the full DCP=8 config; this hasn't been verified, only
inferred from ruobingw's own successful usage.

**Artifacts**: `models/Kimi-K3-DSpark-radixark-raw/` (patched config.json),
`models/Kimi-K3-DSpark-raw/` (Inferact's copy, downloaded but not used for
the final working config), the two non-load-bearing source patches are
still present in `/scratch/chenw/vllm-venv` on both nodes (harmless,
commented `# chenw patch` / `# chenw patch v2`), launch logs in
`run1` through `run13` under `.scratch_kimi_launch/k3-b200-exp/`.

## AgentX live-serving evaluation on the 2-node deployment (2026-10-12/13)

Real agentic-trace replay (`aiperf --scenario inferencex-agentx-mvp`,
corpus `semianalysis_cc_traces_weka_062126`) against the live 2-node
server, `users=1`, `--max-context-length 128000` unless noted. All runs
below target `http://<rank0-node>:8000`, same TP8×PP2×DCP1 base.

| run | draft / block | duration | decode_tok_s | out_tok_s | accept_len | accept_rate | valid |
|---|---|---:|---:|---:|---:|---:|---|
| Concurrency sweep | RadixArk, block=7, full attn | up to users=8 | — | — | — | — | (Bug 5, the old concurrent-load crash, did **not** reproduce — see below) |
| 256K-500K long-context | RadixArk, block=7, full attn | 900s | 134.6 | 84.1 | 2.052 | 0.1502 | false (`--unsafe-override`, 1/63 errored) |
| Full 79-trace corpus | RadixArk, block=7, full attn | 10800s | 139.1 | 98.7 | NA* | NA* | **true** (1257/1319 completed) |
| Block=3 + sliding-window=2048 | `Kimi-K3-DSpark-sliding2048`, block=3 | 1800s | 103.5 | 70.2 | 1.357 | 0.1189 | **true** |

\* The full-corpus run's server was killed mid-flush by a SLURM wall-clock
timeout (see below) right as it finished, so the harness never captured
final cumulative SpecDecoding counters from `/metrics` — throughput
numbers (client-side timing) are unaffected and valid, only
accept_len/accept_rate are missing for that one run.

**Notable finding: Bug 5 (the single-node vLLM 0.29.0 concurrent-load
crash documented in `kimi_k3_draft_eval.md`) did not reproduce** on this
2-node vLLM 0.30.1rc1 stack at `users=8` — a clean 195-request/900s run
completed with the server still alive afterward. Not yet root-caused
(could be the newer vLLM version, the different TP/PP topology, or
something else) — flagged as a positive result, not a confirmed fix.

**Block=3 + sliding-window=2048 vs. block=7 full-attention**: decode
throughput and accept_len are both lower (103.5 vs 139.1 tok_s;
accept_len 1.357). Part of this is structural and expected — block=3
proposes fewer draft tokens/step than block=7, which caps the achievable
speedup regardless of acceptance quality — but 1.357 is also lower than
the offline block=3 baseline's accept_len range (2.5-3.6 across the 24
eval sets, see `dspark_sliding_window_ablation.md`), larger than the
offline sliding=2048 ablation's ~0% measured degradation would predict.
Live AgentX's real agentic traffic is a different distribution than the
offline eval's held-out prompt sets, so some gap is expected, but this
hasn't been confirmed as a stable number vs. a warm-up artifact of the
30-minute window — treat as preliminary.

**Infra incident during the full-corpus run**: the `salloc` behind both
server nodes (job `10368011`/`10368001`) had a default 10-hour
`--time` limit and was killed by SLURM at the 10-hour mark, which landed
almost exactly at the end of the 3-hour full-corpus AgentX run,
producing a burst of `ConnectionRefused` errors in its last ~5 minutes
and the missing accept_len/accept_rate above. Restarted on a fresh
`salloc --time=24:00:00` (job `10478386`) — see
[[kimi-k3-multinode-gloo-hang]] memory for the durable note on this.

**Artifacts**: per-run logs and aiperf exports under
`.scratch_kimi_launch/k3-b200-exp/agentx_2node_{alltraces,longctx,sliding2048_block3}/`;
`serve_k3_host.sh` gained `DRAFT_DIR`/`NUM_SPEC_TOKENS` env overrides
(previously hardcoded to RadixArk's full-attention checkpoint at block=7)
to support switching draft checkpoint and block size per run.

### Isolation test (2026-10-13): sliding-window is the cause, not block=3

The preliminary "block=3+sliding2048 is slower and lower-AL" result above
turned out to understate a real problem. Pulling the full per-second
`SpecDecoding metrics` time series from the server log (not just the
final cumulative number) showed a **bimodal** pattern — bursts of
genuinely good acceptance (up to 85-88% per-position) interleaved
repeatedly with total collapses to exactly `0.000, 0.000, 0.000`
(zero tokens accepted out of 200+ drafted) for multiple consecutive
seconds, recurring throughout the entire run. That shape (snapping
between "works great" and "accepts nothing") looks like a correctness
bug, not a quality tradeoff — and `num_speculative_tokens=3` had never
been tested in **live** vLLM serving before this session (all prior
block=3 testing was offline-harness-only).

Ran a clean isolation test to separate the two variables — RadixArk's
**original full-attention** checkpoint at block=3 (same block size,
sliding window removed):

| config | decode_tok_s | accept_len | accept_rate | zero-collapse rate |
|---|---:|---:|---:|---:|
| block=3, full-attention (isolation) | **163.0** | **2.137** | **0.3789** | **36/1293 = 2.8%** |
| block=3, sliding-window=2048 | 103.5 | 1.357 | 0.1189 | **291/1351 = 21.5%** |

**Conclusion: the sliding window causes the collapse, not block=3.**
Collapse rate jumps ~8x (2.8%→21.5%) purely from adding the
sliding-window config on top of identical block=3. Block=3 alone is
healthy — accept_len=2.137 is a reasonable live-traffic number, and
decode_tok_s=163.0 is even *higher* than the block=7 full-attention
baseline (139.1), a separate interesting finding (fewer draft
tokens/round-trip costs less overhead than it loses in acceptance, net
positive on real AgentX traffic).

**Suspected root cause, now with source-level corroboration:** vLLM's
live serving implements DSpark sliding-window via the **generic
`SlidingWindowSpec` KV-cache-eviction mechanism** (confirmed via a debug
log line tagging the draft's layers `spec_type='SlidingWindowSpec',
num_kv_heads=2, head_size=64` in `kv_cache_utils.py` — plain GQA shape,
**not MLA**, despite `--attention-backend TOKENSPEED_MLA` being set
globally; that flag only governs the target's MLA layers, the draft's
own attention group gets vLLM's separately auto-selected backend for
non-MLA layers, most likely FlashAttention on this hardware). This
differs architecturally from this project's own training/offline-eval
implementation: a custom non-causal block-mask
(`create_anchor_block_mask_mod`, `dflash/attention.py`) restricting
attention from synthetic draft *query* positions specifically to the
target's *context-KV buffer* (5-tap-concatenated hidden states), not a
KV-cache eviction over the draft's own tiny generated-token sequence.

Traced further into `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`
(`DFlashSpeculator.set_attn`, the base class `DSparkSpeculator` extends)
and found vLLM's own maintainers have already hit and partially patched
a **named bug in exactly this class of configuration**, documented
inline (lines 197-209):

```python
# FlashAttention's AOT split schedule is wrong for a windowed drafter,
# and `_get_sliding_window_configs` leaves it on or off depending on
# whether the target also runs FlashAttention. Decide it here instead.
for groups in self.attn_groups:
    for group in groups:
        builder = group.get_metadata_builder()
        if getattr(
            builder, "aot_schedule", False
        ) and get_kv_cache_spec_sliding_window(builder.kv_cache_spec):
            builder.aot_schedule = False
```

**This specific hypothesis was tested and ruled out (2026-10-13).** Added
a one-line debug print right before the fix's `if` check
(`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`, both nodes) and
restarted with `sliding2048`+block=3:

```
CHENW_DEBUG_BACKEND builder_class=FlashAttentionMetadataBuilder
has_aot_schedule=True aot_schedule_val=False is_sliding_window=2048
```

The draft's attention group **is** confirmed FlashAttention (that part
of the hypothesis was right), but `aot_schedule` was **already `False`
before the fix's own `if` condition even runs** — so
`if aot_schedule and is_sliding_window` short-circuits on the first
operand, and the fix never executes. It's a no-op for this config.
Most likely explanation: `_get_sliding_window_configs` (referenced in
the fix's own comment) sets `aot_schedule` based on whether the
**target** model also runs FlashAttention — ours doesn't (target uses
`TOKENSPEED_MLA`) — so AOT scheduling was probably never turned on for
the draft's builder in the first place, regardless of the sliding-window
spec. This named vLLM bug class does not explain the collapse in our
setup.

**Root cause investigation, continued — block-size ruled out
entirely (2026-10-13).** The B300 live-serving verification (above,
"Live-serving verification" section) validated this exact draft
checkpoint at **block=7** (the trained default) on **vLLM 0.29.0,
single-node TP=8** — cleanly, mean AL ratio 0.9994, no collapse issues
reported at all. That's a different config from every B200 test so far
(which used block=3). Re-ran the isolation test at block=7 (trained
default, matching B300's config exactly) on the B200 stack:

| config | decode_tok_s | accept_len | accept_rate | collapse rate |
|---|---:|---:|---:|---:|
| B300, vLLM 0.29.0, single-node TP=8, block=7, sliding2048 | — | — | — | ~0% (AL ratio 0.9994) |
| B200, vLLM 0.30.1rc1, cross-node, block=3, full-attn | 163.0 | 2.137 | 0.3789 | 2.8% |
| B200, vLLM 0.30.1rc1, cross-node, block=3, sliding2048 | 103.5 | 1.357 | 0.1189 | 21.5% |
| **B200, vLLM 0.30.1rc1, cross-node, block=7, sliding2048** | **84.6** | **1.266** | **0.0380** | **68.6%** |

**Block=7 is worse than block=3, not better — this rules out block-size
as the driver entirely.** The collapse is present (and roughly 3x worse)
at the exact trained block size B300 validated cleanly. Two variables
remain between B300's working config and B200's broken one: vLLM
version (0.29.0 → 0.30.1rc1) and topology (single-node TP=8 →
cross-node TP=8×PP=2×DCP=1). Supporting evidence for each:

- **Version**: confirmed earlier that the `set_attn` override containing
  the `aot_schedule`/sliding-window-specific code in
  `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` **does not exist
  at all** in the 0.29.0 install used for the B300 test
  (`/import/ml-sc-scratch6/chenw/conda_env/kimi_k3`, confirmed via
  direct grep, no node allocation needed since it's shared NFS) — this
  code path is new between versions, consistent with a possible
  regression in newer sliding-window-drafter support.
- **Topology**: cross-node PP=2 is itself a fragile, novel part of this
  project's Kimi K3 setup (see the NCCL/DCP issues earlier in this doc)
  — sliding-window KV-cache eviction depends on exact position/slot
  bookkeeping, which is exactly the kind of state that could break under
  pipeline-parallel synchronization. This is the more plausible
  candidate of the two, but **cannot be cleanly isolated**: Kimi K3 does
  not fit on a single B200 node at TP=8 (183 GiB/GPU available vs. ~194
  GiB/GPU needed, see "Why 2 nodes at all" above), so there's no way to
  test single-node TP=8 on B200 without further changes (e.g. a smaller
  quantization, or accepting OOM risk).

**Not yet done**: testing on a single B300 node at TP=8 with vLLM
0.30.1rc1 (same version as B200, same topology as the original working
test) would cleanly separate these two variables — if that combination
is also broken, it's the vLLM version; if clean, it's the cross-node
topology. Requires access to an idle B300 node and building/updating a
vLLM 0.30.1rc1 install there, not attempted given time already invested
this session. **Practical conclusion is unaffected either way: don't
serve the sliding-window checkpoints live until this is actually
root-caused and fixed.**

**Practical takeaway**: don't use the sliding-window checkpoints in live
serving yet — they're validated offline only. Block-size overrides
(block=3) are safe in live serving and may even help throughput on
real agentic traffic.

### Shared vLLM venv for future node pairs (2026-10-13)

The working `/scratch/chenw/vllm-venv` (all patches applied: the
architecture-misrouting config fix, `CHENW_DEBUG_BACKEND` diagnostic
print, etc.) only existed on `cp81`/`cp82`'s node-local disk — every new
node pair would otherwise need a from-scratch rebuild (the original
build hit real bugs: `flashinfer-cubin` version mismatches, CUDA
forward-compat issues). Copied it to shared NFS instead:
**`/import/ml-sc-scratch1/chenw/vllm-venv-b200/`** (18 GiB, `rsync -a`
from `cp81`, ~instant node-to-node over the 100GbE fabric). Any future
node pair can `source /import/ml-sc-scratch1/chenw/vllm-venv-b200/bin/activate`
directly — no rebuild, no rsync needed.

**Caveat — don't move the runtime caches here too.** This only covers
the *static* venv (Python packages, compiled extensions, read-only
during serving). The actual runtime cache directories (FlashInfer,
Triton, TorchInductor — `$CACHE` in `serve_k3_host.sh`, currently
`/scratch/chenw/cache`) must **stay node-local**: NFS caused a
documented deadlock there under concurrent multi-worker file locking
(see `kimi_k3_dspark_fp8_kv_cache.md`,
`FLASHINFER_WORKSPACE_BASE` comment). Expect a modest one-time
Python-import latency cost per server boot from reading the venv over
NFS instead of local disk (many small file reads) — not a correctness
risk, just slower startup.

### Full-attention block=3/5/7 sweep at 256K context (2026-10-13, in progress)

Re-grounding on the vanilla full-attention RadixArk draft (no sliding
window) before further sliding-window root-causing — sweeping
`NUM_SPEC_TOKENS` (3, 5, 7) via live AgentX at `MAX_CONTEXT=256000`
(vs. the 128K ceiling used in all earlier §AgentX runs), same
TP=8×PP=2×DCP=1 topology. Running sequentially on `cp81`/`cp82` (the
only node pair with a working venv at the time this was started — see
above). Collapse-rate sanity check (the same `SpecDecoding metrics`
zero-acceptance pattern tracked throughout the sliding-window
investigation) applied here too, as a control: full-attention should
show ~0% collapse at any block size, confirming the collapse pattern
really is sliding-window-specific and not a general artifact of this
vLLM build/topology.

*(Results to be filled in once all three block sizes complete.)*

---

**Status: blocked on a cross-node NCCL data-plane hang, not yet working end-to-end.**
This doc is the recipe for everything that *does* work, plus the full evidence
trail for the one thing that doesn't, so the next attempt doesn't have to
re-derive any of it.

## Why 2 nodes at all

Kimi K3's weights alone need **~194 GiB per GPU shard at TP=8** (measured on the
known-good 8xB300 node, which has 275 GiB/GPU). B200 nodes in this cluster have
only **183 GiB/GPU** (`gpu:b200m180`) — the model does not fit at TP=8 on a
single B200 node. Splitting across 2 nodes (TP=16, 16 GPUs total) roughly halves
the per-GPU shard to ~97 GiB, which does fit.

## What works

### 1. Book two B200 nodes with real CPU, not SLURM's default

```bash
salloc --partition=gpuonly --nodelist=<node1>,<node2> --nodes=2 \
  --ntasks-per-node=1 --cpus-per-task=128 --gres=gpu:b200m180:8 \
  --time=08:00:00 --no-shell
```

**Must** set `--cpus-per-task` explicitly. A bare `salloc --gres=gpu:...` with no
CPU flags gives a tiny default (we got 4 CPUs total for a 16-GPU/2-node job on
one attempt) — nowhere near enough for 8 TP worker processes per node.

### 2. `srun` into the allocation — always pin `--nodes=1 --ntasks=1`

```bash
srun --jobid=<JOBID> --nodes=1 --ntasks=1 --nodelist=<node1> <script>
```

**This bit us twice.** The job allocation has `NumTasks=2` (one per node). A
bare `srun --jobid=... --nodelist=<node1> <script>` does **not** restrict
execution to that one node — it inherits the job's full 2-task/2-node spec and
runs the script on *both* nodes simultaneously (confirmed via two different
"Local node IP" lines in one log file, from two independently-started "head"
processes). Always add `--nodes=1 --ntasks=1` explicitly.

Also: running a second `srun` step on a node that already has one running step
fails with `Job/step already completing or completed` unless the first step
has actually exited — SLURM does not multiplex steps onto spare CPU by default
on this cluster even when there's plenty of CPU headroom.

### 3. Keep the step alive for the daemon's whole lifetime

`ray start --head`/`ray start --address=...` fork and detach, but the detached
daemon is still inside the `srun` step's cgroup/session — when the step's main
process exits, SLURM tears down everything in it, killing Ray too. Keep the
step open for as long as Ray needs to run:

```bash
ray start --head --port=6379
sleep 5
exec <the actual long-running process (vllm serve, or sleep infinity for a pure worker)>
```

### 4. This node's kernel enforces FIPS — vLLM's `cv2` import crashes

`/proc/sys/crypto/fips_enabled` = `1` on these nodes. `opencv-python-headless`
(an indirect vLLM dependency, pulled in for optional multimodal/video support)
bundles its own vendored OpenSSL 1.1.1k, which fails its legacy FIPS self-test
on this kernel and aborts the whole process (`crypto/fips/fips.c:154: OpenSSL
internal error: FATAL FIPS SELFTEST FAILURE`). Confirmed via `LD_DEBUG=libs`:
`cv2.abi3.so` → bundled `libavformat` → bundled `libssl-*.so.1.1.1k`, whose
`calling init:` step is exactly where the abort happens. `cv2` is unused for
Kimi K3's text-only DSpark serving.

**Fix** (does not touch the shared conda env): a `sitecustomize.py` that poisons
`sys.modules['cv2'] = None` (forcing a clean `ImportError`, which vLLM's own
`PlaceholderModule` fallback already handles gracefully), activated via
`PYTHONPATH` rather than dropped into site-packages — this way it's scoped to
this launch, not a silent permanent change to the shared env.

```bash
mkdir -p pythonpath_shim
cat > pythonpath_shim/sitecustomize.py <<'EOF'
import sys
sys.modules['cv2'] = None
EOF
export PYTHONPATH=/path/to/pythonpath_shim:$PYTHONPATH
```

This must be set for **every** process that imports vLLM — including Ray's own
spawned worker processes (separate interpreters that don't inherit a parent's
in-memory `sys.modules`), not just the one you launch directly. Set it in the
env of both the head and worker launch scripts.

### 5. DSpark does not support pipeline parallelism

`--tensor-parallel-size 8 --pipeline-parallel-size 2` fails immediately with:

```
NotImplementedError: Pipeline parallelism is not supported for this model.
Supported models implement the SupportsPP interface.
```

This is in `draft_model_config.verify_with_parallel_config` — the DSpark draft
class doesn't implement `SupportsPP`. Use pure **`--tensor-parallel-size 16`**
instead (no PP). This doesn't need `SupportsPP` at all, since every rank does
the same thing.

### 6. Ray placement-group strategy: use `SPREAD`, never `STRICT_SPREAD`, for N bundles > N nodes

If you manually pre-create a placement group (rather than letting vLLM create
its own — see caveat below), **`STRICT_SPREAD` requires each bundle on a
strictly different node**. With 16 single-GPU bundles and only 2 nodes, this is
mathematically impossible and hangs until timeout, every time, regardless of
any NCCL/network setting:

```python
# WRONG — guaranteed to time out with 16 bundles on 2 nodes
pg = ray.util.placement_group(bundles=[{"GPU": 1} for _ in range(16)], strategy="STRICT_SPREAD")

# RIGHT — best-effort spread, succeeds immediately
pg = ray.util.placement_group(bundles=[{"GPU": 1, "CPU": 8} for _ in range(16)], strategy="SPREAD")
```

Confirmed via a standalone Ray diagnostic (`debug_pg.py`): `STRICT_SPREAD`
timed out twice (120s each), `SPREAD` with otherwise-identical bundles
succeeded immediately.

**Caveat — this specific bug does not explain the launch hang below.** vLLM's
*own* internal placement-group creation (`vllm/v1/executor/ray_utils.py`, used
whenever you don't pre-create one yourself) uses `strategy="PACK"`, which has
no such impossibility. The `vllm serve` launches in this doc never pre-create a
placement group, so they go through vLLM's own PACK-based path and are
unaffected by this bug — it only matters if you write your own Ray
orchestration script (as we did for diagnostics) rather than calling
`vllm serve` directly.

### 7. The working single-node command (for reference / sanity-checking memory)

```bash
export FLASHINFER_WORKSPACE_BASE=/scratch/$(whoami)/kimi_fi_cache
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=/path/to/pythonpath_shim:$PYTHONPATH

/import/ml-sc-scratch6/chenw/conda_env/kimi_k3/bin/python3 -c "
import sys; sys.modules['cv2']=None
from vllm.entrypoints.cli.main import main
sys.argv=['vllm','serve','/import/ml-sc-scratch5/chenw/models/Kimi-K3-patched',
  '--trust-remote-code','--tensor-parallel-size','8','--max-model-len','8192',
  '--speculative-config','{\"method\":\"dspark\",\"model\":\"/import/ml-sc-scratch5/chenw/models/Kimi-K3-DSpark\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"greedy\"}']
main()
"
```

This is expected to OOM at weight loading (194 GiB shard vs 183 GiB available)
— useful for confirming the real numbers on this hardware, not for actually
serving.

## What does not work yet: cross-node NCCL hangs after rank 0's own init

### The full 2-node launch command

```bash
# Head node
export FLASHINFER_WORKSPACE_BASE=/scratch/$(whoami)/kimi_fi_cache
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=/path/to/pythonpath_shim:$PYTHONPATH
export NCCL_SOCKET_IFNAME=brcm0
export GLOO_SOCKET_IFNAME=brcm0
export NCCL_IB_DISABLE=1     # or 0 -- both hang (see below)
export NCCL_DEBUG=INFO

/path/to/conda/bin/ray start --head --port=6379
sleep 5
exec /path/to/conda/bin/python3 vllm_no_cv2.py serve \
  /import/ml-sc-scratch5/chenw/models/Kimi-K3-patched \
  --trust-remote-code --tensor-parallel-size 16 \
  --distributed-executor-backend ray --max-model-len 8192 \
  --speculative-config '{"method":"dspark","model":"/import/ml-sc-scratch5/chenw/models/Kimi-K3-DSpark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
```

```bash
# Worker node
export FLASHINFER_WORKSPACE_BASE=/scratch/$(whoami)/kimi_fi_cache
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=/path/to/pythonpath_shim:$PYTHONPATH
export NCCL_SOCKET_IFNAME=brcm0
export GLOO_SOCKET_IFNAME=brcm0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

/path/to/conda/bin/ray start --address='<head_ip>:6379'
sleep infinity
```

### The symptom

Ray cluster always forms cleanly (both nodes report, 16 GPUs, correct
resources). vLLM resolves config correctly, Ray workers spawn, rank 0's own
`ncclCommInitRank` always completes (`Init COMPLETE`, ~2s). Then: **silence,
indefinitely** (tested to 30+ min). The step stays alive (not crashed), the
process never progresses, nothing is written to the log past this point.

### What was ruled out, each confirmed independently

| hypothesis | test | result |
|---|---|---|
| Wrong node pair / bad cable | retried on three independent idle node pairs (`cp80`/`cp81` → `cp83`/`cp84` → `cp90`/`cp91`) | **same hang every time** |
| Wrong NIC selection | tried plain TCP socket (`brcm0`), mixed RoCE+IB (auto-detected `bnxt_re0`+`mlx5_*`), and IB restricted to Mellanox only (`NCCL_IB_HCA=mlx5`) | **all three hang at the same point** |
| `STRICT_SPREAD` Ray scheduling bug (real, found and fixed separately) | this launch never creates a manual placement group — goes through vLLM's own `PACK`-strategy path | not applicable to this launch |
| NVLink/P2P path specifically broken | `NCCL_P2P_DISABLE=1` (confirmed active: `isAllDirectP2p 0` in the log) | **still hangs — even one line earlier than before** |
| Two heads both starting (own bug) | fixed by pinning `srun --nodes=1 --ntasks=1` (see §2 above) | fixed, unrelated to this hang |

### Third attempt (2026-10-10/11, fresh node pair `cp90`/`cp91`, job 10023137)

Same recipe as above, run fresh end-to-end (new `salloc`, new Ray cluster,
same env vars). Ray formed correctly (worker joined, "Ray runtime
started" on both sides). vLLM resolved config and the EngineCore process
connected to the Ray cluster (`Connected to Ray cluster` at 23:55:33).

**Then: total silence for 90+ minutes** — zero new bytes written to the
head log (confirmed via file mtime, unchanged from 23:55:36 onward), no
crash, no exit; both SLURM steps stayed `RUNNING` throughout. This stall
point is **earlier in the pipeline than the original hang** documented
above — the original attempts got all the way through weight loading and
rank 0's own `ncclCommInitRank` before stalling; this attempt never
produced any evidence of weight-loading starting at all, stalling
immediately after the Ray-cluster-connect step instead. Whether this is
the exact same underlying block manifesting at an earlier checkpoint, or
a second, distinct stall in the same general area, is unresolved — not
enough signal to distinguish them.

Direct SSH to the allocated compute node was attempted to inspect
process/GPU state independently of the log, but the SSH session closed
immediately after key-based auth succeeded (a PAM/session-policy
rejection unrelated to the job allocation itself — not pursued further,
not likely related to the hang). `sstat` job accounting showed ~213GB
resident memory on the head node, too ambiguous to interpret (could be
genuine weight-loading progress not yet flushed to the log, or just
Ray's own large default object-store allocation).

After 90 minutes of confirmed silence with zero progress, the allocation
was released (`scancel`) rather than held further — this is now a
**third independent confirmation**, across three different node pairs,
of a 2-node launch stall on this cluster. Combined with the earlier
"Reading" analysis (Ray's own cross-node control-plane traffic always
works; only the data-plane/compute-side sequence stalls), this continues
to point at something structural on the cluster side rather than
anything fixable from the launch-script/env-var side.

### Reading (original, superseded below)

Three independent transport configurations, on two independent node pairs, all
converge on the identical failure point: **rank 0 completes its own
`ncclCommInitRank`, then the cluster never finishes whatever barrier/handshake
waits for all 16 ranks.** Ray's own cross-node control-plane traffic (GCS,
resource reporting, actor dispatch) works reliably every single time — only
NCCL's own inter-node data-plane connection setup hangs. This pattern (OOB/
bootstrap succeeds, actual data-channel connection does not) is most
consistent with a **network/firewall-level block on whatever TCP port range
NCCL's proxy service needs for the real inter-node data connections**, not a
misconfigured env var on our end — we've now tried the three standard
env-var-level fixes (socket vs. IB vs. P2P-disabled) and none change the
outcome.

### Revised theory (2026-10-11): not a blanket block — likely idle-connection drop

The "NCCL data-plane is blocked outright" theory above doesn't survive a
direct counter-example: **MiniMax-M3 serves successfully across 2 of these
same B200 nodes**, using SGLang with `--tp-size 4 --pp-size 2` (TP stays
*within* each node on fast NVLink; only sparse pipeline activations cross
the inter-node link) — confirmed by reproducing it fresh on a brand-new
pair (`cp95`/`cp96`, chosen specifically because it had never been used by
either the working MiniMax job or any of our three hung Kimi attempts):
weights loaded, `Uvicorn running on http://0.0.0.0:8000` logged, both
pipeline stages progressed in lockstep. So cross-node communication on
this cluster is **not blanket-blocked** — something more specific is going
on.

**What actually happened on the validation run, and why it's a better
lead than the original hang**: the server came up successfully, then
crashed on its *first real request* — not a hang, a clean Python
exception:

```
RuntimeError: [gloo/transport/tcp/pair.cc:547] Connection closed by peer [10.17.16.68]:43692
```

This is **Gloo** (the separate CPU-side TCP control channel SGLang uses
for pipeline-stage metadata — tensor shapes/dtypes — distinct from NCCL's
GPU data channel) reporting that the *remote peer actively closed* the
connection. Critically, this channel had sat **idle for the full ~40
minutes of weight loading** before this first real use. "Connected but
then actively reset by the peer after sitting idle" is the signature of
a **stateful firewall or NAT dropping idle connections after some
timeout**, not a connection that was never allowed to form.

This reframes the original Kimi K3 hang as plausibly **the same root
cause, manifesting differently**: Kimi K3's checkpoint is ~1.5 TB
(vs. MiniMax's much smaller one), so its NCCL setup/weight-loading phase
runs far longer before any real inter-node data needs to move — more than
enough time for the same idle-connection timeout to fire on NCCL's own
socket. The difference in symptom (silent hang vs. clean exception) would
then just reflect that Gloo surfaces a dead socket as an explicit error,
while NCCL's blocking collective wait apparently does not (it just blocks
forever trying to use a socket the peer already closed, with no timeout
firing — vLLM's default NCCL collective timeout is very long/effectively
unbounded for a single `ncclCommInitRank`-adjacent barrier). Not proven —
the exact point in Kimi K3's pipeline where the hang occurs (right after
rank 0's own init, which is comparatively early) hasn't been confirmed to
coincide with an idle-timeout window the way MiniMax's 40-minute gap does
— but this is now the best-supported lead, not a guess from first
principles.

**Next step this suggests, not yet tried**: TCP keepalive on the relevant
sockets (so periodic small packets go out during long idle windows like
weight loading, preventing any stateful middlebox from expiring the
connection) — either via `NCCL_SOCKET_KEEPALIVE`-style settings, OS-level
`SO_KEEPALIVE` tuning on the compute nodes (shorter `net.ipv4.tcp_keepalive_time`),
or simply keeping some trivial periodic traffic flowing on the relevant
channel during weight load. This is a more targeted, testable fix than
the original "firewall port audit" recommendation — worth trying before
escalating to a network/infra request.

### Fourth and fifth attempts (2026-10-11, same session): both the "more
nodes" and "match MiniMax's topology" theories falsified

Two more fresh attempts, both on a new 4-node pair (`cp92`/`cp93`/`cp94`/`cp97`,
never used before), both killed after ~45+ min of confirmed silence (file
mtime frozen, job still `RUNNING`, SLURM steps alive):

1. **Pure TP=32 across 4 nodes** (smaller per-GPU shard than TP=16 — ~47GB
   vs ~97GB — to test whether a shorter weight-load window avoids whatever
   idle-connection issue might be at play). Hung at the identical point as
   every 2-node attempt: immediately after `Connected to Ray cluster`, no
   further log output.
2. **PP=4 × TP=8, target model only (no DSpark draft attached)** —
   deliberately mirroring MiniMax-M3's exact working topology (TP stays
   intra-node on NVLink, only sparse PP activations cross nodes) and
   removing DSpark's `SupportsPP` restriction entirely by not using it.
   **Hung at the exact same point as every other attempt** — confirmed via
   log mtime frozen at the "Connected to Ray cluster" timestamp for 46
   minutes, `speculative_config=None` and `tensor_parallel_size=8,
   pipeline_parallel_size=4` both confirmed correct in the engine config
   dump (so this really was the intended config, not a launch-script bug).

**This rules out both of today's two leading theories as the primary or
sole cause**: it's not "DSpark forces pure TP, which needs too much
cross-node collective traffic" (attempt 2 used PP with no DSpark at all,
and hung identically); it's not simply "weight loading takes too long,
letting an idle connection expire" either, since **this hang occurs before
any weight loading or further distributed-init logging happens at
all** — immediately after Ray reports a successful connection, with
zero progress for 45+ minutes, regardless of parallelism strategy or
checkpoint size.

**New leading suspect: the hang is in Ray's own actor/placement-group
orchestration (or vLLM's use of it), not in model parallelism strategy or
NCCL data-plane transport at all.** The one thing every single Kimi K3
attempt has in common, that MiniMax-M3's working launch does *not* share,
is **using Ray at all** — MiniMax-M3/SGLang uses its own native
`torch.distributed`-style rendezvous (`--dist-init-addr`), with no Ray
layer in the picture. Every failing Kimi K3 attempt goes through vLLM's
`--distributed-executor-backend ray`. Also worth scrutiny: the
`pythonpath_shim/sitecustomize.py` used in every Kimi K3 attempt
monkey-patches `ray.util.placement_group` (forcing `SPREAD` strategy
universally) and `ray.init` (redirecting `address=None` through
`RAY_ADDRESS`) — these patches have not been tested at `world_size=32`
across 4 nodes before today, and a bug specific to that scale or
topology in the patched placement-group logic hasn't been ruled out.
Neither of these has been tested in isolation yet (e.g., a bare Ray
placement-group/actor-creation script with no vLLM at all, with and
without the shim's patches active).

### Sixth attempt (2026-10-11, job 10072414, `cp99`/`cp100`): Ray removed
entirely — NCCL succeeds, a Gloo `barrier()` hangs instead

To directly test the "it's Ray's orchestration layer" theory from the
previous section, this attempt bypassed Ray completely: vLLM's **native**
multi-node launch (`--nnodes 2 --node-rank {0,1} --master-addr
--master-port`, `distributed_executor_backend` defaulting to `mp`), run
via identical `srun` invocations on `cp99` (rank 0) and `cp100` (rank 1),
TP=16 with the DSpark draft attached. Script:
`.scratch_kimi_launch/native_mp_launch.sh`, logs:
`logs/native_mp_rank{0,1}.log`.

**This got further than every previous attempt.** All 16 ranks across
both nodes completed real NCCL communicator setup:

```
sc3-cp99:354962:354962 [3] NCCL INFO ncclCommInitRank comm 0x55e4cbcb0c00 rank 3 nranks 16 cudaDev 3 nvmlDev 3 busId 73000 commId 0xc15e92efffdc1746 - Init COMPLETE
```
(8 such lines on `cp99` — ranks 0-7 — and 8 more on `cp100` — ranks
8-15, all sharing the same `commId`.) This is the first time in this
entire investigation that a cross-node NCCL TP communicator has actually
formed. **Ray's orchestration layer is no longer implicated as the sole
cause** — the hang reproduces even with Ray completely out of the
picture, and it occurs *after* the exact step (cross-node NCCL setup)
that was previously never reached.

The actual failure is later and much more specific, in
`vllm.distributed.parallel_state.in_the_same_node_as` — a utility that
determines which ranks share a physical machine, used when building the
TP group's shared-memory broadcaster. Walking through what it does
(confirmed by reading `parallel_state.py` directly, lines ~2340-2400):

1. Rank 0 of the group creates a POSIX shared-memory segment and writes
   a magic string into it.
2. Its name is shared with every rank via `torch.distributed.broadcast_object_list`
   over the group's **Gloo** CPU process-group (vLLM creates one of
   these alongside every NCCL device-group, for exactly this kind of
   host-side metadata op — it is a separate transport from the NCCL
   group that just succeeded).
3. Each non-source rank tries to open that shared-memory segment by
   name. Cross-node ranks are *expected* to fail here (POSIX shm doesn't
   exist across machines) — the resulting exception is caught and merely
   logged (`ERROR ... Error ignored in is_in_the_same_node: ...`), not
   fatal, and this is exactly what both logs show first.
4. **After that try/except, every rank calls `torch.distributed.barrier(group=pg)` on the same Gloo group** (`parallel_state.py:2385`). This is
   the actual hang: the barrier never returns, and 30 minutes later Gloo's
   own transport layer throws a clean, un-caught timeout — this exception
   message happens to get logged by the *previous* step's exception
   handler due to log timing, which is what makes the first-glance log
   line slightly misleading:

   ```
   ERROR ... [parallel_state.py:2379] Error ignored in is_in_the_same_node: [.../gloo/transport/tcp/unbound_buffer.cc:78] Timed out waiting 1800000ms for recv operation to complete
   ...
   File ".../vllm/distributed/parallel_state.py", line 2385, in in_the_same_node_as
       torch.distributed.barrier(group=pg)
   ...
   RuntimeError: Application timeout caused pair closure
   ```

   Both nodes hit this at the identical wall-clock timestamp (`15:24:20`,
   exactly 30 min after worker startup at `~14:54:20`), confirming it's
   one synchronized collective timing out together, not two independent
   failures.

**Sharpened conclusion: on this cluster, NCCL's cross-node transport
works; Gloo's does not.** This is now corroborated by two independent
data points from *different* frameworks, hanging in *different* specific
Gloo calls, but always Gloo and never NCCL:
- Kimi K3 / vLLM (this attempt): hangs in a Gloo `barrier()` inside
  `in_the_same_node_as`.
- MiniMax-M3 / SGLang (this morning's crash, same session): hung in Gloo
  during `recv_tensor_dict` (pipeline-parallel metadata exchange) —
  `Connection closed by peer`, also never an NCCL error.

This also means the earlier "Ray orchestration" and "idle-connection
drop" theories (previous two sections) weren't wrong about *where* things
hang being downstream of real communication setup — the common factor
running through all of today's failures turns out to be **any
distributed collective routed over Gloo, cross-node**, regardless of
whether Ray, native `mp`, vLLM, or SGLang is the framework issuing it.

### Seventh attempt (2026-10-11, job 10082774, fresh nodes `cp103`/`cp104`):
external reference recipe, same result

A colleague/reference-provided launch recipe was tried next, adapted to
this cluster: same native `mp`/no-Ray mechanism as the sixth attempt, but
adding `NCCL_DMABUF_ENABLE=0`, `VLLM_ALLREDUCE_USE_FLASHINFER=1`,
`VLLM_USE_RUST_FRONTEND=1` (new — vLLM's Rust-based frontend, previously
untested here), `VLLM_ENGINE_READY_TIMEOUT_S=3600`, `--load-format
fastsafetensors`, `--moe-backend auto`, `--kv-cache-dtype fp8`,
`--max-cudagraph-capture-size 256`, and an `--attention-config` block
(`mla_prefill_backend: FLASHINFER`, prefill query quantization on).
Adapted from the original recipe: HF hub model IDs swapped for this box's
local offline checkpoints (`HF_HUB_OFFLINE=1`), `--tensor-parallel-size`
raised from the recipe's 8 to 16 (the recipe's 8+nnodes=2 implies 4
GPUs/node; this cluster's B200 nodes have 8), `--max-model-len` pinned to
8192 instead of `auto`. Script: `native_mp_recipe_launch.sh`, logs:
`logs/native_recipe_rank{0,1}.log`.

**Reproduced the identical failure**, at the identical place, on a
completely fresh node pair: all 16 ranks completed cross-node NCCL
`Init COMPLETE` again, then the Gloo `barrier()` inside
`in_the_same_node_as` hung and hit the same 30-minute timeout
(`Timed out waiting 1800000ms for recv operation to complete` →
`RuntimeError: Application timeout caused pair closure`), on both nodes,
confirmed via matching log lines. As expected going in: none of this
recipe's new flags touch Gloo transport or distributed-collective
timeout configuration, so this result doesn't change the diagnosis —
it's an additional independent confirmation (third framework-adjacent
variant, fourth distinct node pair total across this investigation) that
the blocker is specifically Gloo's cross-node TCP transport, not
anything about launch flags, loader, quantization, or frontend choice.

## Next steps, in order of expected value

**Superseded a third time — see "Sixth attempt" above.** The evidence
now narrows the target dramatically: this is not about Ray, not about
parallelism topology, and not about NCCL transport — it's specifically
about **Gloo's cross-node TCP transport failing while NCCL's
succeeds**. Revised priority:

1. **Reproduce the Gloo hang in isolation, outside vLLM/SGLang entirely**
   (highest priority — cheap, fast, no multi-minute model load needed
   each iteration): a bare 2-process `torch.distributed` script across
   `cp99`/`cp100` doing `dist.init_process_group(backend="gloo", ...)`
   then a `barrier()` or `broadcast_object_list()`. If this hangs the
   same way, it's conclusively a cluster/Gloo-transport issue, fully
   decoupled from any model-serving framework. Compare directly against
   an identical script using `backend="nccl"` on the same node pair
   (expected to succeed, matching today's observation) to nail down the
   contrast in one script.
2. **Check whether `GLOO_SOCKET_IFNAME=brcm0` is actually being honored**
   by this PyTorch/Gloo build — the env var is set in every launch script,
   but Gloo's interface-selection code path differs from NCCL's, and a
   version-specific bug or precedence issue (e.g. Gloo preferring a
   different interface it auto-detects before consulting the env var)
   would produce exactly this symptom: NCCL on the right interface,
   Gloo on a dead/firewalled one.
3. **Firewall/port audit specifically for Gloo's TCP range** — Gloo uses
   plain TCP sockets (no RDMA/IB path like NCCL can), with its own port
   allocation behavior; if the cluster's inter-node firewall allows
   NCCL's ports/protocol but not Gloo's, that alone explains everything
   seen today across both frameworks.
4. **Ask whoever has gotten 2-node Kimi K3 (or any multi-node
   vLLM job) running on this cluster** whether they hit this, and if so
   how they worked around it — still a high-value, zero-cost question.
5. If none of the above resolves it: fall back to the single-node B200
   path (§7 above) purely to get the real OOM numbers, and otherwise use
   the known working 8xB300 node for any real Kimi K3 serving needs in
   the meantime.

## Artifacts

- `/import/ml-sc-scratch1/chenw/speculators/.scratch_kimi_launch/` — all launch
  scripts (`head_node_2x.sh`/`worker_node_2x.sh` for the 2-node attempts;
  `head_node_4x.sh` for the 4-node pure-TP=32 attempt;
  `head_node_4x_pp4tp8.sh` for the 4-node PP=4×TP=8/no-draft attempt;
  `native_mp_launch.sh` for the 6th attempt and `native_mp_recipe_launch.sh`
  for the 7th, both native `mp`/no-Ray, run identically on both nodes with
  only `$1` node-rank differing), the `pythonpath_shim/`/`cv2_only_shim/`,
  and the `debug_pg.py` Ray diagnostic referenced above. Logs:
  `logs/head_2x.log`/`worker_2x.log` (3rd attempt, 2-node), `logs/head_4x.log`
  (4th attempt, pure TP=32), `logs/head_4x_pp4tp8.log`/`worker_pp4tp8_<node>.log`
  (5th attempt, PP=4×TP=8, no DSpark), `logs/native_mp_rank{0,1}.log` (6th
  attempt, native mp, `cp99`/`cp100`, job 10072414), `logs/native_recipe_rank{0,1}.log`
  (7th attempt, native mp + external recipe flags, `cp103`/`cp104`, job
  10082774). Both the 6th and 7th attempts' allocations are fully
  released, verified via `squeue`.
- MiniMax-M3 cross-node validation (known-good reference point):
  `/import/ml-sc-scratch1/chenw/launch_minimax_pp2_sglang.sbatch` (original,
  `cp78`/`cp79`) and
  `/import/ml-sc-scratch1/chenw/speculators/.scratch_kimi_launch/launch_minimax_pp2_validate.sbatch`
  (fresh repro, `cp95`/`cp96`) — logs at
  `/import/ml-sc-scratch1/chenw/minimax-pp2-9495483.{out,err}` and
  `.scratch_kimi_launch/logs/minimax_pp2_validate-10063008.{out,err}`.
