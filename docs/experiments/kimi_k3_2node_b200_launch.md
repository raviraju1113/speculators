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

**Not yet done**: re-attaching the DSpark speculative draft, and the
Mooncake KV-cache tier (needed only for large prefix-cache-sharing
workloads, per ruobingw's `RECIPE.md`). Both are described in ruobingw's
`RECIPE.md`/`serve_k3_host.sh` and should layer on cleanly now that the
base topology works.

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
