# Design: Mooncake streaming for online training

**Status:** Proposed (design review) · **Scope:** large · **Author:** (fill in)

Adapt [TorchSpec](https://github.com/lightseekorg/TorchSpec)'s Mooncake transfer
layer so Speculators' **online training** streams target hidden states over the
network instead of handing them off through a shared filesystem.

## Motivation

Speculators online training already avoids the terabyte-scale *offline*
hidden-state dump: hidden states are generated on demand by the frozen target
and consumed immediately. But the current transport is a **shared filesystem**:

- **Producer:** vLLM's `ExampleHiddenStatesConnector` (configured in
  `scripts/launch_vllm.py` via `kv_transfer_config`) writes each sample's hidden
  states to `--hidden-states-path` (default `/tmp/hidden_states`).
- **Consumer:** `ArrowDataset._maybe_generate_hs` (`src/speculators/train/data.py`)
  triggers generation through the OpenAI client, reads the file, then deletes it
  (`on_generate=delete`).

This requires the vLLM server and the trainer to **share a filesystem** (same
node or a shared mount), which limits how independently inference and training
can scale. TorchSpec instead streams hidden-state tensors directly from inference
engines to training workers via **Mooncake** (`mooncake-transfer-engine`), with
no shared FS and no disk materialization — enabling inference and training to run
as independently-scaled groups (potentially across nodes/clusters).

Goal: offer Mooncake as an alternative transport for Speculators online training,
selectable via config, without removing the existing FS path.

> **Applicability.** On a single node the FS handoff already works and Mooncake's
> benefit is modest; the payoff is at multi-node / cross-cluster scale. This
> design keeps `--transport file` the default and adds `--transport mooncake`.

## Background: TorchSpec's Mooncake layer

`torchspec/transfer/mooncake/` (MIT, ~2000 LOC):

- `store.py` — `MooncakeHiddenStateStore(ABC)` wrapping `mooncake.store.MooncakeDistributedStore`;
  `setup()` connects to a **Mooncake master** + metadata server
  (`master_server_address`). `MooncakeMaster` is a **Ray actor**.
- `eagle_store.py` — `EagleMooncakeStore` with:
  - `put(key, hidden_states, input_ids, last_hidden_states, target=None) -> {shapes, dtypes}`
    (stores under `{key}_hs`, `{key}_ids`, `{key}_lhs`, `{key}_tgt`; async
    DtoH + background RDMA via `AsyncPutManager`).
  - `get(key, shapes, dtypes, device) -> Eagle3TargetOutput` (GPUDirect RDMA, or
    host-buffer/TCP fallback).
- `buffers.py` (host/GPU buffer pools, async put), `deferred_delete.py` (TTL/GC).

Key facts that shape this design:
- `get()` **requires shapes + dtypes**, which `put()` returns. TorchSpec passes
  that metadata producer→consumer through its **controller**. Speculators has no
  such controller, so we need a metadata side channel.
- The master is a **Ray actor** in TorchSpec. Speculators should not take a Ray
  dependency; use a **standalone Mooncake master** process instead.
- RDMA needs InfiniBand; a single PCIe-A100 box uses the **TCP host-buffer path**
  (works, but a same-node network hop).

## Proposed architecture

| Side | Today | Proposed (`--transport mooncake`) |
|---|---|---|
| Producer | vLLM `ExampleHiddenStatesConnector` → file | vLLM `MooncakeHiddenStatesConnector` → `store.put(key, …)` |
| Consumer | read file, delete | `store.get(key, shapes, dtypes, device)` |
| Coordination | file path encodes the sample | request-id key + shapes/dtypes metadata channel |
| Infra | none | standalone Mooncake master + metadata server |

### Components

1. **Vendored store** — copy `transfer/mooncake/{store,eagle_store,buffers,deferred_delete,helpers,utils}.py`
   into `src/speculators/train/transfer/mooncake/` (MIT; retain LightSeek
   copyright). **De-Ray**: replace `MooncakeMaster(RayActor)` with a thin helper
   that launches/points at a standalone `mooncake_master` process.
2. **Producer connector** — `MooncakeHiddenStatesConnector` (vLLM kv-connector
   plugin), adapted from `ExampleHiddenStatesConnector`: on each request, extract
   the aux hidden states (as today) and `store.put()` them keyed by the request
   id. Configured through `launch_vllm.py`'s `kv_transfer_config`.
3. **Consumer dataset** — `MooncakeHiddenStatesDataset` (or an `ArrowDataset`
   transport mode) in `src/speculators/train/data.py`: still requests generation
   via the OpenAI client, then `store.get(key, …)` instead of a file read; no
   explicit delete (Mooncake TTL / `deferred_delete`).
4. **Metadata channel** — carry per-sample `{shapes, dtypes}` from producer to
   consumer. Options (decide in review): (a) return them in the OpenAI response
   body, (b) a tiny metadata put in Mooncake under `{key}_meta`, (c) fixed shapes
   derived from the verifier config + seq len. (c) is simplest if layouts are
   deterministic.
5. **Config / launch** — `--transport {file,mooncake}` (default `file`) +
   `--mooncake-master <host:port>` on both `scripts/launch_vllm.py` and
   `scripts/train.py`; a `scripts/launch_mooncake_master.sh` helper.

### Key + lifecycle

- **Key** = the training sample's request id (must be identical on both sides).
  The trainer already drives generation per sample; thread that id into the vLLM
  request so the connector uses it as the store key.
- **Lifecycle** = put (producer, async) → get (consumer) → deferred delete
  (TTL), replacing the current read-then-`unlink`.

## Risks / open questions

1. **vLLM producer connector (highest risk).** Whether the kv-connector plugin
   API exposes the aux hidden states *and* a stable per-request key at the right
   point, for the target vLLM version. **De-risk in Phase 0/2.**
2. **Metadata channel.** No controller exists; pick option (a)/(b)/(c) above.
3. **Dependencies / ops.** Adds `mooncake-transfer-engine` and a master +
   metadata server to run; document single-node TCP setup.
4. **Testability.** Not verifiable on a CPU box; needs a GPU node with Mooncake.
   Also gated on the current CUDA-driver blocker (nothing online runs until
   that's fixed).
5. **Single-node value.** Modest vs. the FS path; keep FS the default.

## Phased plan

| Phase | Deliverable | Proves |
|---|---|---|
| **0 — spike** | vendor store (de-Ray) + standalone master + `put`/`get` smoke test with dummy tensors, no vLLM | deps + transport work on our hardware |
| **1 — consumer** | `MooncakeHiddenStatesDataset` + metadata channel; bridge producer (reads existing files, `put`s to Mooncake) | consumer path end-to-end |
| **2 — producer** | real `MooncakeHiddenStatesConnector`; keys wired trainer→vLLM→store | true streaming, no FS |
| **3 — polish** | `--transport` flags, master launch helper, drop FS in mooncake mode, docs | production-ready, config-selectable |

## Alternatives considered

- **Keep FS handoff (status quo).** Zero cost; sufficient single-node. Chosen as
  the default; Mooncake is opt-in.
- **Use TorchSpec directly.** It's a Ray+Mooncake, multi-node framework with no
  Gemma-4 support; running it wholesale is a bigger lift than adding its
  transport to Speculators.

## Dependencies

- `mooncake-transfer-engine>=0.3.10.post1` (TorchSpec pins this).
- A Mooncake master + metadata server (standalone; no Ray).
- MIT-licensed vendored code from TorchSpec (`transfer/mooncake/`) — retain the
  LightSeek copyright header.
