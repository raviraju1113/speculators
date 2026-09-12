# vLLM prototype patches

vLLM changes needed to *serve* an EAGLE-3 speculator whose mixer is Mamba2. Kept here
so the work is reviewable and upstreamable rather than living only in `site-packages`.

| file | what it does |
|---|---|
| `0001-eagle3-mamba2-draft-metadata.patch` | (a) threads `is_prefilling` + `seq_lens_cpu_upper_bound` through the drafting loop's own metadata builder; (b) lets Mamba's full-CUDAGraph capture accept a draft's `query_len == 1` decode shape |
| `orig/` | pristine copies of the touched files (local only, gitignored -- see `.gitignore` to recreate) |

Everything that *can* be done from outside vLLM already is, in
`speculators/integrations/vllm/` (registered via the `vllm.general_plugins` entry
point): the model class, the Eagle3 `model_type -> architecture` map entry, and the
`init_model_state` override. Only the drafting-loop change needs to touch vLLM itself.

## Apply / revert

```bash
VLLM=$(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')
patch -p1 -d "$VLLM" < 0001-eagle3-mamba2-draft-metadata.patch   # apply
# revert: copy the matching file back out of orig/
```

## Serving flags this prototype requires

```
--mamba-block-size 8192        # == max_model_len; see below
--no-enable-prefix-caching     # else "block_size must be divisible by hash_block_size"
--gpu-memory-utilization 0.65  # SSD scan intermediates are not in vLLM's mem profile
--max-num-batched-tokens 4096  # >= max_tokens_per_mm_item (2496) for the Gemma-4 VLM
--speculative-config '{"method": "eagle3", ...}'   # method is NOT auto-detected

CUDA graphs work with the patch applied -- `--enforce-eager` is NOT required.
```

Each corresponds to a gap worth fixing upstream:

1. **`method` is inferred from the checkpoint *path string*** —
   `elif "eagle3" in draft_model_config.model.lower()`. There is no
   speculators-config-aware inference, so a correctly-formatted speculators checkpoint
   whose directory is not named `*eagle3*` silently degrades to `method='draft_model'`
   and the draft is called as a standalone LM. Should read `speculators_model_type`.
2. **`mamba_block_size` is only derived for hybrid *targets***
   (`HybridAttentionMambaModelConfig.verify_and_update_config`), yet
   `MambaBase.get_kv_cache_spec` asserts it is set. A Mamba *draft* on a
   pure-attention target leaves it `None`. It should be derived from the draft too.
3. **`validate_mamba_block_size` + prefix caching** force
   `mamba_block_size == max_model_len` unless prefix caching is on, and with prefix
   caching on the block-hash divisibility assert fires. A draft-owned Mamba state has
   no reason to participate in target prefix caching at all.
4. **Draft chunked-scan intermediates are unaccounted for** in memory profiling: at
   `gpu_memory_utilization 0.9` warmup OOM'd wanting 32 GiB inside `_bmm_chunk_fwd`.

## Status

Works end to end at TP=4 on Gemma-4-31B with a trained Mamba2 Eagle-3 draft, **with
CUDA graphs enabled**. Aggregate `accept_len` 2.51-2.53 (rate ~0.51) at
`num_speculative_tokens=3` over a mixed six-prompt set, ranging 1.90 (translation) to
2.87 (arithmetic). Bit-stable within a server instance.

### Eager vs CUDA graphs: no discrepancy (earlier claim retracted)

Measured per request on a six-prompt set, the two paths agree: aggregate `accept_len`
2.530 (eager) vs 2.512 (graphs), identical on four of six prompts. An earlier note here
claiming 2.484 vs 2.857 was arithmetic error on a bulk metrics read and is withdrawn.
Neither path mishandles the recurrent state.

### Losslessness: confirmed

Against target-only with otherwise identical flags, 5 of 6 prompts are byte-identical.
The single mismatch first diverges at token 7, where the target's top-2 logprobs are
**exactly tied** (`gap = 0.000000 nats`), so the difference is argmax tie-breaking under
a different floating-point reduction order, not the draft. Neither path is consistently
favoured, and output is bit-stable within a server instance.

### Passed on: prefix caching

Not a flag fix. `KVCacheCoordinator` requires every group's `block_size` to be a
multiple of `hash_block_size`, so an 8192-token Mamba page cannot coexist with 16-token
attention blocks. Making it work means snapshotting recurrent state at block boundaries
-- which is what vLLM's experimental `mamba_cache_mode` ("all" / "align") is for, and a
16-token granularity would multiply draft state memory by ~512x. Genuine design work,
orthogonal to whether a recurrent drafter works at all.

### Still open

A systematic losslessness check (greedy target vs speculative, token-for-token across a
suite) and throughput numbers from the eval harness.
