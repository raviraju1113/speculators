# vLLM: speculators P-EAGLE checkpoints crash at startup (no `peagle` branch in method detection)

**Version:** vLLM 0.28.0 (+cu129)

## Summary
A speculators-format **P-EAGLE** checkpoint cannot be served. vLLM's
speculative-method auto-detection has no `peagle` branch, so the checkpoint
falls through to the generic `draft_model` proposer, which does not pass
`hidden_states` to the draft. Startup then dies during warmup.

## Error
```
TypeError: Eagle3Qwen3ForCausalLM.forward() missing 1 required positional argument: 'hidden_states'
  File "vllm/v1/spec_decode/llm_base_proposer.py", line 1686, in dummy_run
    self.model(**kwargs)
RuntimeError: Engine core initialization failed.
```

## Cause
`vllm/config/speculative.py` auto-detects `eagle`, `eagle3`, `dflash`,
`dspark`, `medusa`, `mlp_speculator`, `mtp` — but not `peagle`, even though
`vllm/transformers_utils/configs/speculators/algos.py` has a registered
`@register_speculator("peagle")` handler that maps the checkpoint to
`PeagleQwen3ForCausalLM`/`PeagleLlamaForCausalLM` and sets `pard_token`.
With no branch, `method` stays `draft_model`, whose proposer is constructed
with `pass_hidden_states_to_model=False`, so `dummy_run` omits the argument
the eagle-family model requires.

## Workaround
Force the eagle3 proposer, which reads `pard_token` for parallel drafting:
```yaml
speculative_config:
  method: eagle3
```
Verified working: served correctly and produced sane acceptance.

## Suggested fix
Add a detection branch alongside the others, e.g.
```python
elif "peagle" in self.draft_model_config.model.lower() or any(
    a.startswith("Peagle") for a in self.draft_model_config.architectures
):
    self.method = "eagle3"   # P-EAGLE runs on the eagle3 proposer via pard_token
```

## Impact
Every speculators-format P-EAGLE checkpoint is unservable out of the box, and
the failure is an obscure TypeError rather than "unsupported method".
