"""vLLM integration for speculator architectures vLLM does not ship natively.

Registered through vLLM's ``vllm.general_plugins`` entry point (see
``pyproject.toml``), so no files under ``site-packages/vllm`` are modified and the
wiring survives a vLLM reinstall.
"""

__all__ = ["register"]

_EAGLE3_MAMBA2_ARCH = "Eagle3Mamba2ForCausalLM"


def _patch_eagle3_algo() -> None:
    """Teach vLLM's Eagle3 config translator about ``model_type: mamba2``.

    ``algos.update_eagle3`` resolves the draft architecture through a dict literal
    built inside the function body, so it cannot be reached and extended directly.
    Instead the registered handler is wrapped.

    The wrapper delegates first. The upstream function assigns every Eagle3 field it
    knows about *before* consulting its architecture map, so by the time it raises on
    an unrecognized ``model_type`` those fields are already populated on
    ``pre_trained_config`` and only the architecture (and the aux-layer ids, which it
    sets after the map lookup) remain. Delegating this way means new upstream fields
    are picked up automatically instead of silently going missing here.
    """
    from vllm.transformers_utils.configs.speculators import algos

    registry = algos.SUPPORTED_SPECULATORS_TYPES
    if "eagle3" not in registry:
        raise RuntimeError(
            "vllm.transformers_utils.configs.speculators.algos has no 'eagle3' "
            "handler to extend; vLLM's internals have changed."
        )
    upstream = registry["eagle3"]
    if getattr(upstream, "_speculators_mamba2_patched", False):
        return

    def update_eagle3(config_dict: dict, pre_trained_config: dict) -> None:
        try:
            upstream(config_dict, pre_trained_config)
        except ValueError:
            if pre_trained_config.get("model_type") != "mamba2":
                raise
            pre_trained_config["architectures"] = [_EAGLE3_MAMBA2_ARCH]
            aux_ids = config_dict.get("eagle_aux_hidden_state_layer_ids")
            if aux_ids:
                pre_trained_config["eagle_aux_hidden_state_layer_ids"] = aux_ids

    update_eagle3._speculators_mamba2_patched = True  # noqa: SLF001
    registry["eagle3"] = update_eagle3


def _draft_has_mamba_layers(vllm_config) -> bool:
    """Whether the configured speculator's draft is one of our recurrent arms."""
    spec = getattr(vllm_config, "speculative_config", None)
    draft_cfg = getattr(spec, "draft_model_config", None) if spec else None
    hf_config = getattr(draft_cfg, "hf_config", None) if draft_cfg else None
    if hf_config is None:
        return False
    if getattr(hf_config, "model_type", None) == "mamba2":
        return True
    return _EAGLE3_MAMBA2_ARCH in (getattr(draft_cfg, "architectures", None) or [])


def _patch_model_state_selection() -> None:
    """Make the target's runner use the mamba-aware ModelState for a mamba draft.

    ``init_model_state`` chooses ``MambaHybridModelState`` from
    ``model_config.is_hybrid``, which describes the *target* architecture. With a
    pure-attention target such as Gemma-4 the runner therefore gets
    ``DefaultModelState``, which does not populate ``is_prefilling`` on
    ``CommonAttentionMetadata`` -- yet the runner still builds metadata for *every*
    KV-cache group, including the draft's Mamba group, whose builder asserts on it
    (``mamba_attn.py``: ``assert is_prefilling is not None``).

    ``is_hybrid`` is a read-only property derived from registry info, so the selection
    function is wrapped instead: when the draft contributes Mamba layers, pick the
    mamba-aware state. Its ``is_prefilling`` derivation is computed from the input
    batch and does not assume the *target* owns the Mamba layers.
    """
    from vllm.v1.worker.gpu import model_states

    upstream = model_states.init_model_state
    if getattr(upstream, "_speculators_mamba2_patched", False):
        return

    def init_model_state(vllm_config, model, encoder_cache, device):
        if not vllm_config.model_config.is_hybrid and _draft_has_mamba_layers(
            vllm_config
        ):
            from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
                MambaHybridModelState,
            )

            return MambaHybridModelState(vllm_config, model, encoder_cache, device)
        return upstream(vllm_config, model, encoder_cache, device)

    init_model_state._speculators_mamba2_patched = True  # noqa: SLF001
    model_states.init_model_state = init_model_state

    # The runner imports the symbol directly, so rebind it there too.
    try:
        from vllm.v1.worker.gpu import model_runner

        if getattr(model_runner, "init_model_state", None) is upstream:
            model_runner.init_model_state = init_model_state
    except ImportError:
        pass


def register() -> None:
    """Register speculators' out-of-tree vLLM draft architectures.

    Called by vLLM at startup via the ``vllm.general_plugins`` entry point. A Mamba2
    Eagle3 draft needs two things: the model class in vLLM's ``ModelRegistry``, and an
    entry in the Eagle3 config translator's ``model_type -> architecture`` map.

    Failures are raised rather than swallowed -- a silently unregistered architecture
    resurfaces later as a confusing "unsupported model_type" error at config load.
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        _EAGLE3_MAMBA2_ARCH,
        f"speculators.integrations.vllm.mamba2_eagle3:{_EAGLE3_MAMBA2_ARCH}",
    )
    _patch_eagle3_algo()
    _patch_model_state_selection()
