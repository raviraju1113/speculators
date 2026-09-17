import warnings
from functools import partial

import torch
from transformers import AutoConfig, PretrainedConfig


def conditional_torch_compile(func=None, *args, **kwargs):
    if func is None:
        return partial(conditional_torch_compile, *args, **kwargs)
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        return torch.compile(func, *args, **kwargs)
    return func


def get_verifier_config(
    verifier_name_or_path: str, trust_remote_code: bool = False
) -> PretrainedConfig:
    verifier_config = AutoConfig.from_pretrained(
        verifier_name_or_path, trust_remote_code=trust_remote_code
    )
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config
    return verifier_config


#: Verifier model_types whose full-attention layers are DeepSeek-style MLA and can
#: therefore be drafted with a transformers ``deepseek_v3`` decoder layer.
MLA_VERIFIER_MODEL_TYPES = ("kimi_linear", "kimi_k3")


def translate_verifier_config_for_draft(
    verifier_config: PretrainedConfig,
) -> PretrainedConfig:
    """Translate a trust-remote-code verifier config into a registered draft config.

    Kimi K3 (``kimi_linear``) is a hybrid KDA/MLA model whose modeling code lives
    outside transformers, so its config cannot be used as a draft
    ``transformer_layer_config`` (``AutoConfig.for_model`` cannot re-instantiate it
    on checkpoint reload, and no decoder-layer class is importable). Its
    full-attention layers are DeepSeek-V3-style MLA, so the draft uses a *dense*
    single-layer ``DeepseekV3Config`` carrying the verifier's MLA geometry --
    the same mapping TorchSpec used for the Kimi-K3 EAGLE3 draft.

    Non-Kimi configs pass through unchanged.
    """
    if verifier_config.model_type not in MLA_VERIFIER_MODEL_TYPES:
        return verifier_config

    from transformers import DeepseekV3Config  # noqa: PLC0415

    return DeepseekV3Config(
        vocab_size=verifier_config.vocab_size,
        hidden_size=verifier_config.hidden_size,
        intermediate_size=verifier_config.intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=verifier_config.num_attention_heads,
        num_key_value_heads=verifier_config.num_key_value_heads,
        q_lora_rank=verifier_config.q_lora_rank,
        kv_lora_rank=verifier_config.kv_lora_rank,
        qk_nope_head_dim=verifier_config.qk_nope_head_dim,
        qk_rope_head_dim=verifier_config.qk_rope_head_dim,
        v_head_dim=verifier_config.v_head_dim,
        # Kimi's "situ" activation is not in transformers' ACT2FN; the TorchSpec
        # draft used silu, and drafts need not mirror the verifier MLP exactly.
        hidden_act="silu",
        rms_norm_eps=verifier_config.rms_norm_eps,
        # Kimi MLA layers are NoPE (no rope_theta in the shipped config); the
        # draft still needs positional information, so use the KimiLinearConfig
        # class default (10000.0) -- again matching the TorchSpec draft.
        rope_theta=getattr(verifier_config, "rope_theta", None) or 10000.0,
        rope_scaling=None,
        max_position_embeddings=verifier_config.max_position_embeddings,
        # Dense draft: layers below first_k_dense_replace use the dense MLP, so
        # a high threshold means no MoE regardless of layer_idx (n_routed_experts
        # keeps its class default; transformers 5.x rejects None for it).
        first_k_dense_replace=10_000,
        attention_bias=False,
        tie_word_embeddings=False,
        bos_token_id=verifier_config.bos_token_id,
        eos_token_id=verifier_config.eos_token_id,
        pad_token_id=getattr(verifier_config, "pad_token_id", None),
    )


DEFAULT_TARGET_LAYER_IDS_WARNING = (
    "--target-layer-ids is not explicitly set. Setting target "
    "layers to {target_layer_ids}. If custom target layers were used "
    "when launching vllm datagen, please set them explicitly."
)


def resolve_target_layer_ids(
    target_layer_ids: list[int] | None,
    verifier_name_or_path: str,
    trust_remote_code: bool = False,
) -> list[int]:
    if target_layer_ids is not None:
        return target_layer_ids

    num_layers = get_verifier_config(
        verifier_name_or_path, trust_remote_code=trust_remote_code
    ).num_hidden_layers
    target_layer_ids = [2, num_layers // 2, num_layers - 3]
    warnings.warn(
        DEFAULT_TARGET_LAYER_IDS_WARNING.format(target_layer_ids=target_layer_ids),
        stacklevel=3,
    )
    return target_layer_ids


def resolve_draft_intermediate_size(verifier_config: PretrainedConfig) -> int:
    """Resolve a dense draft MLP ``intermediate_size`` from a verifier config.

    The draft is an independent small *dense* decoder, so its FFN width is a design
    choice rather than something to reconcile with the verifier's routed capacity:

    * Dense verifiers expose ``intermediate_size`` directly; the draft mirrors it.
    * MoE verifiers have no dense ``intermediate_size`` (their FFN is a routed set of
      small experts), so the draft falls back to the widely used ``3 * hidden_size``
      gated-MLP ratio -- the Qwen3 dense convention that the dflash draft decoder
      follows. Pass ``--draft-config`` to set it explicitly instead.

    :raises ValueError: when the verifier config exposes neither ``intermediate_size``
        nor ``hidden_size`` (degenerate config; pass ``--draft-config``).
    """
    dense = getattr(verifier_config, "intermediate_size", None)
    if dense is not None:
        return int(dense)

    hidden_size = getattr(verifier_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(
            "Verifier config exposes neither `intermediate_size` nor `hidden_size`, "
            "so a draft intermediate_size cannot be inferred. Pass --draft-config to "
            "set the draft architecture explicitly."
        )

    intermediate_size = 3 * int(hidden_size)
    warnings.warn(
        "Verifier config has no dense intermediate_size (likely MoE); using draft "
        f"intermediate_size={intermediate_size} (3 x hidden_size = {hidden_size}). "
        "Pass --draft-config to override.",
        stacklevel=3,
    )
    return intermediate_size
