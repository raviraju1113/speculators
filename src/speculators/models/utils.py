import logging
import warnings
from copy import deepcopy
from functools import cache, partial

import torch
from transformers import AutoConfig, PretrainedConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

logger = logging.getLogger(__name__)

# Verifier families whose final norm follows the Gemma convention
# `x_norm * (1 + w)` instead of the plain RMSNorm `x_norm * w`. Matched
# exactly against the verifier's effective `model_type` (after text_config
# extraction) — gemma3n and gemma4 dropped this convention, so prefix
# matching would over-include. Qwen3.5 is in here because transformers'
# Qwen3_5RMSNorm computes `output * (1.0 + weight)` and vLLM's is an alias
# of GemmaRMSNorm.
GEMMA_STYLE_FINAL_NORM_MODEL_TYPES = frozenset(
    (
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "recurrent_gemma",
    )
)


@cache
def _verifier_model_type(name_or_path: str) -> str | None:
    """The verifier's effective model_type, or None when unresolvable."""
    try:
        verifier_config = get_verifier_config(name_or_path)
    except (KeyError, OSError, ValueError):
        logger.warning(
            "Could not resolve a config for verifier %s; assuming the plain "
            "final-norm convention (x * w).",
            name_or_path,
        )
        return None
    return str(getattr(verifier_config, "model_type", "") or "").lower()


def uses_gemma_style_final_norm(config) -> bool:
    """Whether the verifier's final norm applies gain ``1 + w`` (not ``w``)."""
    verifier = getattr(getattr(config, "speculators_config", None), "verifier", None)
    name_or_path = getattr(verifier, "name_or_path", None)
    if not name_or_path:
        return False
    model_type = _verifier_model_type(name_or_path)
    return model_type is not None and model_type in GEMMA_STYLE_FINAL_NORM_MODEL_TYPES


def resolve_verifier_norm_class(config) -> type:
    """The RMSNorm class matching the verifier's final-norm weight convention.

    The frozen ``verifier_norm`` must apply the same gain convention the
    verifier was trained under (`x * (1 + w)` for the Gemma/Qwen3.5
    families, `x * w` otherwise), or the reconstructed verifier targets
    are silently mis-scaled.
    """
    if uses_gemma_style_final_norm(config):
        return Gemma3RMSNorm
    return Qwen3RMSNorm


def conditional_torch_compile(func=None, *args, **kwargs):
    if func is None:
        return partial(conditional_torch_compile, *args, **kwargs)
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        return torch.compile(func, *args, **kwargs)
    return func


def get_verifier_config(
    verifier_name_or_path: str,
    trust_remote_code: bool = False,
) -> PretrainedConfig:
    verifier_config = AutoConfig.from_pretrained(
        verifier_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config
    return verifier_config


def resolve_norm_eps(config: PretrainedConfig) -> float:
    """RMSNorm epsilon, under either of the two names HF configs use.

    Transformer configs expose ``rms_norm_eps``; SSM configs such as
    :class:`~transformers.models.mamba2.configuration_mamba2.Mamba2Config` call the
    same quantity ``layer_norm_epsilon``.

    :raises AttributeError: if the config exposes neither name.
    """
    for name in ("rms_norm_eps", "layer_norm_epsilon"):
        eps = getattr(config, name, None)
        if eps is not None:
            return eps
    raise AttributeError(
        f"{type(config).__name__} exposes neither 'rms_norm_eps' nor "
        "'layer_norm_epsilon'"
    )


DEFAULT_TARGET_LAYER_IDS_WARNING = (
    "--target-layer-ids is not explicitly set. Setting target "
    "layers to {target_layer_ids}. If custom target layers were used "
    "when launching vllm datagen, please set them explicitly."
)


#: Verifier model_types whose full-attention layers are DeepSeek-style MLA and can
#: therefore be drafted with a transformers ``deepseek_v3`` decoder layer.
MLA_VERIFIER_MODEL_TYPES = ("kimi_linear", "kimi_k3")

# Keep in sync with MTP deepseek_v3 registration (imported by model_definitions).
_DEEPSEEK_V3_MIN_TRANSFORMERS = "4.51.0"


def _require_deepseek_v3_config():
    """Import ``DeepseekV3Config``, or raise a version-aware error."""
    from importlib.metadata import version as pkg_version

    from packaging.version import Version

    installed = pkg_version("transformers")
    if Version(installed) < Version(_DEEPSEEK_V3_MIN_TRANSFORMERS):
        raise ImportError(
            "Kimi K3 / MLA verifier drafts map to DeepseekV3Config, which requires "
            f"transformers>={_DEEPSEEK_V3_MIN_TRANSFORMERS} (installed: {installed}). "
            f"Upgrade with: pip install 'transformers>={_DEEPSEEK_V3_MIN_TRANSFORMERS}'"
        )
    try:
        from transformers import DeepseekV3Config  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Failed to import DeepseekV3Config from transformers "
            f"(installed: {installed}; need >={_DEEPSEEK_V3_MIN_TRANSFORMERS}). "
            f"Upgrade with: pip install 'transformers>={_DEEPSEEK_V3_MIN_TRANSFORMERS}'"
        ) from e
    return DeepseekV3Config


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

    DeepseekV3Config = _require_deepseek_v3_config()

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
    num_layers = get_verifier_config(
        verifier_name_or_path,
        trust_remote_code=trust_remote_code,
    ).num_hidden_layers

    if target_layer_ids is None:
        explicit = False
        target_layer_ids = [2, num_layers // 2, num_layers - 3]
    else:
        explicit = True

    # Layer id ``num_layers`` (the final hidden state) is valid, matching the
    # ids scripts/launch_vllm.py emits with --include-last-layer.
    invalid = (
        not target_layer_ids
        or min(target_layer_ids) < 0
        or max(target_layer_ids) > num_layers
        or len(set(target_layer_ids)) != len(target_layer_ids)
    )
    if invalid:
        if explicit:
            raise ValueError(
                f"target_layer_ids must be distinct and within [0, {num_layers}] "
                f"for a verifier with {num_layers} hidden layers, "
                f"got {target_layer_ids}"
            )
        raise ValueError(
            f"Default target layer ids {target_layer_ids} are invalid for a verifier "
            f"with {num_layers} hidden layers; pass --target-layer-ids explicitly."
        )

    if not explicit:
        warnings.warn(
            DEFAULT_TARGET_LAYER_IDS_WARNING.format(target_layer_ids=target_layer_ids),
            stacklevel=3,
        )
    return target_layer_ids


def flatten_rope_parameters(config: PretrainedConfig) -> PretrainedConfig:
    """Flatten nested per-layer-type ``rope_parameters`` for rotary embedding init.

    Models like Laguna store separate rope configs per layer type
    (``sliding_attention``, ``full_attention``). Rotary embedding classes expect
    a flat dict with ``rope_type``/``rope_theta`` at the top level. This helper
    selects the ``sliding_attention`` variant when nested parameters are detected
    and returns a deep-copied config; otherwise returns the original unchanged.
    """
    rope_params = getattr(config, "rope_parameters", None)
    if not rope_params or "sliding_attention" not in rope_params:
        return config
    config = deepcopy(config)
    config.rope_parameters = rope_params["sliding_attention"]
    return config


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
