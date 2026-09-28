"""DSpark with the optional DFlash2 local convolution inside each draft layer."""

from typing import Any

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.dflash2.model_definitions import Qwen3DFlash2DecoderLayer
from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig


def _tiny_config(**overrides) -> DSparkSpeculatorConfig:
    transformer_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    values: dict[str, Any] = {
        "transformer_layer_config": transformer_config,
        "draft_vocab_size": 64,
        "block_size": 4,
        "aux_hidden_state_layer_ids": [0, 1],
        "mask_token_id": 0,
        "markov_rank": 8,
    }
    values.update(overrides)
    return DSparkSpeculatorConfig(**values)


def test_conv_disabled_by_default_keeps_plain_dflash_layers():
    config = _tiny_config()
    assert config.conv_kernel_size == 0
    model = DSparkDraftModel(config)
    for layer in model.layers:
        assert type(layer) is Qwen3DFlashDecoderLayer
    assert not any("attention_conv" in name for name, _ in model.named_parameters())


def test_conv_enabled_builds_dflash2_layers_at_identity():
    model = DSparkDraftModel(_tiny_config(conv_kernel_size=2, conv_group_size=4))
    for layer in model.layers:
        assert isinstance(layer, Qwen3DFlash2DecoderLayer)
        for conv in (layer.attention_conv, layer.mlp_conv):
            assert conv.kernel_size == 2
            assert conv.group_size == 4
            assert conv.block_size == 4
            # Identity init: tap 0 is all ones, later taps and the dynamic
            # projection are zero, so the layer equals a plain DFlash layer.
            assert torch.equal(conv.base_kernel[:, 0], torch.ones(2, 16))
            assert torch.count_nonzero(conv.base_kernel[:, 1:]) == 0
            assert torch.count_nonzero(conv.kernel_projection.weight) == 0
    # Heads are untouched by the conv option.
    assert model.markov_head is not None
    assert model.confidence_head is not None


def test_conv_model_loads_plain_dspark_state_dict_with_only_conv_keys_missing():
    plain = DSparkDraftModel(_tiny_config())
    conv = DSparkDraftModel(_tiny_config(conv_kernel_size=2, conv_group_size=4))
    missing, unexpected = conv.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all("attention_conv" in k or "mlp_conv" in k for k in missing)


def test_conv_config_round_trips():
    config = _tiny_config(conv_kernel_size=2, conv_group_size=4)
    restored = DSparkSpeculatorConfig(**config.model_dump(exclude_none=True))
    assert restored.conv_kernel_size == 2
    assert restored.conv_group_size == 4
    assert restored.speculators_model_type == "dspark"
