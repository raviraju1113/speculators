# SPDX-License-Identifier: Apache-2.0
"""vLLM draft model for an EAGLE-3 speculator whose mixer is Mamba2.

Mirrors ``vllm.model_executor.models.qwen3_eagle3`` (the transformer arm of the same
experiment) so the two are directly comparable, with the attention block swapped for
:class:`~vllm.model_executor.layers.mamba.mamba_mixer2.MambaMixer2`.

Why this file exists
--------------------
vLLM's Eagle3 config translator maps ``transformer_layer_config.model_type`` to a draft
architecture and only knows ``llama`` and ``qwen3``; a Mamba2 draft has no entry and no
model class. ``MambaMixer2`` already subclasses ``MambaBase(AttentionLayerBase)`` and
returns a ``MambaSpec`` from ``get_kv_cache_spec`` (including
``num_speculative_blocks``), so the drafting machinery -- which discovers draft layers
through ``AttentionLayerBase`` and requires them to share one KV-cache group --
accommodates a recurrent mixer without modification. What was missing is only the model
definition and the registry wiring.

Deployment note
---------------
``MambaBase.get_kv_cache_spec`` asserts ``cache_config.mamba_block_size is not None``.
That field is populated by the *target* model's ``verify_and_update_config``, which only
runs for hybrid/mamba architectures -- a pure-attention target such as Gemma-4 leaves it
unset. Pass ``--mamba-block-size <max_model_len>`` when serving until vLLM derives it
from the draft as well.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import HasInnerState
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
)

__all__ = ["Eagle3Mamba2ForCausalLM"]


def _norm_eps(config) -> float:
    """Mamba2Config spells the RMSNorm epsilon ``layer_norm_epsilon``."""
    for name in ("rms_norm_eps", "layer_norm_epsilon"):
        eps = getattr(config, name, None)
        if eps is not None:
            return eps
    return 1e-5


class Mamba2Eagle3DecoderLayer(nn.Module):
    """EAGLE-3 draft block: norm -> Mamba2 mixer -> residual, optionally + SwiGLU MLP.

    The ``W_FC`` projection is fused into the block by widening ``in_proj`` to
    ``2 * hidden_size``, matching how the transformer arms widen ``qkv_proj``. A SwiGLU
    MLP is appended when the config carries an ``intermediate_size``, which is how the
    capacity-matched arm is distinguished from the plain one.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config=None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        config = config or vllm_config.speculative_config.draft_model_config.hf_config
        quant_config = get_draft_quant_config(vllm_config)
        hidden_size = config.hidden_size
        eps = _norm_eps(config)
        self.layer_idx = layer_idx

        intermediate = int(config.expand * hidden_size)
        self.mixer = MambaMixer2(
            hidden_size=hidden_size,
            ssm_state_size=config.state_size,
            conv_kernel_size=config.conv_kernel,
            intermediate_size=intermediate,
            use_conv_bias=config.use_conv_bias,
            use_bias=config.use_bias,
            n_groups=config.n_groups,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            rms_norm_eps=eps,
            activation="silu",
            model_config=vllm_config.speculative_config.draft_model_config,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "mixer"),
        )

        # Fuse W_FC into the block: layer 0 consumes cat([embeds, hidden]).
        if layer_idx == 0:
            self._widen_in_proj(hidden_size, intermediate, quant_config, prefix)

        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.hidden_norm = RMSNorm(hidden_size, eps=eps)

        mlp_intermediate = getattr(config, "intermediate_size", None)
        if mlp_intermediate:
            # Imported lazily so the plain arm does not require Qwen3.
            from vllm.model_executor.models.qwen3 import Qwen3MLP

            self.mlp = Qwen3MLP(
                hidden_size=hidden_size,
                intermediate_size=mlp_intermediate,
                hidden_act=getattr(config, "hidden_act", "silu"),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "mlp"),
            )
            self.post_mixer_layernorm = RMSNorm(hidden_size, eps=eps)
        else:
            self.mlp = None
            self.post_mixer_layernorm = None

        self.norm_before_residual = getattr(config, "norm_before_residual", False)

    def _widen_in_proj(self, hidden_size, intermediate, quant_config, prefix) -> None:
        """Widen ``in_proj`` to a ``2 * hidden_size`` input, same output split.

        Only the ``n_groups % tp_size == 0`` layout is handled. The other branch uses a
        bespoke interleaved sharded weight loader that this widening would invalidate;
        for the shapes in use (``n_groups=8``, TP in {1,2,4,8}) it never applies.
        """
        mixer = self.mixer
        if not isinstance(mixer.in_proj, MergedColumnParallelLinear):
            raise NotImplementedError(
                "Eagle3 Mamba2 draft requires n_groups % tensor_parallel_size == 0 "
                f"(n_groups={mixer.n_groups}); got in_proj of type "
                f"{type(mixer.in_proj).__name__}, whose interleaved sharded weight "
                "loader is incompatible with the Eagle3 2x-width input."
            )
        mixer.in_proj = MergedColumnParallelLinear(
            input_size=2 * hidden_size,  # previous: hidden_size
            output_sizes=[
                intermediate,
                intermediate,
                mixer.groups_ssm_state_size,
                mixer.groups_ssm_state_size,
                mixer.num_heads,
            ],
            bias=mixer.in_proj.bias is not None,
            quant_config=quant_config,
            prefix=maybe_prefix(maybe_prefix(prefix, "mixer"), "in_proj"),
        )

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_idx == 0:
            embeds = self.input_layernorm(embeds)
            normed = self.hidden_norm(hidden_states)
            residual = normed if self.norm_before_residual else hidden_states
            mixer_input = torch.cat([embeds, normed], dim=-1)
        else:
            mixer_input, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.mixer(mixer_input)

        if self.mlp is not None:
            hidden_states, residual = self.post_mixer_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Mamba2Eagle3Model(nn.Module):
    # Training writes the mixer under `layers.0.mixer.*`, which already matches vLLM's
    # naming; only A_log -> A differs (vLLM stores the log-parameterized value under
    # `A` and exponentiates it in the kernel).
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".A_log": ".A", "midlayer.": "layers.0."},
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )

    def __init__(
        self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = ""
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)
        eps = _norm_eps(self.config)
        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleList(
            [
                Mamba2Eagle3DecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{idx + start_layer_id}"),
                    config=self.config,
                    layer_idx=idx,
                )
                for idx in range(self.config.num_hidden_layers)
            ]
        )

        # Fusion of the target's auxiliary hidden states.
        self.use_aux_hidden_state = True
        aux_ids = getattr(self.config, "eagle_aux_hidden_state_layer_ids", None)
        self.num_aux_layers = len(aux_ids) if aux_ids else 3
        target_hidden_size = getattr(
            self.config, "target_hidden_size", self.config.hidden_size
        )
        self.fc_input_size = target_hidden_size * self.num_aux_layers
        self.norm_before_fc = bool(getattr(self.config, "norm_before_fc", False))
        self.input_norm = (
            RMSNorm(self.fc_input_size, eps=eps) if self.norm_before_fc else None
        )
        self.fc_norm = (
            nn.ModuleList(
                [
                    RMSNorm(target_hidden_size, eps=eps)
                    for _ in range(self.num_aux_layers)
                ]
            )
            if getattr(self.config, "fc_norm", False)
            else None
        )
        self.fc = ReplicatedLinear(
            input_size=self.fc_input_size,
            output_size=self.config.hidden_size,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "fc"),
            return_bias=False,
        )

        self.norm_output = getattr(self.config, "norm_output", False)
        self.norm = RMSNorm(self.config.hidden_size, eps=eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        assert hidden_states.shape[-1] == input_embeds.shape[-1]

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                embeds=input_embeds,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, hidden_prenorm = self.norm(hidden_states, residual)
        aux_output = hidden_states if self.norm_output else hidden_prenorm
        return hidden_states, aux_output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )


class Eagle3Mamba2ForCausalLM(nn.Module, HasInnerState):
    """Eagle3 speculator with a Mamba2 mixer, for vLLM's drafting loop."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.config.target_layer_count = target_layer_num

        self.model = Mamba2Eagle3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            quant_config=get_draft_quant_config(vllm_config),
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=getattr(self.config, "logit_scale", 1.0),
        )
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits
        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (logits.shape[0], self.config.vocab_size), -torch.inf
        )
        logits_new[:, targets] = logits
        return logits_new

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        if self.model.norm_before_fc:
            hidden_states = self.model.input_norm(hidden_states)
        if self.model.fc_norm is not None:
            chunks = hidden_states.chunk(self.model.num_aux_layers, dim=-1)
            hidden_states = torch.cat(
                [
                    norm(chunk)
                    for norm, chunk in zip(self.model.fc_norm, chunks, strict=True)
                ],
                dim=-1,
            )
        return self.model.fc(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for orig_name, loaded_weight in weights:
            name = orig_name
            if "t2d" in name or "verifier_" in name:
                # t2d is the inverse map (unused at inference); verifier_* are the
                # frozen target tensors the trainer keeps for the distillation loss.
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight

        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.norm_before_fc:
            skip_substrs.append("input_norm.")
        AutoWeightsLoader(
            self, skip_prefixes=None, skip_substrs=skip_substrs
        ).load_weights(model_weights.items())
