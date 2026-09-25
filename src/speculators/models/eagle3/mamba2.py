"""Mamba2 (SSM) draft block for EAGLE-3.

The EAGLE-3 draft head normally runs one transformer decoder layer. This module
provides a Mamba2 alternative whose per-sequence state is a constant ~5.6 MB
recurrent state instead of a KV cache that grows linearly with context.

Two arms are provided:

* :class:`Mamba2DecoderEagle3FirstLayer` -- norm -> Mamba2 mixer -> residual.
* :class:`Mamba2MLPDecoderEagle3FirstLayer` -- the same plus a SwiGLU MLP, so the
  block's parameter count can be matched against a transformer draft and the
  architecture effect separated from the capacity effect.

Both follow the same convention as the transformer first layers in
``model_definitions.py``: the ``W_FC`` projection is fused into the block by widening
its input projection to ``2 * hidden_size``, and the layer receives
``cat([embeds, hidden])`` which it splits, normalizes and recombines.

Why the mixer is subclassed rather than used directly
-----------------------------------------------------
The trainer packs many documents into one sequence (see
``speculators.train.data.CollateFn``). Attention blocks cross-document leakage with a
document mask; a recurrent scan has no mask, so state would bleed from one document
into the next. ``mamba_chunk_scan_combined`` supports this through its ``seq_idx``
argument, but ``transformers`` hardcodes ``seq_idx=None`` in every call, so the stock
:class:`~transformers.models.mamba2.modeling_mamba2.Mamba2Mixer` cannot express
document boundaries. :class:`Eagle3Mamba2Mixer` overrides only the packed-sequence
scan path to thread ``seq_idx`` through; the single-token decode path is left to the
upstream implementation so serving behaviour stays identical to ``transformers``.

Document boundaries that are not multiples of ``chunk_size`` leave a small numerical
residue in the one chunk that straddles the boundary (the scan is still correct, but
the arithmetic is reordered). Aligning boundaries to ``chunk_size`` makes the
isolation exact.
"""

from typing import Any

import torch
from torch import nn
from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
from transformers.models.mamba2.modeling_mamba2 import (
    Mamba2Mixer,
    Mamba2RMSNorm,
    apply_mask_to_padding_states,
)

from speculators.models.utils import resolve_norm_eps

__all__ = [
    "Eagle3Mamba2Mixer",
    "Mamba2DecoderEagle3FirstLayer",
    "Mamba2MLPDecoderEagle3FirstLayer",
    "build_mamba2_draft_config",
]


def seq_idx_from_document_ids(document_ids: torch.Tensor) -> torch.Tensor:
    """Convert the trainer's ``document_ids`` into a ``seq_idx`` for the scan.

    ``document_ids`` marks padding with ``-1``. Padding must not be folded into the
    last real document, so it is given its own trailing segment id.

    :param document_ids: ``[batch, seq_len]`` document index per position.
    :return: contiguous ``int32`` ``[batch, seq_len]`` segment ids starting at 0.
    """
    if document_ids.dim() == 1:
        document_ids = document_ids.unsqueeze(0)
    # Kept as a device tensor: calling .item() here would force a host sync and break
    # the dynamo graph of the compiled Eagle3 forward.
    pad_id = document_ids.amax() + 1
    return (
        torch.where(document_ids < 0, pad_id, document_ids).to(torch.int32).contiguous()
    )


class Eagle3Mamba2Mixer(Mamba2Mixer):
    """Mamba2 mixer with a ``2 * hidden_size`` input and document-aware scanning.

    :param config: Mamba2 configuration for the draft block.
    :param layer_idx: index of this layer, used to address the cache.
    """

    def __init__(self, config: Mamba2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # Fuse W_FC into the block: the layer is handed cat([embeds, hidden]).
        self.in_proj = nn.Linear(
            2 * config.hidden_size,  # previous: config.hidden_size
            self.in_proj.out_features,
            bias=config.use_bias,
        )

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        cache_params: Any = None,
        attention_mask: torch.Tensor | None = None,
        seq_idx: torch.Tensor | None = None,
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        is_decoding = cache_params is not None and cache_params.has_previous_state(
            self.layer_idx
        )
        # Only the packed multi-document training path needs custom handling; leave
        # decode (and the unpacked case) to the upstream implementation.
        if seq_idx is None or is_decoding:
            return super().forward(
                hidden_states, cache_params=cache_params, attention_mask=attention_mask
            )
        return self._packed_scan_forward(hidden_states, seq_idx, attention_mask)

    def _packed_scan_forward(
        self,
        hidden_states: torch.Tensor,
        seq_idx: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Chunked scan over a packed sequence, isolating documents via ``seq_idx``.

        Mirrors the non-fused prefill branch of ``Mamba2Mixer.cuda_kernels_forward``,
        but forwards ``seq_idx`` to the conv and scan kernels so state does not cross
        document boundaries. Single-capital names (A, B, C, D) are the SSM matrices and
        deliberately match the upstream implementation and the Mamba2 paper.
        """
        # Imported lazily: these are optional extras (pip install speculators[mamba]).
        from causal_conv1d import causal_conv1d_fn  # noqa: PLC0415
        from mamba_ssm.ops.triton.ssd_combined import (  # noqa: PLC0415
            mamba_chunk_scan_combined,
        )

        batch_size, seq_len, _ = hidden_states.shape
        groups_state = self.n_groups * self.ssm_state_size

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        projected_states = self.in_proj(hidden_states)
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * groups_state
            - self.num_heads
        ) // 2
        _, _, gate, hidden_states_B_C, dt = projected_states.split(  # noqa: N806
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads],
            dim=-1,
        )

        # Depthwise causal conv. seq_idx keeps the kernel's window from reaching back
        # into the previous document.
        hidden_states_B_C = causal_conv1d_fn(  # noqa: N806
            x=hidden_states_B_C.transpose(1, 2),
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            seq_idx=seq_idx,
            activation=self.activation,
        ).transpose(1, 2)
        hidden_states_B_C = apply_mask_to_padding_states(  # noqa: N806
            hidden_states_B_C, attention_mask
        )

        ssm_input, B, C = torch.split(  # noqa: N806
            hidden_states_B_C,
            [self.intermediate_size, groups_state, groups_state],
            dim=-1,
        )

        A = -torch.exp(self.A_log.float())  # noqa: N806
        dt_limit_kwargs = (
            {}
            if self.time_step_limit == (0.0, float("inf"))
            else {"dt_limit": self.time_step_limit}
        )
        scan_output = mamba_chunk_scan_combined(
            ssm_input.view(batch_size, seq_len, -1, self.head_dim),
            dt,
            A,
            B.view(batch_size, seq_len, self.n_groups, -1),
            C.view(batch_size, seq_len, self.n_groups, -1),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=seq_idx,
            return_final_states=False,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            **dt_limit_kwargs,
        )
        scan_output = scan_output.view(batch_size, seq_len, -1)
        scan_output = self.norm(scan_output, gate)
        return self.out_proj(scan_output)


class Mamba2DecoderEagle3FirstLayer(nn.Module):
    """EAGLE-3 draft block whose token mixer is Mamba2 instead of attention (Arm B).

    Consumes ``cat([embeds, hidden])`` of width ``2 * hidden_size``, normalizes the two
    halves separately, and returns ``residual + mixer(...)``. Unlike the transformer
    first layers there is no rotary embedding: a recurrence carries position implicitly
    through scan order, so ``position_ids`` and ``position_embeddings`` are ignored.

    Setting ``intermediate_size`` on the config turns this into Arm B2 by appending a
    SwiGLU MLP. Both arms therefore share ``model_type: "mamba2"`` and are told apart
    by the serialized config alone, so a checkpoint always reconstructs the arm it was
    trained as.

    :param config: Mamba2 configuration for the draft block.
    :param layer_idx: index of this layer.
    :param norm_before_residual: store the normalized hidden as the residual instead of
        the raw hidden, matching the transformer first layers' flag of the same name.
    """

    _force_mlp: bool = False

    def __init__(
        self,
        config: Mamba2Config,
        layer_idx: int = 0,
        norm_before_residual: bool = False,
    ):
        super().__init__()
        eps = resolve_norm_eps(config)
        self.norm_before_residual = norm_before_residual
        self.input_layernorm = Mamba2RMSNorm(config.hidden_size, eps=eps)
        self.hidden_norm = Mamba2RMSNorm(config.hidden_size, eps=eps)
        self.mixer = Eagle3Mamba2Mixer(config, layer_idx)
        self.mlp = None
        self.post_mixer_layernorm = None
        intermediate_size = getattr(config, "intermediate_size", None)
        if self._force_mlp and intermediate_size is None:
            intermediate_size = 4 * config.hidden_size
        if intermediate_size is not None:
            # Imported lazily so the plain Mamba2 arm does not depend on Qwen3.
            from transformers.models.qwen3.configuration_qwen3 import (  # noqa: PLC0415
                Qwen3Config,
            )
            from transformers.models.qwen3.modeling_qwen3 import (  # noqa: PLC0415
                Qwen3MLP,
            )

            mlp_config = Qwen3Config(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=getattr(config, "hidden_act", "silu"),
            )
            self.mlp = Qwen3MLP(mlp_config)
            self.post_mixer_layernorm = Mamba2RMSNorm(config.hidden_size, eps=eps)

    def forward(  # noqa: PLR0917
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,  # noqa: ARG002
        position_ids: torch.LongTensor | None = None,  # noqa: ARG002
        past_key_values: Any = None,
        use_cache: bool | None = False,  # noqa: ARG002
        cache_position: torch.LongTensor | None = None,  # noqa: ARG002
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # noqa: ARG002
        seq_idx: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        # hidden_states is cat([embeds, hidden]); the residual is the hidden part.
        mid = hidden_states.shape[-1] // 2
        embeds, hidden = hidden_states.split(mid, dim=-1)
        residual = hidden

        embeds = self.input_layernorm(embeds)
        hidden = self.hidden_norm(hidden)
        if self.norm_before_residual:
            residual = hidden
        mixer_input = torch.cat([embeds, hidden], dim=-1)

        hidden_states = residual + self.mixer(
            mixer_input,
            cache_params=past_key_values,
            attention_mask=padding_mask,
            seq_idx=seq_idx,
        )

        if self.mlp is not None and self.post_mixer_layernorm is not None:
            hidden_states = hidden_states + self.mlp(
                self.post_mixer_layernorm(hidden_states)
            )
        return hidden_states


class Mamba2MLPDecoderEagle3FirstLayer(Mamba2DecoderEagle3FirstLayer):
    """Arm B2: :class:`Mamba2DecoderEagle3FirstLayer` plus a SwiGLU MLP.

    Exists so the Mamba2 block can be parameter-matched against a transformer draft
    block; without it a Mamba2-vs-transformer comparison confounds architecture with
    capacity. Equivalent to setting ``intermediate_size`` on the config; provided as an
    explicit class for tests and for callers that would rather name the arm.
    """

    _force_mlp: bool = True


def build_mamba2_draft_config(
    hidden_size: int,
    vocab_size: int,
    *,
    num_hidden_layers: int = 1,
    expand: int = 2,
    head_dim: int = 64,
    state_size: int = 128,
    n_groups: int = 8,
    conv_kernel: int = 4,
    chunk_size: int = 256,
    intermediate_size: int | None = None,
    **kwargs,
) -> Mamba2Config:
    """Build a :class:`Mamba2Config` for a draft block of a given width.

    ``Mamba2Config`` strictly enforces ``hidden_size * expand == num_heads * head_dim``,
    so ``num_heads`` is derived rather than accepted, and ``head_dim`` must divide
    ``expand * hidden_size``.

    :param hidden_size: draft width, which must match the verifier's hidden size.
    :param vocab_size: verifier vocabulary size.
    :param intermediate_size: SwiGLU width for the B2 arm; ignored by the plain arm.
    :raises ValueError: if ``head_dim`` does not divide ``expand * hidden_size``.
    """
    inner = expand * hidden_size
    if inner % head_dim != 0:
        raise ValueError(
            f"expand * hidden_size ({inner}) must be divisible by head_dim "
            f"({head_dim}); Mamba2Config requires num_heads * head_dim == "
            "expand * hidden_size."
        )
    config = Mamba2Config(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_hidden_layers=num_hidden_layers,
        expand=expand,
        head_dim=head_dim,
        state_size=state_size,
        n_groups=n_groups,
        conv_kernel=conv_kernel,
        chunk_size=chunk_size,
        num_heads=inner // head_dim,
        **kwargs,
    )
    if intermediate_size is not None:
        # Not a native Mamba2Config field; carried for the B2 arm's SwiGLU.
        config.intermediate_size = intermediate_size
    return config
