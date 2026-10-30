"""vLLM general plugin: carry aux hidden states across pipeline-parallel stages.

Needed to extract hidden states from GLM-5.3 bf16 (1.51 TB) served TP8 x PP2
over two B200 nodes. Stock vLLM 0.26.0 loses every aux layer that lives on a
non-last PP rank, twice over:

1. ``GPUModelRunner`` sets ``use_aux_hidden_state_outputs`` and calls
   ``set_aux_hidden_state_layers`` only inside the last-rank drafter branch, so
   earlier stages never learn which layers to capture.
2. ``DeepseekV2Model.forward`` returns only ``hidden_states``/``residual`` from a
   non-last rank, and ``make_empty_intermediate_tensors`` pre-allocates only
   those two receive buffers.

The patch sets the aux layers on every rank for ``extract_hidden_states`` and
ships upstream aux states as extra ``aux_hidden_states_<i>`` tensors, so the
last rank returns the same ordered list it would without PP.

Active only with VLLM_PP_AUX_PATCH=1 and only for vLLM 0.26.0 (the forward
below is a copy of that version's ``DeepseekV2Model.forward``).
"""

import os
from itertools import islice

_AUX_KEY = "aux_hidden_states_{}"


def _num_upstream_aux(model) -> int:
    return sum(1 for i in model.aux_hidden_state_layers if i < model.start_layer)


def _make_empty_intermediate_tensors(model):
    import torch
    from vllm.sequence import IntermediateTensors

    def make(batch_size: int, dtype: torch.dtype, device: torch.device):
        keys = ["hidden_states", "residual"] + [
            _AUX_KEY.format(i) for i in range(_num_upstream_aux(model))
        ]
        return IntermediateTensors(
            {
                k: torch.zeros((batch_size, model.hidden_size), dtype=dtype, device=device)
                for k in keys
            }
        )

    return make


def _patch_model() -> None:
    from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
    from vllm.model_executor.models import deepseek_v2 as dv2
    from vllm.sequence import IntermediateTensors

    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None):
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError(
                        "Either input_ids or inputs_embeds must be provided "
                        "to DeepseekV2Model.forward"
                    )
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
            aux_hidden_states = []
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            # Clone: these are views of the runner's persistent receive buffer,
            # which the next micro-batch overwrites.
            aux_hidden_states = [
                intermediate_tensors[_AUX_KEY.format(i)].clone()
                for i in range(_num_upstream_aux(self))
            ]

        llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
        if llama_4_scaling_config is not None:
            llama_4_scaling = dv2._get_llama_4_scaling(
                original_max_position_embeddings=llama_4_scaling_config[
                    "original_max_position_embeddings"
                ],
                scaling_beta=llama_4_scaling_config["beta"],
                positions=positions,
            )
        else:
            llama_4_scaling = None

        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if (
                hidden_states.shape[0] != positions.shape[0]
                and not layer.use_sequence_parallel_moe
            ):
                combined_states = dv2.torch.cat([hidden_states, residual], dim=-1)
                combined_states = tensor_model_parallel_all_gather(combined_states, 0)
                combined_states = combined_states[: positions.shape[0]]
                hidden_states, residual = combined_states.split(
                    [self.hidden_size, self.hidden_size], dim=-1
                )
                residual = residual.contiguous()
            if idx in self.aux_hidden_state_layers:
                aux_hidden_state = hidden_states + residual
                if aux_hidden_state.shape[0] != positions.shape[0]:
                    aux_hidden_state = tensor_model_parallel_all_gather(
                        aux_hidden_state, 0
                    )
                    aux_hidden_state = aux_hidden_state[: positions.shape[0]]
                aux_hidden_states.append(aux_hidden_state)
            hidden_states, residual = layer(
                positions, hidden_states, residual, llama_4_scaling
            )

        if not get_pp_group().is_last_rank:
            tensors = {"hidden_states": hidden_states, "residual": residual}
            for i, t in enumerate(aux_hidden_states):
                tensors[_AUX_KEY.format(i)] = t
            return IntermediateTensors(tensors)

        if hidden_states.shape[0] != positions.shape[0]:
            combined_states = dv2.torch.cat([hidden_states, residual], dim=-1)
            combined_states = tensor_model_parallel_all_gather(combined_states, 0)
            combined_states = combined_states[: positions.shape[0]]
            hidden_states, residual = combined_states.split(
                [self.hidden_size, self.hidden_size], dim=-1
            )
            residual = residual.contiguous()

        if self.end_layer in self.aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states + residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    dv2.DeepseekV2Model.forward = forward

    orig_set_aux = dv2.DeepseekV2ForCausalLM.set_aux_hidden_state_layers

    def set_aux_hidden_state_layers(self, layers):
        orig_set_aux(self, layers)
        # Both bindings were captured at __init__ time; rebind so the runner's
        # receive buffer gets the upstream aux keys.
        make = _make_empty_intermediate_tensors(self.model)
        self.model.make_empty_intermediate_tensors = make
        self.make_empty_intermediate_tensors = make

    dv2.DeepseekV2ForCausalLM.set_aux_hidden_state_layers = set_aux_hidden_state_layers


def _patch_runner() -> None:
    from vllm.distributed import get_pp_group
    from vllm.logger import init_logger
    from vllm.v1.worker import gpu_model_runner as gmr

    logger = init_logger(__name__)
    orig = gmr.GPUModelRunner._setup_eagle3_aux_hidden_state_outputs

    def setup(self) -> None:
        orig(self)
        spec = self.speculative_config
        if (
            spec is not None
            and spec.method == "extract_hidden_states"
            and not get_pp_group().is_last_rank
        ):
            layers = self._get_eagle3_aux_layers_from_config()
            assert layers, "extract_hidden_states needs explicit aux layer ids"
            self.get_model().set_aux_hidden_state_layers(tuple(layers))
            logger.info("[vllm_pp_aux] non-last PP rank captures aux layers %s", layers)

    gmr.GPUModelRunner._setup_eagle3_aux_hidden_state_outputs = setup


def register() -> None:
    if os.environ.get("VLLM_PP_AUX_PATCH") != "1":
        return
    import vllm

    if vllm.__version__ != "0.26.0":
        raise RuntimeError(
            f"vllm_pp_aux copies vLLM 0.26.0's DeepseekV2Model.forward; "
            f"got vLLM {vllm.__version__}"
        )
    _patch_model()
    _patch_runner()
