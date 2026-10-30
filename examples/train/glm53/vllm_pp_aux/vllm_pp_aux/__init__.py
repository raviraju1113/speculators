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
last rank returns the same ordered list it would without PP. It also lets the
ExtractHiddenStatesModel draft config pass ``verify_with_parallel_config``,
which rejects PP > 1 for any architecture without ``SupportsPP``.

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
            # Non-last ranks build no drafter, yet _dummy_run,
            # _build_attention_metadata, initialize_kv_cache and
            # _check_and_update_cudagraph_mode all touch self.drafter whenever
            # the runner has a speculative_config (smoke2: AttributeError in
            # profile_run). Capturing aux layers is all spec decode needs from
            # these ranks, so drop it from the runner; vllm_config keeps it for
            # the KV connector and scheduler. Safe for prefill-only
            # (max_tokens=1) traffic, which never schedules spec tokens.
            self.speculative_config = None
            logger.info(
                "[vllm_pp_aux] non-last PP rank captures aux layers %s; "
                "runner spec decode disabled on this rank", layers,
            )

    gmr.GPUModelRunner._setup_eagle3_aux_hidden_state_outputs = setup


def _patch_registry() -> None:
    # The draft config inherits the target's PP size, and ExtractHiddenStatesModel
    # does not declare SupportsPP, so config validation rejects PP > 1. The
    # runner builds the whole drafter on the last PP rank regardless, so the
    # check does not apply to it.
    from vllm.model_executor.models.registry import ModelRegistry

    cls = type(ModelRegistry)
    orig = cls.is_pp_supported_model

    def is_pp_supported_model(self, architectures, model_config) -> bool:
        archs = [architectures] if isinstance(architectures, str) else architectures
        if "ExtractHiddenStatesModel" in archs:
            return True
        return orig(self, architectures, model_config)

    cls.is_pp_supported_model = is_pp_supported_model


def _patch_page_size_unification() -> None:
    # GLM-5.3 (DSA) has three KV page sizes at block 64: MLA latent
    # 576 x 2 B = 73,728 B, the fp8 indexer 132 B = 8,448 B (= 2^8 * 3 * 11),
    # and the hidden-states cache 6 x 6144 x 2 B = 4.7 MB. vLLM unifies pages by
    # growing smaller layers' blocks, which needs every page to divide the
    # largest; the indexer's factor 11 never divides a power-of-two block of
    # hidden states, so engine init fails (smoke3). Resize only the
    # hidden-states block so its page is the smallest common multiple of the
    # other pages (11 tokens here -> MLA blocks of 704, indexer blocks of 6144,
    # both multiples of the 64-token kernel block). Its CacheOnlyAttention
    # backend indexes by slot, so any block size works.
    import math
    from dataclasses import replace

    from vllm.logger import init_logger
    from vllm.v1.core import kv_cache_utils as kcu
    from vllm.v1.kv_cache_interface import HiddenStateCacheSpec

    logger = init_logger(__name__)
    orig = kcu.unify_kv_cache_spec_page_size

    def unify(kv_cache_spec):
        hs = {n: s for n, s in kv_cache_spec.items() if isinstance(s, HiddenStateCacheSpec)}
        others = [s.page_size_bytes for n, s in kv_cache_spec.items() if n not in hs]
        if hs and others:
            lcm = math.lcm(*others)
            new = {}
            for name, spec in hs.items():
                per_token = spec.page_size_bytes // spec.block_size
                block = lcm // math.gcd(lcm, per_token)
                while block * per_token < max(others):
                    block *= 2
                if block != spec.block_size:
                    new[name] = replace(spec, block_size=block)
            if new:
                logger.info(
                    "[vllm_pp_aux] hidden-states cache block %s -> %s tokens "
                    "(page %s B) to unify with pages %s",
                    next(iter(hs.values())).block_size,
                    next(iter(new.values())).block_size,
                    next(iter(new.values())).page_size_bytes,
                    sorted(set(others)),
                )
                kv_cache_spec = {**kv_cache_spec, **new}
        return orig(kv_cache_spec)

    kcu.unify_kv_cache_spec_page_size = unify


def register() -> None:
    if os.environ.get("VLLM_PP_AUX_PATCH") != "1":
        return
    import vllm

    if vllm.__version__ != "0.26.0":
        raise RuntimeError(
            f"vllm_pp_aux copies vLLM 0.26.0's DeepseekV2Model.forward; "
            f"got vLLM {vllm.__version__}"
        )
    _patch_registry()
    _patch_page_size_unification()
    _patch_model()
    _patch_runner()
