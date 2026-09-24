"""Load the Gemma4 MTP k=5 CoE PEF and run one greedy generate."""

from __future__ import annotations

import json
import os
import re
import time

from _metrics import vllm_accept_stats

DYT_VALUES = {"BS": 2, "SS_CG": 8192, "SS_KV": 8192}
SS = 8192
PROMPT_HEADROOM = 16

DEFAULT_PEF = (
    "/import/snvm-sc-podscratch4/weip/gemma4/0828_mtp_mattf/apps_persistent/"
    "gemma4_31b_full_layers_tp16_ssss_cg_ss_kv_ss_tg_parallel_sdk_bf16/"
    "coe_pef_bsBS_max8_ssSS_CG_max131072_SS_KV_max131072_SS_TG_max131072/"
    "gemma4_31b_full_layers_TP16_ssSS_CG_SS_KV_SS_TG_parallel_sdk_bf16_CoE_ckpt_sharing_"
    "BSBS_max8_SSSS_CG_max131072_SS_KV_max131072_SS_TG_max131072/"
    "gemma4_31b_full_layers_TP16_ssSS_CG_SS_KV_SS_TG_parallel_sdk_bf16_CoE_ckpt_sharing_"
    "BSBS_max8_SSSS_CG_max131072_SS_KV_max131072_SS_TG_max131072.pef"
)
DEFAULT_CKPT = "/import/mlcp-sc-nlp/gemma-4/gemma-4-31b-it-pad5632-kv8-prefix"
DEFAULT_ASSISTANT = (
    "/import/ml-sc-nlpcheckpoints-scratch3/weip/gemma-4-31b-it-assistant-pad5632-prefix-split"
)


def load_eos_token_ids(ckpt: str, fallback: list[int]) -> list[int]:
    """Load all model termination IDs, matching Transformers generation."""
    config_path = os.path.join(ckpt, "generation_config.json")
    try:
        with open(config_path) as f:
            eos_ids = json.load(f).get("eos_token_id")
    except (OSError, TypeError, ValueError):
        eos_ids = None
    if isinstance(eos_ids, int):
        return [eos_ids]
    if isinstance(eos_ids, list) and eos_ids and all(isinstance(x, int) for x in eos_ids):
        return eos_ids
    return fallback


def stats_through_stop(
    chunks: list[tuple[int, list[int]]], eos_ids: list[int], max_new_tokens: int
) -> tuple[int, int, int, int | None]:
    """Return completion length and draft stats through the first stop token."""
    generated = 0
    num_drafts = 0
    num_accepted = 0
    stop_token_id = None
    eos_set = set(eos_ids)
    for accepted, tokens in chunks:
        remaining = max_new_tokens - generated
        if remaining <= 0:
            break
        visible = tokens[:remaining]
        stop_index = next((i for i, token in enumerate(visible) if token in eos_set), None)
        emitted = len(visible) if stop_index is None else stop_index + 1
        generated += emitted
        if accepted >= 0:
            num_drafts += 1
            # ``accepted`` is the inclusive end index of this verification chunk
            # (tokens[:accepted+1]). The last token is the target/bonus; the
            # preceding ``accepted`` tokens are accepted drafts. Counting
            # ``accepted+1`` would treat the bonus as a draft and inflate
            # accept_length by +1 per step vs the vLLM formula
            # (1 + accepted_drafts / num_drafts). Cap by ``emitted`` when EOS
            # truncates mid-chunk.
            num_accepted += min(accepted, emitted)
        if stop_index is not None:
            stop_token_id = visible[stop_index]
            break
    return generated, num_drafts, num_accepted, stop_token_id


def disable_rdu_profiler() -> None:
    import contextlib
    import rdu_engine

    rdu_engine.profile_context = lambda *a, **k: contextlib.nullcontext()


def build_pipeline(pef: str, ckpt: str, assistant: str, profiler_dir: str):
    from sn_accuracy.dynamic_tensor import _get_max_len
    from sambanova_modelzoo.pipelines import RDUMonolithLLMPipeline
    from sambanova_modelzoo.testing.schema import DynamicTensorConfig, PEFConfig

    dyt = DynamicTensorConfig(
        batch_size_name="BS",
        decode_seq_len_name="SS_TG",
        kv_seq_config_name="SS_KV",
        prefill_seq_len_names=["SS_CG"],
        values_to_sweep=[dict(DYT_VALUES)],
    )
    pef_cfg = PEFConfig(pef=pef, dynamic_tensor_options=dyt)
    pef_cfg.load_pef_metadata()
    k = int(pef_cfg.token_gen_seq_length) - 1
    print(f"PEF token_gen_seq_length={pef_cfg.token_gen_seq_length} -> MTP k={k}", flush=True)
    try:
        static_seq_lens = set(pef_cfg.pef_lengths)
    except TypeError:
        static_seq_lens = pef_cfg.pef_lengths
    pipeline = RDUMonolithLLMPipeline.create(
        model_name_or_path=ckpt,
        pef=pef,
        max_pef_len=_get_max_len(pef_cfg.pef_lengths),
        sn_model_config=pef_cfg.sn_model_config,
        static_seq_lens=static_seq_lens,
        batch_size=pef_cfg.pef_batch_size,
        profiler_output_dir=profiler_dir,
        token_gen_seq_length=pef_cfg.token_gen_seq_length,
        debug_mode=False,
        config_override_path=os.path.join(ckpt, "config.json"),
        lazy_checkpoint_shard_loading=False,
        dynamic_tensor_options=dyt,
        dynamic_dims_config=pef_cfg.dynamic_dims_config,
        assistant_checkpoint=assistant,
    )
    pipeline.reset_dynamic_shapes(dict(DYT_VALUES))
    pipeline.batch_size = 2
    if not isinstance(pipeline.sn_model_config.max_seq_length, int):
        pipeline.sn_model_config.max_seq_length = SS
    if not hasattr(pipeline.config, "page_aligned_sliding_page_size"):
        import sambanova_modelzoo.generation.clm_runtime as clm_runtime

        orig_flatten_paged_kv = clm_runtime.flatten_paged_kv

        def flatten_paged_kv_compat(config, k_cache, v_cache, *, layer_idx):
            sliding_window = config.sliding_window
            config.sliding_window = int(k_cache.shape[-2])
            try:
                return orig_flatten_paged_kv(
                    config, k_cache, v_cache, layer_idx=layer_idx
                )
            finally:
                config.sliding_window = sliding_window

        clm_runtime.flatten_paged_kv = flatten_paged_kv_compat
    dynamic_consume_symbol = "input_ids_consume_cache_True_max_seq_length_SS_KV"
    if any(
        dynamic_consume_symbol in inputs for inputs in pipeline.graph_inputs.values()
    ):
        orig_get_sampling_keys = pipeline._get_mtp_sampling_keys
        orig_sample_mtp = pipeline._sample_mtp

        def get_sampling_keys_dynamic_symbol(*args, **kwargs):
            keys = orig_get_sampling_keys(*args, **kwargs)
            keys.input_ids_key = dynamic_consume_symbol
            return keys

        def sample_mtp_dynamic_symbol(*args, **kwargs):
            max_pef_len = pipeline.max_pef_len
            pipeline.max_pef_len = "SS_KV"
            try:
                return orig_sample_mtp(*args, **kwargs)
            finally:
                pipeline.max_pef_len = max_pef_len

        pipeline._get_mtp_sampling_keys = get_sampling_keys_dynamic_symbol
        pipeline._sample_mtp = sample_mtp_dynamic_symbol
    compiled_max_lens = [
        int(match.group(1))
        for inputs in pipeline.graph_inputs.values()
        for name in inputs
        if (
            match := re.fullmatch(
                r"input_ids_consume_cache_True_max_seq_length_(\d+)", name
            )
        )
    ]
    if compiled_max_lens:
        pipeline.max_pef_len = max(compiled_max_lens)
    orig_coe_run = pipeline._coe_run

    def coe_run_with_declared_inputs(inputs, graph_name, *args, **kwargs):
        # Split graphs in the surviving k=5 PEF accept only a subset of the
        # tensors produced by this newer host pipeline.
        observed_max_lens = [
            int(match.group(1))
            for name in inputs
            if (match := re.search(r"max_seq_length_(\d+)", name))
        ]
        if observed_max_lens:
            pipeline.max_pef_len = max(pipeline.max_pef_len, *observed_max_lens)
        declared = set(pipeline.graph_inputs[graph_name])
        inputs = {key: value for key, value in inputs.items() if key in declared}
        return orig_coe_run(inputs, graph_name, *args, **kwargs)

    pipeline._coe_run = coe_run_with_declared_inputs
    tokenizer_eos = pipeline._get_eos_token_ids()
    eos_ids = load_eos_token_ids(ckpt, tokenizer_eos)
    pipeline._get_eos_token_ids = lambda: eos_ids
    pipeline.eval_eos_token_ids = eos_ids
    print(f"generation EOS token IDs={eos_ids} (tokenizer default={tokenizer_eos})", flush=True)
    print(
        f"DYT reset to {DYT_VALUES}; chosen_length={pipeline.chosen_length}; "
        f"max_pef_len={pipeline.max_pef_len}",
        flush=True,
    )
    return pipeline, k


def run_one(pipeline, prompt: str, max_new_tokens: int, k: int) -> dict:
    """Duplicate prompt to BS=2. Count decode drafts like vLLM counters."""
    ttft_box = {"t": None}
    # accepted=-1 marks the prefill token, which is emitted but is not a draft.
    generated_chunks: list[tuple[int, list[int]]] = []
    orig_prefill = pipeline._prefill
    orig_sample = pipeline._sample_mtp

    def timed_prefill(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig_prefill(*args, **kwargs)
        ttft_box["t"] = time.perf_counter() - t0
        return out

    def counting_sample(is_prefill: bool = False, **kwargs):
        out = orig_sample(is_prefill=is_prefill, **kwargs)
        if not is_prefill:
            accepted = out.accepted_token_index
            while isinstance(accepted, (list, tuple)):
                accepted = accepted[0]
            if hasattr(accepted, "reshape"):
                accepted = accepted.reshape(-1)[0]
            if hasattr(accepted, "item"):
                accepted = accepted.item()
            accepted = int(accepted)
        else:
            accepted = -1
        tokens = out.generated_tokens[0, : max(1, accepted + 1)].tolist()
        generated_chunks.append((accepted, [int(token) for token in tokens]))
        return out

    pipeline._prefill = timed_prefill
    pipeline._sample_mtp = counting_sample
    try:
        prompts = [prompt, prompt]
        prompt_len = int(
            pipeline.get_tokenized_lengths(
                prompts, apply_chat_template=True, add_generation_prompt=True
            )[0]
        )
        if prompt_len > SS - PROMPT_HEADROOM:
            return {
                "skipped": True,
                "skip_reason": f"prompt_tokens={prompt_len} > {SS - PROMPT_HEADROOM}",
                "prompt_tokens": prompt_len,
            }
        gen_budget = min(max_new_tokens, max(1, SS - prompt_len - 8))
        model_inputs = pipeline.preprocess(
            prompts, apply_chat_template=True, add_generation_prompt=True, padding=True
        )
        pipeline._prepare_for_generation()
        t0 = time.perf_counter()
        gen_out = pipeline.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs.get("attention_mask"),
            max_new_tokens=gen_budget,
            do_sample=False,
        )
        t_end = time.perf_counter()
        pipeline._cleanup_after_generation()
    finally:
        pipeline._prefill = orig_prefill
        pipeline._sample_mtp = orig_sample

    seq = gen_out.sequences[0]
    in_ids = model_inputs["input_ids"][0]
    tok = getattr(pipeline.processor, "tokenizer", pipeline.processor)
    pad_id = getattr(tok, "pad_token_id", None)
    prompt_tokens = int((in_ids != pad_id).sum().item()) if pad_id is not None else int(in_ids.shape[0])
    eos_ids = list(pipeline.eval_eos_token_ids)
    completion_tokens, n_drafts, n_accepted, stop_token_id = stats_through_stop(
        generated_chunks, eos_ids, gen_budget
    )
    ttft = ttft_box["t"]
    e2e = t_end - t0
    accept_length, accept_rate = vllm_accept_stats(n_drafts, n_accepted, k)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "decode_time_s": (e2e - ttft) if ttft is not None else None,
        "e2e_s": e2e,
        "k": k,
        "num_drafts": n_drafts,
        "num_accepted_tokens": n_accepted,
        "accept_length": accept_length,
        "accept_rate": accept_rate,
        "max_new_tokens": gen_budget,
        "finish_reason": "stop" if stop_token_id is not None else "length",
        "stop_token_id": stop_token_id,
        "eos_token_ids": eos_ids,
    }
