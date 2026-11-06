"""Load the Gemma4 MTP k=5 CoE PEF and run one greedy generate."""

from __future__ import annotations

import os
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
    print(f"DYT reset to {DYT_VALUES}; chosen_length={pipeline.chosen_length}", flush=True)
    return pipeline, k


def run_one(pipeline, prompt: str, max_new_tokens: int, k: int) -> dict:
    """Duplicate prompt to BS=2. Count decode drafts like vLLM counters."""
    ttft_box = {"t": None}
    draft_counts: list[int] = []
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
            idx = out.accepted_token_index
            draft_counts.append(int(idx[0] if isinstance(idx, (list, tuple)) else idx))
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
    completion_tokens = max(0, int(seq.shape[0] - in_ids.shape[0]))
    ttft = ttft_box["t"]
    e2e = t_end - t0
    n_drafts = len(draft_counts)
    n_accepted = int(sum(draft_counts))
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
    }
