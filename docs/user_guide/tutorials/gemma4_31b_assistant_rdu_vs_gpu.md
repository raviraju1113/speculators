# Gemma-4-31B Google Assistant (MTP) k=5: RDU vs GPU Acceptance

Acceptance-only comparison of the same speculative-decoding configuration,
`gemma-4-31b-it` + Google Assistant (MTP) draft at k=5, greedy, on the
25-benchmark suite (50 samples per bench; AIME / AIME26 = 30), run on two
backends:

| backend | target / draft | k | stack | harness |
|---|---|---:|---|---|
| GPU | `/nvmedata/hf_checkpoints/gemma-4-31b-it` + `/nvmedata/hf_checkpoints/gemma-4-31B-it-assistant` | 5 | 4× A100 80GB, TP=4, vLLM 0.24.0+cu129 | `scripts/evaluate/experiments/gemma4-31b-full.yaml` |
| RDU | `/import/mlcp-sc-nlp/gemma-4/gemma-4-31b-it-pad5632-kv8-prefix` + `/import/ml-sc-nlpcheckpoints-scratch3/weip/gemma-4-31b-it-assistant-pad5632-prefix-split` | 5 | 16× SN40, CoE PEF | `scripts/evaluate/rdu_mtp_eval/` |

The GPU column is the Google Assistant (MTP) k=5 run from
[Gemma-4-31B Full-Suite Spec Decode Results](gemma4_31b_full_spec_decode_results.md).
Raw outputs: GPU `scripts/evaluate/experiments/results/gemma4-31b-full/assistant_k5/`,
RDU `scripts/evaluate/experiments/results/gemma4-31b-rdu-k5/`.

## Method


The corrected RDU harness uses all termination IDs from
`generation_config.json` (`[1, 106, 50]`) and truncates generated and accepted
tokens at the first terminator. It uses the same k=5 formulas as vLLM:
`accept_len = 1 + accepted_draft_tokens / num_drafts` and
`accept_rate = accepted_draft_tokens / (num_drafts * 5)`. It completed on
16× SN40 (`sc3-s240`, Slurm job `4347944`) in 2h16m with 1,210 successful
samples, no errors, and no skips. Ten RDU requests reached the 4,096-token
limit, versus eight on GPU. RDU decode tok/s is intentionally excluded because
it is not comparable to GPU decode tok/s.

## Launch details

- Target checkpoint:
  `/import/mlcp-sc-nlp/gemma-4/gemma-4-31b-it-pad5632-kv8-prefix`
- MTP assistant checkpoint:
  `/import/ml-sc-nlpcheckpoints-scratch3/weip/gemma-4-31b-it-assistant-pad5632-prefix-split`
- k=5 CoE PEF:
  `/import/snvm-sc-podscratch4/weip/gemma4/0908_sampling/apps/gemma4_31b_full_layers_tp16_ssss_cg_ss_kv_ss_tg_parallel_sdk_bf16_mtp_5/coe_pef_bsBS_max8_ssSS_CG_max131072_SS_KV_max131072_SS_TG_max131072/gemma4_31b_full_layers_TP16_ssSS_CG_SS_KV_SS_TG_parallel_sdk_bf16_MTP_5_CoE_ckpt_sharing_BSBS_max8_SSSS_CG_max131072_SS_KV_max131072_SS_TG_max131072.pef`
- SambaFlow installation:
  `/import/snvm-sc-scratch2/weip/sambaflow_oA0pU8PUsZ`

The full evaluation was launched from the `speculators` repository root with:

```bash
export INSTALL_ROOT=/import/snvm-sc-scratch2/weip/sambaflow_oA0pU8PUsZ
export CKPT=/import/mlcp-sc-nlp/gemma-4/gemma-4-31b-it-pad5632-kv8-prefix
export ASSISTANT=/import/ml-sc-nlpcheckpoints-scratch3/weip/gemma-4-31b-it-assistant-pad5632-prefix-split
export PEF=/import/snvm-sc-podscratch4/weip/gemma4/0908_sampling/apps/gemma4_31b_full_layers_tp16_ssss_cg_ss_kv_ss_tg_parallel_sdk_bf16_mtp_5/coe_pef_bsBS_max8_ssSS_CG_max131072_SS_KV_max131072_SS_TG_max131072/gemma4_31b_full_layers_TP16_ssSS_CG_SS_KV_SS_TG_parallel_sdk_bf16_MTP_5_CoE_ckpt_sharing_BSBS_max8_SSSS_CG_max131072_SS_KV_max131072_SS_TG_max131072.pef
export RESULT_DIR=$PWD/scripts/evaluate/experiments/results/gemma4-31b-rdu-k5-corrected
export LOG_DIR=$PWD/scripts/evaluate/rdu_mtp_eval/logs-corrected
export NUM_SAMPLES=50
export MAX_TOKENS=4096
export TIMEOUT=08:00:00
export QOS=5
export NODELIST=sc3-s240
export EXCLUDE=sc3-s339,sc3-s345
./scripts/evaluate/rdu_mtp_eval/submit_snrdu.sh
```

`NODELIST` was an availability-specific pin to a 16-chip SN40 node with
`snruntime=1.38.0`; it can be changed to another compatible SN40-16 node.
Leaving `BENCHMARKS` unset runs the complete 25-benchmark suite. The harness
writes per-sample details and the aggregate summary under `RESULT_DIR`, while
the `snrdu` output is written to `$LOG_DIR/snrdu.log`.

## Results

| benchmark | n GPU / RDU | GPU AL | RDU AL | Δ AL | GPU AR | RDU AR | Δ AR |
|---|---:|---:|---:|---:|---:|---:|---:|
| aime | 30 / 30 | 4.848 | 4.843 | -0.005 | 0.7696 | 0.7687 | -0.0009 |
| gpqa | 50 / 50 | 4.479 | 4.385 | -0.094 | 0.6958 | 0.6769 | -0.0189 |
| livecodebench | 50 / 50 | 4.539 | 4.485 | -0.054 | 0.7077 | 0.6970 | -0.0107 |
| gsm8k | 50 / 50 | 5.068 | 5.005 | -0.063 | 0.8135 | 0.8009 | -0.0126 |
| humaneval | 50 / 50 | 5.183 | 5.168 | -0.015 | 0.8366 | 0.8336 | -0.0030 |
| mbpp | 50 / 50 | 4.524 | 4.444 | -0.080 | 0.7049 | 0.6888 | -0.0161 |
| math500 | 50 / 50 | 5.037 | 5.007 | -0.030 | 0.8075 | 0.8014 | -0.0061 |
| mt-bench | 50 / 50 | 3.412 | 3.408 | -0.004 | 0.4823 | 0.4816 | -0.0007 |
| aime26 | 30 / 30 | 4.769 | 4.816 | +0.047 | 0.7538 | 0.7632 | +0.0094 |
| bfcl | 50 / 50 | 5.587 | 5.420 | -0.167 | 0.9174 | 0.8840 | -0.0334 |
| swe-bench-pro | 50 / 50 | 4.466 | 4.355 | -0.111 | 0.6932 | 0.6710 | -0.0222 |
| speed-coding | 50 / 50 | 4.733 | 4.637 | -0.096 | 0.7466 | 0.7274 | -0.0192 |
| speed-multilingual | 47 / 50 | 4.429 | 4.371 | -0.058 | 0.6859 | 0.6742 | -0.0117 |
| speed-rag | 50 / 50 | 4.387 | 4.449 | +0.062 | 0.6773 | 0.6899 | +0.0126 |
| speed-qa | 50 / 50 | 3.157 | 3.126 | -0.031 | 0.4314 | 0.4252 | -0.0062 |
| speed-writing | 50 / 50 | 3.118 | 3.084 | -0.034 | 0.4235 | 0.4168 | -0.0067 |
| HumanEval | 50 / 50 | 4.902 | 4.859 | -0.043 | 0.7804 | 0.7719 | -0.0085 |
| math_reasoning | 50 / 50 | 5.066 | 4.983 | -0.083 | 0.8131 | 0.7965 | -0.0166 |
| qa | 50 / 50 | 3.157 | 3.126 | -0.031 | 0.4314 | 0.4252 | -0.0062 |
| question | 50 / 50 | 3.408 | 3.408 | +0.000 | 0.4816 | 0.4816 | +0.0000 |
| rag | 50 / 50 | 4.154 | 4.106 | -0.048 | 0.6308 | 0.6211 | -0.0097 |
| summarization | 50 / 50 | 3.290 | 3.225 | -0.065 | 0.4579 | 0.4450 | -0.0129 |
| tool_call | 50 / 50 | 4.012 | 3.900 | -0.112 | 0.6024 | 0.5800 | -0.0224 |
| translation | 50 / 50 | 4.023 | 4.024 | +0.001 | 0.6045 | 0.6048 | +0.0003 |
| writing | 50 / 50 | 3.408 | 3.408 | +0.000 | 0.4817 | 0.4816 | -0.0001 |

- Macro average: GPU `AL=4.286`, `AR=0.6572`; RDU `AL=4.242`,
  `AR=0.6483` (RDU deltas `-0.045`, `-0.0089`).
- RDU is higher on `aime26`, `speed-rag`, and `translation`; 22 of 25
  benchmarks are within `|Δ AL| <= 0.1`.
- The 22 non-drifting benchmark files use identical sample IDs. The data files
  changed between runs for `speed-coding`, `speed-multilingual`, and
  `speed-rag`: only 33/50, 32/47, and 33/50 GPU IDs respectively overlap the
  RDU sample set. Treat those three aggregate deltas as directional.
- The corrected totals are 559,702 GPU and 578,386 RDU completion tokens.
  The table still measures each backend's realized greedy decode trajectory,
  not token-for-token replay.

## Takeaway

`accept_len` and `accept_rate` are properties of the draft/target pair, not of
the hardware: across 25 benchmarks the two backends agree to within 0.17
`accept_len` (median |Δ| 0.05), and 22 of 25 are within 0.1. Decode tok/s is
deliberately not compared, since the two stacks are not throughput-comparable.
