# GLM-5.3 DFlash2 draft on the Kimi K3 recipe (off-policy stopgap)

Goal: start a DFlash2 draft for `zai-org/GLM-5.3` (bf16) now, on the gemma-4-31B-it
completions of nemo782k, while the on-policy GLM-5.3 regeneration
(`run_regen_glm53.sh`, job 14321602 on sc3-c130) finishes. Swap in the GLM data
by warm-starting from the latest checkpoint once it lands.

W&B: <https://forge.coreweave.com/wandb/ravi-raju0594-personal/glm53_dflash2>

## Assets

| what | path |
|---|---|
| verifier (bf16, 282 shards, 1.51 TB) | `/import/ml-sc-nlpcheckpoints-scratch3/mengmengj/glm-5.3-bf16` (reuse; `roshant/GLM-5.3-BF16` on scratch5 is a second full copy -- do not download a third) |
| source corpus | `/import/ml-sc-scratch1/mengmengj/datasets/kimi_nemotron_782k` (`messages` column, gemma answers) |
| conversations JSONL | `/import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2/data/gemma782k_offpolicy.jsonl` (782,413 rows, 90,097 multi-turn) |
| GLM-tokenized dataset | `/import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2/data/gemma782k_glm53_s16k` (`prepare_offpolicy_gemma.sh`, job 14555062, 12 min): 782,413 rows, 775.7M tokens, mean 991 / p50 641 / p99 4,826, 0.02% at the 16k cap; ~6.2k packed steps/epoch at ~126 conv/step |
| draft config | `examples/train/glm53/glm53_dflash2_draft_config.json` |
| env | `/import/ml-sc-scratch1/ravir/envs/glm53_dflash2` = `dspark_sc3_train` (vLLM 0.26.0, torch 2.11 cu130, FIPS-fixed opencv) + `mooncake-transfer-engine 0.3.13.post1` |

## Recipe mapping (Kimi K3 run `8u8zjxez`, TorchSpec -> speculators DFlash2)

| Kimi K3 (TorchSpec config) | here | note |
|---|---|---|
| `learning_rate 5e-5`, AdamW, `weight_decay 0` | `--optimizer adamw --lr 5e-5 --weight-decay 0` | this branch defaults to Muon -- pin AdamW |
| `lr_decay_style cosine`, `warmup_ratio 0.04`, `min_lr 0` | `--scheduler-type cosine --scheduler-warmup-ratio 0.04` | |
| `num_epochs 10` | `--epochs 10` | |
| `max_seq_length 16384` | `prepare_data --seq-length 16384`, `--total-seq-len 16384` | gemma rows: mean ~1.0k tok, p99 5.5k |
| `global_batch_size 128` (8 ranks x mbs 1 x accum 16, unpacked) | 8 ranks x 1 packed 16k-token sequence, no accumulation | packing ~15 conv/sequence -> ~120 conv/step |
| `max_grad_norm 1` | hard-coded `clip_grad_norm_(..., 1.0)` in trainer.py | |
| `dflash_block_size 7`, `dspark_num_anchors 512` | `--block-size 7 --max-anchors 512` | |
| `loss_type forward_kl`, decay gamma 4 | `--loss-fn kl_div --per-position-loss-weight fixed-exp-decay --dflash-decay-gamma 4` | |
| `last_turn_loss_only true` | last-turn assistant regex in `prepare_offpolicy_gemma.sh` | verified: 1 span/row, 0 empty masks |
| draft `gqa_swa`: 5 layers, 64 q / 16 kv heads x 64, MLP 2x hidden, SWA 2048 all layers, non-causal in-block | `glm53_dflash2_draft_config.json` (hidden 6144, MLP 12288, GLM rope theta 8e6) | DFlash2 defaults `sliding_window_non_causal=True` |
| aux layers `[7,23,51,67,83]` of 93 (0-based outputs) | `--target-layer-ids 7 20 44 57 71` of 78 (vLLM ids = output index + 1) | same relative depths; launch_vllm appends layer 78 |
| FSDP FULL_SHARD | DDP (default) | draft ~3.7B incl. frozen embed/lm_head fits one B200 |
| vLLM verifier TP8 x PP2 on 2 nodes + 1 training node, Mooncake tcp | same topology, `--hidden-states-backend mooncake --mooncake-protocol tcp` | see below |

DFlash2 specifics left at defaults: conv kernel 2 / group 16, selector rank 256 /
top-16, selector loss alpha 1, `sample_from_anchor=False`, full verifier vocab
(DFlash2 rejects a pruned draft vocab). Exported block 7 -> 6 speculative tokens.

## Verifier topology

GLM-5.3 bf16 is 1.51 TB; one B200 node holds 8 x 180 GB = 1.44 TB, so the verifier
spans two nodes. sc3-cp nodes talk over `brcm0` 100 GbE sockets only (IB off):

- DP2 x EP16 or TP16 across nodes: per-layer all-to-all / all-reduce over 100 GbE
  (~7 MB/token for EP) -- too slow.
- **TP8 x PP2** (the Kimi layout): one ~24 KB activation hop per token at the stage
  boundary. But vLLM's DeepseekV2/GLM-DSA `forward` returns only
  `hidden_states`/`residual` from a non-last PP rank, so aux layers captured on
  stage 0 are dropped: needs a small vLLM patch to carry them in
  `IntermediateTensors`.
- Fallback: the official fp8 checkpoint (`mengmengj/glm-5.3`, 756 GB) fits TP8 on one
  node with the stock path, at the cost of fp8 instead of bf16 target numerics.

## Making TP8 x PP2 extraction work on vLLM 0.26 (2026-10-30)

Launcher: `run_glm53_dflash2_3node.sh` (one script, role per node), full run via
`sbatch_glm53_dflash2_3node.sh`. Node 0/1 = vLLM TP8 x PP2, node 2 =
`mooncake_master` + 8-rank trainer; hidden states cross nodes through Mooncake (tcp
over brcm0). The `vllm_pp_aux` general plugin (`examples/train/glm53/vllm_pp_aux`,
installed in the env, active only with `VLLM_PP_AUX_PATCH=1`, pinned to vLLM 0.26.0)
plus driver/connector fixes, each found by a smoke run:

| run | symptom | fix |
|---|---|---|
| smoke1 | `Pipeline parallelism is not supported for this model` at config time | plugin: `ExtractHiddenStatesModel` passes the PP check (the drafter lives on the last rank anyway) |
| (design) | aux layers on stage 0 never captured nor sent | plugin: set aux layers on every rank; ship them as extra `IntermediateTensors` keys; receive buffer pre-allocates them |
| smoke2 | stage-0 `AttributeError: ... no attribute 'drafter'` in profile_run | plugin: drop the runner's `speculative_config` on non-last ranks (they only capture) |
| (design) | Mooncake connector asserts one cache layer on stage 0 | connector: non-last PP ranks skip `register_kv_caches` and never write |
| smoke3-5 | KV page sizes 73,728 / 8,448 B cannot be unified | root cause found in debug10: unset block size makes stage 0 pick 64 (DSA indexer) and stage 1 pick 32 (FlashInfer MLA-sparse). Driver pins `--block-size 64`; plugin builds the uniform-type per-layer KV group explicitly (stock `MLAAttentionSpec.merge` would fold the 73,728 B/token hidden-states layer into an MLA-sized page) |
| smoke5 | `[Errno 116] Stale file handle` in Triton warmup | JIT caches on node-local NVMe |
| smoke6 | failed fetches leaked Mooncake space -> evictions | `data.py` deletes a sample whose validation failed |
| smoke7 / debug8 | ~1% NaN samples; load probe: 30% of rows decode to garbage under 128 concurrent requests (clean one at a time) | caused by an interim lcm block-size workaround (704/6144/11-token blocks); replaced by the pinned 64-token layout above |

Gate after the fix (debug11, real weights, 512 random rows, 128 in flight): 0 NaN,
lm_head top-1 agreement with the gemma next token p1 0.44 / p5 0.58 / p50 0.75; the 4
rows below 0.3 (68-243 tokens) score the same fetched alone, i.e. genuine. KV capacity
504,768 tokens (29x concurrency at 17k). `probe_hidden_states.py --verifier` runs as a
gate before every training launch; `--concurrency N` is the load test.
