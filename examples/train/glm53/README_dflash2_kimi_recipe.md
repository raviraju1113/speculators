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
| GLM-tokenized dataset | `/import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2/data/gemma782k_glm53_s16k` (`prepare_offpolicy_gemma.sh`) |
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
