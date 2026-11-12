# Handoff: full-scale GLM-5.3 DFlash2 training, A0 vs A3 (for Mengmeng)

*Written 2026-11-12. Background: [`SUMMARY.md`](SUMMARY.md).*

## Why

- **incoai's weights can't be used for production** (license). That also rules out our best analysis draft, incoai + our fine-tune (8-bench 4.04).
- **Best production-usable draft today:** our own `ftA_ep2best`, at 8-bench 3.14 and 22-bench 2.99 / 2.01×.
- **What the shard-A experiments showed:**
  - incoai's architecture is worth about +11%.
  - The rest of the gap is training budget and breadth.
  - Our on-policy data helps.
- **The next step:** train our own drafts on the **full** GLM-5.3 on-policy regen **plus the multilingual regen**, comparing our current architecture (A0) with incoai's architecture (A3).

**Decisions already made by Ravi:**
- **Runs:** two, A0 and A3, both from scratch.
- **Cut-off replies:** keep the rows that hit the 16k generation cap.
- **Anchors:** keep `--max-anchors 512`.
- **Licensing:** please confirm with whoever owns it that copying incoai's *architecture hyperparameters* (not weights) is fine.

## 1. Data

**Inputs** (in your dir, `/import/ml-sc-scratch1/mengmengj/speculators/data/glm53/`):

| file | rows | notes |
|---|---|---|
| `nemo782k_regen_glm53_all.clean.jsonl` | 777,925 | full nemo782k regen (shards A + redone B1–B3); earlier-turn reasoning stripped, truncated rows **kept** |
| `mling_v2_50k_regen_glm53_h1.jsonl` + `_h2.jsonl` | 25,000 + 25,000 | multilingual regen from Ravi's `prompts_v2`; single-turn, 0 errors |

**Quality scan** (`regen_quality_scan.py`, Nov 12). It reruns the checks that caught the looping 2-node shard B.

| group | rows | hit 16k cap | no `</think>` | repetitive tail (zlib < 0.15) | median / p90 completion tokens |
|---|---|---|---|---|---|
| shard A | 389,289 | 9.8% | 8.7% | 0.1% | 1,503 / 16,052 |
| B1 / B2 / B3 (redone single-node) | ~129.5k each | 12.6–13.5% | 11.5–12.4% | 0.2–0.3% | ~1,440 / 16,384 |
| multilingual 50k | 50,000 | 12.6% | 11.8% | 0.0% | 2,140 / 16,384 |

- The redone shard B is clean: 0.2–0.3% repetitive tails, against **26%** in the old PP=2 rows (`*.pp2raw_20261104`).
- In the multilingual set, de/es/fr/it/ja (the Nemotron prompts) hit the cap about 20% of the time: long math reasoning, not looping.
- A few rare translation directions mostly hit the cap (ka→en ~75%, en→ga ~48%), but each has only about 30 rows.

**Prepare** (CPU only; shuffle so the 0.2% validation tail covers both sources):

```bash
cd <speculators checkout at branch ravir/glm53-dflash2-kimi-recipe>
D=/import/ml-sc-scratch1/mengmengj/speculators/data/glm53
OUT=<your work dir>/data/glm53_onpolicy_full_mling50k_s16k
# ~15 GB through shuf: run on a compute node, not the login node
cat $D/nemo782k_regen_glm53_all.clean.jsonl $D/mling_v2_50k_regen_glm53_h1.jsonl $D/mling_v2_50k_regen_glm53_h2.jsonl \
  | shuf --random-source=<(yes) > ${OUT}.merged.jsonl            # 827,925 rows, fixed shuffle
KEEP_TRUNCATED=1 REGEN_JSONL=${OUT}.merged.jsonl OUTPUT=$OUT \
  bash examples/train/glm53/prepare_onpolicy_glm53.sh            # strip earlier-turn reasoning (idempotent), tokenize at 16k
```

- `prepare_onpolicy_glm53.sh` is our copy of your prep script. It avoids the bash 5.2 `ASSISTANT_PATTERN` parse error on compute nodes.
- `KEEP_TRUNCATED=1` was added for this run.
- The loss mask covers the last assistant turn only, reasoning included.

**Expected size:** about 2.5–2.7B tokens, roughly 5.5–6× shard A (shard A was 141,818 rows / 461M tokens). That's about **20–21k steps per epoch** at 8 ranks × 16k tokens.

## 2. The two runs

Both use the same recipe as the shard-A ablations:
- AdamW, LR 5e-5, cosine schedule, warmup 4%;
- forward-KL loss with `fixed-exp-decay` γ=4, `--max-anchors 512`, `--total-seq-len 16384`, `--train-data-ratio 0.998`;
- from scratch, checkpoint every half epoch.

| | A0: our architecture | A3: incoai's architecture |
|---|---|---|
| draft | 5 layers, 64 heads × head_dim 64, 16 KV heads, SWA 2048 | 6 layers, 64 heads × head_dim 128, 8 KV heads, SWA 2048 |
| `DRAFT_CONFIG` | `glm53_dflash2_draft_config.json` (default) | `glm53_dflash2_draft_config_incoai_shape.json` |
| `TARGET_LAYER_IDS` | `7 20 44 57 71` (default) | `6 20 34 48 62 76` |
| `BLOCK_SIZE` / served k | 7 / 6 | 8 / 7 |
| shard-A result (2 epochs, 8-bench excl. gsm8k) | 2.58 | 2.88 |
| step time on shard A (3 B200 nodes) | ~7.0 s | ~7.7 s |

- **Epochs:** 3 recommended, about 40–45 h per epoch, so about 5–5.5 days per run.
- **When to stop:** on-policy fine-tuning plateaued after about half an epoch, but from-scratch training kept improving through 10 epochs on the Kimi run. Serving-eval each epoch (§3) and stop when it flattens.

**Layout:** per run, 2 nodes for the bf16 verifier (TP8×PP2) and 1 trainer node; the A0 and A3 runs use separate allocations.

**Launch:** use `launch_glm53_dflash2_salloc.sh` → `run_glm53_dflash2_3node.sh` from a frozen copy of this branch.

```bash
salloc -p gpuonly --reservation=sn50-hdi-1 -N3 --ntasks-per-node=1 --gres=gpu:b200m180:8 --exclusive \
  --cpus-per-task=128 --mem=0 --time=2-00:00:00 -J glm53_full_A3 --no-shell
JOBID=<id> REPO=<frozen snapshot> WORK=<your work dir> VENV=/import/ml-sc-scratch1/ravir/envs/glm53_dflash2 \
  WANDB_ENTITY=<yours> RUN_NAME=glm53_full_A3 DATA_PATH=$OUT EPOCHS=3 \
  TARGET_LAYER_IDS="6 20 34 48 62 76" DRAFT_CONFIG=$REPO/examples/train/glm53/glm53_dflash2_draft_config_incoai_shape.json BLOCK_SIZE=8 \
  bash examples/train/glm53/launch_glm53_dflash2_salloc.sh
# A0: same, minus TARGET_LAYER_IDS / DRAFT_CONFIG / BLOCK_SIZE, with RUN_NAME=glm53_full_A0
```

- **Run `SMOKE=1` first** (probes hidden states, then 30 steps).
- **Resume:** allocations max out at 2 days, so rerun the same launch command in a new allocation. The trainer resumes from the newest checkpoint in `SAVE_PATH` with the same W&B run id.
- **Reservation:** `sn50-hdi-1` admits *new* jobs only inside its daily window (09:00–22:00 on Nov 12), so start allocations during the day.
- **Defaults that point at Ravi's setup:** `WORK` and `WANDB_ENTITY` must be overridden. The `glm53_dflash2` env is readable by others, so you can reuse it as `VENV`.

## 3. Evaluation

- **Export** each checkpoint to vLLM format with `export_dflash2_vllm.py --checkpoint <ckpt> --output <dir>`. It sets the k=6 / k=7 config and writes `sample_from_anchor: false`.
- **Serve:** FP8 GLM-5.3, TP8 on one B200 node, vLLM 0.33 (0.28's DFlash2 is broken).
  - 8-bench: `eval_glm53_ablations_fp8_v033.yaml`, about 1.2 h per draft.
  - 22-bench: `eval_glm53_full22_fp8_v033.yaml`, about 4.5 h, including the multilingual and translation categories.
- **Compare means without gsm8k.** It swings ±13% between identical runs at n=50; without it, repeat runs agree within 0.2%.
- **Don't rank recipes by the trainer's `val/accept_len`.** For DFlash2 it measures the unary head's T=1 distribution overlap and ignores the selector. On the incoai fine-tune it fell 5.7% while serving rose 2.3%. It was fine for architecture comparisons. Use serving evals, or greedy `position_k_acc` and the selector metrics.
- **Baselines:**

| draft | 8-bench excl. gsm8k | 22-bench |
|---|---|---|
| A0, shard A, 2 epochs | 2.58 | — |
| A3, shard A, 2 epochs | 2.88 | — |
| `ftA_ep2best` (production today) | 3.14 | 2.99 / 2.01× |
| incoai (reference only) | 3.95 | 3.89 / 2.53× |

- **Results files:** raw per-benchmark results and per-position acceptance for every shard-A eval are in `results/glm53-ablations-fp8-v033/`. Per-position acceptance is aggregated from `server.log` by `results/collect_gap_analysis.py`.

## 4. Gotchas we already paid for

- **Never generate on-policy data on 2-node PP=2 pairs:** those rows loop until max_tokens.
- **vLLM 0.26 hidden-state extraction with PP:**
  - needs the `vllm_pp_aux` plugin and `--block-size 64` (the launcher sets both);
  - load-test extraction (`probe_hidden_states.py --concurrency`), because a workaround once silently corrupted ~30% of hidden states.
- **Mooncake:** reader clients must mount 0 B segments, or about 5% of samples are lost.
- **sc3 compute nodes:**
  - bash 5.2 rejects regexes inside `"${VAR:-...}"` defaults;
  - `~/.local/bin/env` shadows `/usr/bin/env`;
  - system python3 is 3.6;
  - keep JIT caches node-local.
- **sbatch fanning out from sc3-cp81 fails** because of a stale slurm.conf there. Use salloc + srun from the login node, which is what the launcher does.
