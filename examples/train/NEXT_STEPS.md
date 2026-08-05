# Next steps — DSpark Gemma-4 training (as of 2026-08-06)

## Where we are

Two runs in flight, both ~75% through epoch 0, neither has written a checkpoint yet
(see "why" below).

| Run | Node | Layout | Throughput | Epoch | First ckpt |
|---|---|---|---|---|---|
| `dspark_dp_lr1e4` (59017843) | sc3-c98 | 3 engines + **1 rank** | 1,271 seq/h | 47.8 h | ~11 h |
| `dspark_bal_accum` (59236874) | sc3-c81 | **2 engines + 2 ranks**, accum 37, lr 6e-4 | **2,451 seq/h** | 24.8 h | ~6.5 h |

**The balanced 2+2 layout is ~2x faster** — the first throughput change that paid
off. One vLLM engine and one training rank each cap near ~1,250 seq/h, so they are
matched and only scaling BOTH helps.

Measured facts to reason from:
- epoch = **60,708 steps** at 1 rank (30,354 at 2 ranks); **5.2 conversations/step**
- checkpoints are **13 GB** each
- `/import/ml-sc-scratch1` was at 97%; keep an eye on it

---

## Step 1 — wait for the balanced run's epoch-0 checkpoint (~6.5 h)

Nothing to do but watch. `SAVE_BEST=1` makes `maybe_save_checkpoint` bail out
immediately, so the ONLY save path is `maybe_update_best`, which fires after epoch-0
validation (best_val_loss starts at inf, so the first val always wins).

```bash
ls /import/ml-sc-scratch1/mengmengj/output/gemma4_31b_dspark_bal_accum_lr6e4/checkpoints/
grep -iE "Validation epoch|checkpoint_best ->" logs/dspark_bal_accum.txt | tail -3
```

Want to see `0/` plus `checkpoint_best -> 0`. While waiting, confirm lr 6e-4 is
actually applied now that warmup (246 optimizer steps) is long past:

```bash
grep -oE "lr/AdamW=[0-9.eE+-]+" logs/dspark_bal_accum.txt | tail -2      # want ~6e-4
grep -oE "train/(accept_len|position_1_acc)=[0-9.]+" logs/dspark_bal_accum.txt | tail -4
```

If `position_1_acc` is still ~0.0001 with lr at 6e-4, that is NOT an LR problem and
the accumulation patch needs another look before committing more days.

## Step 2 — let the DP run reach its epoch-0 checkpoint too (~11 h), then kill it

It is the slower layout and we are keeping the balanced one, but its epoch-0
checkpoint is a free lr-1e-4 datapoint at un-accumulated batch — worth having for
comparison, and it costs only the remaining ~11 h on a node we are not otherwise
using.

```bash
ls /import/ml-sc-scratch1/mengmengj/output/gemma4_31b_dspark_dp_lr1e4/checkpoints/
scancel 59017843          # once 0/ exists
```

That frees all 4 GPUs on sc3-c98.

## Step 3 — relaunch balanced, with the two corrections

Kill the current balanced run only AFTER its checkpoint exists (step 1), then
relaunch with the SAME RUN_NAME so it resumes from epoch 0:

```bash
scancel 59236874

cd /import/ml-sc-scratch1/mengmengj/speculators
JOBNAME=dspark_bal_accum RUN_NAME=gemma4_31b_dspark_bal_accum_lr6e4 \
  LR=6e-4 EPOCHS=10 VLLM_TP=1 VLLM_DP=2 TRAIN_GPUS_N=2 \
  ACCUM_STEPS=49 \
  CHECKPOINT_FREQ=0.5 SAVE_BEST=0 \
  NODELIST=sc3-c81 \
  bash examples/train/submit_docker.sh
```

Two changes from the run it replaces:

1. **`ACCUM_STEPS=49`** (was 37). Sizing: 5.2 conversations/step x 2 ranks = 10.4 per
   global step, so 512 / 10.4 = 49 to match DeepSpec's global_batch_size 512. At 37
   we were at ~385. The driver recomputes `--scheduler-total-steps` from this
   automatically (30,354 x 10 / 49 = 6,194 optimizer steps) — that rescaling is
   REQUIRED, without it warmup stretches by ACCUM_STEPS and the LR never leaves
   warmup (observed: 1.88e-06 applied when 6e-4 was requested).
2. **`SAVE_BEST=0`** so sub-epoch checkpointing actually happens. With
   `SAVE_BEST=1`, `trainer.py:517` disables sub-epoch saves outright and the only
   saves come from val-loss improvements — meaning if val loss plateaus at epoch 3,
   epochs 4-9 write NOTHING and a crash rewinds to 3.

Expect ~12 h between saves at freq 0.5 (24.8 h epoch), 13 GB each.

## Step 4 — prune checkpoints manually (SAVE_BEST=0 does not)

Run every day or so, keeping the two newest:

```bash
D=/import/ml-sc-scratch1/mengmengj/output/gemma4_31b_dspark_bal_accum_lr6e4/checkpoints
ls "$D" | grep -E '^[0-9]+$' | sort -n | head -n -2 | xargs -r -I{} rm -rf "$D/{}"
find "$D" -xtype l -delete       # drop symlinks to deleted dirs
du -sh "$D"; df -h /import/ml-sc-scratch1 | tail -1
```

## Step 5 — eval the checkpoints on a 595-driver node

Config is ready: `scripts/evaluate/experiments/gemma4-31b-dspark.yaml`
(aime, gpqa_diamond, livecodebench; baseline + ours at k=8 and k=3 + RedHat's
dspark + Google's MTP assistant).

```bash
sngpu --interactive --gpu 1 --gputype a100m80 \
  --exclude sc-c96,sc3-c97,sc-c82,sc-c120 \
  --cpu 16 --mem 200000 --time 8:00:00

nvidia-smi --query-gpu=driver_version --format=csv,noheader   # MUST be 595.x
conda activate /import/ml-sc-scratch1/mengmengj/condaenvs/dspark
cd /import/ml-sc-scratch1/mengmengj/speculators/scripts/evaluate/experiments
python run_experiments.py --config gemma4-31b-dspark.yaml
cat results/gemma4-31b-dspark/results_table.md
```

Point the `draft:` entries at whichever checkpoint you want to score. The driver
check matters: a run landed on sc3-c97 (565) and every server died with
"driver is too old (found version 12070)".

**Numbers to beat** (same target, gemma-4-31B-it): Ravi's EAGLE-3 accept_len
2.4-2.8 / speedup 1.47-1.8x at k=3; Google's MTP assistant accept_len ~3.6.

## Step 6 — read the run with the summary script

```bash
python examples/train/summarize_run.py logs/dspark_bal_accum.txt \
  --ckpt-dir /import/ml-sc-scratch1/mengmengj/output/gemma4_31b_dspark_bal_accum_lr6e4/checkpoints \
  --markdown
```

Also always check for silent data loss — failed hidden-state fetches drop samples
with only a warning:

```bash
grep -c "Failed to load/cache hidden states" logs/dspark_bal_accum.txt
```

---

## Timeline

10 epochs at 24.8 h/epoch = **~248 h (~10 days)**. If that is too long, the honest
options are fewer epochs (3 epochs ~= 75 h and the 3e-4 run's decile curve was
already flattening within epoch 0), or 8 GPUs as 4 engines + 4 ranks — which needs
both nodes and would push hidden states over NFS, so it is not obviously a win.
