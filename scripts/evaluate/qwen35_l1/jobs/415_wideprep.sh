# JOB 415 -- recover the rows the 8192 window threw away. CPU only, so it runs
# alongside 414's training; nice'd and capped so it cannot starve the three
# ranks' dataloader workers.
#
# THE LOSS THIS FIXES, measured on the source corpus:
#   prompt tokens              mean 3148   p50 1745   p90 8583   max 18440
#   prompt+response            mean 4514   p50 2750   p90 11932  max 21593
#   rows with prompt+response > 8192 : 58,704  (22.1%)
#   rows prepare_data dropped        : 59,321  (22.3%)
# preprocessing.py right-truncates (`input_ids[:max_length]`) and then drops any
# row left with fewer than 32 assistant tokens. For a long-prompt row that
# removes the whole response, so the row is dropped. Zero rows failed to render
# -- the window is the entire cause.
#
# Net effect on what the draft could learn from: of 0.3631B response tokens in
# the corpus, the 8192 prep kept 0.1755B (48.3%).
#
# A wider window is the only fix that preserves the task. Left-truncating
# instead would keep the response but cut the head of the prompt, and this
# corpus's prompts ARE the task input (an activity window to extract interests
# from), so a left-cut row teaches the draft to answer from a mutilated input.
set -uo pipefail
export PATH="/root/miniconda3/envs/speculator/bin:$PATH"
PY=/root/miniconda3/envs/speculator/bin/python
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
M=/nvmedata/hf_checkpoints/qwen35-l1234-v25-step6000
E3=/nvmedata/chenw/speculators/output/e3_qwen35
O=/nvmedata/chenw/speculators/output/e4_wide_qwen35
MAIN=/nvmedata/chenw/speculators
SEQ=16384
mkdir -p $O/logs
cd $MAIN
echo "=== JOB 415 START $(date -u) ==="
echo "  re-prepping at --seq-length $SEQ (was 8192)"

for part in train val; do
  if [ -d "$O/prep_$part" ]; then echo "  prep_$part exists, skipping"; continue; fi
  echo "=== prepare_data $part at $SEQ $(date -u) ==="
  nice -n 19 taskset -c 0-31 $PY scripts/prepare_data.py --model "$M" \
    --data $E3/real_$part.jsonl --seq-length $SEQ --minimum-valid-tokens 32 \
    --output $O/prep_$part >> $O/logs/prep_$part.log 2>&1 \
    || { echo "FATAL: prepare_data $part failed"; tail -6 $O/logs/prep_$part.log; exit 1; }
  grep -oE "Dropped [0-9]+/[0-9]+ samples" $O/logs/prep_$part.log | sed 's/^/    /'
done

RATIO=$($PY - "$O" <<'PY'
import shutil, sys
from pathlib import Path
from datasets import load_from_disk, concatenate_datasets
O = Path(sys.argv[1])
tr = load_from_disk(str(O / "prep_train")); va = load_from_disk(str(O / "prep_val"))
ntr, nva, total = len(tr), len(va), len(tr) + len(va)
if ntr == 0 or nva == 0:
    print("EMPTY train=%d val=%d" % (ntr, nva), file=sys.stderr); sys.exit(1)
out = O / "data_prep"
if not (out / "dataset_info.json").exists():
    concatenate_datasets([tr, va]).save_to_disk(str(out))
    src = O / "prep_train" / "token_freq.pt"
    if src.exists(): shutil.copy2(src, out / "token_freq.pt")
merged = load_from_disk(str(out))
if len(merged) != total:
    print("MERGE MISMATCH %d != %d" % (len(merged), total), file=sys.stderr); sys.exit(1)
for extra in range(0, 1001):
    s = "%.10f" % ((ntr + extra / 1000.0) / total)
    if 0.0 < float(s) < 1.0 and int(total * float(s)) == ntr:
        print("%s %d %d %d" % (s, ntr, nva, total)); break
else:
    print("NO EXACT BOUNDARY", file=sys.stderr); sys.exit(1)
PY
) || { echo "FATAL: could not build an exact train/val boundary"; exit 1; }
set -- $(printf '%s\n' "$RATIO" | tail -1)
R=$1; PTR=$2; PVA=$3; PTOT=$4
echo "  prepared at $SEQ: $PTR train + $PVA val = $PTOT rows, ratio=$R"

# ---------- what the wider window actually recovered ----------
$PY - "$O" "$R" <<'PY'
import sys, numpy as np
sys.path.insert(0, "/nvmedata/chenw/speculators/src")
from datasets import load_from_disk
from speculators.train.distributed_batch_sampler import MultipackDistributedBatchSamplerV2
O, R = sys.argv[1], float(sys.argv[2])
d = load_from_disk(O + "/data_prep")
tr = d.select(range(int(len(d) * R)))
tok = 0
for b in tr.with_format("numpy").iter(batch_size=2000):
    for m in b["loss_mask"]:
        tok += int(np.sum(m))
print("  trainable response tokens: %.4fB   (8192 prep had 0.1755B of 0.3631B = 48.3%%)" % (tok / 1e9))
print("  recovery: %.1f%% of the corpus's response tokens" % (100 * tok / 0.3631e9))
s = np.array(tr["seq_len"])
for nproc in (3,):
    for bml in (8192, 16384):
        n = len(MultipackDistributedBatchSamplerV2(batch_max_length=bml, lengths=np.clip(s, 0, bml),
              num_replicas=nproc, rank=0)._generate_batches(0))
        print("  one epoch at total-seq-len %5d on %d ranks: %6d steps" % (bml, nproc, n))
print("  NOTE: rows longer than total-seq-len are TRUNCATED by the sampler, so")
print("        training must use 16384 to actually see the recovered tails.")
PY
echo "=== JOB 415 DONE $(date -u) ==="
