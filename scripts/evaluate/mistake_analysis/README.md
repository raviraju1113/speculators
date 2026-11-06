# Draft-model mistake analysis

Find **which tokens the EAGLE3 draft gets wrong** on eval benchmarks (gpqa,
livecodebench, aime, …) so you can decide **what training corpus would help**.

The existing eval only reports *aggregate* acceptance. This adds per-token,
per-TTT-step draft-vs-target comparison and three analysis lenses:

1. **Token categories** — code punctuation / numbers / LaTeX / identifiers / prose.
2. **Rare-in-training cross-reference** — missed tokens that were rare/absent in
   `token_freq.pt`. The direct "add this corpus" signal.
3. **Position & TTT-depth** — where in the sequence / at which depth it fails.

## Files

| file | runs where | purpose |
|---|---|---|
| `score_mistakes.py` | GPU box, vLLM up | generates `mistakes.jsonl` (the heavy step) |
| `mistake_lib.py` | anywhere | pure-Python analysis (no GPU/server) |
| `mistake_analysis.ipynb` | anywhere | loads the JSONL, plots the 3 lenses |

## Step 1 — score (GPU box)

Start the verifier exactly like training (it must run the hidden-states
connector), then run the scorer against it:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/launch_vllm.py \
    /sms-scratch/checkpoints/gemma-4-31B-it/ --port 8000 \
    --tensor-parallel-size 4 --served-model-name google/gemma-4-31B-it \
    --hidden-states-path /sms-scratch/ravira/hidden_states

# NOTE: the verifier above occupies GPUs 0-3. Put the scorer on a FREE gpu (4),
# or it will OOM trying to share a GPU with vLLM.
CUDA_VISIBLE_DEVICES=4 python scripts/evaluate/mistake_analysis/score_mistakes.py \
    --draft-checkpoint /sms-scratch/ravira/checkpoints/gemma4_draft_model_300k_eagle3/checkpoint_best \
    --vllm-endpoint http://localhost:8000/v1 \
    --hidden-states-path /sms-scratch/ravira/hidden_states \
    --benchmarks gpqa_diamond,livecodebench,aime \
    --ttt-steps 3 --num-samples 50 \
    --out scripts/evaluate/mistake_analysis/out/run1_mistakes.jsonl
```

Notes:
- **Run the scorer on a GPU the vLLM server is *not* using** (`CUDA_VISIBLE_DEVICES=4`).
  It picks `cuda:0` within the visible set; sharing a GPU with the verifier OOMs.
- The draft forward runs eager (dynamo disabled in-script) to avoid compile
  startup/memory; export `TORCHDYNAMO_DISABLE=0` to re-enable compilation.
- Use the **same `--ttt-steps`** the checkpoint was trained with (run 1 = 3).
- `--total-seq-len` defaults to 8192 (run 1's cap); sequences are trimmed to it.
- Greedy generation means the target's next token *is* `input_ids[i+1]`, which is
  exactly the greedy speculative-decoding accept criterion — so no verifier
  `lm_head` pass is needed to define "correct".

## Step 2 — analyze (anywhere)

Open `mistake_analysis.ipynb`, set the three paths in the config cell, run all.
Or use the library directly:

```python
import mistake_lib as ML
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/sms-scratch/checkpoints/gemma-4-31B-it")
df = ML.load_mistakes("out/run1_mistakes.jsonl")
df = ML.add_training_freq(ML.add_categories(ML.attach_token_strings(df, tok)),
                          ML.load_token_freq(".../token_freq.pt"))
ML.accuracy_by_freq_bucket(df)      # lens 2 — the corpus signal
ML.top_missed_rare_tokens(df)       # shortlist of tokens to get more data for
```

## `mistakes.jsonl` schema

One row per scored token (per TTT step):

```
benchmark, id, ttt_step, aligned_pos, seq_len, rel_pos,
pred_id, target_id, correct, cond_correct
```

`correct` = draft's step-k prediction matched the target token; `cond_correct`
chains correctness across shallower steps (the trainer's `cond_acc` semantics).

## Comparing checkpoints — freeze the targets

**Always pass `--reference` when comparing checkpoints.** The scorer generates the
target continuations greedily, but greedy decoding is not reliably deterministic
across server runs — regenerating per checkpoint scores each draft on *different*
text, which invalidates the comparison (symptom: token-level accuracy that
contradicts the serving `accept_rate`).

`--reference ref.jsonl` fixes this: the first run generates the continuations and
saves them; every later run *loads* the same continuations and only swaps the
draft checkpoint. So all checkpoints are scored on identical targets.

```bash
# first checkpoint: creates the reference
CUDA_VISIBLE_DEVICES=4 python .../score_mistakes.py \
  --draft-checkpoint .../gemma4_draft_model_300k_eagle3/checkpoint_best \
  --vllm-endpoint http://localhost:8000/v1 --hidden-states-path /sms-scratch/ravira/hidden_states \
  --benchmarks gpqa_diamond,livecodebench,aime --ttt-steps 3 \
  --reference .../out/reference.jsonl \
  --out .../out/run1_mistakes.jsonl

# every other checkpoint: reuses the SAME targets (only --draft-checkpoint/--out change)
CUDA_VISIBLE_DEVICES=4 python .../score_mistakes.py \
  --draft-checkpoint .../gemma4_draft_model_300k_eagle3_kimi_mtp_stem_code/checkpoint_best \
  --vllm-endpoint http://localhost:8000/v1 --hidden-states-path /sms-scratch/ravira/hidden_states \
  --benchmarks gpqa_diamond,livecodebench,aime --ttt-steps 3 \
  --reference .../out/reference.jsonl \
  --out .../out/stem_code_mistakes.jsonl
```

Then load several `mistakes.jsonl` into the notebook with a `run` column to
compare. Sanity check: offline step-0 accuracy should track the serving
`accept_rate` — if it doesn't, the reference wasn't shared.
