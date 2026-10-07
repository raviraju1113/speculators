---
name: dspark-train-serve-parity
description: The sample_from_anchor train/serve mismatch that silently halves DSpark acceptance, plus how to diagnose any train-vs-serve gap in speculative decoding. Use when a DSpark or DFlash draft trains with good metrics but evaluates far worse in vLLM, when accept_len in eval is roughly half what training reported, when deciding sample_from_anchor / --no-sample-from-anchor for a new run, or before trusting any eval of a speculators-format checkpoint. Triggers include sample_from_anchor, dspark_bonus_anchor, accept_len lower than training, position_0_acc mismatch, speculative_tokens 7 vs 8, algos.py patch.
---

# DSpark train/serve parity — the `sample_from_anchor` trap

## TL;DR

**speculators trains DSpark with `sample_from_anchor=True` by default. vLLM's
speculators-format loader hardcodes the opposite. The default configuration
produces a draft that decodes one slot off, loads with no error, and reports
roughly half the acceptance it should.**

Measured on gemma-4-31B-it, same weights, same harness, one-line vLLM change:

| | accept_len (aime) | decode tok/s | speedup vs 21.0 baseline |
|---|---|---|---|
| before | 1.838 | 33.9 | 1.61x |
| after | **3.569** | **66.4** | **3.16x** |

Fix one of two ways (see §5). Default to training with
`--no-sample-from-anchor`; patch vLLM only for checkpoints you already have.

---

## 1. What `sample_from_anchor` actually is

DSpark drafts a whole block in ONE parallel forward. The draft gets a block of
query slots hanging off the **anchor** — the last real, verified token.
`sample_from_anchor` decides what occupies those slots and which token each one
is responsible for.

| | block layout | slot *k* predicts | draft tokens from `block_size=8` |
|---|---|---|---|
| `True` | `[anchor, noise, noise, ...]` — N slots | token *p+k+1* (next-token from each slot) | **8** |
| `False` | `[anchor(bonus), mask@1, mask@2, ...]` — 1+N | token *p+k* (mask sits AT its position) | **7** |

Both fill exactly `block_size` query slots. `True` gets one extra draft token
per block for the same compute, which is why speculators defaults DSpark to it —
it is mildly BETTER on paper, not a crutch.

It is **not** a training technique. It defines what the model's output slots
MEAN. Training and serving must agree or every prediction is read out of the
wrong box.

Source of truth: `src/speculators/models/dspark/core.py` (search
`sample_from_anchor`) — the branch also changes the Markov head's conditioning:

```python
if self.config.sample_from_anchor:
    prev_token_ids = block_tokens                               # unshifted
else:
    prev_token_ids = torch.cat([block_tokens[:, :1],
                                block_tokens[:, :-1]], dim=1)   # shifted
```

So a mismatch is a DOUBLE shift — backbone slot mapping AND Markov conditioning.
That is why acceptance degrades hard rather than losing a single token.

## 2. The incompatibility

- **Training default**: `src/speculators/models/dflash/core.py` —
  `default_sample_from_anchor = algorithm == "dspark"` → **True**.
- **Serving**: `vllm/transformers_utils/configs/speculators/algos.py` —
  `pre_trained_config["dspark_bonus_anchor"] = True`, unconditional, with the
  comment "Speculators DSpark uses the 1+N fill-in block".
- `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` then does
  `self.sample_from_anchor = not getattr(hf_config, "dspark_bonus_anchor", False)`
  → **False**, always, for every speculators-format checkpoint. No config knob.

vLLM never reads the checkpoint's `sample_from_anchor` field. It is descriptive
documentation that nothing at serving time consults. **Editing config.json does
nothing** — the convention lives in what the weights learned.

Also note `speculative_tokens = block_size if sample_from_anchor else block_size - 1`,
so the field is visible indirectly: a `True` checkpoint ships **8**, a `False`
checkpoint ships **7**.

## 3. Detect it in two minutes

Diff your exported config against a known-good published checkpoint for the same
verifier. This is the check that would have saved days:

```bash
python - <<'PY'
import json
a=json.load(open('<reference>/config.json'))    # e.g. RedHatAI/...dspark
b=json.load(open('<ours>/config.json'))
for k in sorted(set(a)|set(b)):
    if k in ('transformer_layer_config','speculators_config'): continue
    if a.get(k)!=b.get(k): print(f'  {k}: ref={a.get(k)}  ours={b.get(k)}')
PY
```

**Do this BEFORE the first long run, not after.** For gemma-4-31B the only
substantive difference was `sample_from_anchor` — everything else (aux layers,
block_size, draft_vocab_size, markov_rank, all 5 layer dims) matched exactly.

Symptoms that should trigger the check:
- eval `accept_len` roughly HALF what training reported
- **position 0 acceptance halved** (0.78 train → 0.42 serving). Position 0
  involves no chaining, no compounding — train and serve must agree there. A
  position-0 gap is a serving bug, not distribution shift.
- degraded but not destroyed (random would be ~0)

Read vLLM's own per-position line from the server log — it is the ground truth:
```
grep "SpecDecoding metrics" server.log | tail -3
# Per-position acceptance rate: 0.420, 0.222, 0.136, ...
```

## 4. Dead ends — do not re-walk these

Three hypotheses that felt right and were all wrong. Eliminate them fast:

- **"The metrics are defined differently."** No. Chain the training per-position
  marginals multiplicatively: `1 + p0 + p0p1 + p0p1p2 + ...`. For us that gave
  **3.161** vs training's reported 3.237 — training ALREADY accounts for
  sequential rejection. Not an apples-to-oranges artifact.
- **"The reduced draft vocab can't represent math/code."** No. Measured coverage
  was 92.5–97.9% on aime/gpqa/livecodebench, which caps position 0 near 0.95,
  nowhere near 0.42.
- **"Training captures 6 aux tensors, serving captures 5."** This asymmetry is
  BY DESIGN and correct. `launch_vllm.py --include-last-layer` (default True)
  appends the verifier's last layer so `data.py` can peel it off as
  `verifier_last_hidden_states`; at serving the verifier produces that final
  hidden state naturally, so only the 5 aux layers need to be in config.
  **Do NOT turn off `--include-last-layer`** — that would make
  `hidden_states[-1]` become layer 58 instead of layer 60 and introduce a real
  bug.

**The control that actually settles it**: run a PUBLISHED checkpoint for the same
verifier through your exact harness. If it scores well and yours doesn't, the
serving path is fine and the problem is your checkpoint. That one run
(~20 min) collapsed the search space immediately.

## 5. The two fixes

### 5a. Train to the runtime's convention (DEFAULT — do this)

```bash
python scripts/train.py --no-sample-from-anchor ...
# or via the driver:
SAMPLE_FROM_ANCHOR=0 bash examples/train/dspark_online_gemma4_31b.sh
```

`examples/train/dspark_online_gemma4_31b.sh` defaults `SAMPLE_FROM_ANCHOR=0` and
`submit_docker.sh` passes it through. Verify it took:

```bash
grep -o "\-\-\(no-\)\?sample-from-anchor" <save-path>/checkpoints/train_command.txt
```

Costs one draft token per block (8 → 7). RedHat reaches `accept_len 4.626` with
7, so it is not the binding constraint. **Buys portability**: the checkpoint
decodes correctly on stock vLLM anywhere.

### 5b. Patch vLLM (only for checkpoints already trained with True)

```python
# vllm/transformers_utils/configs/speculators/algos.py
- pre_trained_config["dspark_bonus_anchor"] = True
+ pre_trained_config["dspark_bonus_anchor"] = not config_dict.get("sample_from_anchor", False)
```

Backup the original as `algos.py.orig`; revert with `cp algos.py.orig algos.py`.
Clear `__pycache__/algos*.pyc` after editing.

**Safe to leave applied permanently.** For a `False` checkpoint,
`not False == True` — the exact constant upstream hardcoded, so the patched line
assigns the same value and behaviour is byte-identical. It diverges ONLY for
`True` checkpoints. Verified empirically: RedHat's aime k=8 read 4.562 both
pre- and post-patch.

Resolution table:

| trained | unpatched serves | patched serves | native k |
|---|---|---|---|
| False | False OK | False OK | 7 |
| True | False **WRONG** | True OK | 8 |

**Do not ship a `True` checkpoint.** It loads on stock vLLM with no error and
silently reports ~half its real acceptance. Nobody downstream gets a warning.

### Worth reporting upstream
Either speculators should default DSpark to `False`, or vLLM should read
`sample_from_anchor` from the config. As-is, the default training config cannot
be served by the path it exports for, and fails silently.

## 6. Eval facts learned alongside this

- **`k` (`num_speculative_tokens`) is a serving knob**, not baked in. Set it per
  experiment. It IS the number of tokens proposed — not reduced by one. What
  differs by convention is slots consumed: `True` needs `N`, `False` needs `1+N`.
  Native k = 8 for `True`, 7 for `False` (both fill `block_size` 8). Running
  RedHat at k=8 asks 9 slots from a model trained for 8 — out of spec, but it ran
  and scored well.
- **Measure the baseline in the SAME config as the drafts.** A backbone baseline
  borrowed from an older run gave 55 tok/s; measured under identical settings it
  was **21.0**. That is the difference between reporting 1.20x and 3.16x. Put
  `baseline` FIRST in `experiments:` — `run_experiments.py` uses experiment #0 as
  the speedup denominator, so a config without one produces a meaningless column
  (the first draft divided by itself = 1.00x).
- **Eval accept_rate = `(accept_len - 1) / k`.** Not comparable to training's
  `accept_rate`, and it changes with k for the same checkpoint. **Compare
  `accept_len` only.**
- **Editing `benchmarks:` and rerunning OVERWRITES** the previous per-experiment
  summaries in the same `output_dir`. Two benchmark sets were lost that way. Use
  a fresh `output_dir` per variation.
- **Run evals as BATCH jobs.** `scripts/evaluate/submit_eval.sh` wraps it with a
  driver guard and generous `--time`; an interactive session's 8 h limit killed a
  run mid-`baseline`.
- Accuracy scores are NOT a quality signal — at `temperature: 0.0` speculative
  decoding is lossless, so GPQA/AIME/LCB scores should match the backbone
  exactly. If they shift, that is a bug signal. The benchmarks are measuring
  `accept_len` and `decode_tok_s` on three output distributions.

## 7. Reference numbers (gemma-4-31B-it, 1x A100-80, TP=1, --enforce-eager, n=30)

Baseline decode: **21.0** (aime) / **21.2** (gpqa) / **21.9** (lcb) tok/s.

| config | aime accept_len | speedup | gpqa | lcb |
|---|---|---|---|---|
| ours ep0 (0.99 ep, lr1e-4, True+patch) k=8 | 3.569 | 3.16x | 2.820 | 3.235 |
| ours ep0 k=3 | 2.907 | 2.61x | 2.460 | 2.687 |
| RedHat k=7 (native) | **4.626** | **4.06x** | 4.012 | 3.559 |
| RedHat k=8 | 4.562 | 4.00x | 3.920 | 3.871 |
| RedHat k=3 | 2.254 | 1.99x | 2.612 | 2.754 |

RedHat = `RedHatAI/gemma-4-31B-it-speculator.dspark`, architecturally IDENTICAL
to ours (5 layers, 5376 hidden, 21504 FFN, 4.20 B params, aux `[1,17,29,47,58]`,
`draft_vocab_size 32000`, `markov_rank 256`) — it differs ONLY in
`sample_from_anchor` and training budget. Its config is the reference to diff
against; copy it somewhere durable, the HF cache is not.

Draft size is NOT the problem: RedHat's is the same 4.20 B and reaches 4.06x.
`--enforce-eager` penalises both equally (it disables the CUDA graphs that the
DSpark speculator explicitly captures over the whole draft step INCLUDING the
sequential Markov sampling) — worth testing with graphs on, but it does not
affect acceptance.

## 8. Reading the gap that remains

Once conventions match, a train-vs-serve gap is expected and benign:
- training metrics are **teacher-forced** on ground-truth prefixes and
  hidden states; serving is free-running on the model's own output
- training averaged over chat data; benchmarks are math/science/code

What is NOT benign is a **position-0** gap. No compounding happens there, so
train and serve must agree. Check position 0 first, always.
