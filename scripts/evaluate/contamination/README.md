# Train/eval contamination audit

Checks whether the EAGLE3 training corpora overlap the eval benchmarks, and — the
part that actually decides anything — whether any overlap **moved the measured
numbers** enough to justify retraining or re-running an experiment.

## Why a speculator needs its own framing

The draft is trained to predict the **target's** next tokens. Two facts make
contamination sharper here than in ordinary benchmark reporting:

* the training corpora's assistant turns were regenerated **by the same target**
  (`gemma-4-31B-it`) whose acceptance the eval measures, and
* eval runs greedy (`temperature: 0.0`).

So an eval prompt present in training means the draft trained on *exactly the
continuation being scored*. The metric at risk is `accept_rate` / `accept_length`,
not task accuracy — and answer leakage is not the concern, since the original
reference answers were discarded and regenerated.

## Policy: exact-only (default)

This project counts an item as contaminated **only when its prompt matches a
training prompt exactly**, after tier-0 template stripping and whitespace / case /
punctuation folding. A confirmed copy of the same problem that was reformatted or
reworded is reported as a near-duplicate but **not** counted as contamination.

`--policy graded` applies the stricter rule (long verbatim run, containment, or
MinHash near-duplicate also count). Both are always computed: the `graded_verdict`
column and `review_pairs.jsonl` stay populated under exact-only, so the decision
is auditable and revisiting it needs no re-scan.

Note "exact" must mean *normalized* exact. A byte-exact rule is vacuous here,
because the harness wraps every problem in scaffolding (`AIME_TMPL`, `LCB_TMPL`,
`REASONING_SUFFIX`) that no training row carries — it would report zero matches on
any corpus, including one containing the whole benchmark verbatim.

## Method

Tiers 0–4 find candidates; tier 5 decides.

| Tier | What | Why |
|---|---|---|
| 0 | **Template stripping** | Eval prompts wrap problems in harness scaffolding (`AIME_TMPL`, `GPQA_TMPL`, `LCB_TMPL`, `REASONING_SUFFIX`). Compare the wrapped forms and every detector below returns ~0 hits *by construction*. This is the easiest way to get a falsely clean report. |
| 1 | **Exact normalized hash** | Zero-false-positive floor. |
| 2 | **13-gram containment + longest verbatim run** | GPT-3/Llama convention. The run length is the number a human can actually judge. An 8-gram pass carries recall for rephrased copies. |
| 3 | **MinHash Jaccard** | Near-duplicates that share no long verbatim run. |
| 4 | **Manual adjudication** | Auto-flags are candidates, never verdicts. Decisions are recorded in `adjudicated.jsonl` so the final numbers are reproducible. |
| 5 | **Impact analysis + control** | Counting hits decides nothing. See below. |

### Noise suppression (learned from real false positives)

Three filters, each added after a confirmed false positive:

* **Eval-side boilerplate** — an n-gram in ≥5 distinct eval items is template
  phrasing, not problem content. `aime/65`'s entire 40-token "verbatim run" was
  the AIME answer-format sentence (*"...where m and n are relatively prime
  positive integers. Find m+n."*), with none of its actual problem in training.
* **All-numeric n-grams** — sample grids and I/O tables collide across unrelated
  problems. `livecodebench/abc327_c` shared **424** 13-grams with an unrelated
  Aizu Sudoku *solver* purely through digit rows, at MinHash Jaccard 0.000.
* **Minimum containment denominator** — a 9-token mbpp prompt has two 8-gram
  positions, so one chance match on generic phrasing gives containment 0.50.
  Below 10 positions, only exact match / long run / MinHash may flag an item.

A genuinely duplicated problem always shares prose too, so none of these mask
real hits — verified by regression-checking known true positives after each.

### Tier 5: the control that prevents a wrong retrain

A raw dirty-vs-clean acceptance gap is **confounded**. Flagged items are not a
random sample: the LiveCodeBench items matching Nemotron's code split are
Codeforces problems, whose formulaic phrasing is intrinsically easier for *any*
draft to predict. Measured naively, flagged LCB items accept ~5 points higher —
which reads as memorization but is not.

The fix is to compare against drafts that were **never trained** on the split an
item is attributed to:

* **within-draft** (preferred) — exposed-dirty vs unexposed-dirty items in the
  *same* draft, so item difficulty and draft quality both cancel;
* **cross-draft DiD** (fallback) — treated drafts vs never-exposed controls,
  including RedHat's published draft, which saw none of these corpora.

Only the *excess* gap over the control is attributable to contamination. Both
tests refuse to report a verdict when a group has fewer than 3 items, because a
clustered bootstrap over one item returns a degenerate zero-width interval that
reads as high confidence while being pure noise.

## Running it

```bash
conda activate speculators          # needs numpy; xxhash strongly recommended

# ~14 s for 1.12M training rows x 3,523 eval prompts on 48 workers
python detect_contamination.py --workers 48

# decision step; reuses existing mistake_analysis/out/*.jsonl (no GPU needed)
python impact_analysis.py
```

`detect_contamination.py --limit-rows 300` gives a fast smoke test.

Outputs land in `out/`:

| file | contents |
|---|---|
| `contamination_report.md` / `.json` | per-benchmark summary, affected corpora |
| `eval_item_scores.jsonl` | every eval item with all four scores + attribution |
| `review_pairs.jsonl` | flagged items with matching training text and the **longest common span** — read that field first, it decides most adjudications |
| `impact_analysis.json` | per-draft acceptance split and the control statistics |

Corpora are scanned from their **source** files (`train_regen.jsonl` + the three
`nemotron-v2__*_regen_*.jsonl`) rather than the merged `train_regen_*_merged.jsonl`,
because the merged files' UUID row ids cannot distinguish stem from code — and
per-split attribution is what maps a hit to the experiments it affects.

Attribution is strength-aware: a split owns an item only if that split *on its
own* clears the dirty bar. Keying on "any shared 13-gram" implicates every corpus
via boilerplate (it wrongly credited `kimi` with the Codeforces duplicates).

## Findings (scan of 2026-08-14, 1,118,854 training rows, exact-only policy)

**No retraining or re-running is required.**

Exact prompt matches: **3 of 3,523** eval prompts (0.09%), all verified genuine
(containment 1.000, MinHash 1.000):

| item | corpus | in an evaluated sample? |
|---|---|---|
| `math500/101` | `nemotron_math` | yes — but only in `gemma4-31b` / `glm52-kvcache-ablation` |
| `math500/145` | `nemotron_math` | no |
| `gsm8k/530` | `kimi` | no |

**Zero** exact matches in `aime`, `gpqa` and `livecodebench` — the only three
benchmarks any of our draft experiments report. And the two experiments that do
score `math500` evaluate third-party drafts (Google's published
`gemma-4-31B-it-assistant`, GLM-5.2's own MTP head) which were never trained on
`nemotron_math`, so the one evaluated exact match cannot affect them either.

`gpqa` is clean under *any* policy: zero 13-gram hits across all 198 items.
Verified not to be an indexing artifact — a positive control confirms GPQA items
index normally and 8-gram containment does register partial matches (max 0.19).

### Under `--policy graded`, for reference

The stricter rule flags substantially more, and this is what the exact-only policy
consciously sets aside:

* **livecodebench** — Codeforces problems appear near-verbatim in `nemotron_code`
  (`1899_B`: containment 0.93, 322-token run, Jaccard 0.97, confirmed by grep;
  differs only in `$n$` vs `$$$n$$$`). Independently shown *not* to inflate
  acceptance: drafts that never saw `nemotron_code` show the same +5 point
  elevation on those items (+0.0508 / +0.0544 / +0.0540) as drafts that did
  (+0.0567 / +0.0346), giving **DiD = −0.0011**. So even under the strict rule the
  conclusion is unchanged — the elevation is intrinsic to formulaic
  competitive-programming phrasing, not memorization.
* **aime** — `aime/61` is a confirmed reworded copy (4 copies in `nemotron_math`,
  variables renamed). Dropping it moves AIME acceptance by **+0.0007**.
* Off the experiment path: `math500` (57), `humaneval` (42), `gsm8k` (6). If any
  of these is ever promoted to a reported metric, re-audit before reporting.

## Extending

* Re-run after adding any corpus; add the new split to `DEFAULT_SOURCES` and to
  `CORPUS_COMPOSITION` / `DRAFT_SPLITS` so attribution and the control stay valid.
* Adding a benchmark whose prompts use new scaffolding means adding its template
  to `_EVAL_PREFIXES` / `_EVAL_SUFFIXES` / `_EVAL_CUTS` in `contamination_lib.py`
  — otherwise tier 0 silently under-reports (see the table above).
