# Train/eval contamination report

Rows scanned: {'kimi': 349389, 'nemotron_stem': 355000, 'nemotron_code': 174998, 'nemotron_math': 239467}

Policy **exact-only**: a benchmark item counts as contaminated only when its prompt matches a training prompt exactly (after tier-0 template stripping and whitespace/case/punctuation folding). Near-duplicates -- a reformatted or reworded copy of the same problem -- are scored below and listed in `review_pairs.jsonl`, but are NOT counted as contamination. The `graded *` columns show what the stricter rule would have flagged.

n-gram orders: 13 (primary), 8 (short items). Thresholds: {'run_dirty': 25, 'run_suspect': 13, 'ngram13_dirty': 0.2, 'ngram8_dirty': 0.5, 'minhash_dirty': 0.7, 'minhash_suspect': 0.5}

## Per-benchmark

| benchmark | items | dirty | exact | evaluated | eval dirty | graded dirty | graded suspect | eval graded dirty | max run (tok) | max 13-gram | max minhash |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aime | 30 | 0 | 0 | 30 | 0 | 2 | 8 | 2 | 40 | 0.373 | 0.000 |
| gpqa | 198 | 0 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.000 |
| gsm8k | 1319 | 1 | 1 | 50 | 0 | 6 | 3 | 0 | 61 | 1.000 | 1.000 |
| humaneval | 164 | 0 | 0 | 50 | 0 | 42 | 27 | 11 | 76 | 1.000 | 0.984 |
| livecodebench | 1055 | 0 | 0 | 50 | 0 | 19 | 493 | 0 | 379 | 1.000 | 0.969 |
| math500 | 500 | 2 | 2 | 50 | 1 | 57 | 29 | 7 | 118 | 1.000 | 1.000 |
| mbpp | 257 | 0 | 0 | 50 | 0 | 4 | 1 | 1 | 23 | 0.647 | 0.750 |

## Affected trained corpora

- `gemma4_draft_model_300k_eagle3`: 1 dirty eval items
- `gemma4_draft_model_300k_eagle3_kimi_mtp_stem`: 1 dirty eval items
- `gemma4_draft_model_300k_eagle3_kimi_mtp_stem_code`: 1 dirty eval items
- `gemma4_draft_model_900k_eagle3_kimi_mtp_stem_code_math`: 3 dirty eval items

## Next steps

1. Adjudicate `review_pairs.jsonl` by hand -- these are candidates, not
   verdicts. Shared boilerplate and common code idioms do produce
   false positives at the `suspect` level.
2. Run `impact_analysis.py` to split *measured* per-token acceptance into
   flagged vs clean. Only a material gap justifies a retrain/rerun.
