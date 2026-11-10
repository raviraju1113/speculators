# Multilingual prompts v2 for GLM-5.3 on-policy regeneration (2026-11-10)

**Prompts only, no responses.** Every row needs a fresh GLM-5.3 reply. These are for training
the GLM-5.3 DFlash2 speculative draft, so the replies must be GLM-5.3's own (on-policy).

## Files

All files are JSONL in the same format as `prompts_nemo782k*.jsonl`, so `run_regen_glm53.sh`
reads them as-is:

```json
{"id": "mling:<lang>:<source>:<n>", "source": "<source>", "conversations": [{"from": "human", "value": "..."}]}
```

- `prompts_multilingual_all.jsonl`: all 152,496 rows, shuffled. Use this, or the per-language files.
- `prompts_multilingual_<lang>.jsonl`: one file per language. `translation.jsonl` holds the
  translation instructions.
- `manifest.json`: rows and sources per file, plus how many rows were dropped.

| file | rows | sources |
|---|---:|---|
| es / fr / de / ja | 15,000 each | 10k Nemotron-v2 + 5k WildChat-4.8M |
| it | 7,500 | 5k Nemotron-v2 + 2.5k WildChat |
| zh | 25,000 | WildChat |
| ru / ar / ko / pt / vi | 7,500 each | WildChat |
| th | 7,500 | 486 WildChat + 7,014 WangchanThaiInstruct |
| translation | 14,996 | OPUS-100 train: 2k German->English, plus both directions for 38 languages incl. long-tail ones (Basque, Nynorsk, Gaelic, Malagasy, Gujarati, ...) |

## How the prompts were chosen

- **Nemotron-Post-Training-Dataset-v2** (multilingual_{es,fr,de,ja,it}): only the user turns are
  used, never its responses. Its reasoning traces are English-only and were written by other
  models. The raw split is about 60% math, so each language is sampled stratified: 40% chat,
  40% math, 15% code, 5% multiple choice.
- **WildChat-4.8M**: the first user turn of real chats. Language comes from WildChat's own
  `language` field. Rows flagged toxic or redacted are dropped. These are real user prompts, so
  expect some spam or ad copy and some sensitive topics.
- Prompts are single-turn, 20 to 6,000 characters, and de-duplicated on normalized text.
- **Decontaminated** against all 7,588 prompts in our eval suite (`mtp_server_eval/data`): exact
  matches are removed, and for non-Nemotron sources so are substring matches.

## Regeneration settings

Match the earlier nemo782k regen:

- GLM-5.3 at `reasoning_effort=max`, T=1.0, top_p=0.95, the same `max_tokens` as before (16384).
  Drop truncated replies later (`prepare_onpolicy_glm53.sh` does this).
- **Single-node serving only (TP8, PP=1).** Two-node PP=2 pair serving produced looping,
  degenerate replies in the old shard B (37% truncated vs 8.6% for single-node shard A).
- Keep `id` and `source` so each language can be tracked through training and eval.

Estimated cost: about 3k output tokens per reply, so roughly 450M tokens in total, or about
4-6 node-hours per 10k prompts on one FP8 node.

## Licenses

- Nemotron-v2: CC BY 4.0, gated. Its WildChat-derived prompts are ODC-BY.
- WildChat-4.8M: ODC-BY.
- WangchanThaiInstruct: CC BY-SA 4.0.
- OPUS-100: per-corpus licenses.
- Aya and KULLM-v2 were not needed in v2.

## Reproducing

```
python examples/train/glm53/multilingual/build_multilingual_prompts.py \
    --raw /import/ml-sc-scratch1/ravir/spec_decoding_work/multilingual_data/raw \
    --out /import/ml-sc-scratch1/ravir/spec_decoding_work/multilingual_data/prompts_v2
```

The builder is on branch `ravir/glm53-dflash2-kimi-recipe`. Nemotron needs an HF token that has
accepted its terms. Re-running with seed 0 gives the same rows.
