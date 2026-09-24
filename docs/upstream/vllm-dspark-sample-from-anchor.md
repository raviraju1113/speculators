# vLLM + speculators: DSpark `sample_from_anchor=True` checkpoints silently serve at ~half acceptance

**Versions:** speculators (DSpark training default) + vLLM speculators loader

## Summary
speculators trains DSpark with `sample_from_anchor=True` by default, but the
vLLM speculators loader hardcodes the opposite convention and never reads the
field from the checkpoint. The result loads with **no error or warning** and
reports roughly **half** the acceptance it should.

Measured on gemma-4-31B-it, same weights, one-line change:

| | accept_len (aime) | decode tok/s | speedup |
|---|---|---|---|
| default (mismatched) | 1.838 | 33.9 | 1.61x |
| conventions matched | **3.569** | **66.4** | **3.16x** |

## Cause
`sample_from_anchor` defines what the draft's output slots *mean*:

| | block layout | slot k predicts | draft tokens (block_size=8) |
|---|---|---|---|
| `True` | `[anchor, noise, ...]` | token p+k+1 | 8 |
| `False` | `[anchor(bonus), mask@1, ...]` | token p+k | 7 |

It also flips the Markov head's conditioning (shifted vs unshifted), so a
mismatch is a double shift — hence acceptance degrades hard rather than losing
one token.

- Training default: `speculators/models/dflash/core.py` —
  `default_sample_from_anchor = algorithm == "dspark"` → **True**
- Serving: `vllm/transformers_utils/configs/speculators/algos.py` sets
  `dspark_bonus_anchor = True` unconditionally, and the runtime derives
  `sample_from_anchor = not dspark_bonus_anchor` → **always False**

Editing the checkpoint's `config.json` does nothing — the convention lives in
what the weights learned.

## Suggested fix (either side)
vLLM: read the field the checkpoint already ships.
```python
pre_trained_config["dspark_bonus_anchor"] = not config_dict.get("sample_from_anchor", False)
```
This is byte-identical for `False` checkpoints (`not False == True`, the value
currently hardcoded) and only changes behaviour for `True` ones.

Or speculators: default DSpark to `sample_from_anchor=False` so exported
checkpoints match the serving convention.

## Impact
The default training configuration cannot be served correctly by the path it
exports for, and fails **silently** — users see a working draft at half its
real speedup, with no warning to investigate.
