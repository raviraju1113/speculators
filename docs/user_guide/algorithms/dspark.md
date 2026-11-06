# DSpark

DSpark extends [DFlash](dflash.md) with two heads on top of the same block-parallel draft backbone: a low-rank Markov head that biases each draft position by the token before it, and a confidence head that predicts per-position acceptance probability. Pure block-parallel drafting has no dependency between tokens inside a block, so acceptance decays toward the end of the block -- the Markov head restores that dependency, and the confidence signal indicates how far a block is worth verifying. The draft model subclasses DFlash, so the architecture and training pipeline are otherwise unchanged, and it can be paired with any supported verifier. Serving uses vLLM's own `dspark` method (`"method": "dspark"` in `--speculative-config`).

## How It Works

### Markov Head

The head adds a low-rank logit bias `B = W1 @ W2` to the DFlash logits: `W1` embeds the previous block token (verifier vocabulary) into `markov_rank` dimensions and `W2` projects to the draft vocabulary. Three variants are available:

- **`vanilla` (default)**: bias from the previous token alone
- **`gated`**: bias gated by the backbone hidden state
- **`rnn`**: recurrent state carried across positions within the block

Setting `--markov-rank 0` disables the head, leaving pure DFlash drafting. It must be paired with `--no-confidence-head-with-markov`, since that option requires a Markov head.

### Confidence Head

A linear head predicts each position's acceptance probability from the backbone hidden state, concatenated with the Markov previous-token embedding when `--confidence-head-with-markov` is set. It is trained with a BCE term weighted by `--confidence-head-alpha`.

### Sample From Anchor

DSpark defaults to `sample_from_anchor: True` -- the anchor and all mask positions predict future tokens, producing `block_size` speculative tokens. See [DFlash](dflash.md#sample-from-anchor) for details.

## Key Parameters

| Parameter                       | Default   | Description                                                       |
| ------------------------------- | --------- | ----------------------------------------------------------------- |
| `--markov-rank`                 | 256       | Low-rank dimension of the Markov logit bias (0 disables the head) |
| `--markov-head-type`            | `vanilla` | Sequential head variant: `vanilla`, `gated`, or `rnn`             |
| `--enable-confidence-head`      | enabled   | Attach the per-position acceptance head                           |
| `--confidence-head-with-markov` | enabled   | Feed the Markov previous-token embedding into the confidence head |
| `--confidence-head-alpha`       | 1.0       | Weight of the confidence-head BCE term                            |
| `--topk-context`                | 0         | DSA-style top-k context selection: positions per block (0 = dense)|
| `--topk-layers`                 | auto      | Full-attention layers that use the selection (default: all but 0) |
| `--topk-local-window`           | 128       | Context positions before the anchor always attended               |
| `--indexer-heads` / `--indexer-head-dim` | 4 / 64 | Lightning-indexer size                                         |
| `--indexer-loss-weight`         | 1.0       | Weight of the indexer KL term                                     |
| `--topk-warmup-steps`           | 0         | Dense steps while the indexer trains before going sparse          |

All DFlash parameters (`--block-size`, `--max-anchors`, `--num-layers`, ...) apply unchanged.

## Pretrained Models

Pretrained DSpark speculator models are available on HuggingFace from the [RedHatAI speculator models collection](https://huggingface.co/collections/RedHatAI/speculator-models):

| Verifier              | Speculator                                                                                                        |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `zai-org/GLM-5.2-FP8` | [`RedHatAI/GLM-5.2-speculator.dspark-preview`](https://huggingface.co/RedHatAI/GLM-5.2-speculator.dspark-preview) |

To train your own, see `examples/train/dspark_qwen3_0_6b_sharegpt_online.sh`.

## Top-k Context Selection (DSA-style, experimental)

DSpark's draft layers attend from a block of queries (anchor + mask tokens) to
the verifier's projected hidden states for every earlier token in the document,
the "context KV". With `--topk-context K` the full-attention layers listed in
`--topk-layers` attend only to the `K` context positions a small **lightning
indexer** ranks highest for that block, plus an always-on local window
(`--topk-local-window`, default 128). This is the DeepSeek Sparse Attention
recipe (DeepSeek-V3.2-Exp) applied to the draft:

- **Indexer** (per top-k layer): `I[block, s] = Σ_q Σ_h w_h(q) · ReLU(q_h(q) · k(s))`
  with `--indexer-heads` small query heads of `--indexer-head-dim` and a single
  key projection of the context. Scores are pooled over the block's queries
  because one decoding step drafts one block and shares one KV selection.
- **Indexer training**: KL from the layer's *dense* attention distribution over
  the context (averaged over heads and block queries) to `softmax(I)`, weighted
  by `--indexer-loss-weight`. The indexer's inputs are detached, so the draft
  loss never reaches it and the KL never reaches the draft.
- **Two stages, one run**: for the first `--topk-warmup-steps` steps attention
  stays dense while the indexer learns; after that the top-k layers switch to
  the selected set and the KL is computed over the selected positions only
  (DSA's sparse stage). `topk_sparse` in the metrics shows which stage a step
  used.
- **Default layers**: every full-attention layer except layer 0, whose queries
  are still bare anchor/mask embeddings and carry little to rank with.

Metrics: `indexer_kl`, `topk_recall` (share of the dense attention mass that
falls inside the selected set; the number to watch), `topk_density`
(selected / candidate positions), `topk_sparse`.

```bash
# fresh draft with top-k layers (dense warm-up for 1k steps, then sparse)
python scripts/train.py --speculator-type dspark ... \
    --topk-context 512 --topk-local-window 128 --topk-warmup-steps 1000

# warm-start a dense DSpark checkpoint: indexers initialise fresh, rest loads
python scripts/train.py --speculator-type dspark --from-pretrained <dense-dspark-dir> ... \
    --topk-context 512 --topk-warmup-steps 500
```

Caveats:

- A checkpoint with `topk_context > 0` carries `context_indexers.*` weights and
  needs engine support for the indexer at serving time. vLLM's DSpark proposer
  does not have it yet; the offline harness in
  `scripts/evaluate/kimi_k3_offline_eval/` evaluates such checkpoints through the
  training forward and is the way to measure acceptance until then.
- Evidence so far says the draft's useful context is mostly local: a 2048-token
  sliding window cost a full-attention Kimi-K3 DSpark checkpoint at most 0.5%
  accept_len on any of 24 sets, and Gemma-4 DSpark acceptance is flat from 1k to
  32k context. Top-k selection is therefore an efficiency and long-context
  hypothesis to test, not an established win. Compare `topk_recall` and
  accept_len against a dense run with the same budget.
- `--topk-warmup-steps` counts from process start; a resumed run restarts the
  count, so pass `0` when resuming a run that already reached the sparse stage.
- `val/loss` includes the indexer KL, which has heavy outliers (unnormalised
  scores); select checkpoints by `val/accept_len`, not `val/loss` or `--save-best`.

First result: [DSpark top-k context selection on Gemma-4-26B-A4B](../../experiments/gemma4_26b_moe_results.md) (§18, and Appendix A for the first A/B)
(32% density on the full-attention layer costs 1.2% accept_len; indexer contribution not yet isolated).

## Research & Citation

DSpark is based on research from DeepSeek: [arXiv Paper](https://arxiv.org/abs/2607.05147)

```bibtex
@article{cheng2026dspark,
  title={DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation},
  author={Cheng, Xin and Yu, Xingkai and Shao, Chenze and Li, Jiashi and Xiong, Yunfan and Qian, Yi and Zhu, Jiaqi and Ma, Shirong and Zhang, Xiaokang and Ye, Jiasheng and others},
  journal={arXiv preprint arXiv:2607.05147},
  year={2026}
}
```

## See Also

- [DFlash](dflash.md) -- The base algorithm DSpark extends
- [Train a Speculator](../tutorials/train.md) -- Step-by-step training guide (select DSpark, then online, offline, or hybrid)
- [Gemma-4-31B Full-Suite Spec Decode Results](../tutorials/gemma4_31b_full_spec_decode_results.md) -- DSpark Qwen (Mengmeng) vs Google Assistant (MTP) / Eagle-3 on the 25-bench suite
