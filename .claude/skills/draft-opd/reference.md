# Draft-OPD reference: loss, data formats, interfaces

## 1. Loss (paper eqs. 10-13, App. B)

For anchor m with drafted block `d_m` (length K) and `r_m` accepted tokens:

```
I_acc = {k : 1 <= k <= r_m}          I_rej = {k : r_m < k <= K}
L_acc = mean_{k in I_acc}  KL( p_{m,k} || q_{m,k} )            # forward KL
L_rej = sum_{k in I_rej} w_k KL( q_{m,k} || p_{m,k} ) / sum w_k  # reverse KL
w_k   = gamma^(k-1), gamma = 0.8      (paper: insensitive in 0.5..0.95)
L     = (lambda_acc L_acc + lambda_rej L_rej) / (lambda_acc + lambda_rej),  lambdas = 1
```

`p_{m,k} = p_theta(. | c_m, d_{m,<k})` (target, full vocab), `q_{m,k} = q_phi(. | c_m, d_{m,<k})`
(draft). Forward KL on accepted = target-weighted cross-entropy (eq. 16);
reverse KL on rejected = draft-weighted disagreement, penalizes the draft's own
high-probability modes the target does not support (eq. 18).

**First-rejected-only variant (Sec. 6.1, use this):** `I_rej = {r_m + 1}` only
(if `r_m < K`). Same tau as the full objective; the target needs scoring only on
`d_{m,1..r_m+1}`.

**Draft vocab:** q lives on the 32k draft vocab, p on the 262k target vocab.
Compute the KLs on the draft-vocab support mapped through `d2t` (as the TV loss
does: target probs gathered at `d2t` indices; fold the out-of-vocab mass into a
"rest" bucket so p sums to 1 - keep it identical in the parity check).
Out-of-vocab drafted tokens cannot occur (the draft never proposes them); an
out-of-vocab *target* token at an accepted position cannot occur either (it
would have been rejected).

**Where it plugs in:** `src/speculators/models/dspark/core.py` computes
`verifier_logits` from `verifier_last_hidden_states` and the draft logits
(`hidden -> lm_head`, plus `markov_head` bias) per block position. The OPD loss
replaces the per-position `{ce: 0.1, tv: 0.9}` on OPD samples; the confidence
head keeps its TV-based labels. Keep `w_k = exp(-(k-1)/gamma_dflash)` OFF for
OPD positions (the paper decays rejected tokens only, with its own gamma).

## 2. Rollout record format (one jsonl line per request)

```json
{
  "rollout_id": "r-000123", "draft_ckpt": "conv_blk12/checkpoints/33", "round": 3,
  "prompt_ids": [..], "sampling": {"temperature": 0.0, "max_tokens": 2048},
  "response_ids": [..],
  "steps": [
    {"anchor": 0,  "draft_ids": [t1..tK], "accepted": 5},
    {"anchor": 6,  "draft_ids": [..],     "accepted": 0},
    ...
  ],
  "k": 8, "finish": "eos|length"
}
```

`anchor` = number of response tokens already verified when the block was
proposed (`a_m`); `draft_ids` = the K proposals *before* verification;
`accepted = r_m`. vLLM's verify step also emits one target ("bonus") token, so
the next anchor is `anchor + accepted + 1`; the bonus token is part of
`response_ids`, not of the block. Invariant to assert when reading:
`response_ids[anchor : anchor+accepted] == draft_ids[:accepted]`.

Source of truth in vLLM: the DSpark speculator (`v1/worker/gpu/spec_decode/dspark/speculator.py`,
`_generate_draft`) produces the block for step t; the model runner's rejection
sampler at step t+1 yields the accepted length. The hook must emit (request id,
step, draft ids, accepted) off the CUDA-graph path: copy after the step, on the
host, write asynchronously. Enabled by env var `DSPARK_ROLLOUT_LOG=<dir>`.

## 3. Replay sample format (what the trainer consumes)

Per anchor, one training sample:

```
input_ids      = prompt_ids + response_ids[:anchor] + draft_ids[:s]     # s = min(accepted+1, K)
anchor_pos     = len(prompt_ids) + anchor                                # block starts here
block          = draft_ids[:K]      (draft forward: K mask slots at anchor_pos..anchor_pos+K-1)
labels         = draft_ids[:s]      (positions to score); accepted mask = k <= accepted
teacher_hidden = last hidden state at positions anchor_pos-1 .. anchor_pos+s-2  (p for slots 1..s)
```

Hidden states come from the hidden-state server prefill of `input_ids`
(aux layers 1/17/29/47/58 + last, exactly as SFT). The draft forward uses the
context part (`< anchor_pos`) as K/V and K mask slots as queries; q at slot k adds
`markov_head(draft_ids[k-1])` (slot 1 uses the anchor token).

Packing: group all anchors of one rollout; their contexts share the prefix, so
with prefix caching the incremental prefill is `s` tokens per anchor. Without it,
cap rollouts at 2048 tokens and accept ~N_anchors x len prefill per rollout.

## 4. Round loop (sketch)

```
for round in range(R):
    serve = start_vllm(target, draft=ckpt[round], k=K, DSPARK_ROLLOUT_LOG=...)
    rollouts = spec_decode(serve, prompt_pool.sample(P))
    stop(serve)
    samples = replay(rollouts, hidden_state_server)               # first-rejected-only
    ckpt[round+1] = train(ckpt[round], samples, steps=N, loss=OPD)  # speculators trainer
```

Paper: P x R x epochs ~ 16K prompts x 8 passes, ~5.3K steps at batch 24 prompts.
Start at 1/10 of that for the pilot. Between rounds the data is slightly
off-policy; keep N small (<= 1/8 of an epoch over the pool).

## 5. Parity checks (gates, in order)

1. Rollout invariant above holds on 100% of logged steps.
2. Replayed q (trainer) == serving draft logits at the same anchor on 20 anchors
   (max abs diff at bf16 tolerance), with and without the Markov bias.
3. Replayed accepted count recomputed from p and q (greedy verify) == logged r_m
   on >= 99% of anchors at temperature 0.
4. Pilot: 1K prompts x 2 rounds; val accept_len on the SFT val split must not
   drop > 0.05, and the first-rejection histogram (App. D style) must shift right.
