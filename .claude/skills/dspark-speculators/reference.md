# DSpark math and pseudo-code reference

Source: "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation" (arXiv:2607.05147). Notation: γ = block size, V = vocab, d = backbone hidden dim, r = Markov rank (default 256), x_0 = anchor token, h_k = backbone hidden state at position k, U_k = base logits at position k.

## 1. Markov head (sequential stage)

Draft distribution with transition bias:

```
p_k(v | x_0, x_<k) = softmax(U_k + B_k)[v]
B(x_{k-1}, ·) = W1[x_{k-1}] @ W2        # W1: V×r lookup, W2: r×V projection
```

Position 1 conditions only on the anchor (its "previous token" is x_0).

```python
class MarkovHead(nn.Module):
    def __init__(self, vocab, rank=256):
        self.W1 = nn.Embedding(vocab, rank)
        self.W2 = nn.Linear(rank, vocab, bias=False)

    def forward(self, prev_tokens):          # [B, γ] ground-truth-shifted at train time
        return self.W2(self.W1(prev_tokens)) # [B, γ, V] bias added to base logits
```

Training uses teacher forcing: `prev_tokens[:, k] = x*_{k-1}` (ground truth), so no loop.
Inference (vLLM side) samples left-to-right, feeding each sampled token as the next `prev`.

Optional RNN head (paper §3.1) exists but adds marginal gains; skip it — Markov is the paper's default and production choice.

## 2. Confidence head

```
c_k = sigmoid(w^T [h_k ; W1[x_{k-1}]])    # single linear layer on concat, reuses Markov W1
```

```python
class ConfidenceHead(nn.Module):
    def __init__(self, hidden, rank=256):
        self.proj = nn.Linear(hidden + rank, 1)

    def forward(self, h, prev_emb):           # h: [B, γ, d], prev_emb: [B, γ, r]
        return torch.sigmoid(self.proj(torch.cat([h, prev_emb], -1))).squeeze(-1)
```

`c_k` models the CONDITIONAL probability that position k survives verification given
positions 1..k−1 survived. Prefix survival = cumulative product `a_j = ∏_{i≤j} c_i`.

## 3. Losses

Position weights (emphasize early positions): `w_k = exp(-(k-1)/γ)`.

```
L_ce   = − Σ_k w_k · log p_draft_k(x*_k)                      # x*_k = regenerated ground truth
L_tv   =   Σ_k w_k · || p_draft_k − p_target_k ||₁            # p_draft includes Markov bias
L_conf = − Σ_k w_k · [ c*_k · log c_k + (1−c*_k) · log(1−c_k) ]
c*_k   = 1 − ½ · || p_draft_k.detach() − p_target_k ||₁       # soft label; per-step analytical acceptance
L      = 0.1·L_ce + 0.9·L_tv + 1.0·L_conf
```

Notes:
- `p_target_k` = softmax of the frozen shared LM head applied to the verifier's hidden
  state at the corresponding position (available from online hidden-state streaming).
- TV distance is a direct proxy for acceptance: per-step acceptance rate = 1 − ½·TV
  (Leviathan et al. 2023), so minimizing L_tv maximizes expected accepted length.
- Detach the draft distribution inside `c*_k` — confidence labels must not backprop
  through the drafter.
- Frozen: verifier, shared embeddings, shared LM head. Trained: backbone, Markov head,
  confidence head.

## 4. Sequential Temperature Scaling (STS)

Post-training, on a held-out split. Collect per-position raw confidences `c_k` and the
empirical accept/reject outcomes from actual verification rollouts (or analytical c*_k).

```
for k in 1..γ:                      # strictly left to right
    grid-search T_k over e.g. [0.25, 4.0]:
        c_k_cal = sigmoid(logit(c_k) / T_k)
        a_k     = (∏_{i<k} c_i_cal_frozen) · c_k_cal      # cumulative product
        score   = ECE(a_k vs empirical prefix-survival at position k)
    fix T_k = argmin score; freeze position k's calibrated scores
store [T_1..T_γ] in checkpoint config
```

Temperature scaling is order-preserving: it fixes absolute magnitudes (needed by the
throughput scheduler) without changing token rankings. Paper reports ECE dropping from
5.7–8.2% to 0.4–2.0% after STS.

## 5. Default hyperparameters (paper / DeepSpec release)

| Setting | Value |
|---|---|
| Drafter layers | 5 (2 already beats 5-layer DFlash) |
| Block size γ | 7 (offline benchmarks); 5 (DeepSeek production) |
| Markov rank r | 256 |
| Loss weights (ce/tv/conf) | 0.1 / 0.9 / 1.0 |
| Position weight | w_k = exp(−(k−1)/γ) |
| Epochs | 10 (DeepSpec) — speculators DFlash tutorial uses 5 with lr 3e-4 as a starting point |
| Eval temperature | 1.0, chain drafting, τ includes bonus token |

## 6. Parity targets (DeepSpec released checkpoints, paper Table 1, block 7)

| Benchmark | Qwen3-8B DSpark | Gemma4-12B DSpark |
|---|---|---|
| GSM8K | 6.17 | 6.05 |
| MATH500 | 5.78 | 5.78 |
| MBPP | 5.16 | 5.11 |
| HumanEval | 5.52 | 5.64 |
| MT-Bench | 3.72 | 3.49 |
| Alpaca | 3.58 | 3.35 |

(Accepted length τ per round, temp 1.0, incl. bonus token. Reference checkpoints:
deepseek-ai/dspark_qwen3_8b_block7, deepseek-ai/dspark_gemma4_12b_block7.)
