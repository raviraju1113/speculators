"""Sequential (Markov) and confidence heads for the DSpark draft model."""

import torch
from torch import nn

__all__ = [
    "ConfidenceHead",
    "ContextIndexer",
    "MarkovHead",
]


class MarkovHead(nn.Module):
    """Low-rank sequential logit bias ``B = W1 @ W2``.

    ``W1`` indexes the verifier vocabulary (the previous token id); ``W2`` projects
    to the draft vocabulary so the bias adds onto the DFlash logits.
    """

    def __init__(
        self,
        *,
        verifier_vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        hidden_size: int,
        head_type: str = "vanilla",
    ) -> None:
        super().__init__()
        if markov_rank <= 0:
            raise ValueError(f"markov_rank must be > 0, got {markov_rank}")
        if head_type not in ("vanilla", "gated", "rnn"):
            raise ValueError(f"Unsupported markov_head_type: {head_type!r}")
        self.head_type = head_type
        self.markov_rank = markov_rank
        self.markov_w1 = nn.Embedding(verifier_vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, draft_vocab_size, bias=False)
        if head_type == "gated":
            self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)
        elif head_type == "rnn":
            # Joint [gate; candidate; output] projection over [state; prev_emb; hidden].
            self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up W1 embeddings for the given previous-token ids."""
        return self.markov_w1(token_ids.long())

    def block_bias(
        self,
        *,
        prev_token_ids: torch.Tensor,  # [N, block_size]
        hidden_states: torch.Tensor,  # [N, block_size, hidden]
        prev_emb: torch.Tensor | None = None,  # [N, block_size, r]
    ) -> torch.Tensor:
        """Return the per-position logit bias, shape [N, block_size, draft_vocab]."""
        if prev_emb is None:
            prev_emb = self.prev_embeddings(prev_token_ids)
        prev_emb = prev_emb.to(self.markov_w2.weight.dtype)

        if self.head_type == "vanilla":
            return self.markov_w2(prev_emb)

        if self.head_type == "gated":
            hidden_states = hidden_states.to(prev_emb.dtype)
            gate = torch.sigmoid(
                self.gate_proj(torch.cat([hidden_states, prev_emb], dim=-1))
            )
            return self.markov_w2(gate * prev_emb)

        # rnn: maintain a recurrent state across block positions.
        hidden_states = hidden_states.to(prev_emb.dtype)
        num_blocks, block_size, _ = prev_emb.shape
        state = prev_emb.new_zeros(num_blocks, self.markov_rank)
        outputs = []
        for k in range(block_size):
            z = torch.cat([state, prev_emb[:, k], hidden_states[:, k]], dim=-1)
            gate_raw, cand_raw, out_raw = self.joint_proj(z).chunk(3, dim=-1)
            gate = torch.sigmoid(gate_raw)
            state = gate * state + (1.0 - gate) * torch.tanh(cand_raw)
            outputs.append(self.markov_w2(torch.tanh(out_raw)))
        return torch.stack(outputs, dim=1)


class ConfidenceHead(nn.Module):
    """Per-position acceptance-probability predictor (linear -> scalar logit)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


class ContextIndexer(nn.Module):
    """Lightning indexer (DeepSeek Sparse Attention) for one draft layer.

    Scores every context position ``s`` for a draft block::

        I[block, s] = sum_{q in block} sum_h  w_h(q) * ReLU(q_h(q) . k(s))

    with a handful of small query heads and a single key projection. The score
    is pooled over the block's queries because the whole block shares one KV
    selection (one decoding step drafts one block). Inputs are expected to be
    detached: the indexer learns only through its KL loss, never through the
    draft loss (DSA keeps the two gradient paths separate).
    """

    def __init__(
        self, hidden_size: int, num_heads: int = 4, head_dim: int = 64
    ) -> None:
        super().__init__()
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError(
                "indexer needs num_heads > 0 and head_dim > 0, "
                f"got {num_heads}, {head_dim}"
            )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.w_proj = nn.Linear(hidden_size, num_heads, bias=False)

    def forward(
        self,
        block_hidden: torch.Tensor,  # [num_blocks, block_size, hidden]
        ctx_hidden: torch.Tensor,  # [total_seq_len, hidden]
        chunk_blocks: int = 256,
    ) -> torch.Tensor:  # [num_blocks, total_seq_len] float32
        num_blocks, block_size, _ = block_hidden.shape
        k = self.k_proj(ctx_hidden)  # [T, d]
        q = self.q_proj(block_hidden).view(
            num_blocks, block_size, self.num_heads, self.head_dim
        )
        w = self.w_proj(block_hidden)  # [N, B, H]
        out = []
        for start in range(0, num_blocks, chunk_blocks):
            end = min(start + chunk_blocks, num_blocks)
            scores = torch.einsum("nbhd,td->nbht", q[start:end], k).relu()
            scores = (scores * w[start:end].unsqueeze(-1)).sum(dim=(1, 2))  # [n, T]
            out.append(scores.float())
        return torch.cat(out, dim=0)
