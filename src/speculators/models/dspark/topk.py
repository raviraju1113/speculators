"""DSA-style top-k context selection for the DSpark draft.

DeepSeek Sparse Attention (DeepSeek-V3.2-Exp) scores every cached key with a
cheap "lightning indexer" and lets each query attend only to its top-k keys.
Here the same idea is applied to the DFlash/DSpark draft: a block of draft
queries (anchor + mask tokens) attends to the verifier's projected hidden states
(the "context KV"). For the layers listed in ``topk_layers`` the block attends
only to the ``topk_context`` context positions its indexer ranks highest, plus
an always-on local window, instead of the whole document prefix.

Pieces (all pure functions except the indexer module in ``model_definitions``):

* :func:`build_context_candidates` -- which context positions a block may attend
  at all (same document, before the anchor) and which are in the local window.
* :func:`select_topk_context` -- top-k over indexer scores among the candidates.
* :func:`create_selected_context_mask_mod` -- flex-attention mask for a layer
  that attends only to the selected context positions and its own block.
* :func:`dense_context_attention_probs` -- the layer's *dense* attention
  distribution over the context, pooled over heads and block queries. This is
  the indexer's training target (DSA trains the indexer by KL to the main
  attention's head-summed, L1-normalised scores).
* :func:`indexer_kl` -- that KL, optionally restricted to the selected set
  (DSA's sparse stage computes it over the selected tokens only).
"""

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import or_masks

from speculators.models.dflash.model_definitions import _rotate_half

__all__ = [
    "build_context_candidates",
    "create_selected_context_mask_mod",
    "dense_context_attention_probs",
    "indexer_kl",
    "select_topk_context",
    "selection_recall",
]

_NEG_INF = float("-inf")
_MASK_VALUE = -1.0e30  # finite stand-in for -inf where gradients must stay finite


def build_context_candidates(
    document_ids: torch.Tensor,  # [total_seq_len], pad = -1
    anchor_positions: torch.Tensor,  # [num_anchors]
    total_seq_len: int,
    local_window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(valid, local)`` boolean maps of shape ``[num_anchors, total_seq_len]``.

    ``valid[j, t]`` mirrors DFlash's ``base_prefix_mod``: position ``t`` is in the
    same document as anchor ``j`` and strictly before it. ``local[j, t]`` is the
    subset within ``local_window`` positions of the anchor (empty if
    ``local_window <= 0``); those are always attended, never ranked.
    """
    device = document_ids.device
    anchor_positions = anchor_positions.to(device=device, dtype=torch.long)
    pos = torch.arange(total_seq_len, device=device)
    anchor_doc = document_ids[anchor_positions]  # [N]
    valid = (
        (document_ids[None, :] == anchor_doc[:, None])
        & (anchor_doc[:, None] != -1)
        & (pos[None, :] < anchor_positions[:, None])
    )
    if local_window > 0:
        local = valid & (pos[None, :] >= (anchor_positions[:, None] - local_window))
    else:
        local = torch.zeros_like(valid)
    return valid, local


def select_topk_context(
    scores: torch.Tensor,  # [num_anchors, total_seq_len]
    valid: torch.Tensor,  # [num_anchors, total_seq_len] bool
    local: torch.Tensor,  # [num_anchors, total_seq_len] bool
    k: int,
) -> torch.Tensor:
    """Select ``k`` ranked context positions per block (plus the local window).

    Ranking is over ``valid & ~local``; the local window is always included.
    Blocks with fewer than ``k`` candidates keep all of them.
    """
    if k <= 0:
        return local.clone()
    candidates = valid & ~local
    masked = scores.float().masked_fill(~candidates, _NEG_INF)
    k_eff = min(k, masked.shape[-1])
    top_vals, top_idx = masked.topk(k_eff, dim=-1)
    selected = torch.zeros_like(valid)
    selected.scatter_(1, top_idx, torch.isfinite(top_vals))
    return selected | local


def create_selected_context_mask_mod(
    total_seq_len: int,
    num_anchors: int,
    block_size: int,
    selected: torch.Tensor,  # [num_anchors, total_seq_len] bool
):
    """Flex mask: each query block sees its selected context positions and its own
    block (bidirectional, as in DFlash full attention). Returns
    ``(mask_mod, q_len, kv_len)`` like ``create_anchor_block_mask_mod``.
    """
    q_len = num_anchors * block_size
    kv_len = total_seq_len + q_len

    def selected_ctx_mod(_b, _h, q_idx, kv_idx):
        q_block = q_idx // block_size
        kv_is_base = kv_idx < total_seq_len
        kv_base_pos = torch.remainder(kv_idx, total_seq_len)
        return kv_is_base & selected[q_block, kv_base_pos]

    def same_block_mod(_b, _h, q_idx, kv_idx):
        q_block = q_idx // block_size
        kv_is_block = kv_idx >= total_seq_len
        kv_block = (kv_idx - total_seq_len) // block_size
        return kv_is_block & (q_block == kv_block)

    return or_masks(selected_ctx_mod, same_block_mod), q_len, kv_len


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [1, H, L, d]; cos/sin: [1, L, d]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + _rotate_half(x) * sin


@torch.no_grad()
def dense_context_attention_probs(
    attn: torch.nn.Module,
    layer_input: torch.Tensor,  # [1, num_anchors*block_size, hidden], post-layernorm
    ctx_hidden: torch.Tensor,  # [1, total_seq_len, hidden] (fc_output)
    position_embeddings: tuple[torch.Tensor, torch.Tensor],  # cos/sin [1, T+Q, d]
    valid: torch.Tensor,  # [num_anchors, total_seq_len] bool
    block_size: int,
    chunk_blocks: int = 32,
) -> torch.Tensor:
    """The layer's dense attention over the context, pooled to ``[num_anchors, T]``.

    Uses the attention module's own q/k projections, norms and RoPE, scores every
    valid context position, averages the softmax over heads and over the block's
    queries, and renormalises per block. Block-internal keys are excluded, so the
    result is the distribution the indexer should imitate.
    """
    cos, sin = position_embeddings
    total_seq_len = ctx_hidden.shape[1]
    num_anchors = valid.shape[0]
    head_dim = attn.head_dim

    k = attn.k_proj(ctx_hidden).view(1, total_seq_len, -1, head_dim)
    k = attn.k_norm(k).transpose(1, 2)  # [1, Hk, T, d]
    k = _rope(k, cos[:, :total_seq_len], sin[:, :total_seq_len])

    q = attn.q_proj(layer_input).view(1, layer_input.shape[1], -1, head_dim)
    q = attn.q_norm(q).transpose(1, 2)  # [1, Hq, Q, d]
    q = _rope(q, cos[:, total_seq_len:], sin[:, total_seq_len:])

    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
    k_t = k.transpose(-1, -2)  # [1, Hq, d, T]

    out = torch.zeros(num_anchors, total_seq_len, device=q.device, dtype=torch.float32)
    for start in range(0, num_anchors, chunk_blocks):
        end = min(start + chunk_blocks, num_anchors)
        qc = q[:, :, start * block_size : end * block_size]  # [1, Hq, qc, d]
        scores = torch.matmul(qc, k_t).float() * attn.scaling  # [1, Hq, qc, T]
        valid_q = valid[start:end].repeat_interleave(block_size, dim=0)  # [qc, T]
        scores = scores.masked_fill(~valid_q[None, None], _NEG_INF)
        probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        pooled = probs.mean(dim=1)[0]  # [qc, T]
        pooled = pooled.view(end - start, block_size, total_seq_len).sum(dim=1)
        out[start:end] = pooled / pooled.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return out


def indexer_kl(
    scores: torch.Tensor,  # [num_anchors, T] indexer scores (with grad)
    target_probs: torch.Tensor,  # [num_anchors, T] dense attention probs
    restrict: torch.Tensor,  # [num_anchors, T] bool: positions the KL ranges over
    block_valid: torch.Tensor,  # [num_anchors] bool
) -> torch.Tensor:
    """KL(target || softmax(scores)) over ``restrict``, averaged over valid blocks.

    Masked positions get a large finite negative score rather than ``-inf`` so
    rows with nothing to attend (padded anchors, anchor at position 0) stay
    finite in forward and backward instead of producing NaN gradients.
    """
    logq = torch.log_softmax(scores.float().masked_fill(~restrict, _MASK_VALUE), dim=-1)
    p = target_probs.float() * restrict
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    kl = (p * (torch.log(p.clamp_min(1e-12)) - logq)).masked_fill(~restrict, 0.0)
    kl = kl.sum(dim=-1)  # [N]
    weight = (block_valid & restrict.any(dim=-1)).float()
    return (kl * weight).sum() / weight.sum().clamp_min(1.0)


def selection_recall(
    target_probs: torch.Tensor,  # [num_anchors, T]
    selected: torch.Tensor,  # [num_anchors, T] bool
    block_valid: torch.Tensor,  # [num_anchors] bool
) -> torch.Tensor:
    """Fraction of the dense attention mass that falls inside the selected set."""
    captured = (target_probs.float() * selected).sum(dim=-1)
    weight = block_valid.float()
    return (captured * weight).sum() / weight.sum().clamp_min(1.0)
