"""Loss and metrics for the DSpark draft model.

loss = compound_loss(logits, targets) + conf_alpha * BCE(confidence, accept_rate)

The confidence target ``accept_rate = sum_v min(q_v, p_v) = 1 - d_TV`` is the
analytical acceptance rate (the overlap ``tv_loss`` already computes).
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from speculators.losses import (
    LossConfig,
    compound_loss,
    dflash_loss_decay,
    dpace_loss_decay,
    tv_loss,
)
from speculators.models.metrics import compute_accuracy_multi_step

__all__ = [
    "compute_metrics",
    "opd_loss",
]

_EPS = 1e-8


def _forward_kl(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per position, p = softmax(targets), q = softmax(logits). [1, T]"""
    logq = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
    logp = torch.nn.functional.log_softmax(targets, dim=-1, dtype=torch.float32)
    return (logp.exp() * (logp - logq)).sum(dim=-1)


def _reverse_kl(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """KL(q || p) per position. [1, T]"""
    logq = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
    logp = torch.nn.functional.log_softmax(targets, dim=-1, dtype=torch.float32)
    return (logq.exp() * (logq - logp)).sum(dim=-1)


def opd_slot_masks(
    accepted: torch.Tensor,  # [num_blocks], accepted count r per block, -1 = not OPD
    block_size: int,
    first_rejected_only: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draft-OPD slot masks for ``sample_from_anchor=True`` blocks: slot j
    (0-based) predicts drafted token j+1, so slots ``j < r`` were accepted and
    slot ``j == r`` is the first rejected one (when ``r < block_size``).
    Returns ``(acc_mask, rej_mask)`` of shape ``[num_blocks, block_size]``."""
    slots = torch.arange(block_size, device=accepted.device).unsqueeze(0)
    r = accepted.unsqueeze(1)
    valid = r >= 0
    acc_mask = (slots < r) & valid
    rej_mask = ((slots == r) if first_rejected_only else (slots >= r)) & valid
    return acc_mask, rej_mask


def opd_loss(
    logits: torch.Tensor,  # [1, T, V]
    targets: torch.Tensor,  # [1, T, V]
    loss_mask: torch.Tensor,  # [1, T]
    accepted: torch.Tensor,  # [num_blocks]
    block_size: int,
    lambda_rej: float = 1.0,
    rej_gamma: float = 0.8,
    first_rejected_only: bool = True,
    rej_valid: torch.Tensor | None = None,  # [num_blocks, block_size] bool
    rej_kl: str = "reverse",  # "reverse" (paper) | "forward" (ablation)
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Acceptance-aware distillation (arXiv 2605.29343 eqs. 10-13): forward KL
    on accepted slots, ``gamma^(j-r)``-weighted reverse KL on rejected slots
    (just the first one by default), ``(L_acc + lambda L_rej) / (1 + lambda)``.

    ``rej_valid`` (False = the target's actual next token at that rejected slot
    is outside the draft vocabulary) drops such slots from the rejected term:
    ``targets`` are renormalized over the draft vocab, so there the reverse KL
    would pull the draft toward a mode the target never produced (28% of
    first-rejected slots on the sci16k pool, 2026-10-28).
    """
    num_blocks = accepted.shape[0]
    acc_mask, rej_mask = opd_slot_masks(accepted, block_size, first_rejected_only)
    valid = loss_mask.to(torch.bool).view(num_blocks, block_size)
    acc_w = (acc_mask & valid).to(torch.float32)
    slots = torch.arange(block_size, device=logits.device).unsqueeze(0)
    rej_w = (rej_mask & valid).to(torch.float32) * (
        rej_gamma ** (slots - accepted.clamp_min(0).unsqueeze(1)).clamp_min(0).to(torch.float32)
    )
    rej_oov = torch.zeros((), device=logits.device, dtype=torch.float32)
    if rej_valid is not None:
        rej_oov = ((rej_w > 0) & ~rej_valid.to(torch.bool)).sum().to(torch.float32)
        rej_w = rej_w * rej_valid.to(torch.float32)
    fwd = _forward_kl(logits, targets).view(num_blocks, block_size)
    rev = (_reverse_kl if rej_kl == "reverse" else _forward_kl)(logits, targets).view(
        num_blocks, block_size
    )
    l_acc = (fwd * acc_w).sum() / (acc_w.sum() + _EPS)
    l_rej = (rev * rej_w).sum() / (rej_w.sum() + _EPS)
    loss = (l_acc + lambda_rej * l_rej) / (1.0 + lambda_rej)
    scored = ((acc_w + rej_w) > 0).view(1, -1)
    terms = {
        "opd_acc_loss": l_acc.detach(),
        "opd_rej_loss": l_rej.detach(),
        "opd_acc_slots": acc_w.sum().detach(),
        "opd_rej_slots": (rej_w > 0).sum().detach().to(torch.float32),
        "opd_rej_oov_slots": rej_oov,
        "opd_scored_mask": scored,
    }
    return loss, terms


def _masked_decayed_mean(
    elementwise: torch.Tensor,  # [1, T]
    loss_mask: torch.Tensor,  # [1, T]
    pos_idx: torch.Tensor,  # [1, T]
    decay_fn: Callable[..., torch.Tensor] | None,
) -> torch.Tensor:
    """Masked, optionally position-decayed mean of a precomputed per-position term."""
    loss_mask = loss_mask.to(elementwise.dtype)
    weighted = elementwise * loss_mask
    if decay_fn is not None:
        weighted = weighted * decay_fn(
            pos_idx.to(weighted.dtype), elementwise_loss=elementwise
        )
    denominator = loss_mask.sum(dim=1) + _EPS
    return (weighted.sum(dim=1) / denominator).mean()


def compute_metrics(
    logits: torch.Tensor,  # [1, T, draft_vocab_size] (Markov-corrected)
    targets: torch.Tensor,  # [1, T, draft_vocab_size]
    confidence_logits: torch.Tensor | None,  # [1, T] or None
    loss_mask: torch.Tensor,  # [1, T]
    block_size: int,
    loss_config: LossConfig,
    tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
    gamma: float = 4.0,
    confidence_head_alpha: float = 1.0,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = True,
    opd_accepted: torch.Tensor | None = None,  # [num_blocks] Draft-OPD accepted counts
    opd_lambda_rej: float = 1.0,
    opd_rej_valid: torch.Tensor | None = None,  # [num_blocks, block_size], see opd_loss
    opd_rej_kl: str = "reverse",
) -> tuple[torch.Tensor, dict]:
    """Compute the DSpark loss and a metrics dict (``*_sum``/``*_total`` pairs)."""

    device = logits.device
    seq_len = logits.shape[1]
    pos_idx = (torch.arange(seq_len, device=device) % block_size).unsqueeze(0)
    start_pos = 0 if sample_from_anchor else 1
    if per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
    else:
        decay_fn = partial(
            dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
        )

    opd_terms = None
    if opd_accepted is not None:
        # Draft-OPD batch: acceptance-aware KL on the rollout's own slots; no
        # position decay (the paper decays rejected slots only, via rej_gamma).
        loss, opd_terms = opd_loss(
            logits, targets, loss_mask, opd_accepted, block_size,
            lambda_rej=opd_lambda_rej, rej_valid=opd_rej_valid, rej_kl=opd_rej_kl,
        )
        term_losses = {}
        # Confidence-head labels only where the draft was actually scored.
        loss_mask = loss_mask * opd_terms["opd_scored_mask"].to(loss_mask.dtype)
        decay_fn = None
    else:
        loss, term_losses = compound_loss(
            logits, targets, loss_mask, pos_idx, loss_config=loss_config, decay_fn=decay_fn
        )

    # Analytical per-position acceptance rate = distributional overlap
    # = 1 - TV; the fused kernel avoids the two full-vocab fp32 softmaxes.
    with torch.no_grad():
        accept_rate = 1.0 - tv_loss_fn(logits, targets)  # [1, T]
        # Per-block cumulative acceptance product over the draft slots (slot 0
        # is the anchor), shared by the accept-length and calibration metrics.
        num_blocks = seq_len // block_size
        accept_blocks = accept_rate.view(num_blocks, block_size)
        draft_mask = loss_mask.to(accept_rate.dtype).view(num_blocks, block_size)[
            :, start_pos:
        ]
        accept_prefix = (accept_blocks[:, start_pos:] * draft_mask).cumprod(dim=-1)

    metrics: dict[str, Any] = {}
    if confidence_logits is not None:
        c_star = accept_rate.detach().to(confidence_logits.dtype)
        bce = binary_cross_entropy_with_logits(
            confidence_logits, c_star, reduction="none"
        )  # [1, T]
        conf_loss = _masked_decayed_mean(bce, loss_mask, pos_idx, decay_fn)
        loss = loss + confidence_head_alpha * conf_loss

        with torch.no_grad():
            mask_f = loss_mask.to(accept_rate.dtype)
            mask_total = mask_f.sum().clamp_min(1.0)
            conf_prob = confidence_logits.float().sigmoid()
            metrics["confidence_loss_sum"] = conf_loss.detach().clone()
            metrics["confidence_loss_total"] = torch.ones((), device=device)
            metrics["confidence_abs_error_sum"] = (
                (conf_prob - accept_rate).abs() * mask_f
            ).sum()
            metrics["confidence_abs_error_total"] = mask_total
            # Mean predicted vs. observed acceptance — a calibration sanity check.
            metrics["confidence_pred_mean_sum"] = (conf_prob * mask_f).sum()
            metrics["confidence_pred_mean_total"] = mask_total.clone()
            # Calibration of the cumulative acceptance product, which is what
            # dynamic draft-length thresholding consumes (signed pred - target).
            conf_prefix = (
                conf_prob.view(num_blocks, block_size)[:, start_pos:] * draft_mask
            ).cumprod(dim=-1)
            metrics["confidence_cumprod_bias_sum"] = (
                (conf_prefix - accept_prefix) * draft_mask
            ).sum()
            metrics["confidence_cumprod_bias_total"] = draft_mask.sum().clamp_min(1.0)

    ones = torch.ones((), device=device)
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    # Mean acceptance rate of the (Markov-corrected) drafter.
    with torch.no_grad():
        mask_f = loss_mask.to(accept_rate.dtype)
        metrics["accept_rate_sum"] = (accept_rate * mask_f).sum()
        metrics["accept_rate_total"] = mask_f.sum().clamp_min(1.0)

    # Expected accepted draft length per block (DSpark's tau): the cumulative
    # acceptance product summed over draft slots, plus the always-emitted anchor.
    with torch.no_grad():
        per_block_len = accept_prefix.sum(dim=-1) + 1.0
        block_valid = (draft_mask.sum(dim=-1) > 0).to(accept_rate.dtype)
        metrics["accept_len_sum"] = (per_block_len * block_valid).sum()
        metrics["accept_len_total"] = block_valid.sum().clamp_min(1.0)

    # Per-position greedy accuracy
    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)
    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )
    metrics["full_acc_sum"] = correct_per_pos[start_pos:].sum()
    metrics["full_acc_total"] = total_per_pos[start_pos:].sum()
    for pos in range(start_pos, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]

    if opd_terms is not None:
        for name in ("opd_acc_loss", "opd_rej_loss", "opd_acc_slots", "opd_rej_slots"):
            metrics[f"{name}_sum"] = opd_terms[name]
            metrics[f"{name}_total"] = ones.clone()

    return loss, metrics
