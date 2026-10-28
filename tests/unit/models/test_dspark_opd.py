"""Draft-OPD loss and anchor selection (speculators.models.dspark.metrics,
speculators.models.dflash.utils). CPU-only, no model weights."""

import torch

from speculators.losses.eager import tv_loss  # eager: the fused kernel needs CUDA tensors
from speculators.models.dflash.utils import select_given_anchors
from speculators.models.dspark.metrics import (
    _forward_kl,
    _reverse_kl,
    compute_metrics,
    opd_loss,
    opd_slot_masks,
)


def test_opd_slot_masks_first_rejected_only():
    accepted = torch.tensor([0, 3, 8, -1])  # block 8: r=0, r=3, full accept, not OPD
    acc, rej = opd_slot_masks(accepted, block_size=8)
    assert acc[0].sum() == 0 and rej[0].tolist() == [True] + [False] * 7
    assert acc[1].tolist() == [True] * 3 + [False] * 5
    assert rej[1].tolist() == [False] * 3 + [True] + [False] * 4
    assert acc[2].all() and rej[2].sum() == 0  # all accepted, nothing rejected
    assert acc[3].sum() == 0 and rej[3].sum() == 0  # non-OPD block ignored


def test_opd_slot_masks_full_tail():
    acc, rej = opd_slot_masks(torch.tensor([2]), block_size=5, first_rejected_only=False)
    assert rej[0].tolist() == [False, False, True, True, True]


def test_kl_directions_zero_at_match_and_asymmetric():
    torch.manual_seed(0)
    a = torch.randn(1, 4, 50)
    b = torch.randn(1, 4, 50)
    assert torch.allclose(_forward_kl(a, a), torch.zeros(1, 4), atol=1e-6)
    assert torch.allclose(_reverse_kl(a, a), torch.zeros(1, 4), atol=1e-6)
    f, r = _forward_kl(a, b), _reverse_kl(a, b)
    assert (f > 0).all() and (r > 0).all()
    # forward KL(p||q) with p from b, q from a == reverse KL(q||p) with roles swapped
    assert torch.allclose(f, _reverse_kl(b, a), atol=1e-5)


def test_opd_loss_zero_when_draft_matches_target_and_respects_masks():
    torch.manual_seed(1)
    block, nb, v = 8, 3, 64
    targets = torch.randn(1, nb * block, v)
    loss_mask = torch.ones(1, nb * block)
    accepted = torch.tensor([2, 8, -1])
    loss, terms = opd_loss(targets.clone(), targets, loss_mask, accepted, block)
    assert loss.abs() < 1e-5
    assert terms["opd_acc_slots"] == 2 + 8  # block 0: 2 accepted, block 1: all 8
    assert terms["opd_rej_slots"] == 1  # only block 0 has a first rejected slot
    # scored mask: block 0 slots 0..2, block 1 all, block 2 none
    scored = terms["opd_scored_mask"].view(nb, block)
    assert scored[0].tolist() == [True] * 3 + [False] * 5
    assert scored[1].all() and not scored[2].any()


def test_opd_loss_lambda_weighting():
    torch.manual_seed(2)
    block, nb, v = 4, 2, 32
    logits = torch.randn(1, nb * block, v)
    targets = torch.randn(1, nb * block, v)
    loss_mask = torch.ones(1, nb * block)
    accepted = torch.tensor([1, 2])
    l1, t = opd_loss(logits, targets, loss_mask, accepted, block, lambda_rej=1.0)
    l0, _ = opd_loss(logits, targets, loss_mask, accepted, block, lambda_rej=0.0)
    assert torch.allclose(l0, t["opd_acc_loss"], atol=1e-6)
    assert torch.allclose(l1, (t["opd_acc_loss"] + t["opd_rej_loss"]) / 2, atol=1e-6)


def test_opd_loss_rej_valid_drops_out_of_vocab_rejected_slots():
    torch.manual_seed(4)
    block, nb, v = 4, 2, 32
    logits = torch.randn(1, nb * block, v)
    targets = torch.randn(1, nb * block, v)
    loss_mask = torch.ones(1, nb * block)
    accepted = torch.tensor([1, 2])  # rejected slots: block0 slot1, block1 slot2
    rej_valid = torch.ones(nb, block, dtype=torch.bool)
    rej_valid[1, 2] = False  # block1's bonus token is outside the draft vocab
    l_all, t_all = opd_loss(logits, targets, loss_mask, accepted, block)
    l_msk, t_msk = opd_loss(logits, targets, loss_mask, accepted, block, rej_valid=rej_valid)
    assert t_all["opd_rej_slots"] == 2 and t_msk["opd_rej_slots"] == 1
    assert t_msk["opd_rej_oov_slots"] == 1 and t_all["opd_rej_oov_slots"] == 0
    assert torch.allclose(t_msk["opd_acc_loss"], t_all["opd_acc_loss"])
    # the surviving rejected term is exactly block 0's reverse KL
    from speculators.models.dspark.metrics import _reverse_kl
    rev = _reverse_kl(logits, targets).view(nb, block)
    assert torch.allclose(t_msk["opd_rej_loss"], rev[0, 1], atol=1e-6)
    assert not torch.allclose(l_msk, l_all)


def test_compute_metrics_opd_branch_runs():
    torch.manual_seed(3)
    block, nb, v = 8, 4, 48
    logits = torch.randn(1, nb * block, v, requires_grad=True)
    targets = torch.randn(1, nb * block, v)
    conf = torch.randn(1, nb * block)
    loss_mask = torch.ones(1, nb * block)
    accepted = torch.tensor([0, 3, 7, -1])
    loss, metrics = compute_metrics(
        logits, targets, conf, loss_mask, block, loss_config={},
        tv_loss_fn=tv_loss, opd_accepted=accepted,
    )
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None
    assert "opd_acc_loss_sum" in metrics and "opd_rej_loss_sum" in metrics
    assert "accept_len_sum" in metrics


def test_select_given_anchors_subsample_and_pad():
    torch.manual_seed(4)
    T, block = 64, 8
    loss_mask = torch.zeros(1, T, dtype=torch.bool)
    loss_mask[:, 10:] = True
    anchor_mask = torch.zeros(1, T, dtype=torch.bool)
    anchor_mask[0, [5, 12, 20, 33, 40, 60]] = True  # 5 is outside loss_mask, 60 in the last block
    anchors, valid = select_given_anchors(anchor_mask, loss_mask, num_anchors=8, block_size=block)
    assert valid.sum() == 4 and sorted(anchors[valid].tolist()) == [12, 20, 33, 40]
    anchors, valid = select_given_anchors(anchor_mask, loss_mask, num_anchors=2, block_size=block)
    assert valid.sum() == 2 and set(anchors[valid].tolist()) <= {12, 20, 33, 40}
