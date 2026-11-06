"""Unit tests for DSpark's DSA-style top-k context selection."""

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.losses import resolve_loss_config
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark.model_definitions import ContextIndexer
from speculators.models.dspark.topk import (
    build_context_candidates,
    create_selected_context_mask_mod,
    dense_context_attention_probs,
    indexer_kl,
    select_topk_context,
    selection_recall,
)


def _document_ids(lengths, total_seq_len):
    doc = torch.full((total_seq_len,), -1, dtype=torch.long)
    doc[: sum(lengths)] = torch.repeat_interleave(
        torch.arange(len(lengths)), torch.tensor(lengths)
    )
    return doc


class TestCandidates:
    def test_valid_matches_same_doc_before_anchor(self):
        doc = _document_ids([6, 5], 14)  # positions 11..13 are padding
        anchors = torch.tensor([3, 8, 10])
        valid, local = build_context_candidates(doc, anchors, 14, local_window=2)
        for j, a in enumerate(anchors.tolist()):
            for t in range(14):
                expect = doc[t] == doc[a] and t < a and doc[a] != -1
                assert valid[j, t].item() == bool(expect), (j, t)
                assert local[j, t].item() == bool(expect and t >= a - 2), (j, t)

    def test_zero_window_has_no_local(self):
        doc = _document_ids([8], 8)
        _, local = build_context_candidates(doc, torch.tensor([5]), 8, local_window=0)
        assert not local.any()


class TestSelection:
    def test_topk_picks_highest_scores_outside_local(self):
        doc = _document_ids([10], 10)
        anchors = torch.tensor([9])
        valid, local = build_context_candidates(doc, anchors, 10, local_window=2)
        scores = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.0]])
        selected = select_topk_context(scores, valid, local, k=2)
        # local window = positions 7, 8; top-2 among 0..6 = positions 1 and 3
        assert selected[0].nonzero().flatten().tolist() == [1, 3, 7, 8]

    def test_never_selects_invalid_positions(self):
        doc = _document_ids([4, 6], 12)
        anchors = torch.tensor([2, 7])
        valid, local = build_context_candidates(doc, anchors, 12, local_window=0)
        scores = torch.randn(2, 12)
        selected = select_topk_context(scores, valid, local, k=100)
        assert torch.equal(selected, valid)  # k larger than candidates -> all valid

    def test_k_zero_is_local_only(self):
        doc = _document_ids([10], 10)
        valid, local = build_context_candidates(doc, torch.tensor([9]), 10, 3)
        selected = select_topk_context(torch.randn(1, 10), valid, local, k=0)
        assert torch.equal(selected, local)


class TestMask:
    def test_mask_mod_matches_selection_and_block(self):
        total_seq_len, block = 12, 3
        doc = _document_ids([12], 12)
        anchors = torch.tensor([4, 9])
        valid, local = build_context_candidates(doc, anchors, total_seq_len, 1)
        selected = select_topk_context(torch.randn(2, 12), valid, local, k=2)
        mask_mod, q_len, kv_len = create_selected_context_mask_mod(
            total_seq_len, anchors.numel(), block, selected
        )
        dense = create_mask(mask_mod, None, None, q_len, kv_len, device="cpu")[0, 0]
        for q in range(q_len):
            j = q // block
            for kv in range(kv_len):
                if kv < total_seq_len:
                    expect = selected[j, kv].item()
                else:
                    expect = (kv - total_seq_len) // block == j
                assert dense[q, kv].item() == expect, (q, kv)


class TestIndexer:
    def test_scores_shape_and_relu(self):
        idx = ContextIndexer(hidden_size=16, num_heads=2, head_dim=4)
        scores = idx(torch.randn(3, 4, 16), torch.randn(20, 16), chunk_blocks=2)
        assert scores.shape == (3, 20)
        assert scores.dtype == torch.float32

    def test_kl_zero_when_indexer_matches_target(self):
        target = torch.softmax(torch.randn(2, 10), dim=-1)
        restrict = torch.ones(2, 10, dtype=torch.bool)
        kl = indexer_kl(
            torch.log(target), target, restrict, torch.ones(2, dtype=torch.bool)
        )
        assert kl.item() == pytest.approx(0.0, abs=1e-5)

    def test_kl_restricted_ignores_outside_positions(self):
        scores = torch.randn(1, 6)
        target = torch.softmax(torch.randn(1, 6), dim=-1)
        restrict = torch.tensor([[True, True, True, False, False, False]])
        kl_r = indexer_kl(scores, target, restrict, torch.ones(1, dtype=torch.bool))
        # Changing scores/target outside the restricted set must not matter.
        scores2 = scores.clone()
        scores2[0, 3:] += 5.0
        target2 = target.clone()
        target2[0, 3:] *= 0.1
        kl_r2 = indexer_kl(scores2, target2, restrict, torch.ones(1, dtype=torch.bool))
        assert kl_r.item() == pytest.approx(kl_r2.item(), abs=1e-6)

    def test_recall_is_captured_mass(self):
        target = torch.tensor([[0.5, 0.3, 0.2]])
        selected = torch.tensor([[True, False, True]])
        r = selection_recall(target, selected, torch.ones(1, dtype=torch.bool))
        assert r.item() == pytest.approx(0.7)


def _tiny_dspark(topk_context, topk_layers=None, local_window=1, num_layers=2):
    tl_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        layer_types=["full_attention"] * num_layers,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    config = DSparkSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        markov_rank=8,
        topk_context=topk_context,
        topk_layers=topk_layers,
        topk_local_window=local_window,
        indexer_heads=2,
        indexer_head_dim=4,
    )
    model = DSparkDraftModel(config)
    # Verifier-shared weights are normally loaded from the verifier; give the
    # tiny test model finite values instead of uninitialised memory.
    torch.nn.init.normal_(model.embed_tokens.weight)
    torch.nn.init.normal_(model.lm_head.weight)
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    torch.nn.init.ones_(model.verifier_norm.weight)
    return model


def _loss_kwargs():
    """Eager losses so the tests run on CPU (the default kl_div is a Triton kernel)."""
    return {
        "loss_config": resolve_loss_config("kl_div", "eager"),
        "tv_loss_fn": resolve_loss_config("tv", "eager")["tv"][0],
    }


def _batch(seq_len=40):
    torch.manual_seed(0)
    return {
        "hidden_states": torch.randn(1, seq_len, 2 * 16),
        "input_ids": torch.randint(0, 64, (1, seq_len)),
        "loss_mask": torch.ones(1, seq_len),
        "verifier_last_hidden_states": torch.randn(1, seq_len, 16),
        "document_ids": torch.zeros(1, seq_len, dtype=torch.long),
    }


class TestDSparkTopK:
    def test_default_layers_skip_layer_zero(self):
        model = _tiny_dspark(topk_context=4, num_layers=3)
        assert model.topk_layer_indices == [1, 2]
        assert set(model.context_indexers) == {"1", "2"}

    def test_single_layer_falls_back_to_layer_zero(self):
        model = _tiny_dspark(topk_context=4, num_layers=1)
        assert model.topk_layer_indices == [0]

    def test_off_when_topk_context_zero(self):
        model = _tiny_dspark(topk_context=0)
        assert not model.topk_enabled
        assert len(model.context_indexers) == 0
        assert model.topk_mode() == "off"

    def test_rejects_sliding_layer(self):
        tl_config = Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=64,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            _attn_implementation="eager",  # type: ignore[call-arg]
        )
        config = DSparkSpeculatorConfig(
            transformer_layer_config=tl_config,
            draft_vocab_size=64,
            block_size=4,
            aux_hidden_state_layer_ids=[0, 1],
            mask_token_id=0,
            topk_context=4,
            topk_layers=[0],
        )
        with pytest.raises(ValueError, match="not full-attention"):
            DSparkDraftModel(config)

    @pytest.mark.parametrize("warmup_steps", [0, 100])
    def test_forward_reports_topk_metrics(self, warmup_steps):
        model = _tiny_dspark(topk_context=4).train()
        _, loss, metrics = model(
            **_batch(), **_loss_kwargs(), max_anchors=6, topk_warmup_steps=warmup_steps
        )
        assert torch.isfinite(loss)
        for name in ("indexer_kl", "topk_recall", "topk_density", "topk_sparse"):
            assert f"{name}_sum" in metrics
            assert f"{name}_total" in metrics
        assert metrics["topk_sparse_sum"].item() == (0.0 if warmup_steps else 1.0)
        assert 0.0 <= metrics["topk_recall_sum"].item() <= 1.0 + 1e-6
        assert 0.0 < metrics["topk_density_sum"].item() <= 1.0 + 1e-6

    def test_warmup_counter_switches_to_sparse(self):
        model = _tiny_dspark(topk_context=4).train()
        batch = _batch()
        for _ in range(2):
            model(**batch, **_loss_kwargs(), max_anchors=6, topk_warmup_steps=2)
        assert model.topk_mode() == "sparse"
        _, _, metrics = model(
            **batch, **_loss_kwargs(), max_anchors=6, topk_warmup_steps=2
        )
        assert metrics["topk_sparse_sum"].item() == 1.0

    def test_indexer_grad_only_from_kl(self):
        model = _tiny_dspark(topk_context=4).train()
        _, loss, _ = model(
            **_batch(),
            **_loss_kwargs(),
            max_anchors=6,
            indexer_loss_weight=0.0,
            topk_warmup_steps=0,
        )
        loss.backward()
        for p in model.context_indexers.parameters():
            assert p.grad is None or torch.all(p.grad == 0)
        model.zero_grad()
        _, loss, _ = model(
            **_batch(),
            **_loss_kwargs(),
            max_anchors=6,
            indexer_loss_weight=1.0,
            topk_warmup_steps=0,
        )
        loss.backward()
        grads = [p.grad for p in model.context_indexers.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(torch.any(g != 0) for g in grads)

    def test_warmup_matches_dense_model_logits(self):
        """In warmup the draft path is exactly the dense DSpark forward."""
        torch.manual_seed(1)
        dense = _tiny_dspark(topk_context=0).eval()
        sparse = _tiny_dspark(topk_context=4).eval()
        sparse.load_state_dict(dense.state_dict(), strict=False)
        batch = _batch()
        torch.manual_seed(0)
        _, loss_dense, m_dense = dense(**batch, **_loss_kwargs(), max_anchors=6)
        torch.manual_seed(0)
        _, loss_sparse, m_sparse = sparse(
            **batch,
            **_loss_kwargs(),
            max_anchors=6,
            indexer_loss_weight=0.0,
            topk_warmup_steps=10,
        )
        torch.testing.assert_close(loss_sparse, loss_dense, atol=1e-5, rtol=0)
        torch.testing.assert_close(
            m_sparse["full_acc_sum"], m_dense["full_acc_sum"], atol=0, rtol=0
        )

    def test_dense_probs_normalised_over_valid(self):
        model = _tiny_dspark(topk_context=4).eval()
        layer = model.layers[1]
        total_seq_len, block, n = 20, 4, 3
        anchors = torch.tensor([5, 10, 15])
        doc = _document_ids([20], 20)
        valid, _ = build_context_candidates(doc, anchors, total_seq_len, 0)
        layer_input = torch.randn(1, n * block, 16)
        ctx = torch.randn(1, total_seq_len, 16)
        pos = torch.arange(total_seq_len + n * block).unsqueeze(0)
        cos, sin = model.rotary_emb(ctx, pos)
        probs = dense_context_attention_probs(
            layer.self_attn, layer_input, ctx, (cos, sin), valid, block, chunk_blocks=2
        )
        assert probs.shape == (n, total_seq_len)
        torch.testing.assert_close(probs.sum(-1), torch.ones(n), atol=1e-5, rtol=0)
        assert torch.all(probs[~valid] == 0)
