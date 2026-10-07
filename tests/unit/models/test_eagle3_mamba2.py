"""Tests for the Mamba2 EAGLE-3 draft block.

The three equivalence tests here guard failure modes that do not raise: a wrong
document mask, a wrong prefill/decode split, or a wrong state rollback all show up
only as a quietly degraded acceptance rate, which would bias an architecture
comparison rather than break a run.
"""

import pytest
import torch
from transformers import DynamicCache

from tests.conftest import requires_cuda

mamba2 = pytest.importorskip(
    "speculators.models.eagle3.mamba2", reason="mamba2 draft block unavailable"
)
pytest.importorskip("mamba_ssm", reason="mamba_ssm kernels required")
pytest.importorskip("causal_conv1d", reason="causal_conv1d kernels required")

from speculators.models.eagle3.mamba2 import (  # noqa: E402
    Eagle3Mamba2Mixer,
    Mamba2DecoderEagle3FirstLayer,
    Mamba2MLPDecoderEagle3FirstLayer,
    build_mamba2_draft_config,
    seq_idx_from_document_ids,
)

HIDDEN = 512
CHUNK = 128


def _config(**kwargs):
    defaults = {
        "hidden_size": HIDDEN,
        "vocab_size": 2048,
        "head_dim": 32,
        "n_groups": 4,
        "state_size": 64,
        "conv_kernel": 4,
        "chunk_size": CHUNK,
    }
    defaults.update(kwargs)
    return build_mamba2_draft_config(**defaults)


def _mixer(dtype=torch.float32, **kwargs):
    torch.manual_seed(0)
    return Eagle3Mamba2Mixer(_config(**kwargs), 0).cuda().to(dtype).eval()


class TestConfig:
    def test_num_heads_derived_to_satisfy_strict_constraint(self):
        cfg = _config(expand=2, head_dim=32)
        assert cfg.num_heads * cfg.head_dim == cfg.expand * cfg.hidden_size

    def test_rejects_head_dim_that_does_not_divide_inner(self):
        with pytest.raises(ValueError, match="divisible by head_dim"):
            build_mamba2_draft_config(hidden_size=100, vocab_size=32, head_dim=64)

    def test_intermediate_size_selects_the_mlp_arm(self):
        assert getattr(_config(), "intermediate_size", None) is None
        assert _config(intermediate_size=1024).intermediate_size == 1024


class TestSeqIdx:
    def test_padding_gets_its_own_segment(self):
        docs = torch.tensor([[0, 0, 1, 1, -1, -1]])
        out = seq_idx_from_document_ids(docs)
        assert out.dtype == torch.int32
        # padding must not be folded into document 1
        assert out[0, 4].item() == out[0, 5].item() != out[0, 3].item()

    def test_accepts_unbatched_input(self):
        assert seq_idx_from_document_ids(torch.tensor([0, 0, 1])).shape == (1, 3)


@requires_cuda
class TestInputWidth:
    def test_in_proj_is_widened_to_two_hidden(self):
        """W_FC is fused into the block, as for the transformer first layers."""
        assert _mixer().in_proj.in_features == 2 * HIDDEN

    def test_layer_consumes_concatenated_embeds_and_hidden(self):
        layer = Mamba2DecoderEagle3FirstLayer(_config(), 0).cuda().eval()
        x = torch.randn(1, CHUNK, 2 * HIDDEN, device="cuda")
        with torch.no_grad():
            out = layer(x)
        assert out.shape == (1, CHUNK, HIDDEN)


@requires_cuda
class TestArmB2:
    def test_mlp_present_only_when_intermediate_size_set(self):
        assert Mamba2DecoderEagle3FirstLayer(_config(), 0).mlp is None
        with_mlp = Mamba2DecoderEagle3FirstLayer(_config(intermediate_size=1024), 0)
        assert with_mlp.mlp is not None

    def test_explicit_mlp_class_forces_the_arm(self):
        assert Mamba2MLPDecoderEagle3FirstLayer(_config(), 0).mlp is not None

    def test_b2_has_more_parameters_than_b(self):
        layer_b = Mamba2DecoderEagle3FirstLayer(_config(), 0)
        layer_b2 = Mamba2DecoderEagle3FirstLayer(_config(intermediate_size=1024), 0)
        b = sum(p.numel() for p in layer_b.parameters())
        b2 = sum(p.numel() for p in layer_b2.parameters())
        assert b2 > b


@requires_cuda
class TestDocumentIsolation:
    """P1: a packed scan must not leak state across document boundaries."""

    @staticmethod
    def _run(mixer, x, docs):
        with torch.no_grad():
            return mixer(x, seq_idx=seq_idx_from_document_ids(docs))

    def test_chunk_aligned_boundaries_isolate_exactly(self):
        mixer = _mixer()
        lengths = [CHUNK, 2 * CHUNK]
        total = sum(lengths)
        docs = (
            torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
            .unsqueeze(0)
            .cuda()
        )

        torch.manual_seed(1)
        x = torch.randn(1, total, 2 * HIDDEN, device="cuda")
        packed = self._run(mixer, x, docs)

        # rewrite document 0 entirely; document 1's output must not move
        x2 = x.clone()
        x2[:, : lengths[0]] = torch.randn(1, lengths[0], 2 * HIDDEN, device="cuda")
        packed2 = self._run(mixer, x2, docs)

        torch.testing.assert_close(
            packed[:, lengths[0] :], packed2[:, lengths[0] :], atol=0, rtol=0
        )

    def test_without_seq_idx_documents_do_leak(self):
        """Control: proves the isolation test above is not vacuous."""
        mixer = _mixer()
        lengths = [CHUNK, 2 * CHUNK]
        total = sum(lengths)
        torch.manual_seed(1)
        x = torch.randn(1, total, 2 * HIDDEN, device="cuda")
        x2 = x.clone()
        x2[:, : lengths[0]] = torch.randn(1, lengths[0], 2 * HIDDEN, device="cuda")
        with torch.no_grad():
            a = mixer(x)
            b = mixer(x2)
        assert (a[:, lengths[0] :] - b[:, lengths[0] :]).abs().max() > 1e-3

    def test_packed_matches_independent_scans(self):
        mixer = _mixer()
        lengths = [CHUNK, 2 * CHUNK, CHUNK]
        total = sum(lengths)
        docs = (
            torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
            .unsqueeze(0)
            .cuda()
        )
        torch.manual_seed(2)
        x = torch.randn(1, total, 2 * HIDDEN, device="cuda")

        packed = self._run(mixer, x, docs)
        offset = 0
        for length in lengths:
            piece = x[:, offset : offset + length]
            with torch.no_grad():
                single = torch.zeros(1, length, dtype=torch.int32, device="cuda")
                alone = mixer(piece, seq_idx=single)
            torch.testing.assert_close(
                packed[:, offset : offset + length], alone, atol=2e-4, rtol=2e-4
            )
            offset += length


@requires_cuda
class TestCacheAndRollback:
    """P3: speculation requires an exact state rewind, which HF's cache cannot do."""

    @staticmethod
    def _snapshot(cache, layer_idx=0):
        layer = cache.layers[layer_idx]
        return (
            layer.conv_states.clone(),
            layer.recurrent_states.clone(),
            layer.has_previous_state,
        )

    @staticmethod
    def _restore(cache, snap, layer_idx=0):
        layer = cache.layers[layer_idx]
        conv, recurrent, has_prev = snap
        # copy_ rather than assignment: the cache relies on stable addresses
        layer.conv_states.copy_(conv)
        layer.recurrent_states.copy_(recurrent)
        layer.has_previous_state = has_prev

    def test_prefill_then_decode_matches_a_single_scan(self):
        """Catches the dt[:, 0, :] multi-token trap and the conv_state off-by-one."""
        cfg = _config()
        mixer = Eagle3Mamba2Mixer(cfg, 0).cuda().float().eval()
        torch.manual_seed(3)
        prefix = torch.randn(1, 64, 2 * HIDDEN, device="cuda")
        extra = torch.randn(1, 4, 2 * HIDDEN, device="cuda")

        with torch.no_grad():
            stepwise_cache = DynamicCache(config=cfg)
            mixer(prefix, cache_params=stepwise_cache)
            stepped = [
                mixer(extra[:, i : i + 1], cache_params=stepwise_cache)
                for i in range(4)
            ]
            stepped_out = torch.cat(stepped, dim=1)

            whole = mixer(torch.cat([prefix, extra], dim=1))[:, prefix.shape[1] :]

        torch.testing.assert_close(stepped_out, whole, atol=2e-3, rtol=2e-3)

    def test_rollback_after_rejection_is_exact(self):
        cfg = _config()
        mixer = Eagle3Mamba2Mixer(cfg, 0).cuda().float().eval()
        torch.manual_seed(4)
        prefix = torch.randn(1, 64, 2 * HIDDEN, device="cuda")
        speculated = torch.randn(1, 5, 2 * HIDDEN, device="cuda")
        accepted = 3

        with torch.no_grad():
            rolled = DynamicCache(config=cfg)
            mixer(prefix, cache_params=rolled)
            snap = self._snapshot(rolled)
            for i in range(5):  # draft 5
                mixer(speculated[:, i : i + 1], cache_params=rolled)
            self._restore(rolled, snap)  # verifier rejected after 3
            for i in range(accepted):
                out_rolled = mixer(speculated[:, i : i + 1], cache_params=rolled)

            clean = DynamicCache(config=cfg)
            mixer(prefix, cache_params=clean)
            for i in range(accepted):
                out_clean = mixer(speculated[:, i : i + 1], cache_params=clean)

        torch.testing.assert_close(out_rolled, out_clean, atol=0, rtol=0)
        torch.testing.assert_close(
            rolled.layers[0].recurrent_states,
            clean.layers[0].recurrent_states,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            rolled.layers[0].conv_states, clean.layers[0].conv_states, atol=0, rtol=0
        )

    def test_without_rollback_state_is_corrupted(self):
        """Control: proves the rollback test above is not vacuous."""
        cfg = _config()
        mixer = Eagle3Mamba2Mixer(cfg, 0).cuda().float().eval()
        torch.manual_seed(4)
        prefix = torch.randn(1, 64, 2 * HIDDEN, device="cuda")
        speculated = torch.randn(1, 5, 2 * HIDDEN, device="cuda")

        with torch.no_grad():
            dirty = DynamicCache(config=cfg)
            mixer(prefix, cache_params=dirty)
            for i in range(5):
                mixer(speculated[:, i : i + 1], cache_params=dirty)
            for i in range(3):
                out_dirty = mixer(speculated[:, i : i + 1], cache_params=dirty)

            clean = DynamicCache(config=cfg)
            mixer(prefix, cache_params=clean)
            for i in range(3):
                out_clean = mixer(speculated[:, i : i + 1], cache_params=clean)

        assert (out_dirty - out_clean).abs().max() > 1e-3
