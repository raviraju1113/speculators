from collections.abc import Callable
from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.losses import (
    LossConfig,
    kl_div_loss,
    resolve_loss_config,
    tv_loss,
)
from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.dspark.model_definitions import (
    ConfidenceHead,
    ContextIndexer,
    MarkovHead,
)
from speculators.models.dspark.topk import (
    build_context_candidates,
    create_selected_context_mask_mod,
    dense_context_attention_probs,
    indexer_kl,
    select_topk_context,
    selection_recall,
)
from speculators.models.utils import conditional_torch_compile

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

__all__ = [
    "DSparkDraftModel",
]


@SpeculatorModel.register("dspark")
class DSparkDraftModel(DFlashDraftModel):
    """DFlash backbone plus a Markov logit-bias head and a confidence head.

    After the base draft logits are produced, the Markov head biases position
    ``k`` using the previous block token and the confidence head predicts each
    position's acceptance probability. Everything else is inherited from DFlash.

    Optionally (``topk_context > 0``) the full-attention layers listed in
    ``topk_layers`` attend only to the context positions a per-layer lightning
    indexer ranks highest (DeepSeek-Sparse-Attention style, see ``topk.py``).
    The indexer is trained by KL to the layer's dense attention; the draft loss
    never reaches it. ``topk_warmup_steps`` keeps attention dense while the
    indexer learns, then the top-k layers switch to the selected set.
    """

    config_class: ClassVar[type[DSparkSpeculatorConfig]] = DSparkSpeculatorConfig  # type: ignore[misc,assignment]

    def __init__(self, config: DSparkSpeculatorConfig) -> None:
        super().__init__(config=config)

        hidden_size = config.transformer_layer_config.hidden_size

        self.markov_head: MarkovHead | None = None
        if config.markov_rank > 0:
            self.markov_head = MarkovHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                markov_rank=config.markov_rank,
                hidden_size=hidden_size,
                head_type=config.markov_head_type,
            )

        self.confidence_head: ConfidenceHead | None = None
        if config.enable_confidence_head:
            if config.confidence_head_with_markov and self.markov_head is None:
                raise ValueError(
                    "confidence_head_with_markov=True requires markov_rank > 0."
                )
            input_dim = hidden_size + (
                config.markov_rank if config.confidence_head_with_markov else 0
            )
            self.confidence_head = ConfidenceHead(input_dim)

        # DSA-style top-k context selection.
        self.topk_layer_indices: list[int] = []
        self.context_indexers = torch.nn.ModuleDict()
        if config.topk_context > 0:
            num_layers = len(self.layers)
            full_layers = [
                i for i in range(num_layers) if i not in self.sliding_window_indices
            ]
            if not full_layers:
                raise ValueError(
                    "topk_context > 0 needs at least one full-attention layer."
                )
            if config.topk_layers is None:
                layers = [i for i in full_layers if i != 0] or full_layers
            else:
                layers = list(config.topk_layers)
                bad = [i for i in layers if i not in full_layers]
                if bad:
                    raise ValueError(
                        f"topk_layers {bad} are not full-attention layers "
                        f"(full-attention layers: {full_layers})."
                    )
            self.topk_layer_indices = layers
            for i in layers:
                self.context_indexers[str(i)] = ContextIndexer(
                    hidden_size, config.indexer_heads, config.indexer_head_dim
                )
        self._topk_steps_seen = 0
        self._topk_warmup_steps = 0

    @property
    def topk_enabled(self) -> bool:
        return bool(self.topk_layer_indices)

    def topk_mode(self) -> str:
        """``"off"``, ``"warmup"`` (dense attention, indexer trains) or ``"sparse"``."""
        if not self.topk_enabled:
            return "off"
        if self._topk_steps_seen < self._topk_warmup_steps:
            return "warmup"
        return "sparse"

    @torch.compiler.disable
    def _create_selected_mask(
        self,
        total_seq_len: int,
        num_anchors: int,
        selected: torch.Tensor,
        device: torch.device,
    ):
        mask_mod, q_len, kv_len = create_selected_context_mask_mod(
            total_seq_len=total_seq_len,
            num_anchors=num_anchors,
            block_size=self.block_size,
            selected=selected,
        )
        return self._create_mask_fn(
            mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device
        )

    def _run_layers(
        self,
        noise_embedding: torch.Tensor,
        fc_output: torch.Tensor,
        full_attn_mask,
        sliding_window_attn_mask,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        document_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        anchor_valid: torch.Tensor,
        total_seq_len: int,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mode = self.topk_mode()
        if mode == "off":
            return super()._run_layers(
                noise_embedding,
                fc_output,
                full_attn_mask,
                sliding_window_attn_mask,
                position_ids,
                position_embeddings,
                document_ids=document_ids,
                anchor_positions=anchor_positions,
                anchor_valid=anchor_valid,
                total_seq_len=total_seq_len,
                **kwargs,
            )

        num_anchors = anchor_positions.numel()
        block = self.block_size
        valid, local = build_context_candidates(
            document_ids.squeeze(0).to(noise_embedding.device),
            anchor_positions,
            total_seq_len,
            self.config.topk_local_window,
        )
        kl_terms, recalls, densities = [], [], []
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.sliding_window_indices:
                attention_mask = sliding_window_attn_mask
            elif layer_idx in self.topk_layer_indices:
                indexer = self.context_indexers[str(layer_idx)]
                # Indexer inputs are detached: it learns only from its KL term.
                block_hidden = noise_embedding.detach().view(num_anchors, block, -1)
                scores = indexer(block_hidden, fc_output.detach()[0])  # [N, T]
                target = dense_context_attention_probs(
                    layer.self_attn,
                    layer.input_layernorm(noise_embedding),
                    fc_output,
                    position_embeddings,
                    valid,
                    block,
                )
                selected = select_topk_context(
                    scores.detach(), valid, local, self.config.topk_context
                )
                if mode == "sparse":
                    attention_mask = self._create_selected_mask(
                        total_seq_len, num_anchors, selected, noise_embedding.device
                    )
                    restrict = selected
                else:
                    attention_mask = full_attn_mask
                    restrict = valid
                kl_terms.append(indexer_kl(scores, target, restrict, anchor_valid))
                recalls.append(selection_recall(target, selected, anchor_valid))
                n_valid = valid.sum(dim=-1).float()
                density = selected.sum(dim=-1).float() / n_valid.clamp_min(1.0)
                w = (anchor_valid & (n_valid > 0)).float()
                densities.append((density * w).sum() / w.sum().clamp_min(1.0))
            else:
                attention_mask = full_attn_mask
            noise_embedding = layer(
                hidden_states=noise_embedding,
                target_hidden=fc_output,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        aux: dict[str, torch.Tensor] = {}
        if kl_terms:
            aux = {
                "indexer_kl": torch.stack(kl_terms).mean(),
                "topk_recall": torch.stack(recalls).mean().detach(),
                "topk_density": torch.stack(densities).mean().detach(),
                "topk_sparse": torch.tensor(
                    1.0 if mode == "sparse" else 0.0, device=noise_embedding.device
                ),
            }
        return noise_embedding, aux

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSparkDraftModel":
        """Create a DSpark model from training arguments (mirrors DFlash)."""
        enable_confidence_head_arg = kwargs.get("enable_confidence_head")
        confidence_head_with_markov_arg = kwargs.get("confidence_head_with_markov")
        config = DSparkSpeculatorConfig(
            **cls._build_base_config_kwargs("dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=(
                True
                if enable_confidence_head_arg is None
                else enable_confidence_head_arg
            ),
            confidence_head_with_markov=(
                True
                if confidence_head_with_markov_arg is None
                else confidence_head_with_markov_arg
            ),
            topk_context=kwargs.get("topk_context", 0) or 0,
            topk_layers=kwargs.get("topk_layers"),
            topk_local_window=kwargs.get("topk_local_window", 128),
            indexer_heads=kwargs.get("indexer_heads", 4),
            indexer_head_dim=kwargs.get("indexer_head_dim", 64),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve DSpark's compound loss from ``--loss-fn``."""
        implementation = kwargs.get("loss_implementation", "fused")
        loss_config = resolve_loss_config(kwargs["loss_fn"], implementation)
        tv_loss_fn = resolve_loss_config("tv", implementation)["tv"][0]
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 3072)
        confidence_head_alpha = kwargs.get("confidence_head_alpha", 1.0)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        shared = {
            "loss_config": loss_config,
            "tv_loss_fn": tv_loss_fn,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "confidence_head_alpha": confidence_head_alpha,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
            "indexer_loss_weight": kwargs.get("indexer_loss_weight", 1.0),
            "topk_warmup_steps": kwargs.get("topk_warmup_steps", 0),
        }
        return dict(shared), dict(shared)

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        loss_config: LossConfig | None = None,
        tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
        gamma: float = 4.0,
        max_anchors: int = 3072,
        confidence_head_alpha: float = 1.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        truncate_k: int | None = None,
        indexer_loss_weight: float = 1.0,
        topk_warmup_steps: int = 0,
        **kwargs,
    ):
        self._topk_warmup_steps = topk_warmup_steps
        hidden, logits, targets, aligned_loss_mask, anchored_block_indices, aux = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                **kwargs,
            )
        )

        # DSpark: add the Markov logit bias and predict per-position confidence.
        num_blocks = max_anchors
        block = self.block_size
        mask_tokens_size = num_blocks * block
        # Ground-truth block tokens (verifier vocab); position 0 is the anchor.
        block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block)
        if self.config.sample_from_anchor:
            # With sample_from_anchor=True (DSpark default), slot k predicts
            # token p+k+1 and the inference Markov chain conditions slot k's
            # bias on the token at the previous position p+k.
            prev_token_ids = block_tokens
        else:
            # With sample_from_anchor=False (Dflash default), slot k predicts
            # token p+k, so the previous token within the block is
            # block_tokens[:, k-1] (shifted).
            prev_token_ids = torch.cat(
                [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
            )  # [num_blocks, block]
        hidden_blocks = hidden.view(num_blocks, block, -1)

        confidence_logits = None
        prev_emb = None
        if self.markov_head is not None:
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            logits = (logits.view(num_blocks, block, -1) + markov_bias).view(
                1, mask_tokens_size, -1
            )

        if self.confidence_head is not None:
            # confidence_head_with_markov requires markov_rank > 0 (enforced in
            # __init__), so prev_emb is always set when the flag is on.
            if self.config.confidence_head_with_markov and prev_emb is not None:
                conf_features = torch.cat(
                    [hidden_blocks, prev_emb.to(hidden_blocks.dtype)],
                    dim=-1,
                )
            else:
                conf_features = hidden_blocks
            confidence_logits = self.confidence_head(conf_features).reshape(
                1, mask_tokens_size
            )

        loss, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            aligned_loss_mask,
            self.block_size,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            tv_loss_fn=tv_loss_fn,
            gamma=gamma,
            confidence_head_alpha=confidence_head_alpha,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
            truncate_k=truncate_k,
        )

        if aux:
            loss = loss + indexer_loss_weight * aux["indexer_kl"]
            ones = torch.ones((), device=loss.device)
            for name in ("indexer_kl", "topk_recall", "topk_density", "topk_sparse"):
                metrics[f"{name}_sum"] = aux[name].detach().clone()
                metrics[f"{name}_total"] = ones.clone()
            metrics["loss_sum"] = loss.detach().clone()
            if self.training:
                self._topk_steps_seen += 1
        return None, loss, metrics
