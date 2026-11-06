from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DSparkSpeculatorConfig",
]


@SpeculatorModelConfig.register("dspark")
class DSparkSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash config plus a Markov logit-bias head and a confidence head.

    The Markov head lets each draft position condition on previously sampled
    tokens within the block; the confidence head predicts the per-position
    acceptance probability. All DFlash fields are inherited unchanged.
    """

    speculators_model_type: Literal["dspark"] = "dspark"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DSparkSpeculator"],
        description="Model architectures that can load these weights",
    )

    sample_from_anchor: bool = Field(
        default=True,
        description=(
            "Whether to sample from the anchor position. "
            "False: anchor is the bonus token, only mask tokens predict "
            "(block_size-1 speculative tokens). "
            "True: sample from anchor and all mask positions "
            "(block_size speculative tokens). "
            "Default True matches DeepSeek/DeepSpec convention."
        ),
    )

    # Sequential (Markov) head.
    markov_rank: int = Field(
        default=256,
        description=(
            "Low-rank dimension of the Markov logit-bias factorization B = W1 @ W2. "
            "Set to 0 to disable the sequential head (pure DFlash drafting)."
        ),
    )
    markov_head_type: Literal["vanilla", "gated", "rnn"] = Field(
        default="vanilla",
        description=(
            "Sequential head variant: 'vanilla' (first-order Markov bias), 'gated' "
            "(hidden-gated bias), or 'rnn' (recurrent state over the block)."
        ),
    )

    # Confidence head.
    enable_confidence_head: bool = Field(
        default=True,
        description="Whether to attach the per-position acceptance-probability head.",
    )
    confidence_head_with_markov: bool = Field(
        default=True,
        description=(
            "Concatenate the Markov previous-token embedding with the backbone "
            "hidden state as the confidence-head input."
        ),
    )

    # DSA-style top-k context selection (see models/dspark/topk.py).
    topk_context: int = Field(
        default=0,
        description=(
            "Number of context positions each draft block attends in the top-k "
            "layers, chosen by a lightning indexer (DeepSeek Sparse Attention "
            "style). 0 disables selection: every full-attention layer sees the "
            "whole document prefix, as in DFlash. Serving a checkpoint with "
            "topk_context > 0 needs engine support for the indexer."
        ),
    )
    topk_layers: list[int] | None = Field(
        default=None,
        description=(
            "Draft layer indices that use top-k context selection. Must be "
            "full-attention layers. Default: every full-attention layer except "
            "layer 0, whose queries are still bare anchor/mask embeddings."
        ),
    )
    topk_local_window: int = Field(
        default=128,
        description=(
            "Context positions immediately before the anchor that top-k layers "
            "always attend, in addition to the ranked selection. 0 disables."
        ),
    )
    indexer_heads: int = Field(
        default=4, description="Query heads of each layer's lightning indexer."
    )
    indexer_head_dim: int = Field(
        default=64, description="Head dimension of the lightning indexer."
    )
