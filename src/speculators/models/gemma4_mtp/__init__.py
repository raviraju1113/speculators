"""Gemma 4 MTP assistant fine-tuning (thin wrapper around the stock model).

This package fine-tunes Google's native ``Gemma4AssistantForCausalLM`` (the MTP
draft/assistant) with a Training-Time Test (TTT) multi-step distillation
objective, so the result stays a bit-for-bit drop-in replacement for the stock
assistant in vLLM. Unlike the EAGLE3/MTP algorithms in this repo, it does NOT
subclass ``SpeculatorModel`` -- the stock assistant is loaded via
``AutoModelForCausalLM`` and trained directly, with this package supplying only
the data pipeline, the target-signal cache, and the training step / loss.

Requires ``transformers>=5.10.2`` (ships ``Gemma4AssistantForCausalLM``) and
``torch>=2.5``.

Ported from the gemma4-mtp-trainer project; internal-infrastructure references
have been genericized. See ``scripts/gemma4_mtp/`` for the runnable entry points
(``prepare_cache.py`` then ``train.py``).
"""

from speculators.models.gemma4_mtp.data import (
    DataConfig,
    Gemma4ConversationParser,
    build_dataset,
    collate,
    iter_jsonl,
)
from speculators.models.gemma4_mtp.target_cache import (
    CacheDataset,
    collate_cache,
)
from speculators.models.gemma4_mtp.training_step import (
    MTPLossConfig,
    build_target_signals,
    compute_step_weights,
    locate_target_parts,
    training_step,
    training_step_from_cache,
)

__all__ = [
    "CacheDataset",
    "DataConfig",
    "Gemma4ConversationParser",
    "MTPLossConfig",
    "build_dataset",
    "build_target_signals",
    "collate",
    "collate_cache",
    "compute_step_weights",
    "iter_jsonl",
    "locate_target_parts",
    "training_step",
    "training_step_from_cache",
]
