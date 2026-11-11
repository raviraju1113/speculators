"""Vocabulary mapping utilities for draft model training."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset as HFDataset
from tqdm import tqdm  # type: ignore[import-untyped]
from transformers import AutoConfig

__all__ = [
    "build_vocab_mappings_from_distribution",
    "save_token_frequency_distribution",
]


def save_token_frequency_distribution(
    dataset: HFDataset,
    output_path: Path | str = "./token_freq.pt",
):
    """Save token frequency distribution from the dataset.

    Args:
        dataset: HuggingFace dataset with input_ids and loss_mask
        output_path: Path where to save the token frequency distribution

    Returns:
        Path to the saved frequency distribution file
    """
    path = Path(output_path)
    if path.exists():
        return

    # Batched numpy pass (same counts as the old per-row loop, far faster on
    # large sets: the per-row torch path ran at ~10 rows/s on 777k rows).
    token_freq: Counter[int] = Counter()
    plain = dataset.with_format(None) if hasattr(dataset, "with_format") else dataset
    batch_size = 2048
    total = len(plain)
    for start in tqdm(range(0, total, batch_size), desc="Counting token frequencies"):
        batch = plain[start : start + batch_size]
        ids = np.concatenate(
            [
                np.asarray(i, dtype=np.int64)[np.asarray(m, dtype=bool)]
                for i, m in zip(batch["input_ids"], batch["loss_mask"], strict=True)
            ]
        )
        if ids.size == 0:
            continue
        unique_ids, counts = np.unique(ids, return_counts=True)
        token_freq.update(dict(zip(unique_ids.tolist(), counts.tolist(), strict=True)))

    token_freq_dict = dict(token_freq)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(token_freq_dict, path)


def combine_token_frequency_distributions(
    token_freq_paths: list[str | Path],
    output_path: str | Path,
):
    """Combine multiple token frequency distributions into a single file."""
    token_freq_dicts = [
        torch.load(path, weights_only=True) for path in token_freq_paths
    ]
    combined_token_freq: Counter[str] = Counter()
    for token_freq_dict in token_freq_dicts:
        combined_token_freq.update(token_freq_dict)
    combined_token_freq_dict = dict(combined_token_freq)
    torch.save(combined_token_freq_dict, output_path)


def build_vocab_mappings_from_distribution(
    token_freq_dict: dict[int, int],
    draft_vocab_size: int,
    target_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build vocabulary mappings for draft model from token frequency distribution."""

    sorted_tokens = sorted(
        token_freq_dict, key=lambda tid: (-token_freq_dict[tid], tid)
    )

    num_tokens_to_select = min(draft_vocab_size, len(sorted_tokens))
    selected_token_ids = sorted_tokens[:num_tokens_to_select]

    if len(selected_token_ids) < draft_vocab_size:
        current_ids = set(selected_token_ids)
        for tid in range(draft_vocab_size):
            if tid not in current_ids:
                selected_token_ids.append(tid)
            if len(selected_token_ids) >= draft_vocab_size:
                break

    selected_token_ids.sort()

    # Store offset: target_token_id = draft_idx + draft_to_target[draft_idx]
    draft_to_target = torch.tensor(selected_token_ids, dtype=torch.long) - torch.arange(
        draft_vocab_size, dtype=torch.long
    )

    target_to_draft = torch.zeros(target_vocab_size, dtype=torch.bool)
    target_to_draft[selected_token_ids] = True

    return draft_to_target, target_to_draft


def get_target_vocab_size(target_vocab_size, target_model_path):
    has_vocab = target_vocab_size is not None
    has_model = target_model_path is not None

    if has_vocab and has_model:
        raise ValueError("Cannot specify both target-vocab-size and target-model-path")

    if not has_vocab and not has_model:
        raise ValueError("Must specify either target-vocab-size or target-model-path")

    if has_vocab:
        return target_vocab_size

    config = AutoConfig.from_pretrained(target_model_path)

    # For multimodal models (Qwen3VL, etc.), extract text_config
    if hasattr(config, "text_config"):
        config = config.text_config

    return config.vocab_size
