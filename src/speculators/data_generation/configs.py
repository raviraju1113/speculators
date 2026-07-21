"""Configuration registries for data generation pipeline."""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "DATASET_CONFIGS",
    "DatasetConfig",
]


@dataclass(kw_only=True)
class DatasetConfig:
    """Configuration for loading a dataset"""

    name: str
    hf_path: str
    subset: str | None = None
    split: str
    filter_fn: Callable[[dict], bool] | None = None
    normalize_fn: Callable[[dict], dict] | None = None


def _normalize_ultrachat(example: dict) -> dict:
    if "messages" in example:
        return {"conversations": example["messages"]}
    return example


def _normalize_gsm8k(example: dict) -> dict:
    return {
        "conversations": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


def get_coco_dir():
    return os.getenv("COCO_DIR") or "coco/"


def _parse_sharegpt4v_part(part: str, image_path: str):
    if part == "<image>":
        return {"type": "image", "path": image_path}

    return {"type": "text", "text": part}


def _parse_sharegpt4v_user_content(content: str, image_path: str):
    return [_parse_sharegpt4v_part(part, image_path) for part in content.split("\n")]


def _parse_sharegpt4v_assistant_content(content: str):
    return [{"type": "text", "text": content}]


def _filter_sharegpt4v_coco(example: dict) -> bool:
    return example["image"].startswith("coco/")


def _normalize_sharegpt4v_coco(example: dict) -> dict:
    coco_dir = get_coco_dir()
    image_path = os.path.join(coco_dir, example["image"].removeprefix("coco/"))

    if not os.path.exists(image_path):
        state_str = "set to" if os.getenv("COCO_DIR") else "default"

        raise ValueError(
            f"No image found at <{image_path}>. "
            f"Please download COCO 2017 Train Images from "
            f"<http://images.cocodataset.org/zips/train2017.zip> and place the "
            f"extracted folder under `COCO_DIR` ({state_str}: `{coco_dir}`)."
        )

    messages = [
        (
            turn
            | {
                "value": (
                    _parse_sharegpt4v_user_content(turn["value"], image_path)
                    if turn["from"] in ("human", "user")
                    else _parse_sharegpt4v_assistant_content(turn["value"])
                )
            }
        )
        for turn in example["conversations"]
    ]

    return {"conversations": messages}


def _adapt_openai_content_part(part: dict) -> dict:
    """Map an OpenAI-style content part to the pipeline's internal part format.

    The pipeline represents media internally as
    ``{"type": <modality>, "url"|"path": ...}`` (see
    ``preprocessing._adapt_part_for_vllm``). OpenAI chat parts instead nest the
    payload, e.g. ``{"type": "image_url", "image_url": {"url": ...}}``.
    """
    part_type = part.get("type")
    if part_type == "text":
        return {"type": "text", "text": part.get("text", "")}

    for modality in ("image", "video", "audio"):
        if part_type == f"{modality}_url":
            payload = part.get(f"{modality}_url") or {}
            url = payload.get("url") if isinstance(payload, dict) else payload
            return {"type": modality, "url": url}

    # Already in the internal format (or an unknown part we leave untouched).
    return part


def _normalize_kimi_mtp(example: dict) -> dict:
    """Normalize ``lightseekorg/kimi-mtp-dataset`` rows.

    The dataset mixes three shapes under a single ShareGPT-style ``conversations``
    field of ``{"from", "value"}`` turns:

    * text-only turns (``value`` is a plain string),
    * ``continual_tool_kimi`` turns (plain-string system prompts / Kimi tool-call
      markers -- handled downstream as text), and
    * ``llava_instruct`` multimodal turns whose ``value`` is a JSON-encoded string
      of OpenAI content parts, e.g.
      ``'[{"type":"image_url","image_url":{"url":...}},{"type":"text","text":...}]'``.

    Downstream (``preprocessing._normalize_conversation``) already maps
    ``from``/``value``, ``system``, and ``tool`` roles, so here we only decode the
    multimodal JSON-string values into the pipeline's internal content-part format.
    """
    conversations = []
    for turn in example.get("conversations", []):
        value = turn.get("value")
        if isinstance(value, str):
            stripped = value.lstrip()
            if stripped.startswith("["):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list) and all(
                    isinstance(part, dict) and "type" in part for part in parsed
                ):
                    value = [_adapt_openai_content_part(part) for part in parsed]
        conversations.append(turn | {"value": value})
    return {"conversations": conversations}


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "sharegpt": DatasetConfig(
        name="sharegpt",
        hf_path="Aeala/ShareGPT_Vicuna_unfiltered",
        split="train",
    ),
    "ultrachat": DatasetConfig(
        name="ultrachat",
        hf_path="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        normalize_fn=_normalize_ultrachat,
    ),
    "gsm8k": DatasetConfig(
        name="gsm8k",
        hf_path="openai/gsm8k",
        subset="main",
        split="train",
        normalize_fn=_normalize_gsm8k,
    ),
    # TorchSpec's Kimi-K2.5 EAGLE3 training corpus (~477k ShareGPT-style rows).
    # Mixes text (mostly open-perfectblend), llava_instruct multimodal turns whose
    # `value` is a JSON-encoded list of OpenAI content parts, and continual_tool_kimi
    # tool-call/system turns. Multimodal rows reference remote COCO image URLs, so a
    # text-only target processor will skip/mishandle them -- use a multimodal target
    # (and a vLLM server allowed to fetch those URLs) to train on the image turns.
    "kimi_mtp": DatasetConfig(
        name="kimi_mtp",
        hf_path="lightseekorg/kimi-mtp-dataset",
        split="train",
        normalize_fn=_normalize_kimi_mtp,
    ),
    # NOTE: You need to serve vLLM with `--allowed-local-media-path /path/to/coco`
    "sharegpt4v_coco": DatasetConfig(
        name="sharegpt4v_coco",
        hf_path="Lin-Chen/ShareGPT4V",
        subset="ShareGPT4V",
        split="train",
        filter_fn=_filter_sharegpt4v_coco,
        normalize_fn=_normalize_sharegpt4v_coco,
    ),
}
