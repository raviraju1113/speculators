#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import sys
import time
from collections import deque
from collections.abc import Callable
from typing import Any, cast

import aiohttp
from datasets import load_dataset
from tqdm import tqdm

# NOTE: `datasets` is imported lazily inside iter_input_items (HF-dataset branch
# only). The --input-jsonl (local conversations) path does not need it, so a
# broken/absent `datasets`/`pyarrow` install won't block regenerating a JSONL.

# Characters of each turn shown per item by --dry-run.
DRY_RUN_PREVIEW_CHARS = 400

# A dataset config declares EITHER `prompt_field` (single-prompt datasets: one
# column holds the instruction) OR `messages_field` (conversation datasets: one
# column holds a `[{role, content}]` list, regenerated turn-by-turn like
# --input-jsonl). `min_prompt_chars` sets a per-dataset floor on total user-turn
# length, overridable with --min-prompt-chars.
DATASET_CONFIGS = {
    "magpie": {
        "id": "Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered",
        "prompt_field": "instruction",
        "default_split": "train",
    },
    "ultrachat": {
        "id": "HuggingFaceH4/ultrachat_200k",
        "prompt_field": "prompt",
        "default_split": "train_sft",
    },
    "gsm8k": {
        "id": "openai/gsm8k",
        "prompt_field": "question",
        "default_split": "train",
        "subset": "main",
    },
    # Gated: accept the terms at
    # https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2
    # and export HF_TOKEN before running. Splits: stem / math / code / chat /
    # multilingual_{de,es,fr,it,ja}.
    #
    # Rows are single-turn (user, assistant) with the assistant answer wrapped in
    # a DeepSeek-R1 `<think>` block. We drop the original assistant turns and
    # regenerate, so no R1 reasoning trace leaks into the corpus.
    #
    # min_prompt_chars guards against placeholder prompts: the whole v1 `code`
    # split ships `-` as the user turn (the TACO/CodeForces statements are not
    # redistributable). Verify with --dry-run before a full run.
    "nemotron-v2": {
        "id": "nvidia/Nemotron-Post-Training-Dataset-v2",
        "messages_field": "messages",
        "default_split": "stem",
        "min_prompt_chars": 16,
    },
    "nemotron-v1": {
        "id": "nvidia/Nemotron-Post-Training-Dataset-v1",
        "messages_field": "messages",
        "default_split": "stem",
        "min_prompt_chars": 16,
    },
}


def parse_args():
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Regenerate dataset responses via a vLLM Chat API endpoint."
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="vLLM OpenAI-compatible Chat Completions endpoint",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name exposed by vLLM (auto-detected if not specified)",
    )
    parser.add_argument(
        "--dataset",
        default="ultrachat",
        type=_dataset_choice,
        choices=REGEN_DATASETS,
        help="Dataset to process",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split (defaults to dataset-specific split)",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help=(
            "Dataset subset/config name "
            "(auto-detected from dataset config if not specified)"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N rows")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Max concurrent requests",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="max_tokens for generation",
    )
    parser.add_argument(
        "--sampling-params",
        default=None,
        help=(
            "JSON object merged into each chat-completion request, "
            'e.g. \'{"temperature": 0.6, "top_p": 0.95, "seed": 0}\''
        ),
    )
    parser.add_argument(
        "--outfile",
        default=None,
        help="Output JSONL path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already in outfile (by stable primary id)",
    )
    parser.add_argument(
        "--language-filter",
        default=None,
        help="Only process rows where language==this (e.g., EN)",
    )
    parser.add_argument(
        "--min-prompt-chars",
        type=int,
        default=None,
        help=(
            "Skip rows whose user turns total fewer than N characters. Filters "
            "out placeholder prompts (e.g. the '-' user turns in the Nemotron "
            "code split). Defaults to the dataset config's value, else 0."
        ),
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Inspect the source instead of generating: print the first N items "
            "(plus skip counts) and exit. No endpoints are contacted. Use this to "
            "sanity-check prompts before spending GPU hours."
        ),
    )
    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use in filenames."""
    name = re.sub(r'[/\\:*?"<>|]', "_", name)
    name = name.replace(" ", "_")
    return name.strip("._")


# ---------------------------------------------------------------------------
# Row ingestion: user/system turns, tool schema, cached tool results
# ---------------------------------------------------------------------------


def _conversation_messages(row: dict[str, Any]) -> list:
    """The ``messages`` or ``conversations`` list from a row, else []."""
    convs = row.get("messages")
    if not (isinstance(convs, list) and convs):
        convs = row.get("conversations")
    return convs if isinstance(convs, list) else []


def _message_role_content(m: dict) -> tuple[str | None, Any]:
    """Canonical ``(role, content)`` for a message across the role/content and
    from/value schemas. ``role`` collapses ``human`` to ``user``; ``system`` and
    ``tool`` pass through; anything else (assistant/gpt) returns ``None``."""
    role = m.get("role") or m.get("from")
    content = m.get("content")
    if content is None:
        content = m.get("value")
    if role in ("user", "human"):
        return "user", content
    if role in ("system", "tool"):
        return role, content
    return None, content


def extract_conversation(
    row: dict[str, Any], prompt_field: str | None
) -> tuple[list[dict[str, Any]], list[tuple[Any, list[str]]]]:
    """Read the regeneration turns and the cached tool results in one pass.

    Walks a ``messages``/``conversations`` field (role/content or from/value
    schema). System and user turns drive regeneration; the original assistant
    turns are dropped and regenerated. Each tool-result turn is captured as a
    ``(content, tool_names)`` pair, where ``tool_names`` are the tools that
    result answers (read from its ``<tool_response>`` payload) -- used to guard
    the positional splice against a call for a different tool. Rows without a
    usable conversation fall back to a single ``prompt_field`` user turn and
    carry no results.
    """
    turns: list[dict[str, Any]] = []
    results: list[tuple[Any, list[str]]] = []
    for m in _conversation_messages(row):
        if not isinstance(m, dict):
            continue
        role, content = _message_role_content(m)
        if role in ("system", "user") and content:
            turns.append({"role": role, "content": content})
        elif role == "tool" and content is not None:
            results.append((content, _tool_result_names(content)))
        # original assistant/gpt turns are dropped and regenerated
    if any(turn["role"] == "user" for turn in turns):
        return turns, results

    # no usable user turn: fall back to the prompt_field
    prompt = row.get(prompt_field) if prompt_field else None
    if prompt:
        return [{"role": "user", "content": prompt}], []
    return [], []


def prepare_row(
    row: dict[str, Any], config: DatasetConfig
) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[Any, list[str]]]] | None:
    """The normalized row, its regeneration turns, and cached tool results.

    ``filter_fn`` sees the raw row; ``normalize_fn`` is merged over it (HF
    ``map`` keeps raw columns). Turns and results are read from that one
    normalized row so they stay paired; ``None`` to skip the row.
    """
    if config.filter_fn is not None and not config.filter_fn(row):
        return None
    normalized = {**row, **config.normalize_fn(row)} if config.normalize_fn else row
    turns, tool_results = extract_conversation(normalized, config.prompt_field)
    if not turns:
        return None
    return normalized, turns, tool_results


def _maybe_json(value: Any):
    """Best-effort JSON decode; return None if the value is not valid JSON."""
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _list_field(row, key) -> list | None:
    """Non-empty list from ``row[key]`` (JSON-decoded if a string), else None."""
    value = row.get(key)
    if isinstance(value, str):
        value = _maybe_json(value)
    return value if isinstance(value, list) and value else None


def extract_tools(row) -> list | None:
    """Return the OpenAI-style ``tools`` schema for a row, or ``None``.

    Reads the ``tools`` column -- a list, or a JSON-string encoding one, as the
    Hermes function-calling dataset stores it. A row that declares a ``tools``
    field we cannot read as a list raises ``ValueError`` rather than silently
    regenerating tool-free; a tool-free row returns ``None``.
    """
    tools = _list_field(row, "tools")
    if tools:
        return tools
    if row.get("tools") not in (None, "", [], {}):
        raise ValueError("a tools field is present but not a usable list")
    return None


_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)


def _tool_result_names(content: Any) -> list[str]:
    """Tool names a cached result answers, for the splice name-match guard.

    Hermes embeds the answering tool's name in each ``<tool_response>`` payload.
    Returns ``[]`` when no name can be read, which disables the guard for that
    result rather than blocking the splice on our own parse miss.
    """
    if not isinstance(content, str):
        return []
    names = []
    for block in _TOOL_RESPONSE_RE.findall(content):
        obj = _maybe_json(block)
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            names.append(obj["name"])
    return names


# ---------------------------------------------------------------------------
# Resume state & vLLM server IO
# ---------------------------------------------------------------------------


def _is_present(value: Any) -> bool:
    """Return True for a usable identifier (not None / not empty string)."""
    return value not in (None, "")


def _content_hash(row: dict[str, Any]) -> str:
    """Deterministic hash of a row, used when it has no explicit id."""
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
    return "hash_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _primary_identifier(row: dict[str, Any]) -> str:
    """Return a stable primary id for a dataset row.

    Prefers an explicit ``id``/``uuid``; otherwise a deterministic content hash.
    Unlike a streaming enumeration index, this key does not shift when
    ``--limit``/``--language-filter`` or the input order change, so ``--resume``
    stays correct across runs.
    """
    for field in ("id", "uuid"):
        value = row.get(field)
        if _is_present(value):
            return str(value)
    return _content_hash(row)


def load_seen(path: str) -> set[str]:
    """Load previously completed conversation ids from the output file.

    A conversation fans out to one row per target generation -- a tool call or a
    final answer -- whose ``id`` carries a ``_gen<N>`` suffix; the conversation's
    own :func:`_primary_identifier` is kept alongside it as ``primary_id``.
    Resume keys on that, since the suffixed ids never match a recomputed one.
    Rows are written only after the conversation finishes, so one row is enough
    to mark it done.

    ``id`` is the fallback for output files written before the fan-out, where the
    top-level ``id`` *was* the primary identifier.
    """
    seen: set[str] = set()
    if not os.path.isfile(path):
        return seen

    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = obj.get("primary_id")
            if not _is_present(key):
                key = obj.get("id")
            if _is_present(key):
                seen.add(str(key))
    return seen


async def detect_model(endpoint: str) -> str:
    """Automatically detect the model name from the vLLM server."""
    models_endpoint = endpoint.replace("/v1/chat/completions", "/v1/models")

    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(models_endpoint) as response,
        ):
            data = await response.json()
            models = data.get("data", [])
            if models:
                model_name = models[0]["id"]
                print(f"Auto-detected model: {model_name}")
                return model_name
            raise ValueError("No models found at endpoint")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Failed to auto-detect model from {models_endpoint}: {e}\n"
            f"Please specify model with --model argument"
        ) from e


def build_detokenizer(model: str) -> Callable[[list[int]], str]:
    """Return a decoder for the review-only ``text`` twin (see _sample_from_response).

    Loads ``model``'s tokenizer so the twin is exactly ``decode(input_ids)``;
    ``skip_special_tokens=False`` keeps the chat/control tokens (``<|im_start|>``,
    ``<think>``, ``<|im_end|>``) visible. Pass a tokenizer path as ``--model``
    (the checkpoint, not a ``--served-model-name`` alias).
    """
    print(f"Loading tokenizer: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model)

    def detokenize(token_ids: list[int]) -> str:
        # decode() is typed str | list[str]; a 1-D id list always yields str.
        return cast("str", tokenizer.decode(token_ids, skip_special_tokens=False))

    return detokenize


# Transient statuses worth retrying: request timeout, conflict, too-early, and
# rate limiting, plus all 5xx. Other non-2xx replies (e.g. 400/401/404) are
# permanent config/client errors and fail fast.
SERVER_ERROR_STATUS = 500
RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429}


@with_retries
async def _post_chat(
    session: aiohttp.ClientSession,
    endpoint: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """POST one chat-completion request and return the parsed response.

    Wrapped by ``with_retries`` (adds a ``max_retries`` kwarg): transient
    failures — network errors and transient HTTP statuses (408/409/425/429/5xx)
    — are retried with exponential backoff. Permanent non-2xx replies (e.g.
    400/404) raise ``InvalidResponseError``, which ``with_retries`` never
    retries, so they fail fast. A non-2xx reply is surfaced with its status and
    a short body so the caller does not record a bare ``KeyError('choices')``.
    """
    async with session.post(endpoint, json=payload) as response:
        if not response.ok:
            body = (await response.text())[:500]
            message = f"HTTP {response.status} from {endpoint}: {body}"
            # Retry transient statuses (408/409/425/429/5xx); fail fast otherwise.
            if (
                response.status >= SERVER_ERROR_STATUS
                or response.status in RETRYABLE_HTTP_STATUSES
            ):
                raise RuntimeError(message)
            raise InvalidResponseError(message)
        return await response.json()


# ---------------------------------------------------------------------------
# Regeneration: model response -> boundary training samples
# ---------------------------------------------------------------------------


def build_boundary_sample(
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
) -> tuple[list[int], list[int]]:
    """Build one training sample: prompt (loss_mask 0) + generated tokens (1).

    The generation boundary is the mask -- no ``{% generation %}`` markers, no regex.
    """
    input_ids = [*prompt_token_ids, *completion_token_ids]
    loss_mask = [0] * len(prompt_token_ids) + [1] * len(completion_token_ids)
    return input_ids, loss_mask


def _tool_result_message(tool_call: dict, content: str) -> dict[str, Any]:
    """Build the ``tool`` message that feeds a cached (off-policy) result back to
    the target, paired to the id of the call the target just generated."""
    message: dict[str, Any] = {"role": "tool", "content": content}
    call_id = tool_call.get("id")
    if call_id:
        message["tool_call_id"] = call_id
    return message


def _sample_from_response(
    data: dict[str, Any],
    *,
    detokenize: Callable[[list[int]], str],
    conv_id: str,
    sample_index: int,
    idx: int,
    endpoint: str,
    sampling_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list | None]:
    """Turn one chat-completion response into a boundary sample and the assistant
    message to append to the running prefix.

    Returns None for unusable rows (multimodal/list content, unknown roles, or no
    user turn) so the caller can skip them.

    Turns with empty content are dropped rather than disqualifying the row: every
    Nemotron row opens with an empty `system` turn, and rejecting the whole
    conversation over it would discard the entire dataset. An empty turn carries
    no context, so dropping it loses nothing; rows left without a user turn are
    still rejected below.
    """
    if not isinstance(conv, list) or not conv:
        return None
    messages: list[dict[str, str]] = []
    for turn in conv:
        role = _ROLE_MAP.get(turn.get("from") or turn.get("role"))
        content = turn.get("value")
        if content is None:
            content = turn.get("content")
        # Reject multimodal (list content) and malformed turns.
        if role is None or not isinstance(content, str):
            return None
        if not content:
            continue
        messages.append({"role": role, "content": content})
    if not any(m["role"] == "user" for m in messages):
        return None
    return messages


def _choice_content(data: dict) -> str:
    """Return the assistant message content, or raise a clear error.

    vLLM returns a body without ``choices`` on failure (e.g. a 400 when the
    conversation exceeds ``max_model_len``); surface that instead of a cryptic
    ``KeyError('choices')``.
    """
    choices = data.get("choices")
    if not choices:
        detail = data.get("error") or data.get("message") or data
        raise RuntimeError(f"server returned no choices: {str(detail)[:200]}")
    return choices[0]["message"]["content"]


async def regenerate_conversation(
    sem, session, endpoint, args, messages
) -> tuple[list[dict[str, str]], dict | None, list[str | None]]:
    """Walk a conversation and regenerate each assistant turn with the target.

    Keeps system/user turns; for every user turn, generates a fresh assistant
    reply conditioned on the accumulated context (original assistant turns are
    replaced). Returns (regenerated_messages, last_usage, finish_reasons).

    finish_reasons is per generated turn: a "length" entry means the reply hit
    --max-tokens and is truncated mid-sentence, which is worth filtering out of a
    training corpus. Without it, truncation is invisible in the output.
    """
    regenerated: list[dict[str, str]] = []
    last_usage = None
    finish_reasons: list[str | None] = []
    for turn in messages:
        if turn["role"] not in ("system", "user"):
            continue  # drop original assistant turns; we regenerate them
        regenerated.append(turn)
        if turn["role"] != "user":
            continue
        payload = {
            "model": args.model,
            "messages": regenerated,
            "max_tokens": args.max_tokens,
        }
        async with sem, session.post(endpoint, json=payload) as response:
            data = await response.json()
        content = _choice_content(data)
        finish_reasons.append(data["choices"][0].get("finish_reason"))
        regenerated.append({"role": "assistant", "content": content})
        last_usage = data.get("usage")
    return regenerated, last_usage, finish_reasons


async def _regenerate_conversation_item(
    sem, session, queue_item, args, out_fh, err_fh, endpoint, progress, stats
):
    """Regenerate one conversation item and write it out (conversations mode)."""
    idx = queue_item["idx"]
    sample_id = queue_item.get("uuid") or f"sample_{idx}"
    start_time = time.time()
    try:
        regen, usage, finish_reasons = await regenerate_conversation(
            sem, session, endpoint, args, queue_item["messages"]
        )
        output = {
            "id": sample_id,
            "conversations": [
                {"from": _TO_FROM[m["role"]], "value": m["content"]} for m in regen
            ],
            "metadata": {
                "idx": idx,
                "num_turns": len(regen),
                "latency_s": round(time.time() - start_time, 3),
                "usage": usage,
                "endpoint": endpoint,
                "finish_reasons": finish_reasons,
                # Convenience flag: any turn that hit --max-tokens is truncated.
                "truncated": "length" in finish_reasons,
            },
        }
        out_fh.write(json.dumps(output, ensure_ascii=False) + "\n")
        out_fh.flush()
        stats["ok"] += 1
        if output["metadata"]["truncated"]:
            stats["truncated"] += 1
    except Exception as e:  # noqa: BLE001
        # Errors go to a sibling .errors.jsonl so the training output stays clean.
        err_fh.write(
            json.dumps(
                {"id": sample_id, "metadata": {"idx": idx, "error": repr(e)}},
                ensure_ascii=False,
            )
            + "\n"
        )
        err_fh.flush()
        stats["errors"] += 1
    finally:
        progress.set_postfix(ok=stats["ok"], errors=stats["errors"], refresh=False)
        progress.update(1)


async def worker(
    session: aiohttp.ClientSession,
    queue: "asyncio.Queue[dict[str, Any]]",
    args,
    out_fh,
    err_fh,
    endpoint: str,
    progress,
    stats: dict[str, Any],
    detokenize: Callable[[list[int]], str],
):
    """Pull conversations off the queue and regenerate them into boundary rows.

    Each target generation becomes one boundary sample; tool calls reuse the
    source data's cached results (see ``regenerate_conversation``). Truncated
    conversations still emit the rows completed before the cut.
    """

    async def post(payload: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        result = await _post_chat(
            session, endpoint, payload, max_retries=args.max_retries
        )
        latency = time.perf_counter() - t0
        stats["total_request_s"] += latency
        stats["requests"] += 1
        logger.debug("vLLM request completed in %.0f ms", latency * 1000)
        return result

    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        conv_id = item["primary_id"]
        # Held by the caller so a mid-conversation failure can still report how
        # many rows had been completed.
        samples: list[dict[str, Any]] = []
        try:
            truncated = await regenerate_conversation(
                post,
                item,
                model=args.model,
                max_tokens=args.max_tokens,
                endpoint=endpoint,
                sampling_params=args.sampling_params,
                samples=samples,
                detokenize=detokenize,
            )
            # Written only after the conversation finishes -- a clean truncation
            # included, since rerunning it would truncate again. An exception
            # writes nothing, so any row in the output file means the
            # conversation needs no rerun (see load_seen).
            for sample in samples:
                out_fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
                usage = sample.get("metadata", {}).get("usage", {})
                stats["completion_tokens"] += usage.get("completion_tokens", 0)
            out_fh.flush()
            stats["ok"] += 1
            if finish_reason == "length":
                stats["truncated"] += 1
        except Exception as e:  # noqa: BLE001
            # Failures go to a separate error file, not the training output.
            error_output = {
                "id": conv_id,
                "metadata": {
                    "idx": item["idx"],
                    "error": repr(e),
                    "generations_completed": len(samples),
                    "endpoint": endpoint,
                },
            }
            err_fh.write(json.dumps(error_output, ensure_ascii=False) + "\n")
            err_fh.flush()
            stats["errors"] += 1
        finally:
            elapsed = time.perf_counter() - stats["start_time"]
            postfix = {
                "ok": stats["ok"],
                "err": stats["errors"],
                "trunc": stats["truncated"],
            }
            if elapsed > 0 and stats["requests"] > 0:
                postfix["rps"] = f"{stats['requests'] / elapsed:.1f}"
                postfix["tps"] = f"{stats['completion_tokens'] / elapsed:.0f}"
            progress.set_postfix(postfix, refresh=False)
            progress.update(1)
            queue.task_done()


def resolve_split(args) -> str | None:
    """Return the effective HF split (None in --input-jsonl mode)."""
    if args.input_jsonl:
        return None
    cfg = DATASET_CONFIGS[args.dataset]
    return args.split if args.split is not None else cfg["default_split"]


def resolve_min_prompt_chars(args) -> int:
    """Return the user-turn length floor: CLI flag, else dataset default, else 0."""
    if args.min_prompt_chars is not None:
        return args.min_prompt_chars
    if args.input_jsonl:
        return 0
    return DATASET_CONFIGS[args.dataset].get("min_prompt_chars", 0)


def prompt_chars(messages: list[dict[str, str]]) -> int:
    """Total characters across the user turns of a normalized conversation."""
    return sum(len(m["content"]) for m in messages if m["role"] == "user")


def load_hf_dataset(dataset_id: str, subset: str | None, split: str):
    """Stream an HF dataset, turning an auth failure into an actionable error."""
    from datasets import load_dataset  # noqa: PLC0415  (lazy: HF path only)

    try:
        return load_dataset(dataset_id, name=subset, split=split, streaming=True)
    except Exception as e:
        text = str(e)
        if any(s in text for s in ("401", "403", "gated", "restricted", "authenticat")):
            raise SystemExit(
                f"Cannot read {dataset_id} (split={split}): {text}\n\n"
                f"This dataset is gated. Accept the terms at\n"
                f"  https://huggingface.co/datasets/{dataset_id}\n"
                f"then authenticate (`hf auth login`, or export HF_TOKEN=hf_...)."
            ) from e
        raise


class _Budget:
    """Row-examination counter shared by the source iterators.

    ``max_scan`` has to be enforced inside the iterators rather than by the
    caller: a split that filters out every row yields nothing, so a caller-side
    check would never run and the scan would never stop.
    """

    def __init__(self, skipped: dict[str, int] | None, max_scan: int | None):
        self.skipped = skipped
        self.max_scan = max_scan
        self.examined = 0

    def exhausted(self) -> bool:
        """Return True once max_scan rows have been examined; counts this row."""
        if self.max_scan is not None and self.examined >= self.max_scan:
            return True
        self.examined += 1
        return False

    def skip(self, reason: str) -> None:
        """Tally one omitted row under `reason`."""
        if self.skipped is not None:
            self.skipped[reason] = self.skipped.get(reason, 0) + 1


def _iter_jsonl_items(args, budget: _Budget, min_chars: int):
    """Yield conversation items from a local JSONL (--input-jsonl)."""
    skip_sources = {
        s.strip() for s in (args.skip_sources or "").split(",") if s.strip()
    }
    only_sources = (
        {s.strip() for s in args.sources.split(",") if s.strip()}
        if args.sources
        else None
    )
    with open(args.input_jsonl, encoding="utf-8") as f:
        for index, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            if budget.exhausted():
                return
            row = json.loads(stripped)
            source = row.get("source")
            if source in skip_sources:
                budget.skip("skipped_source")
                continue
            if only_sources is not None and source not in only_sources:
                budget.skip("not_in_sources")
                continue
            messages = normalize_input_conversation(row.get("conversations"))
            if messages is None:
                budget.skip("multimodal_or_malformed")
                continue
            if prompt_chars(messages) < min_chars:
                budget.skip("short_prompt")
                continue
            yield index, row.get("id") or row.get("uuid"), {"messages": messages}


def _hf_payload(row, cfg, budget: _Budget, min_chars: int) -> dict | None:
    """Convert one HF row to a queue payload, or None if it should be skipped."""
    # Conversation-shaped dataset (e.g. Nemotron `messages`): regenerate every
    # assistant turn in-context, exactly like the --input-jsonl path.
    messages_field = cfg.get("messages_field")
    if messages_field:
        messages = normalize_input_conversation(row.get(messages_field))
        if messages is None:
            budget.skip("multimodal_or_malformed")
            return None
        if prompt_chars(messages) < min_chars:
            budget.skip("short_prompt")
            return None
        return {"messages": messages}

    prompt = row.get(cfg["prompt_field"])
    if not prompt:
        budget.skip("empty_prompt")
        return None
    if len(prompt) < min_chars:
        budget.skip("short_prompt")
        return None
    return {"prompt": prompt}


def _iter_hf_items(args, budget: _Budget, min_chars: int):
    """Yield items from a streamed HF dataset in DATASET_CONFIGS."""
    cfg = DATASET_CONFIGS[args.dataset]
    subset = args.subset if args.subset is not None else cfg.get("subset")
    dataset = load_hf_dataset(cfg["id"], subset, resolve_split(args))

    for index, row in enumerate(dataset):
        if budget.exhausted():
            return
        if args.language_filter and row.get("language") != args.language_filter:
            budget.skip("language")
            continue
        payload = _hf_payload(row, cfg, budget, min_chars)
        if payload is not None:
            yield index, row.get("uuid"), payload


def iter_input_items(args, skipped: dict[str, int] | None = None, max_scan=None):
    """Yield (index, id, payload) rows from the selected source.

    payload is ``{"messages": [...]}`` for a conversation source (a local JSONL
    via ``--input-jsonl``, or an HF dataset whose config declares
    ``messages_field``) and ``{"prompt": ...}`` for a single-prompt HF dataset.
    Rows to skip (multimodal, empty, filtered) are omitted; if ``skipped`` is
    given, the reason for each omission is tallied into it. ``max_scan`` stops
    after that many rows have been examined (kept + skipped).
    """
    budget = _Budget(skipped, max_scan)
    min_chars = resolve_min_prompt_chars(args)
    source = _iter_jsonl_items if args.input_jsonl else _iter_hf_items
    yield from source(args, budget, min_chars)


def dry_run(args) -> bool:
    """Print the first --dry-run items and the skip tally, without generating.

    Returns whether the split looks regenerable, so a driver script can gate a
    long GPU run on `script.py --dry-run N` exiting 0.
    """
    limit = args.dry_run
    # Cap rows *examined*, not just kept: if a split filters out everything (as
    # the Nemotron v1 code split does), an uncapped scan would stream millions of
    # rows looking for a keeper it will never find.
    scan_cap = max(1000, 50 * limit)
    print(
        f"=== DRY RUN: first {limit} item(s) within {scan_cap} rows scanned; "
        f"no requests will be sent ===\n"
    )
    skipped: dict[str, int] = {}
    kept = 0
    # closing() matters: HF streaming spawns a reader thread, and abandoning the
    # generator mid-stream can crash the interpreter during finalization.
    items = iter_input_items(args, skipped, max_scan=scan_cap)
    with contextlib.closing(items):
        for index, uuid, payload in items:
            kept += 1
            turns = payload.get("messages") or [
                {"role": "user", "content": payload["prompt"]}
            ]
            print(f"--- item {kept} (idx={index}, id={uuid})")
            print(f"    user chars: {prompt_chars(turns)}")
            for turn in turns:
                content = turn["content"]
                preview = content[:DRY_RUN_PREVIEW_CHARS]
                if len(content) > DRY_RUN_PREVIEW_CHARS:
                    preview += " […]"
                print(f"    [{turn['role']}] len={len(content)} :: {preview!r}")
            print()
            if kept >= limit or kept + sum(skipped.values()) >= scan_cap:
                break
            if index and index % 20000 == 0:
                print(f"    ... scanned {index} rows, kept {kept} so far")

    examined = kept + sum(skipped.values())
    print(f"kept {kept} of {examined} row(s) examined")
    if skipped:
        print("skipped:")
        for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>8}  {reason}")
    else:
        print("skipped: none")

    short = skipped.get("short_prompt", 0)
    if short and short >= 0.5 * examined:
        print(
            f"\n!! WARNING: {short}/{examined} rows have placeholder user turns "
            f"(e.g. '-') and cannot be regenerated -- the real prompts are not in "
            f"this release. Do NOT start a full run on this split."
        )
        return False
    if kept == 0:
        print("\n!! WARNING: no usable rows found in the scanned prefix.")
        return False
    return True


def default_outfile(args) -> str:
    """Build the default output path for the current source and model."""
    model_name = sanitize_filename(
        args.model.split("/")[-1] if "/" in args.model else args.model
    )
    if args.input_jsonl:
        stem = os.path.splitext(os.path.basename(args.input_jsonl))[0]
        return f"{stem}_regen_{model_name}.jsonl"
    # Include the split: one file per split keeps --resume correct and lets the
    # splits be mixed independently downstream.
    name = sanitize_filename(f"{args.dataset}_{resolve_split(args)}")
    return f"{name}_{model_name}.jsonl"


def print_run_header(args) -> None:
    """Print the resolved source, mode, prompt floor and destination."""
    if args.input_jsonl:
        source, mode = args.input_jsonl, "conversations"
    else:
        cfg = DATASET_CONFIGS[args.dataset]
        source = f"{cfg['id']} split={resolve_split(args)} (HF)"
        mode = "conversations" if cfg.get("messages_field") else "single-prompt"
    print(f"Source: {source}  [{mode} mode]")
    print(f"Min prompt chars: {resolve_min_prompt_chars(args)}")
    print(f"Output file: {args.outfile}")
    print()


def print_run_summary(
    stats: dict[str, int], skipped: dict[str, int], outfile: str
) -> None:
    """Print generation counts plus why any rows never reached the model."""
    print(f"\nDone: ok={stats['ok']} errors={stats['errors']} -> {outfile}")
    truncated = stats.get("truncated", 0)
    if truncated:
        print(
            f"  {truncated} response(s) hit --max-tokens and are truncated "
            f"(metadata.truncated == true). Raise --max-tokens or filter them out "
            f"before training."
        )
    if skipped:
        print("Rows skipped before generation:")
        for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>8}  {reason}")


async def main():
    """Main async function to process dataset through vLLM endpoints."""
    args = parse_args()

    # Inspect-only: never touches the endpoints, so it works before servers exist.
    # Exits 3 on an unusable split so callers can gate a GPU run on it.
    if args.dry_run is not None:
        if not dry_run(args):
            hard_exit(3)
        return

    endpoints = await resolve_endpoints(args)

    # Auto-detect model if not specified
    if args.model is None:
        args.model = await detect_model(endpoint)

    print(f"Using model: {args.model}")

    if args.outfile is None:
        args.outfile = default_outfile(args)
    print_run_header(args)

    seen_ids = load_seen(args.outfile) if args.resume else set()
    dataset = load_dataset(dataset_id, name=subset, split=split, streaming=True)

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)

    ensure_parent_dirs(args.outfile, error_outfile)

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=90, sock_read=None)
    connector = aiohttp.TCPConnector(
        limit=None, force_close=False, enable_cleanup_closed=True
    )
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, headers=headers
    ) as session:
        with (
            open(args.outfile, "a", encoding="utf-8") as output_file,  # noqa: ASYNC230
            open(error_outfile, "a", encoding="utf-8") as error_file,  # noqa: ASYNC230
            tqdm(
                total=args.limit,
                desc="Generating responses",
                unit="sample",
                dynamic_ncols=True,
            ) as progress,
        ):
            stats = {"ok": 0, "errors": 0, "truncated": 0}
            # Round-robin each worker onto an endpoint so load spreads evenly
            # across all reachable servers.
            workers = [
                asyncio.create_task(
                    worker(
                        session,
                        queue,
                        args,
                        output_file,
                        error_file,
                        endpoint,
                        progress,
                        stats,
                        detokenize,
                    )
                )
                for _ in range(args.concurrency)
            ]

            processed_count = 0
            skipped: dict[str, int] = {}
            for index, uuid, payload in iter_input_items(args, skipped):
                if args.limit is not None and processed_count >= args.limit:
                    break

                if args.language_filter and row.get("language") != args.language_filter:
                    continue

                prepared = prepare_row(row, dataset_config)
                if prepared is None:
                    continue
                normalized, turns, tool_results = prepared

                primary_id = _primary_identifier(row)
                if primary_id in seen_ids:
                    continue

                # Broken input tool schema: record and skip (don't crash the run).
                try:
                    tools = extract_tools(normalized)
                except ValueError as exc:
                    logger.warning(
                        "Skipping row %s: input tool schema is broken (%s)",
                        primary_id,
                        exc,
                    )
                    error_output = {
                        "id": primary_id,
                        "metadata": {
                            "idx": index,
                            "error": repr(exc),
                            "generations_completed": 0,
                            "endpoint": endpoint,
                        },
                    }
                    error_file.write(
                        json.dumps(error_output, ensure_ascii=False) + "\n"
                    )
                    error_file.flush()
                    stats["errors"] += 1
                    progress.update(1)
                    continue

                await queue.put(
                    {
                        "idx": index,
                        "primary_id": primary_id,
                        "turns": turns,
                        "tools": tools,
                        "tool_results": tool_results,
                    }
                )
                processed_count += 1

            # Signal workers to stop
            for _ in range(len(workers)):
                await queue.put(None)
            await asyncio.gather(*workers)

    print_run_summary(stats, skipped, args.outfile)


def hard_exit(code: int) -> None:
    """Flush and exit without running interpreter finalization.

    `datasets`>=5 leaves a pyarrow reader thread behind when a streaming dataset
    is abandoned (e.g. --limit, or any early break), and finalizing it aborts with
    "Fatal Python error: PyGILState_Release" *after* all work is done. That noise
    at the tail of a multi-hour log reads like a failed run, so we skip it. Safe
    here: main() has returned, so every output file is already closed, and rows
    are flushed as they are written regardless.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        hard_exit(130)
    hard_exit(0)
