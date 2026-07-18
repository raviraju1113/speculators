#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import re
import sys
import time
from http import HTTPStatus
from typing import Any

import aiohttp
from tqdm import tqdm

# NOTE: `datasets` is imported lazily inside iter_input_items (HF-dataset branch
# only). The --input-jsonl (local conversations) path does not need it, so a
# broken/absent `datasets`/`pyarrow` install won't block regenerating a JSONL.

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
}


def parse_args():
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Regenerate responses from Magpie instructions via vLLM Chat API."
    )
    parser.add_argument(
        "--endpoint",
        nargs="+",
        default=["http://127.0.0.1:8000/v1/chat/completions"],
        help=(
            "One or more vLLM OpenAI-compatible Chat Completions endpoints. "
            "Requests are round-robined across all reachable endpoints, so you "
            "can point at several servers (e.g. one per GPU) for higher "
            "throughput."
        ),
    )
    parser.add_argument(
        "--skip-endpoint-validation",
        action="store_true",
        help=(
            "Do not probe endpoints before starting. By default every endpoint "
            "is health-checked and unreachable ones are dropped."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name exposed by vLLM (auto-detected if not specified)",
    )
    parser.add_argument(
        "--dataset",
        default="ultrachat",
        choices=list(DATASET_CONFIGS.keys()),
        help="Built-in HF dataset to process (ignored if --input-jsonl is set)",
    )
    parser.add_argument(
        "--input-jsonl",
        default=None,
        help=(
            "Path to a local conversations JSONL to regenerate (each row has a "
            "`conversations` list of {from,value} or {role,content} turns, e.g. "
            "the kimi-mtp-dataset). Every assistant turn is regenerated in-context "
            "by the target. Overrides --dataset; multimodal rows are skipped."
        ),
    )
    parser.add_argument(
        "--skip-sources",
        default="llava_instruct,continual_tool_kimi",
        help=(
            "Comma-separated `source` values to skip when using --input-jsonl. "
            "Default drops multimodal (llava_instruct) and tool-call "
            "(continual_tool_kimi) rows, which can't be cleanly text-regenerated. "
            "Pass '' to keep all."
        ),
    )
    parser.add_argument(
        "--sources",
        default=None,
        help=(
            "If set, only regenerate rows whose `source` is in this comma-separated "
            "allowlist (applied after --skip-sources). --input-jsonl only."
        ),
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
        "--outfile",
        default=None,
        help="Output JSONL path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already in outfile (by uuid or idx)",
    )
    parser.add_argument(
        "--language-filter",
        default=None,
        help="Only process rows where language==this (e.g., EN)",
    )
    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use in filenames."""
    name = re.sub(r'[/\\:*?"<>|]', "_", name)
    name = name.replace(" ", "_")
    return name.strip("._")


def load_seen(path: str):
    """Load previously processed record IDs from output file."""
    seen = set()
    if not os.path.isfile(path):
        return seen

    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Skip failed rows so --resume retries them (errors are recorded in
            # a sibling .errors.jsonl; older runs may have written them inline).
            if (obj.get("metadata") or {}).get("error") is not None:
                continue
            # An output row is identified by any of: top-level uuid/idx (legacy
            # single-prompt), top-level id, or metadata.idx (conversations mode).
            # The consumer key is `str(uuid if uuid is not None else index)`, and
            # for a local JSONL that index is stored as metadata.idx — so collect
            # all of these to make --resume robust across both output shapes.
            for key in (
                obj.get("uuid"),
                obj.get("idx"),
                obj.get("id"),
                (obj.get("metadata") or {}).get("idx"),
            ):
                if key is not None:
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


async def validate_endpoints(endpoints: list[str]) -> list[str]:
    """Probe each endpoint's /v1/models and return only the reachable ones."""

    async def probe(endpoint: str) -> tuple[str, bool]:
        models_endpoint = endpoint.replace("/v1/chat/completions", "/v1/models")
        timeout = aiohttp.ClientTimeout(total=10)
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.get(models_endpoint) as response,
            ):
                return endpoint, response.status == HTTPStatus.OK
        except Exception:  # noqa: BLE001
            return endpoint, False

    results = await asyncio.gather(*(probe(e) for e in endpoints))
    reachable = [endpoint for endpoint, ok in results if ok]
    for endpoint, ok in results:
        print(f"  [{'ok' if ok else 'skip'}] {endpoint}")
    if not reachable:
        raise ValueError("No reachable endpoints. Check your servers and try again.")
    return reachable


async def resolve_endpoints(args) -> list[str]:
    """Normalize --endpoint to a list and optionally drop unreachable servers."""
    endpoints = args.endpoint if isinstance(args.endpoint, list) else [args.endpoint]
    if args.skip_endpoint_validation:
        print(f"Using {len(endpoints)} endpoint(s) (validation skipped):")
        for endpoint in endpoints:
            print(f"  {endpoint}")
    else:
        print(f"Validating {len(endpoints)} endpoint(s)...")
        endpoints = await validate_endpoints(endpoints)
    print(f"Using {len(endpoints)} endpoint(s): {endpoints}")
    return endpoints


# Map ShareGPT-style `from` (or OpenAI `role`) to chat roles, and back.
_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}
_TO_FROM = {"user": "human", "assistant": "gpt", "system": "system"}


def normalize_input_conversation(conv: Any) -> list[dict[str, str]] | None:
    """Normalize a ShareGPT/OpenAI conversation to `[{role, content}]`.

    Returns None for unusable rows (multimodal/list content, missing roles, or no
    user turn) so the caller can skip them.
    """
    if not isinstance(conv, list) or not conv:
        return None
    messages: list[dict[str, str]] = []
    for turn in conv:
        role = _ROLE_MAP.get(turn.get("from") or turn.get("role"))
        content = turn.get("value")
        if content is None:
            content = turn.get("content")
        # Skip multimodal (list content) and malformed turns.
        if role is None or not isinstance(content, str) or not content:
            return None
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
) -> tuple[list[dict[str, str]], dict | None]:
    """Walk a conversation and regenerate each assistant turn with the target.

    Keeps system/user turns; for every user turn, generates a fresh assistant
    reply conditioned on the accumulated context (original assistant turns are
    replaced). Returns (regenerated_messages, last_usage).
    """
    regenerated: list[dict[str, str]] = []
    last_usage = None
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
        regenerated.append({"role": "assistant", "content": content})
        last_usage = data.get("usage")
    return regenerated, last_usage


async def _regenerate_conversation_item(
    sem, session, queue_item, args, out_fh, err_fh, endpoint, progress, stats
):
    """Regenerate one conversation item and write it out (conversations mode)."""
    idx = queue_item["idx"]
    sample_id = queue_item.get("uuid") or f"sample_{idx}"
    start_time = time.time()
    try:
        regen, usage = await regenerate_conversation(
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
            },
        }
        out_fh.write(json.dumps(output, ensure_ascii=False) + "\n")
        out_fh.flush()
        stats["ok"] += 1
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
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    queue: "asyncio.Queue[dict[str, Any]]",
    args,
    out_fh,
    err_fh,
    endpoint: str,
    progress,
    stats: dict[str, int],
):
    """Worker that pulls items from queue and sends them to the vLLM endpoint."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        # Conversations mode (multi-turn regeneration from a local JSONL).
        if "messages" in item:
            await _regenerate_conversation_item(
                sem, session, item, args, out_fh, err_fh, endpoint, progress, stats
            )
            queue.task_done()
            continue

        idx = item["idx"]
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "max_tokens": args.max_tokens,
        }

        start_time = time.time()
        try:
            async with sem, session.post(endpoint, json=payload) as response:
                data = await response.json()

            generated_text = _choice_content(data)
            choice = data["choices"][0]
            message = choice["message"]
            reasoning_content = message.get("reasoning_content")
            if reasoning_content is None:
                reasoning_content = message.get("reasoning")
            finish_reason = choice.get("finish_reason")
            latency = time.time() - start_time

            # Format output in conversations structure
            metadata = {
                "idx": idx,
                "finish_reason": finish_reason,
                "latency_s": round(latency, 3),
                "usage": data.get("usage"),
                "endpoint": endpoint,
            }

            # Only include reasoning_content if it exists
            if reasoning_content is not None:
                metadata["reasoning_content"] = reasoning_content

            output = {
                "id": item.get("uuid") or f"sample_{idx}",
                "conversations": [
                    {"from": "human", "value": item["prompt"]},
                    {"from": "gpt", "value": generated_text},
                ],
                "metadata": metadata,
            }
            out_fh.write(json.dumps(output, ensure_ascii=False) + "\n")
            out_fh.flush()
            stats["ok"] += 1
        except Exception as e:  # noqa: BLE001
            error_output = {
                "id": item.get("uuid") or f"sample_{idx}",
                "metadata": {
                    "idx": idx,
                    "error": repr(e),
                    "endpoint": endpoint,
                },
            }
            err_fh.write(json.dumps(error_output, ensure_ascii=False) + "\n")
            err_fh.flush()
            stats["errors"] += 1
        finally:
            progress.set_postfix(
                ok=stats["ok"],
                errors=stats["errors"],
                refresh=False,
            )
            progress.update(1)
            queue.task_done()


def iter_input_items(args):
    """Yield (index, id, payload) rows from the selected source.

    payload is ``{"messages": [...]}`` for a local conversations JSONL
    (``--input-jsonl``) or ``{"prompt": ...}`` for a built-in HF dataset.
    Rows to skip (multimodal, empty, filtered) are omitted.
    """
    if args.input_jsonl:
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
                row = json.loads(stripped)
                source = row.get("source")
                if source in skip_sources:
                    continue
                if only_sources is not None and source not in only_sources:
                    continue
                messages = normalize_input_conversation(row.get("conversations"))
                if messages is None:
                    continue  # multimodal / malformed
                yield index, row.get("id") or row.get("uuid"), {"messages": messages}
        return

    from datasets import load_dataset  # noqa: PLC0415  (lazy: HF path only)

    cfg = DATASET_CONFIGS[args.dataset]
    split = args.split if args.split is not None else cfg["default_split"]
    subset = args.subset if args.subset is not None else cfg.get("subset")
    prompt_field = cfg["prompt_field"]
    dataset = load_dataset(cfg["id"], name=subset, split=split, streaming=True)
    for index, row in enumerate(dataset):
        if args.language_filter and row.get("language") != args.language_filter:
            continue
        prompt = row.get(prompt_field)
        if not prompt:
            continue
        yield index, row.get("uuid"), {"prompt": prompt}


async def main():
    """Main async function to process dataset through vLLM endpoints."""
    args = parse_args()

    endpoints = await resolve_endpoints(args)

    # Auto-detect model if not specified
    if args.model is None:
        args.model = await detect_model(endpoints[0])

    print(f"Using model: {args.model}")

    # Output filename default depends on the source.
    if args.outfile is None:
        model_name = sanitize_filename(
            args.model.split("/")[-1] if "/" in args.model else args.model
        )
        if args.input_jsonl:
            stem = os.path.splitext(os.path.basename(args.input_jsonl))[0]
            args.outfile = f"{stem}_regen_{model_name}.jsonl"
        else:
            args.outfile = f"{args.dataset}_{model_name}.jsonl"

    mode = "conversations" if args.input_jsonl else "single-prompt"
    print(f"Source: {args.input_jsonl or args.dataset + ' (HF)'}  [{mode} mode]")
    print(f"Output file: {args.outfile}")
    print()

    seen_ids = load_seen(args.outfile) if args.resume else set()

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    semaphore = asyncio.Semaphore(args.concurrency)

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
            open(  # noqa: ASYNC230
                args.outfile + ".errors.jsonl", "a", encoding="utf-8"
            ) as error_file,
            tqdm(
                total=args.limit,
                desc="Generating responses",
                unit="sample",
                dynamic_ncols=True,
            ) as progress,
        ):
            stats = {"ok": 0, "errors": 0}
            # Round-robin each worker onto an endpoint so load spreads evenly
            # across all reachable servers.
            workers = [
                asyncio.create_task(
                    worker(
                        semaphore,
                        session,
                        queue,
                        args,
                        output_file,
                        error_file,
                        endpoints[i % len(endpoints)],
                        progress,
                        stats,
                    )
                )
                for i in range(args.concurrency)
            ]

            processed_count = 0
            for index, uuid, payload in iter_input_items(args):
                if args.limit is not None and processed_count >= args.limit:
                    break

                key = str(uuid if uuid is not None else index)
                if key in seen_ids:
                    continue

                await queue.put({"idx": index, "uuid": uuid, **payload})
                processed_count += 1

            # Signal workers to stop
            for _ in range(len(workers)):
                await queue.put(None)
            await asyncio.gather(*workers)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
