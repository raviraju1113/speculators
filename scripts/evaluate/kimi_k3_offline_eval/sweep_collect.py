#!/usr/bin/env python3
"""Collect on-policy hidden states for the aa-lcr length sweep.

Two phases, run against two DIFFERENT servers (they can't run concurrently on
an 8-GPU box since both need tensor-parallel-size 8):

  --phase generate: render each `{"id", "prompt"}` row as a Kimi chat turn,
      let the target model greedily generate an answer against a PLAIN server
      (no speculative_config), and cache {prompt_ids, gen_ids} to
      <output>/gen_cache/gen_<i>.json.

  --phase extract: read the cached generations, re-submit each full
      prompt+answer token sequence (max_tokens=1) to an `extract_hidden_states`
      server to capture hidden states for every position, and write the
      sample_<i>.safetensors files (hs_stack schema, same as
      prep_and_collect.py). The loss mask marks only the generated answer
      tokens, so the TTT eval measures acceptance on the target's own output
      distribution — matching deployment.

Generation MUST run against a plain server, not the extract_hidden_states one:
confirmed by direct comparison (2026-09-17) that extract_hidden_states mode
corrupts greedy generation into repetition loops after ~10-12 tokens on the
same prompts that generate cleanly on a plain server (same vLLM version, same
sampling settings). Using the extraction server for generation silently bakes
degenerate "ground truth" answers into the collected samples.
"""

import argparse
import fcntl
import json
import time
from pathlib import Path

import openai
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from speculators.data_generation.vllm_client import generate_hidden_states


def load_hs_file(handle: str, timeout_s: float = 300.0) -> dict[str, torch.Tensor]:
    lock_path = handle + ".lock"
    deadline = time.monotonic() + timeout_s
    while not (Path(lock_path).exists() or Path(handle).exists()):
        if time.monotonic() > deadline:
            raise TimeoutError(f"hidden-states file never appeared: {handle}")
        time.sleep(0.2)
    if Path(lock_path).exists():
        with open(lock_path) as lf:
            fcntl.flock(lf, fcntl.LOCK_SH)
    return load_file(handle)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["generate", "extract"])
    ap.add_argument("--sweep-jsonl", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--base-url",
        default="http://127.0.0.1:8082/v1",
        help="server to talk to for this phase (plain for generate, "
        "extract_hidden_states for extract)",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-gen-tokens", type=int, default=256)
    ap.add_argument("--max-rows", type=int, default=None)
    return ap.parse_args()


def run_generate(args, tokenizer, out_dir: Path):
    client = openai.Client(base_url=args.base_url, api_key="none", timeout=1800)
    model_id = client.models.list().data[0].id
    cache_dir = out_dir / "gen_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.sweep_jsonl).open() as f:
        for i, line in enumerate(f):
            if args.max_rows is not None and i >= args.max_rows:
                break
            cache_path = cache_dir / f"gen_{i}.json"
            if cache_path.exists():
                print(f"[{i}] gen cached, skipping")
                continue

            row = json.loads(line)
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if hasattr(prompt_ids, "input_ids"):
                prompt_ids = prompt_ids.input_ids
            if isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]

            try:
                gen = client.completions.create(
                    model=model_id,
                    prompt=prompt_ids,
                    max_tokens=args.max_gen_tokens,
                    temperature=0.0,
                    extra_body={"return_token_ids": True},
                )
            except openai.BadRequestError as e:
                # e.g. prompt longer than the server's max_model_len
                print(f"[{i}] skipped ({str(e)[:120]})")
                continue
            gen_ids = getattr(gen.choices[0], "token_ids", None)
            if gen_ids is None:
                gen_ids = tokenizer.encode(gen.choices[0].text)
            if not gen_ids:
                print(f"[{i}] empty generation, skipping")
                continue

            cache_path.write_text(
                json.dumps({"prompt_ids": list(prompt_ids), "gen_ids": list(gen_ids)})
            )
            print(f"[{i}] ok: prompt={len(prompt_ids)}, gen={len(gen_ids)}")

    print(f"generate phase done -> {cache_dir}")


def run_extract(args, out_dir: Path):
    client = openai.Client(base_url=args.base_url, api_key="none", timeout=1800)
    model_id = client.models.list().data[0].id
    cache_dir = out_dir / "gen_cache"

    gen_files = sorted(
        cache_dir.glob("gen_*.json"), key=lambda p: int(p.stem.split("_")[1])
    )
    for gen_path in gen_files:
        i = int(gen_path.stem.split("_")[1])
        if args.max_rows is not None and i >= args.max_rows:
            continue
        out_path = out_dir / f"sample_{i}.safetensors"
        if out_path.exists():
            print(f"[{i}] exists, skipping")
            continue

        cached = json.loads(gen_path.read_text())
        prompt_ids, gen_ids = cached["prompt_ids"], cached["gen_ids"]

        full_ids = list(prompt_ids) + list(gen_ids)
        handle = generate_hidden_states(client, model_id, {"input_ids": full_ids})
        data = load_hs_file(handle)
        hs = data["hidden_states"]  # [S, L, H]
        if hs.shape[0] != len(full_ids):
            raise ValueError(
                f"sample {i}: hs seq {hs.shape[0]} != input len {len(full_ids)}"
            )
        loss_mask = torch.zeros(len(full_ids), dtype=torch.int64)
        loss_mask[len(prompt_ids) :] = 1
        save_file(
            {
                "input_ids": torch.tensor(full_ids, dtype=torch.int64),
                "loss_mask": loss_mask,
                "hs_stack": hs.contiguous(),
                "last_hidden_states": hs[:, -1, :].contiguous(),
            },
            out_path,
        )
        Path(handle).unlink(missing_ok=True)
        print(f"[{i}] ok: prompt={len(prompt_ids)}, gen={len(gen_ids)}")

    print(f"extract phase done -> {out_dir}")


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "generate":
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        run_generate(args, tokenizer, out_dir)
    else:
        run_extract(args, out_dir)


if __name__ == "__main__":
    main()
