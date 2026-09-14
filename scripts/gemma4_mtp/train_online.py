#!/usr/bin/env python3
"""Online fine-tuning for Gemma 4 assistant (MTP draft).

This script fine-tunes the Gemma4AssistantForCausalLM with the target model
running on the same GPU(s) as training (not on a separate vLLM server).
This is the "online" variant where target forward passes happen live during
training (not from precomputed cache).

A YAML ``drafts:`` list can name N recipes. Each draft is trained **in
isolation** (own target signals, own loss I/O, own optimizer, own
``<output_dir>/<name>/`` checkpoints). Drafts are never mixed in one step —
different drafts can require different inputs/outputs, so we do not share a
target-signal cache across them.

Usage:
    # Multi-draft config (sequential, isolated runs):
    python scripts/gemma4_mtp/train_online.py \\
        --config examples/train/gemma4_26b_mtp_online_multi.yaml

    # Single-draft CLI (backward compatible):
    bash examples/train/gemma4_26b_mtp_online.sh

Prerequisites:
    - Regenerated training data (JSONL with conversations)
    - conda env: speculator
    - pyyaml (for --config)
"""

from __future__ import annotations

import argparse
import copy
import os
from typing import Any

# Silence/avoid HF fast-tokenizer fork deadlock when DataLoader workers tokenize
# on the fly (workers fork after the tokenizer has been used in the main proc).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config with target/data + drafts: list (dynamic N drafts)",
    )
    # Single-draft CLI (ignored when --config is set, except as fallback fields).
    ap.add_argument("--target", default=None, help="target model path")
    ap.add_argument("--assistant", default=None, help="assistant/draft model path")
    ap.add_argument("--data", default=None, help="regenerated conversations JSONL")
    ap.add_argument("--output", default=None, help="output dir for checkpoints")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers for on-the-fly tokenization (0 = main process)",
    )
    ap.add_argument("--ttt-steps", type=int, default=5)
    ap.add_argument("--step-weight-beta", type=float, default=0.8)
    ap.add_argument("--soft-ce-weight", type=float, default=0.5)
    ap.add_argument("--hard-ce-weight", type=float, default=0.0)
    ap.add_argument(
        "--feature-l1-weight",
        type=float,
        default=0.0,
        help="EAGLE/DSpark feature (hidden) smooth-L1 distillation weight",
    )
    ap.add_argument("--bf16", action="store_true", help="load models in bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument(
        "--random-init",
        action="store_true",
        help="initialize assistant from scratch instead of pretrained checkpoint",
    )
    return ap.parse_args()


def _load_yaml(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(
            "pyyaml is required for --config; pip install pyyaml"
        ) from e
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit(f"config must be a mapping: {path}")
    return cfg


def resolve_train_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build a normalized train config from --config YAML and/or CLI flags.

    Normalized shape::

        {
          target, data, output_dir, epochs, batch_size, ...,
          drafts: [{name, assistant, lr, soft_ce_weight, ...}, ...]
        }
    """
    if args.config:
        cfg = _load_yaml(args.config)
    else:
        cfg = {}

    # Shared fields: YAML wins if present, else CLI.
    def shared(key: str, cli_attr: str | None = None, default=None):
        if key in cfg and cfg[key] is not None:
            return cfg[key]
        if cli_attr is not None:
            return getattr(args, cli_attr, default)
        return default

    output_dir = shared("output_dir", None, None) or shared("output", "output", None)
    if output_dir is None and args.output:
        output_dir = args.output

    out: dict[str, Any] = {
        "target": shared("target", "target"),
        "data": shared("data", "data"),
        "output_dir": output_dir,
        "epochs": int(shared("epochs", "epochs", 3)),
        "batch_size": int(shared("batch_size", "batch_size", 2)),
        "grad_accum": int(shared("grad_accum", "grad_accum", 8)),
        "max_length": int(shared("max_length", "max_length", 8192)),
        "max_samples": int(shared("max_samples", "max_samples", 0)),
        "bf16": bool(shared("bf16", "bf16", False)),
        "num_workers": int(shared("num_workers", "num_workers", 4)),
        "log_every": int(shared("log_every", "log_every", 10)),
        "save_every": int(shared("save_every", "save_every", 0)),
        "device": shared("device", "device", "cuda"),
        "warmup_steps": int(shared("warmup_steps", "warmup_steps", 100)),
        "weight_decay": float(shared("weight_decay", "weight_decay", 0.0)),
    }

    defaults = dict(cfg.get("defaults") or {})
    # CLI single-draft defaults fill gaps when no YAML defaults block.
    cli_defaults = {
        "assistant": args.assistant,
        "lr": args.lr,
        "ttt_steps": args.ttt_steps,
        "step_weight_beta": args.step_weight_beta,
        "soft_ce_weight": args.soft_ce_weight,
        "hard_ce_weight": args.hard_ce_weight,
        "feature_l1_weight": args.feature_l1_weight,
        "random_init": bool(args.random_init),
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
    }
    for k, v in cli_defaults.items():
        defaults.setdefault(k, v)

    raw_drafts = cfg.get("drafts")
    if raw_drafts is None:
        # Single-draft CLI mode.
        if not args.assistant and not defaults.get("assistant"):
            raise SystemExit(
                "provide --config with drafts:, or --assistant for single-draft mode"
            )
        raw_drafts = [{"name": "assistant"}]

    if not isinstance(raw_drafts, list) or not raw_drafts:
        raise SystemExit("config.drafts must be a non-empty list")

    drafts = []
    names_seen: set[str] = set()
    for i, entry in enumerate(raw_drafts):
        if not isinstance(entry, dict):
            raise SystemExit(f"drafts[{i}] must be a mapping")
        name = str(entry.get("name") or f"draft{i}")
        if name in names_seen:
            raise SystemExit(f"duplicate draft name: {name!r}")
        names_seen.add(name)
        merged = {**defaults, **{k: entry[k] for k in entry if k != "name"}}
        merged["name"] = name
        if not merged.get("assistant"):
            raise SystemExit(f"draft {name!r}: assistant path required")
        drafts.append(merged)

    out["drafts"] = drafts

    missing = [k for k in ("target", "data", "output_dir") if not out.get(k)]
    if missing:
        raise SystemExit(f"missing required config fields: {missing}")
    return out


def set_trainable(target, assistant, log=print):
    """Apply freeze policy: target frozen, assistant partial freeze."""
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()

    for p in assistant.parameters():
        p.requires_grad_(True)

    frozen_names = []
    lm_head = getattr(assistant, "lm_head", None)
    if lm_head is not None:
        for p in lm_head.parameters():
            p.requires_grad_(False)
        frozen_names.append("lm_head")

    asst_base = getattr(assistant, "model", None)
    embed = getattr(asst_base, "embed_tokens", None) if asst_base is not None else None
    if embed is not None:
        for p in embed.parameters():
            p.requires_grad_(False)
        frozen_names.append("model.embed_tokens")

    trainable = [p for p in assistant.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in assistant.parameters())
    log("[freeze] target: fully frozen")
    log(f"[freeze] assistant frozen submodules: {frozen_names}")
    log(
        f"[freeze] trainable params: {n_train:,} / {n_total:,} "
        f"({100.0 * n_train / max(n_total, 1):.1f}%)"
    )
    if n_train == 0:
        raise RuntimeError("no trainable params after freeze")
    return trainable


def build_dataset(data_path, tokenizer, max_length, max_samples=0):
    """Build dataset from regenerated JSONL conversations."""
    import json
    from speculators.models.gemma4_mtp.data import Gemma4ConversationParser

    parser = Gemma4ConversationParser(tokenizer, max_length=max_length)

    conversations = []
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if max_samples > 0 and i >= max_samples:
                break
            data = json.loads(line)
            conversations.append(data)

    print(f"[data] loaded {len(conversations)} conversations from {data_path}")

    def convert_conversation(conv_list):
        result = []
        for msg in conv_list:
            fr = msg.get("from", "")
            val = msg.get("value", "")
            if fr == "human":
                result.append({"role": "user", "content": val})
            elif fr == "gpt":
                result.append({"role": "assistant", "content": val})
            elif fr == "system":
                result.append({"role": "system", "content": val})
        return result

    converted_convs = []
    for conv in conversations:
        convs = conv.get("conversations", conv.get("messages", []))
        converted = convert_conversation(convs)
        if not converted:
            continue
        converted_convs.append(converted)

    print(
        f"[data] {len(converted_convs)} conversations ready "
        "(lazy tokenization in dataloader workers)"
    )

    class LazyTokenizedDataset(torch.utils.data.Dataset):
        def __init__(self, convs, parser):
            self.convs = convs
            self.parser = parser

        def __len__(self):
            return len(self.convs)

        def __getitem__(self, idx):
            n = len(self.convs)
            for off in range(n):
                parsed = self.parser.parse(self.convs[(idx + off) % n])
                if parsed is not None:
                    return parsed
            raise RuntimeError("no parseable conversation in dataset")

    return LazyTokenizedDataset(converted_convs, parser)


def collate_fn(batch, pad_token_id):
    """Collate variable-length sequences into padded tensors."""
    from speculators.models.gemma4_mtp.data import collate as base_collate

    return base_collate(batch, pad_token_id=pad_token_id)


def build_target_cache_batch(target, batch, target_device, asst_device):
    """Frozen target forward → tensors for ``training_step_from_cache``.

    Built per micro-batch for **one** draft only (never shared across drafts).
    """
    from speculators.models.gemma4_mtp.training_step import locate_target_parts

    input_ids = batch["input_ids"].to(target_device, non_blocking=True)
    attn = batch.get("attention_mask")
    attn_t = attn.to(target_device, non_blocking=True) if attn is not None else None

    target_base, _, _, _ = locate_target_parts(target)
    with torch.no_grad():
        base_out = target_base(
            input_ids=input_ids,
            attention_mask=attn_t,
            return_shared_kv_states=True,
            use_cache=False,
        )
        last_hidden = base_out.last_hidden_state
        shared_kv_states = base_out.shared_kv_states
    if shared_kv_states is None:
        raise RuntimeError("target returned shared_kv_states=None")

    def to_a(t):
        return t.to(asst_device, non_blocking=True)

    return {
        "input_ids": batch["input_ids"].to(asst_device, non_blocking=True),
        "loss_mask": batch["loss_mask"].to(asst_device, non_blocking=True),
        "last_hidden": to_a(last_hidden),
        "shared_kv_states": {
            k: (to_a(kv[0]), to_a(kv[1])) for k, kv in shared_kv_states.items()
        },
    }


def patch_causal_shared_kv_masks(assistant, log=print):
    """Replace the assistant's bidirectional shared-KV mask with block-causal."""
    import types

    holder = None
    for m in assistant.modules():
        if hasattr(type(m), "create_attention_masks"):
            holder = m
            break
    if holder is None:
        raise RuntimeError("no create_attention_masks found to patch")
    cfg = holder.config.get_text_config()
    window = getattr(cfg, "sliding_window", None)

    def causal_create_attention_masks(self, inputs_embeds, attention_mask, shared_kv_states):
        q_len = inputs_embeds.shape[1]
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        neg = torch.finfo(dtype).min

        def build(kv_len, win=None):
            qpos = torch.arange(q_len, device=device)
            kv = torch.arange(kv_len, device=device)
            allow = kv[None, :] <= qpos[:, None]
            if win is not None:
                allow = allow & (kv[None, :] > qpos[:, None] - win)
            m = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
            return m.masked_fill(~allow, neg)[None, None]

        kv_full = shared_kv_states["full_attention"][0][:, 0].shape[1]
        kv_swa = shared_kv_states["sliding_attention"][0][:, 0].shape[1]
        return {
            "full_attention": build(kv_full),
            "sliding_attention": build(kv_swa, window),
        }

    holder.create_attention_masks = types.MethodType(
        causal_create_attention_masks, holder
    )
    log(
        f"[mask-fix] patched {type(holder).__name__}.create_attention_masks -> "
        f"block-causal (sliding_window={window})"
    )
    return holder


def patch_hidden_shift(log=print):
    """No-op: hidden shift is folded into training_step / training_step_from_cache."""
    log("[hidden-shift] folded into training_step.py (no-op here)")


def load_assistant(path: str, dtype, device, random_init: bool, log=print):
    if random_init:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        assistant = AutoModelForCausalLM.from_config(cfg, dtype=dtype)

        def init_weights(mod):
            if hasattr(mod, "_init_weights"):
                try:
                    mod._init_weights(mod)
                except TypeError:
                    pass

        assistant.apply(init_weights)
        log(f"[random-init] {path} initialized from scratch")
    else:
        assistant = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, trust_remote_code=True
        )
    return assistant.to(device)


def _make_target_heads(target, asst_device):
    """Frozen embed / lm_head on the draft device (for one draft's training)."""
    from speculators.models.gemma4_mtp.training_step import locate_target_parts

    _, tgt_lm_head, tgt_embed, _ = locate_target_parts(target)
    target_embed_a = copy.deepcopy(tgt_embed).to(asst_device).eval()
    for p in target_embed_a.parameters():
        p.requires_grad_(False)
    if tgt_lm_head is not None:
        target_lm_head_a = copy.deepcopy(tgt_lm_head).to(asst_device).eval()
        for p in target_lm_head_a.parameters():
            p.requires_grad_(False)
    else:
        import torch.nn.functional as _F

        _w = target_embed_a.weight
        target_lm_head_a = lambda h: _F.linear(h, _w)  # noqa: E731
    return target_embed_a, target_lm_head_a


def _train_one_draft(
    *,
    dcfg: dict[str, Any],
    cfg: dict[str, Any],
    target,
    target_embed_a,
    target_lm_head_a,
    loader,
    sampler,
    tokenizer,
    dtype,
    target_device: str,
    asst_device: str,
    ddp: bool,
    local_rank: int,
    is_main: bool,
    log,
):
    """Full isolated train loop for a single draft (own I/O, never mixed)."""
    from transformers import get_cosine_schedule_with_warmup
    from speculators.models.gemma4_mtp.training_step import (
        MTPLossConfig,
        training_step_from_cache,
    )

    name = dcfg["name"]
    out_dir = os.path.join(cfg["output_dir"], name)
    if is_main:
        os.makedirs(out_dir, exist_ok=True)

    log(
        f"=== [{name}] start === assistant={dcfg['assistant']} "
        f"random_init={dcfg.get('random_init')} lr={dcfg['lr']} "
        f"soft={dcfg['soft_ce_weight']} hard={dcfg['hard_ce_weight']} "
        f"feat={dcfg['feature_l1_weight']} ttt={dcfg['ttt_steps']} -> {out_dir}"
    )

    assistant = load_assistant(
        dcfg["assistant"],
        dtype,
        asst_device,
        bool(dcfg.get("random_init", False)),
        log=log,
    )
    patch_causal_shared_kv_masks(assistant, log=log)
    patch_hidden_shift(log=log)
    trainable = set_trainable(target, assistant, log=log)
    module = assistant
    model = assistant
    if ddp:
        model = DDP(
            assistant,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    loss_cfg = MTPLossConfig(
        ttt_steps=int(dcfg["ttt_steps"]),
        step_weight_beta=float(dcfg["step_weight_beta"]),
        soft_ce_weight=float(dcfg["soft_ce_weight"]),
        hard_ce_weight=float(dcfg["hard_ce_weight"]),
        feature_l1_weight=float(dcfg["feature_l1_weight"]),
    )
    total_steps = (len(loader) // max(cfg["grad_accum"], 1)) * cfg["epochs"]
    optim = torch.optim.AdamW(
        trainable,
        lr=float(dcfg["lr"]),
        weight_decay=float(dcfg.get("weight_decay", cfg["weight_decay"])),
    )
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=int(dcfg.get("warmup_steps", cfg["warmup_steps"])),
        num_training_steps=max(total_steps, 1),
    )
    module.train()
    optim.zero_grad()

    step = 0
    run: dict[str, float] = {}

    for epoch in range(cfg["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for i, batch in enumerate(loader):
            # Target signals are built for THIS draft only — not reused elsewhere.
            cache_batch = build_target_cache_batch(
                target, batch, target_device, asst_device
            )

            use_no_sync = (
                ddp
                and isinstance(model, DDP)
                and (i + 1) % cfg["grad_accum"] != 0
            )
            ctx = model.no_sync() if use_no_sync else torch.enable_grad()
            with ctx:
                loss, metrics = training_step_from_cache(
                    module if not ddp else model,
                    target_embed_a,
                    target_lm_head_a,
                    cache_batch,
                    loss_cfg,
                )
                (loss / cfg["grad_accum"]).backward()

            for _k, _v in metrics.items():
                run[_k] = run.get(_k, 0.0) + float(_v)
            run["_n"] = run.get("_n", 0) + 1

            if (i + 1) % cfg["grad_accum"] == 0:
                step += 1
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()

                if step % cfg["log_every"] == 0:
                    n = run.pop("_n", 1)
                    avg = {k: run[k] / n for k in run}
                    run = {}
                    lr = sched.get_last_lr()[0]
                    msg = " ".join(f"{k}={v:.4f}" for k, v in avg.items())
                    log(
                        f"[{name}] epoch {epoch} step {step}/{total_steps} "
                        f"lr={lr:.2e} {msg} (mean/{n})"
                    )

                if is_main and cfg["save_every"] and step % cfg["save_every"] == 0:
                    _save(module, tokenizer, os.path.join(out_dir, f"step{step}"))

    if is_main:
        _save(module, tokenizer, out_dir)

    # Free this draft before the next one (different I/O; do not keep co-resident).
    del model, module, assistant, optim, sched, trainable
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"=== [{name}] done ===")


def main():
    args = parse_args()
    cfg = resolve_train_config(args)

    # --- Distributed setup ---
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = cfg["device"]

    is_main = rank == 0

    def log(*a, **k):
        if is_main:
            print(*a, **k, flush=True)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    if is_main and args.config:
        try:
            import shutil

            shutil.copy2(args.config, os.path.join(cfg["output_dir"], "train_config.yaml"))
        except OSError:
            pass

    dtype = torch.bfloat16 if cfg["bf16"] else torch.float32

    # Device placement: backbone on one GPU, current draft on another when possible.
    split = (not ddp) and torch.cuda.device_count() >= 2
    if split:
        target_device = "cuda:0"
        asst_device = "cuda:1"
    else:
        target_device = asst_device = device
    n_drafts = len(cfg["drafts"])
    log(
        f"=== device placement: backbone={target_device} draft={asst_device} "
        f"(split={split}, n_drafts={n_drafts}, sequential/isolated) ==="
    )

    log("=== Loading target model ===")
    target = AutoModelForCausalLM.from_pretrained(
        cfg["target"],
        dtype=dtype,
        trust_remote_code=True,
    ).to(target_device)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    # Embed/lm_head copies for the MTP draft path (rebuilt only if needed later).
    target_embed_a, target_lm_head_a = _make_target_heads(target, asst_device)

    log("=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(cfg["target"], trust_remote_code=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    log("=== Building dataset ===")
    dataset = build_dataset(
        cfg["data"], tokenizer, cfg["max_length"], cfg["max_samples"]
    )
    if len(dataset) == 0:
        raise RuntimeError("empty dataset")

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if ddp
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=cfg["num_workers"],
        persistent_workers=(cfg["num_workers"] > 0),
        prefetch_factor=(4 if cfg["num_workers"] > 0 else None),
        pin_memory=True,
    )

    log(
        f"=== Training {n_drafts} draft(s) sequentially "
        f"({len(dataset)} samples, {len(loader)} batches/epoch) ==="
    )
    for di, dcfg in enumerate(cfg["drafts"]):
        log(f"--- draft {di + 1}/{n_drafts}: {dcfg['name']} ---")
        _train_one_draft(
            dcfg=dcfg,
            cfg=cfg,
            target=target,
            target_embed_a=target_embed_a,
            target_lm_head_a=target_lm_head_a,
            loader=loader,
            sampler=sampler,
            tokenizer=tokenizer,
            dtype=dtype,
            target_device=target_device,
            asst_device=asst_device,
            ddp=ddp,
            local_rank=local_rank,
            is_main=is_main,
            log=log,
        )

    log("=== Done (all drafts) ===")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


def _save(assistant, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    assistant.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"[save] wrote checkpoint to {path}", flush=True)


if __name__ == "__main__":
    main()
