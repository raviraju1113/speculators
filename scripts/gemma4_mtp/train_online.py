#!/usr/bin/env python3
"""Online fine-tuning for Gemma 4 assistant (MTP draft).

This script fine-tunes the Gemma4AssistantForCausalLM with the target model
running on the same GPU(s) as training (not on a separate vLLM server).
This is the "online" variant where target forward passes happen live during
training (not from precomputed cache).

Key differences from train.py (offline cache):
  - Target forward passes happen live each step (no cache generation phase)
  - Both target and assistant fit on 4x80GB for Gemma4-26B-MoE
  - Uses training_step() from training_step.py directly

Usage:
    bash examples/train/gemma4_26b_mtp_online.sh              # both stages
    STAGE=train bash examples/train/gemma4_26b_mtp_online.sh  # only train

Prerequisites:
    - Regenerated training data (JSONL with conversations)
    - conda env: speculator
"""

from __future__ import annotations

import argparse
import copy
import os

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
    ap.add_argument("--target", required=True, help="target model path")
    ap.add_argument("--assistant", required=True, help="assistant/draft model path")
    ap.add_argument("--data", required=True, help="regenerated conversations JSONL")
    ap.add_argument("--output", required=True, help="output dir for checkpoints")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader workers for on-the-fly tokenization (0 = main process)")
    ap.add_argument("--ttt-steps", type=int, default=5)
    ap.add_argument("--step-weight-beta", type=float, default=0.8)
    ap.add_argument("--soft-ce-weight", type=float, default=0.5)
    ap.add_argument("--hard-ce-weight", type=float, default=0.0)
    ap.add_argument("--feature-l1-weight", type=float, default=0.0,
                    help="EAGLE/DSpark feature (hidden) smooth-L1 distillation weight")
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


def set_trainable(target, assistant):
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
    print(f"[freeze] target: fully frozen", flush=True)
    print(f"[freeze] assistant frozen submodules: {frozen_names}", flush=True)
    print(
        f"[freeze] trainable params: {n_train:,} / {n_total:,} "
        f"({100.0 * n_train / max(n_total, 1):.1f}%)",
        flush=True,
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

    # Convert from {"from": "human"/"gpt", "value": "..."} to
    # {"role": "user"/"assistant", "content": "..."} format
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
            # skip unknown roles
        return result

    # Cheap pre-pass: convert role format and drop empties only (NO tokenization).
    # Tokenization is deferred to __getitem__ so it runs in DataLoader workers,
    # in parallel and overlapped with GPU compute (see LazyTokenizedDataset).
    converted_convs = []
    for conv in conversations:
        convs = conv.get("conversations", conv.get("messages", []))
        converted = convert_conversation(convs)
        if not converted:
            continue
        converted_convs.append(converted)

    print(f"[data] {len(converted_convs)} conversations ready (lazy tokenization in dataloader workers)")

    class LazyTokenizedDataset(torch.utils.data.Dataset):
        """Tokenizes on the fly in __getitem__ (runs inside DataLoader workers).

        Avoids the ~50 min single-threaded upfront tokenization (4x redundant
        under DDP). A conversation whose parse returns None (e.g. no valid
        assistant label) is skipped by deterministically advancing to the next
        index, keeping __len__ stable for DistributedSampler.
        """

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


def training_step_split(
    target,
    assistant,
    target_embed_a,
    target_lm_head_a,
    batch,
    cfg,
    target_device,
    asst_device,
):
    """Device-split TTT step: backbone on one GPU, draft on another.

    The frozen target forward runs on ``target_device`` (produces last_hidden +
    shared_kv_states, no grad); those signals are copied to ``asst_device`` and
    the assistant is trained there via ``training_step_from_cache`` (backward and
    optimizer state stay on the draft GPU). ``target_embed_a`` / ``target_lm_head_a``
    are frozen copies of the target's embed / lm_head placed on ``asst_device``.
    """
    from speculators.models.gemma4_mtp.training_step import (
        locate_target_parts,
        training_step_from_cache,
    )

    input_ids = batch["input_ids"].to(target_device, non_blocking=True)
    attn = batch.get("attention_mask")
    attn_t = attn.to(target_device, non_blocking=True) if attn is not None else None

    # Backbone forward on target_device -> last_hidden + shared_kv_states only.
    # (Soft labels are recomputed on the draft GPU inside training_step_from_cache,
    # so we skip the full [B,T,V] target_logits matmul here.)
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

    cache_batch = {
        "input_ids": batch["input_ids"].to(asst_device, non_blocking=True),
        "loss_mask": batch["loss_mask"].to(asst_device, non_blocking=True),
        "last_hidden": to_a(last_hidden),
        "shared_kv_states": {
            k: (to_a(kv[0]), to_a(kv[1]))
            for k, kv in shared_kv_states.items()
        },
    }
    return training_step_from_cache(
        assistant, target_embed_a, target_lm_head_a, cache_batch, cfg
    )


def patch_causal_shared_kv_masks(assistant, log=print):
    """Replace the assistant's *bidirectional* shared-KV mask with a block-CAUSAL
    one, fixing the training-time future-KV label leak.

    The stock ``create_attention_masks`` builds an all-ones (bidirectional) mask
    over the target's full teacher-forced KV. With q_len>1 (TTT training) that
    lets query row t attend the target KV of the token it is predicting -> leak.
    (At inference it is harmless: q_len==1 and the KV cache holds only the past.)

    Fix: query row t is absolute position k+t and may attend target KV j only for
    j <= k+t (full-attn layers) or k+t-window < j <= k+t (sliding layers). The
    step offset k need not be passed: in the loop kv_len == full seq length T and
    q_len == T-k-1, so k = kv_len - q_len - 1.
    """
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
            # Query row t is a draft rollout that started at target position t
            # (its recurrent hidden traces back to target_hidden[t]); at inference
            # such a rollout attends only the VERIFIED prefix KV[0..t] with a
            # CONSTANT position/KV-range across draft steps (constant_draft_positions).
            # So row t attends KV[0..t] -- offset 0, i.e. query-pos t attends
            # KV-pos <= t, the SAME for every TTT step k. (Using offset=k would let
            # row t attend KV[t+1..k+t] = the target's KV for the tokens being
            # drafted -> a leak not available at inference.)
            qpos = torch.arange(q_len, device=device)  # row t -> position t
            kv = torch.arange(kv_len, device=device)  # kv index == target position
            allow = kv[None, :] <= qpos[:, None]  # attend verified prefix only
            if win is not None:
                allow = allow & (kv[None, :] > qpos[:, None] - win)  # + SWA window
            m = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
            return m.masked_fill(~allow, neg)[None, None]  # (1,1,q_len,kv_len)

        kv_full = shared_kv_states["full_attention"][0][:, 0].shape[1]
        kv_swa = shared_kv_states["sliding_attention"][0][:, 0].shape[1]
        return {
            "full_attention": build(kv_full),
            "sliding_attention": build(kv_swa, window),
        }

    holder.create_attention_masks = types.MethodType(
        causal_create_attention_masks, holder
    )
    # NOTE: do NOT inject advancing RoPE positions here. vLLM inference uses
    # `constant_draft_positions=True` (all draft steps in a rollout share the
    # last target-model position), and the stock training default
    # (position_ids=None -> arange(L), i.e. row t -> position t, constant across
    # TTT step k) already matches that. Advancing positions (k+t) would create a
    # train/inference RoPE mismatch. Mask fix only.
    log(f"[mask-fix] patched {type(holder).__name__}.create_attention_masks -> "
        f"block-causal (sliding_window={window})")
    return holder


def patch_hidden_shift(log=print):
    """Fix the train/inference HIDDEN-STATE alignment (root cause of the vLLM
    accept-length collapse for from-scratch drafts).

    vLLM / the EAGLE-MTP recipe feed the draft the target hidden of the PREVIOUS
    position plus the current token embedding:  (h_{t-1}, embed(x_t)) -> x_{t+1}.
    But build_target_signals returns h_t, so training aligned the draft to h_t
    (same position). A from-scratch draft then collapses when vLLM feeds h_{t-1}
    (accept ~1.0), even though HF looks perfect. Empirically: trained draft
    step-0 argmax-match on held-out AIME is 0.94 with h_t but 0.06 with h_{t-1}
    (= vLLM); vanilla prefers h_{t-1} (0.92).

    Fix: shift the draft's INPUT hidden right by one (row t -> h_{t-1}) WITHOUT
    touching target_logits (labels stay computed from the unshifted hidden).
    training_step uses last_hidden only as the step-0 draft input + recurrent
    pad, so patching build_target_signals' return is safe and needs no edit to
    training_step.py. The TTT recurrence is self-consistent after this fix.
    """
    # The h_{t-1} shift is now folded DIRECTLY into training_step.py (the draft's
    # step-0 input uses the shifted target hidden; feature-distillation labels use
    # the UNSHIFTED hidden). Patching build_target_signals here too would
    # DOUBLE-shift, so this is a no-op now (kept so launchers don't break).
    log("[hidden-shift] folded into training_step.py (no-op here)")


def main():
    import json
    from transformers import get_cosine_schedule_with_warmup
    from speculators.models.gemma4_mtp.training_step import (
        MTPLossConfig,
        training_step,
    )

    args = parse_args()

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
        device = args.device

    is_main = rank == 0

    def log(*a, **k):
        if is_main:
            print(*a, **k, flush=True)

    os.makedirs(args.output, exist_ok=True)
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    # Device placement: backbone (target) and draft (assistant) on separate
    # GPUs when >=2 are visible and we are single-process. The frozen target
    # forward runs on target_device; the assistant is trained on asst_device.
    split = (not ddp) and torch.cuda.device_count() >= 2
    if split:
        target_device = "cuda:0"
        asst_device = "cuda:1"
    else:
        target_device = asst_device = device
    log(f"=== device placement: backbone={target_device} draft={asst_device} "
        f"(split={split}) ===")

    log("=== Loading target model ===")
    target = AutoModelForCausalLM.from_pretrained(
        args.target,
        dtype=dtype,
        trust_remote_code=True,
    ).to(target_device)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    log("=== Loading assistant model ===")
    if args.random_init:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.assistant, trust_remote_code=True)
        assistant = AutoModelForCausalLM.from_config(cfg, dtype=dtype)

        def init_weights(mod):
            if hasattr(mod, "_init_weights"):
                try:
                    mod._init_weights(mod)
                except TypeError:
                    pass

        assistant.apply(init_weights)
        log("[random-init] assistant initialized from scratch (proper _init_weights)")
    else:
        assistant = AutoModelForCausalLM.from_pretrained(
            args.assistant,
            dtype=dtype,
            trust_remote_code=True,
        )
    assistant = assistant.to(asst_device)

    # Fix the training-time future-KV label leak: make the assistant's shared-KV
    # attention block-causal (query t attends target KV j <= k+t only).
    patch_causal_shared_kv_masks(assistant, log=log)

    # Fix the train/inference hidden-state alignment (draft must consume h_{t-1},
    # not h_t) — the root cause of the from-scratch vLLM accept-length collapse.
    patch_hidden_shift(log=log)

    trainable = set_trainable(target, assistant)
    assistant_module = assistant  # raw module: mask holder, .train(), save_pretrained

    # Data-parallel across GPUs: wrap the (trainable) assistant in DDP so its
    # gradients all-reduce across ranks. Each rank holds a full frozen target +
    # trained draft co-resident on cuda:local_rank (both fit in 80GB), so there
    # is no device split and no cross-GPU ping-pong. The frozen target is not
    # DDP-wrapped (no grads). broadcast_buffers=False: the assistant has no
    # buffers requiring sync; find_unused_parameters=False: every trainable param
    # is used in each TTT forward.
    if ddp:
        assistant = DDP(
            assistant,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    # For the device-split path, place frozen copies of the target's embedding
    # and lm_head on the draft GPU so the assistant's input construction and
    # soft-label matmul run entirely on asst_device (no per-op cross-device).
    target_embed_a = target_lm_head_a = None
    if split:
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
            # Tied head: reuse the (frozen) embedding weight on asst_device.
            import torch.nn.functional as _F

            _w = target_embed_a.weight
            target_lm_head_a = lambda h: _F.linear(h, _w)  # noqa: E731

    log("=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    log("=== Building dataset ===")
    dataset = build_dataset(args.data, tokenizer, args.max_length, args.max_samples)

    if len(dataset) == 0:
        raise RuntimeError("empty dataset")

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if ddp
        else None
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(4 if args.num_workers > 0 else None),
        pin_memory=True,
    )

    log(f"=== Training: {len(dataset)} samples, {len(loader)} batches/epoch ===")
    log(
        f"[loss-config] soft_ce_weight={args.soft_ce_weight} "
        f"hard_ce_weight={args.hard_ce_weight} "
        f"feature_l1_weight={args.feature_l1_weight}"
    )

    loss_cfg = MTPLossConfig(
        ttt_steps=args.ttt_steps,
        step_weight_beta=args.step_weight_beta,
        soft_ce_weight=args.soft_ce_weight,
        hard_ce_weight=args.hard_ce_weight,
        feature_l1_weight=args.feature_l1_weight,
    )
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(loader) // max(args.grad_accum, 1)) * args.epochs
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

    assistant_module.train()
    step = 0
    run = {}  # accumulates micro-batch metrics for a windowed (readable) log
    optim.zero_grad()

    log("=== Starting training ===")

    def to_device(batch):
        out = {}
        for k, v in batch.items():
            if k == "shared_kv_states":
                out[k] = {
                    kt: (kv[0].to(device), kv[1].to(device))
                    for kt, kv in v.items()
                }
            else:
                out[k] = v.to(device)
        return out

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for i, batch in enumerate(loader):
            if split:
                # batch stays on CPU; training_step_split moves each piece to
                # target_device (backbone) / asst_device (draft).
                loss, metrics = training_step_split(
                    target, assistant_module, target_embed_a, target_lm_head_a,
                    batch, loss_cfg, target_device, asst_device,
                )
                (loss / args.grad_accum).backward()
            elif ddp and (i + 1) % args.grad_accum != 0:
                batch = to_device(batch)
                # no_sync: skip the all-reduce on grad-accumulation micro-steps.
                # Pass the DDP-wrapped `assistant` so the forward/backward sync.
                with assistant.no_sync():
                    loss, metrics = training_step(
                        target, assistant, batch, loss_cfg
                    )
                    (loss / args.grad_accum).backward()
            else:
                batch = to_device(batch)
                # `assistant` is the DDP wrapper (ddp) or the raw model (1-GPU);
                # either way calling it runs the (mask-patched) forward.
                loss, metrics = training_step(
                    target, assistant, batch, loss_cfg
                )
                (loss / args.grad_accum).backward()

            # Accumulate EVERY micro-batch's metrics so the log reports a mean
            # over the window, not one (batch-size-1) conversation's loss.
            for _k, _v in metrics.items():
                run[_k] = run.get(_k, 0.0) + float(_v)
            run["_n"] = run.get("_n", 0) + 1

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1

                if step % args.log_every == 0:
                    # mean over all micro-batches since the last log -> smooth,
                    # representative curve (vs. a single noisy conversation)
                    n = run.pop("_n", 1)
                    avg = {k: run[k] / n for k in run}
                    run = {}
                    lr = sched.get_last_lr()[0]
                    msg = " ".join(f"{k}={v:.4f}" for k, v in avg.items())
                    log(f"epoch {epoch} step {step}/{total_steps} "
                        f"lr={lr:.2e} {msg} (mean/{n})")

                if is_main and args.save_every and step % args.save_every == 0:
                    _save(assistant_module, tokenizer, os.path.join(args.output, f"step{step}"))

    if is_main:
        _save(assistant_module, tokenizer, args.output)

    log("=== Done ===")
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