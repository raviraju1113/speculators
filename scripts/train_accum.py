#!/usr/bin/env python3
"""Run scripts/train.py with GRADIENT ACCUMULATION, without modifying it.

WHY
    DeepSpec's gemma-4 DSpark recipe uses global_batch_size=512 (accumulated from
    local_batch_size=1), and their lr 6.0e-4 is tuned for that. This tree has no
    gradient accumulation -- upstream PR #859 is still open and does not merge
    cleanly -- so our effective batch is one packed sequence per rank per step,
    which is where the acceptance plateau most likely comes from.

    This wrapper adds accumulation by monkey-patching the Trainer at runtime, so
    scripts/train.py and src/speculators/train/trainer.py stay untouched and keep
    merging cleanly with upstream.

HOW MANY STEPS?
    Do NOT copy DeepSpec's 512 literally -- our batching unit differs. Our
    dataloader multipacks, and the measured epoch is 45,531 steps over a 314,224
    -sample train split (0.9 of 349,138), i.e. ~6.9 conversations per step. To
    match ~512 conversations per optimizer step:

        512 / 6.9  ~=  74          <- per rank
        with N ranks, divide by N: 74 / 3 ~= 25 for 3 ranks

    --accumulation-steps is PER RANK, because each rank contributes its own
    micro-batches to the same optimizer step.

WHAT IT PATCHES
    1. model forward   -> scales loss by 1/accum so accumulated grads average
                          rather than sum
    2. zero_grad       -> only at the START of an accumulation window
    3. optimizer.step  -> only at the END of a window
    4. scheduler.step  -> only at the END (so the LR schedule counts optimizer
                          steps, not micro-batches)
    5. clip_grad_norm_ -> only at the END; clipping partially-accumulated
                          gradients every micro-step would shrink early
                          contributions more than late ones

CAVEATS
    * global_step still counts MICRO-batches, so an "epoch" is the same number of
      logged steps as before; only the number of optimizer updates changes.
    * Wall-clock per epoch is UNCHANGED -- accumulation buys gradient quality, not
      throughput. It does not fix the ~1,250 sequences/hour ceiling.
    * With DDP, gradients are all-reduced on every backward, so accumulation does
      not save communication here. no_sync() would, but it is not wired through
      this patch.

USAGE
    Identical to scripts/train.py, plus --accumulation-steps:

      python scripts/train_accum.py --accumulation-steps 74 \\
        --verifier-name-or-path <model> --data-path <data> ... --lr 6e-4

      torchrun --nnodes=1 --nproc_per_node=2 \\
        --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29500 --local-addr=127.0.0.1 \\
        scripts/train_accum.py --accumulation-steps 37 ... --lr 6e-4
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _pop_accum_arg(argv: list[str]) -> int:
    """Remove --accumulation-steps from argv and return it (default 1)."""
    accum = 1
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--accumulation-steps":
            accum = int(argv[i + 1])
            i += 2
            continue
        if a.startswith("--accumulation-steps="):
            accum = int(a.split("=", 1)[1])
            i += 1
            continue
        out.append(a)
        i += 1
    argv[:] = out
    return accum


def patch_trainer(accum: int) -> None:
    from speculators.train.trainer import Trainer

    if accum <= 1:
        print("[train_accum] accumulation-steps <= 1: no patch applied", flush=True)
        return

    # --- 1. scale the loss so accumulated gradients AVERAGE ------------------
    orig_setup_model = Trainer.setup_model

    def setup_model(self, *a, **kw):
        out = orig_setup_model(self, *a, **kw)
        model = self.model
        orig_forward = model.forward

        def scaled_forward(*fa, **fkw):
            draft_tokens, loss, metrics = orig_forward(*fa, **fkw)
            # metrics are reported unscaled so logged loss stays comparable to
            # non-accumulated runs; only the backward path is scaled.
            return draft_tokens, loss / accum, metrics

        model.forward = scaled_forward
        self._accum_i = 0
        return out

    Trainer.setup_model = setup_model

    # --- 2..4. gate the optimizer/scheduler on window boundaries -------------
    orig_zero = Trainer._optimizers_zero_grad
    orig_step = Trainer._optimizers_step
    orig_sched = Trainer._schedulers_step

    def _at_window_start(self) -> bool:
        return getattr(self, "_accum_i", 0) % accum == 0

    def _at_window_end(self) -> bool:
        return (getattr(self, "_accum_i", 0) + 1) % accum == 0

    def zero_grad(self):
        # The loop calls zero_grad immediately before backward, so clearing only
        # at a window start is what makes gradients accumulate across the window.
        if _at_window_start(self):
            orig_zero(self)

    def step(self):
        if _at_window_end(self):
            _ACCUM_STATE["clip_now"] = True
            try:
                # clip once, on the fully accumulated gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            finally:
                _ACCUM_STATE["clip_now"] = False
            orig_step(self)

    def sched(self):
        # Advance the LR schedule per OPTIMIZER step, not per micro-batch,
        # otherwise warmup/decay finish accum-times too early.
        if _at_window_end(self):
            orig_sched(self)
        self._accum_i = getattr(self, "_accum_i", 0) + 1

    Trainer._optimizers_zero_grad = zero_grad
    Trainer._optimizers_step = step
    Trainer._schedulers_step = sched

    # --- 5. suppress the trainer's per-micro-step clipping -------------------
    # trainer.py calls torch.nn.utils.clip_grad_norm_ directly every iteration.
    # Clipping a partially accumulated gradient would down-weight the earlier
    # micro-batches, so make it a no-op except when we invoke it above.
    orig_clip = torch.nn.utils.clip_grad_norm_

    def gated_clip(parameters, max_norm, *a, **kw):
        if _ACCUM_STATE["clip_now"]:
            return orig_clip(parameters, max_norm, *a, **kw)
        return torch.zeros((), device="cpu")

    torch.nn.utils.clip_grad_norm_ = gated_clip

    print(
        f"[train_accum] gradient accumulation ON: {accum} micro-batches per "
        f"optimizer step (loss scaled by 1/{accum}; scheduler and grad-clip "
        f"advance per optimizer step)",
        flush=True,
    )


_ACCUM_STATE = {"clip_now": False}


def main() -> None:
    accum = _pop_accum_arg(sys.argv)
    patch_trainer(accum)
    # Delegate to the untouched scripts/train.py, which parses the remaining argv.
    runpy.run_path(str(REPO / "scripts" / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
