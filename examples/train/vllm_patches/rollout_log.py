# SPDX-License-Identifier: Apache-2.0
"""Draft-OPD rollout logger (LOCAL PATCH, 2026-10-21, mengmengj).

Records, for every request served with speculative decoding, each draft step as
seen by the scheduler: the block the draft proposed (the tokens scheduled for
verification), how many of them the target accepted, and the anchor (number of
output tokens already verified when the block was proposed). On finish it writes
one JSON line per request with prompt ids, the full verified output ids, the
steps, and the finish reason -- the rollout record consumed by
``scripts/opd/assemble_rollouts.py`` (see ``.claude/skills/draft-opd/reference.md``).

Enabled by ``DSPARK_ROLLOUT_LOG=<dir>``; one file per engine-core process. The
hook sits in ``Scheduler.update_from_output`` (host side, single thread): it
reads what the scheduler already computes for its spec-decode metrics and never
touches the model runner or the CUDA-graph path.

Invariant for a step with ``accepted = r``: the verifier appends
``r + 1`` tokens (``r`` accepted draft tokens + 1 target token), so
``output_ids[anchor : anchor + r] == draft_ids[:r]`` and the next anchor is
``anchor + r + 1`` (unless the request stopped). An empty ``draft_ids`` marks a
step with no draft (e.g. right after prefill).
"""

import json
import os
import time
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_VAR = "DSPARK_ROLLOUT_LOG"


class RolloutLogger:
    def __init__(self, out_dir: str | None = None, async_scheduling: bool = False) -> None:
        self.out_dir = out_dir if out_dir is not None else os.environ.get(ENV_VAR)
        self.enabled = bool(self.out_dir)
        if self.enabled and async_scheduling:
            # Under async scheduling the scheduler only sees -1 placeholders and
            # the handler's host copy races the pipelined next step (measured:
            # the captured block belonged to step t+1 in >99% of cases). In the
            # sync path vLLM feeds the real ids back via update_draft_token_ids,
            # so the scheduler hook is exact. Refuse rather than log garbage.
            raise RuntimeError(
                f"{ENV_VAR} requires async scheduling to be disabled "
                "(vllm serve ... --no-async-scheduling)"
            )
        self._fh = None
        self._steps: dict[str, list[dict[str, Any]]] = {}
        self._started: dict[str, float] = {}
        self._pending_draft: dict[str, list[int]] = {}
        if self.enabled:
            os.makedirs(self.out_dir, exist_ok=True)
            path = os.path.join(self.out_dir, f"rollout_{os.getpid()}.jsonl")
            self._fh = open(path, "a", buffering=1)  # line-buffered
            logger.info("Draft-OPD rollout logging to %s", path)

    def on_draft(self, draft_token_ids: Any) -> None:
        """Real draft ids for the blocks proposed in the step just processed
        (``DraftTokenIds``: parallel ``req_ids`` / ``draft_token_ids``). Under
        async scheduling the scheduler only ever sees ``-1`` placeholders, so the
        engine core calls this right after ``update_from_output`` and the next
        ``on_step`` substitutes the stored block."""
        if not self.enabled or draft_token_ids is None:
            return
        for rid, ids in zip(draft_token_ids.req_ids, draft_token_ids.draft_token_ids):
            self._pending_draft[rid] = list(ids)

    def on_step(
        self,
        request: Any,
        draft_ids: list[int] | None,
        generated_ids: list[int],
        num_sampled: int = 1,
    ) -> None:
        """Called once per scheduled request per step, BEFORE the new output
        tokens are appended to the request (so ``num_output_tokens`` is the
        anchor). ``accepted`` mirrors the scheduler's own metric:
        ``len(generated) - num_sampled`` (the extra token is the target's)."""
        if not self.enabled:
            return
        rid = request.request_id
        steps = self._steps.get(rid)
        if steps is None:
            steps = self._steps[rid] = []
            self._started[rid] = time.time()
        has_draft = bool(draft_ids)
        ids = list(draft_ids) if has_draft else []
        pending = self._pending_draft.pop(rid, None)
        if has_draft and any(t < 0 for t in ids):
            # async-scheduling placeholders: use the block captured by on_draft
            if pending is not None and len(pending) >= len(ids):
                ids = pending[: len(ids)]
            else:
                ids = [-1] * len(ids)  # unknown; assembler flags it
        steps.append(
            {
                "anchor": request.num_output_tokens,
                "draft_ids": ids,
                "accepted": max(len(generated_ids) - num_sampled, 0) if has_draft else 0,
                "gen": list(generated_ids),
            }
        )

    def on_finish(self, request: Any, finish_reason: Any) -> None:
        if not self.enabled:
            return
        rid = request.request_id
        steps = self._steps.pop(rid, [])
        started = self._started.pop(rid, None)
        sp = request.sampling_params
        record = {
            "request_id": rid,
            "prompt_ids": list(request.prompt_token_ids or []),
            "output_ids": list(request.output_token_ids),
            "steps": steps,
            "k": max((len(s["draft_ids"]) for s in steps), default=0),
            "finish": str(finish_reason) if finish_reason is not None else None,
            "sampling": {
                "temperature": getattr(sp, "temperature", None),
                "top_p": getattr(sp, "top_p", None),
                "top_k": getattr(sp, "top_k", None),
                "max_tokens": getattr(sp, "max_tokens", None),
            },
            "wall_s": (time.time() - started) if started else None,
        }
        assert self._fh is not None
        self._fh.write(json.dumps(record) + "\n")

    def on_abort(self, request_id: str) -> None:
        self._steps.pop(request_id, None)
        self._started.pop(request_id, None)
        self._pending_draft.pop(request_id, None)
