"""Offline per-token mistake scorer for an EAGLE3 draft model.

Answers "which tokens does the draft get wrong" on a set of eval benchmarks
(gpqa, livecodebench, ...), so we can decide what training corpus would help.

How it works (reuses the training/eval primitives exactly):

  1. For each eval prompt, greedily generate the *target* model's answer via the
     live vLLM server. Because generation is greedy, the target's next token at
     position i is exactly ``input_ids[i+1]`` -- which is precisely the greedy
     speculative-decoding acceptance criterion. No verifier lm_head needed.
  2. Extract the draft's input hidden states for that full sequence through the
     same vLLM hidden-states connector used in training (ArrowDataset with
     ``on_missing="generate"``).
  3. Run the trained draft's ``forward`` and read back the ``draft_tokens`` it
     predicts at each TTT step (on-policy, exactly as in training/inference).
  4. Reproduce ``align_for_step``: at TTT step ``k`` and aligned position ``j``,
     the draft predicts ``draft_tokens[k][j]`` and the ground truth is
     ``input_ids[j + k + 1]``. Emit one JSONL record per scored token.

The output ``mistakes.jsonl`` is consumed by ``mistake_analysis.ipynb`` /
``mistake_lib.py``.

Run this ON THE GPU BOX with the vLLM verifier server already up (same server
you use to generate training data), e.g.:

  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/launch_vllm.py \
      /sms-scratch/checkpoints/gemma-4-31B-it/ --port 8000 \
      --tensor-parallel-size 4 --served-model-name google/gemma-4-31B-it \
      --hidden-states-path /sms-scratch/ravira/hidden_states

  python scripts/evaluate/mistake_analysis/score_mistakes.py \
      --draft-checkpoint /sms-scratch/ravira/checkpoints/gemma4_draft_model_300k_eagle3/checkpoint_best \
      --vllm-endpoint http://localhost:8000/v1 \
      --hidden-states-path /sms-scratch/ravira/hidden_states \
      --benchmarks gpqa_diamond,livecodebench,aime \
      --ttt-steps 3 --num-samples 50 \
      --out scripts/evaluate/mistake_analysis/out/run1_mistakes.jsonl
"""

import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

# The draft forward is @torch.compile'd (conditional_torch_compile). For a
# one-shot scoring job, compilation only adds startup latency and inductor
# autotuning memory spikes with no throughput payoff -- run eager instead.
# Must be set before importing torch. Override by exporting TORCHDYNAMO_DISABLE=0.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import openai
import torch
from datasets import Dataset

from speculators.model import SpeculatorModel
from speculators.models.eagle3.metrics import align_for_step
from speculators.train.data import ArrowDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("score_mistakes")

# Repo default: the eval prompts live here as {benchmark, id, prompt} JSONL.
DEFAULT_EVAL_DIR = Path(__file__).resolve().parents[1] / "mtp_server_eval" / "data"


def load_prompts(benchmark: str, eval_dir: Path, num_samples: int | None) -> list[dict]:
    """Load ``{benchmark, id, prompt}`` records for a benchmark."""
    path = eval_dir / f"{benchmark}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No eval file for benchmark {benchmark!r} at {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if num_samples is not None:
        rows = rows[:num_samples]
    logger.info("Loaded %d prompts for %s", len(rows), benchmark)
    return rows


def greedy_generate(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    max_new_tokens: int,
) -> tuple[list[int], list[bool]] | None:
    """Greedily generate an answer; return (full_input_ids, loss_mask).

    ``loss_mask`` is True over the generated (assistant) tokens only -- these are
    the positions where acceptance is actually measured. Requires the vLLM server
    to honor ``return_token_ids`` (the same flag the training data pipeline uses).
    """
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_new_tokens,
        extra_body={"return_token_ids": True},
    )
    choice = res.choices[0]
    # vLLM returns prompt + completion token ids under these extra fields.
    prompt_ids = getattr(res, "prompt_token_ids", None)
    completion_ids = getattr(choice, "token_ids", None)
    if prompt_ids is None or completion_ids is None:
        # Fall back to the model_extra bag if the SDK stashed them there.
        prompt_ids = (res.model_extra or {}).get("prompt_token_ids", prompt_ids)
        completion_ids = (choice.model_extra or {}).get("token_ids", completion_ids)
    if not prompt_ids or not completion_ids:
        logger.warning("Missing token ids in response; skipping sample")
        return None
    input_ids = list(prompt_ids) + list(completion_ids)
    loss_mask = [False] * len(prompt_ids) + [True] * len(completion_ids)
    return input_ids, loss_mask


def build_generated_dataset(
    client: openai.OpenAI,
    model: str,
    prompts: list[dict],
    max_new_tokens: int,
    total_seq_len: int,
) -> Dataset:
    """Greedy-generate answers and pack into an Arrow dataset for ArrowDataset."""
    rows = {"input_ids": [], "loss_mask": [], "seq_len": [], "benchmark": [], "id": []}
    for i, rec in enumerate(prompts):
        out = greedy_generate(client, model, rec["prompt"], max_new_tokens)
        if out is None:
            continue
        input_ids, loss_mask = out
        if len(input_ids) > total_seq_len:  # keep sequences within the draft's cap
            input_ids = input_ids[:total_seq_len]
            loss_mask = loss_mask[:total_seq_len]
        if not any(loss_mask):  # nothing generated / all truncated away
            continue
        rows["input_ids"].append(input_ids)
        rows["loss_mask"].append(loss_mask)
        rows["seq_len"].append(len(input_ids))
        rows["benchmark"].append(rec.get("benchmark", "unknown"))
        rows["id"].append(str(rec.get("id", i)))
        if (i + 1) % 10 == 0:
            logger.info("  generated %d/%d", i + 1, len(prompts))
    return Dataset.from_dict(rows)


@torch.no_grad()
def score_sample(
    model: SpeculatorModel,
    item: dict,
    ttt_steps: int,
    device: torch.device,
) -> list[dict]:
    """Run the draft over one item and emit per-token records for every step.

    Mirrors ``compute_metrics`` / ``align_for_step`` exactly, but records each
    token instead of aggregating. ``cond_correct`` chains correctness across
    steps (a token only counts as conditionally correct if every shallower step
    at that draft-chain position was also correct) -- the same ``prev_correct``
    logic the trainer uses for ``cond_acc``.
    """
    hs = item["hidden_states"]  # [S, 3H]
    ids = item["input_ids"]  # [S]
    lm = item["loss_mask"].bool()  # [S]
    true_len = ids.shape[0]

    # The flex_attention draft-token mask extension requires the sequence length
    # to be a multiple of the block size (128) -- same constraint as training,
    # which the collate satisfies by padding to total_seq_len. Here we pad each
    # sample to the next 128-multiple (efficient for short samples). Padded
    # positions get document_id = -1 so the mask excludes them and loss_mask =
    # False so they are never scored.
    FLEX_BLOCK = 128
    padded_len = ((true_len + FLEX_BLOCK - 1) // FLEX_BLOCK) * FLEX_BLOCK
    pad = padded_len - true_len

    hidden_states = torch.nn.functional.pad(hs, (0, 0, 0, pad)).unsqueeze(0).to(device)
    input_ids = torch.nn.functional.pad(ids, (0, pad)).unsqueeze(0).to(device)
    loss_mask = torch.nn.functional.pad(lm, (0, pad), value=False).unsqueeze(0).to(device)
    # Single real document (id 0) followed by padding (id -1).
    document_ids = torch.cat(
        [torch.zeros(true_len, dtype=torch.long),
         -torch.ones(pad, dtype=torch.long)]
    ).unsqueeze(0).to(device)
    position_ids = torch.arange(padded_len, dtype=torch.long, device=device).unsqueeze(0)

    # With verifier_last_hidden_states=None the forward skips loss/metrics and
    # returns ONLY the draft_tokens list (see eagle3/core.py). The predicted
    # tokens are identical either way (same argmax + on-policy feed), so we take
    # the cheaper path and define "correct" from the greedy input_ids shifts.
    out = model(
        hidden_states=hidden_states,
        input_ids=input_ids,
        document_ids=document_ids,
        loss_mask=loss_mask,
        position_ids=position_ids,
        ttt_steps=ttt_steps,
        verifier_last_hidden_states=None,  # targets come from input_ids shifts
    )
    draft_tokens = out[0] if isinstance(out, tuple) else out

    # Map a draft-vocab prediction to verifier-vocab id (identity when the draft
    # shares the verifier vocabulary, i.e. draft_vocab_size is None).
    d2t = getattr(model, "d2t", None)

    def to_verifier_ids(pred: torch.Tensor) -> torch.Tensor:
        if d2t is None:
            return pred
        return pred + d2t.to(pred.device)[pred]

    records: list[dict] = []
    # prev_correct chains across steps at a fixed draft-chain start position.
    prev_correct = loss_mask.clone()  # [1, S]
    for k in range(ttt_steps):
        pred_full = to_verifier_ids(draft_tokens[k])  # [1, S] verifier-space
        # align_for_step on token-id tensors: logits[:, :-k] vs targets[:, k:].
        # targets_token[i] == input_ids[i+1] (greedy), so use input_ids shifted.
        pred_aligned, _t, mask_aligned, prev_aligned = align_for_step(
            pred_full.unsqueeze(-1),  # fake a vocab dim for the shared helper
            input_ids.unsqueeze(-1),
            loss_mask,
            prev_correct,
            k,
        )
        pred_aligned = pred_aligned.squeeze(-1)  # [1, S-k]
        # ground truth at aligned pos j is input_ids[j + k + 1]
        gt_aligned = input_ids[:, k + 1 : k + 1 + pred_aligned.shape[1]]
        n = min(pred_aligned.shape[1], gt_aligned.shape[1])
        pred_aligned = pred_aligned[:, :n]
        gt_aligned = gt_aligned[:, :n]
        mask_aligned = mask_aligned[:, :n]
        prev_aligned = prev_aligned[:, :n]

        correct = pred_aligned == gt_aligned  # [1, n]
        # update chain for the NEXT step (in place, same semantics as trainer)
        new_prev = torch.logical_and(prev_aligned, correct)

        idxs = mask_aligned[0].nonzero(as_tuple=True)[0].tolist()
        for j in idxs:
            records.append(
                {
                    "benchmark": item["benchmark"],
                    "id": item["id"],
                    "ttt_step": k,
                    "aligned_pos": int(j),
                    "seq_len": int(true_len),
                    "rel_pos": float(j) / max(1, true_len),
                    "pred_id": int(pred_aligned[0, j].item()),
                    "target_id": int(gt_aligned[0, j].item()),
                    "correct": bool(correct[0, j].item()),
                    "cond_correct": bool(new_prev[0, j].item()),
                }
            )
        # carry the chain forward, re-aligned to full length for the next shift
        prev_correct = torch.zeros_like(loss_mask)
        prev_correct[:, : new_prev.shape[1]] = new_prev
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft-checkpoint", required=True, help="Saved SpeculatorModel dir")
    ap.add_argument("--vllm-endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--hidden-states-path", required=True, help="Shared connector dir")
    ap.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    ap.add_argument("--benchmarks", default="gpqa_diamond,livecodebench,aime")
    ap.add_argument("--ttt-steps", type=int, default=3)
    ap.add_argument("--num-samples", type=int, default=50, help="Per benchmark; -1=all")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--total-seq-len", type=int, default=8192)
    ap.add_argument("--out", required=True, help="Output mistakes.jsonl path")
    ap.add_argument(
        "--reference",
        default=None,
        help=(
            "Path to a frozen reference of target continuations (JSONL of "
            "{benchmark,id,input_ids,loss_mask}). If it exists, targets are LOADED "
            "from it (no regeneration) so every checkpoint is scored on identical "
            "text -- required for valid cross-checkpoint comparison. If it does not "
            "exist, targets are generated once and saved here for reuse."
        ),
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = None if args.num_samples < 0 else args.num_samples
    eval_dir = Path(args.eval_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = openai.OpenAI(base_url=args.vllm_endpoint, api_key="EMPTY", max_retries=2)
    model_id = client.models.list().data[0].id
    logger.info("Server model: %s", model_id)

    logger.info("Loading draft model from %s", args.draft_checkpoint)
    model = SpeculatorModel.from_pretrained(args.draft_checkpoint)
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    # Load a frozen reference of target continuations if one exists, so every
    # checkpoint is scored on identical text (greedy generation is not reliably
    # deterministic across server runs, and comparing checkpoints scored on
    # different targets is invalid).
    ref_path = Path(args.reference) if args.reference else None
    ref_by_benchmark: dict[str, list[dict]] = {}
    if ref_path and ref_path.exists():
        for line in ref_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ref_by_benchmark.setdefault(r["benchmark"], []).append(r)
        logger.info("Loaded frozen reference from %s (%d benchmarks)",
                    ref_path, len(ref_by_benchmark))

    n_written = 0
    with out_path.open("w") as fout:
        for benchmark in [b.strip() for b in args.benchmarks.split(",") if b.strip()]:
            if benchmark in ref_by_benchmark:
                rows = ref_by_benchmark[benchmark]
                gen_ds = Dataset.from_dict({
                    "input_ids": [r["input_ids"] for r in rows],
                    "loss_mask": [r["loss_mask"] for r in rows],
                    "seq_len": [len(r["input_ids"]) for r in rows],
                    "benchmark": [benchmark] * len(rows),
                    "id": [str(r["id"]) for r in rows],
                })
                logger.info("Using %d frozen reference samples for %s",
                            len(rows), benchmark)
            else:
                prompts = load_prompts(benchmark, eval_dir, num_samples)
                gen_ds = build_generated_dataset(
                    client, model_id, prompts, args.max_new_tokens, args.total_seq_len
                )
                if ref_path is not None and len(gen_ds) > 0:
                    with ref_path.open("a") as rf:
                        for row in gen_ds:
                            rf.write(json.dumps({
                                "benchmark": benchmark,
                                "id": row["id"],
                                "input_ids": row["input_ids"],
                                "loss_mask": [bool(x) for x in row["loss_mask"]],
                            }) + "\n")
                    logger.info("Saved %d reference samples for %s to %s",
                                len(gen_ds), benchmark, ref_path)
            if len(gen_ds) == 0:
                logger.warning("No usable samples for %s; skipping", benchmark)
                continue

            tmp_dir = Path(tempfile.mkdtemp(prefix=f"mistake_{benchmark}_"))
            try:
                # ArrowDataset relies on the persisted torch format to return
                # tensors (build_client_item calls .tolist(); loss_mask is used as
                # a tensor). output_all_columns keeps our string benchmark/id.
                gen_ds = gen_ds.with_format(
                    "torch",
                    columns=["input_ids", "loss_mask", "seq_len"],
                    output_all_columns=True,
                )
                gen_ds.save_to_disk(str(tmp_dir / "arrow"))
                ds = ArrowDataset(
                    max_len=args.total_seq_len,
                    datapath=str(tmp_dir / "arrow"),
                    hidden_states_path=args.hidden_states_path,
                    vllm_endpoint=args.vllm_endpoint,
                    on_missing="generate",
                    on_generate="delete",
                    model=model_id,
                )
                for i in range(len(ds)):
                    item = ds[i]
                    if item is None:
                        continue
                    # ArrowDataset drops the arrow metadata columns; re-attach.
                    item["benchmark"] = gen_ds[i]["benchmark"]
                    item["id"] = gen_ds[i]["id"]
                    for rec in score_sample(model, item, args.ttt_steps, device):
                        fout.write(json.dumps(rec) + "\n")
                        n_written += 1
                    if (i + 1) % 10 == 0:
                        logger.info("  scored %d/%d (%s)", i + 1, len(ds), benchmark)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("Wrote %d token records to %s", n_written, out_path)


if __name__ == "__main__":
    main()
