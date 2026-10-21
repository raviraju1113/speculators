"""Convert Draft-OPD rollout records (assemble_rollouts.py output) into a
prepared training dataset in the same arrow format as scripts/prepare_data.py,
plus two per-token columns the OPD loss consumes:

  opd_anchor_mask [seq_len] bool  - 1 at every logged anchor position
                                    (anchor_pos = len(prompt_ids) + anchor - 1:
                                    the last verified token, whose embedding
                                    fills slot 0 of the draft block)
  opd_accepted    [seq_len] int32 - at anchor positions: the accepted count r
                                    (0..K); -1 elsewhere

Columns input_ids / loss_mask / seq_len / messages mirror prepare_data.py, so
the online hidden-state path, packing and collation work unchanged; the two
extra per-token columns concatenate like loss_mask when packing.

Steps are dropped when (a) the block does not fit before the row end
(anchor_pos + K > seq_len: the last few anchors of a rollout) or (b) the whole
block was accepted (nothing to learn from, under first-rejected-only the
accepted slots alone would only reinforce; keep with --keep-full-accept).
`finish=length` rollouts keep their steps except the last (its tail is clipped).

    python scripts/opd/rollouts_to_dataset.py --rollouts <rollouts.jsonl> \
        --model <backbone> --token-freq <sft dataset>/token_freq.pt \
        --output <dir> [--seq-length 8192]
"""

import argparse
import json
import shutil
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value
from tokenizers import Tokenizer  # not transformers: its import scans every installed package (minutes on NFS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--model", required=True, help="backbone path (tokenizer)")
    ap.add_argument("--token-freq", required=True,
                    help="token_freq.pt of the SFT dataset the checkpoint was trained on; its "
                    "d2t.npy / t2d.npy (same dir) are copied too -- the draft vocab mapping MUST "
                    "be the checkpoint's. Rebuilding it from token_freq gives a different "
                    "mapping (measured: 27755/32000 overlap) and a garbage draft.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seq-length", type=int, default=8192)
    ap.add_argument("--keep-full-accept", action="store_true")
    ap.add_argument("--min-anchors", type=int, default=1)
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    rows = {"input_ids": [], "loss_mask": [], "seq_len": [], "messages": [],
            "opd_anchor_mask": [], "opd_accepted": []}
    n_in = n_rows = n_anchors = n_drop_fit = n_drop_full = n_drop_bad = 0
    for line in open(args.rollouts):
        rec = json.loads(line)
        n_in += 1
        if not rec.get("ok", True):
            n_drop_bad += 1
            continue
        prompt, out = rec["prompt_ids"], rec["output_ids"]
        k = rec["k"]
        ids = (prompt + out)[: args.seq_length]
        seq_len = len(ids)
        plen = len(prompt)
        anchor_mask = [False] * seq_len
        accepted = [-1] * seq_len
        steps = rec["steps"]
        last = len(steps) - 1
        for i, s in enumerate(steps):
            if not s["draft_ids"]:
                continue
            if rec.get("finish") == "length" and i == last:
                continue  # tail clipped by max_tokens
            # Training anchor = position of the LAST VERIFIED token (slot 0
            # carries that token's embedding and sees base tokens strictly before
            # it; it predicts the next token = draft token 1). The rollout's
            # `anchor` counts verified output tokens, so the last one sits at
            # plen + anchor - 1 (for anchor == 0 that is the last prompt token,
            # which loss_mask excludes; such blocks do not occur in our logs
            # because the first decode step after prefill carries no draft).
            pos = plen + s["anchor"] - 1
            r = s["accepted"]
            if pos < plen or pos + k >= seq_len:
                n_drop_fit += 1
                continue
            if r >= k and not args.keep_full_accept:
                n_drop_full += 1
                continue
            anchor_mask[pos] = True
            accepted[pos] = r
        n_a = sum(anchor_mask)
        if n_a < args.min_anchors:
            continue
        loss_mask = [False] * plen + [True] * (seq_len - plen)
        rows["input_ids"].append(ids)
        rows["loss_mask"].append(loss_mask)
        rows["seq_len"].append(seq_len)
        rows["messages"].append([
            {"role": "user", "content": tok.decode(prompt, skip_special_tokens=True)},
            {"role": "assistant", "content": tok.decode(out, skip_special_tokens=True)},
        ])
        rows["opd_anchor_mask"].append(anchor_mask)
        rows["opd_accepted"].append(accepted)
        n_rows += 1
        n_anchors += n_a

    # Mirror the SFT dataset's feature types (datasets>=5 `List(struct)` for
    # messages; `Sequence(dict)` would store a struct of lists instead).
    try:
        from datasets.features import List as HFList  # datasets >= 5
    except ImportError:  # pragma: no cover
        HFList = Sequence
    features = Features({
        "input_ids": HFList(Value("int32")),
        "loss_mask": HFList(Value("bool")),
        "seq_len": Value("int64"),
        "messages": HFList({"role": Value("string"), "content": Value("string")}),
        "opd_anchor_mask": HFList(Value("bool")),
        "opd_accepted": HFList(Value("int32")),
    })
    ds = Dataset.from_dict(rows, features=features)
    # The trainer expects tensor rows (prepare_data.py saves with the torch
    # format, which save_to_disk persists in state.json); strings stay strings.
    ds = ds.with_format("torch")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    shutil.copy(args.token_freq, out / "token_freq.pt")
    vocab_dir = Path(args.token_freq).parent
    for f in ("d2t.npy", "t2d.npy"):
        if not (vocab_dir / f).exists():
            raise SystemExit(f"!! {vocab_dir / f} missing: the draft-vocab mapping must come from the "
                             "checkpoint's training dataset, not be rebuilt here")
        shutil.copy(vocab_dir / f, out / f)
    print(f"rollouts in: {n_in}  rows out: {n_rows}  anchors: {n_anchors} "
          f"({n_anchors / max(n_rows, 1):.1f}/row)  dropped steps: fit={n_drop_fit} "
          f"full_accept={n_drop_full}; bad rollouts skipped: {n_drop_bad}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
