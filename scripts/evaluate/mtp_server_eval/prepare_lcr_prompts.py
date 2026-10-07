#!/usr/bin/env python3
"""Build AA-LCR long-context prompts, bucketed by target token length.

AA-LCR = 100 reasoning questions over document sets of ~72k-115k tokens
(ArtificialAnalysis/AA-LCR, Apache-2.0). This assembles each record's documents
("BEGIN DOCUMENT n" markers, filename order) + question, then emits TRUNCATED
variants at the requested buckets (documents are cut, question always kept).
Truncated tiers may lose the evidence needed for a correct ANSWER -- fine here:
we measure speculation acceptance/throughput on real long-context reasoning,
not answer accuracy.

Output: one jsonl per bucket under --output-dir:
  lcr_<bucket>.jsonl  rows: {question_id, bucket, prompt_tokens, prompt}

Usage:
  python prepare_lcr_prompts.py --lcr-root /sms-scratch/mengmengj/data/aa_lcr/lcr \
      --buckets 16384,32768,65536,full --num-questions 30 \
      --output-dir /sms-scratch/mengmengj/data/aa_lcr/prompts
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

TOKENIZER_PATH = "/sms-scratch/checkpoints/gemma-4-31B-it"


def assemble(rec, lcr_root):
    set_dir = lcr_root / rec["document_category"] / rec["document_set_id"]
    files = {f.name: f for f in set_dir.glob("*.txt")}
    ordered, seen = [], set()
    for want in str(rec["data_source_filenames"]).split(";"):
        want = want.strip()
        hit = files.get(want) or files.get(want + ".txt") or next(
            (f for n, f in files.items() if n.startswith(want[:40])), None)
        if hit and hit.name not in seen:
            ordered.append(hit)
            seen.add(hit.name)
    ordered += [f for n, f in sorted(files.items()) if n not in seen]
    parts = []
    for i, f in enumerate(ordered, 1):
        parts.append(f"BEGIN DOCUMENT {i}: {f.name}\n\n"
                     + f.read_text(errors="ignore").strip()
                     + f"\n\nEND DOCUMENT {i}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcr-root", required=True)
    ap.add_argument("--buckets", default="16384,32768,65536,full")
    ap.add_argument("--num-questions", type=int, default=30)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    lcr_root = Path(args.lcr_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("ArtificialAnalysis/AA-LCR", split="test")
    recs = list(ds)[: args.num_questions]

    buckets = [b if b == "full" else int(b) for b in args.buckets.split(",")]
    writers = {b: (out / f"lcr_{b}.jsonl").open("w") for b in buckets}
    for rec in recs:
        docs = assemble(rec, lcr_root)
        tail = (f"\n\nBased on the documents above, answer the following question."
                f"\nQuestion: {rec['question']}\nAnswer:")
        doc_ids = tok(docs, add_special_tokens=False).input_ids
        tail_n = len(tok(tail, add_special_tokens=False).input_ids)
        for b in buckets:
            keep = len(doc_ids) if b == "full" else max(0, b - tail_n - 32)
            prompt = tok.decode(doc_ids[:keep]) + tail
            n_tok = min(len(doc_ids), keep) + tail_n
            writers[b].write(json.dumps({
                "question_id": rec["question_id"], "bucket": str(b),
                "prompt_tokens": n_tok, "prompt": prompt}) + "\n")
    for b, w in writers.items():
        w.close()
        print(f"wrote {out}/lcr_{b}.jsonl")


if __name__ == "__main__":
    main()
