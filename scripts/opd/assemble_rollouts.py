"""Validate and merge the engine-side rollout logs (DSPARK_ROLLOUT_LOG) into one
rollouts.jsonl, and print the acceptance statistics (gate 1 of the Draft-OPD plan).

Checks per request:
  * reconstruction: concatenating every step's ``gen`` equals ``output_ids``
  * anchors: step[i+1].anchor == step[i].anchor + len(step[i].gen)
  * invariant: output_ids[anchor : anchor + accepted] == draft_ids[:accepted]
  * accepted < k on every step except possibly the last of a block run
    (accepted == k means the whole block was accepted; that is fine)

    python scripts/opd/assemble_rollouts.py --log-dir <dir with rollout_*.jsonl> \
        [--responses <rollout.py out>/responses.jsonl] --out rollouts.jsonl
"""

import argparse
import collections
import glob
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--responses", default=None, help="rollout.py responses.jsonl (optional join)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--draft-ckpt", default="", help="tag stored in every record")
    ap.add_argument("--round", type=int, default=0)
    args = ap.parse_args()

    # Join key: the engine-side request id. The API server prefixes the client's
    # X-Request-Id (e.g. "chatcmpl-smoke-000001"), and returns it as the
    # response "id"; index both forms.
    responses = {}
    if args.responses:
        for line in open(args.responses):
            r = json.loads(line)
            responses[r["request_id"]] = r
            if r.get("engine_request_id"):
                responses[r["engine_request_id"]] = r

    n = n_bad = 0
    bad_kinds = collections.Counter()
    first_rej = collections.Counter()
    acc_hist = collections.Counter()
    tot_draft = tot_acc = tot_steps = 0
    out = open(args.out, "w")
    for path in sorted(glob.glob(str(Path(args.log_dir) / "rollout_*.jsonl"))):
        for line in open(path):
            rec = json.loads(line)
            n += 1
            steps, out_ids = rec["steps"], rec["output_ids"]
            problems = []
            gen_cat = [t for s in steps for t in s["gen"]]
            if gen_cat != out_ids:
                problems.append("reconstruction")
            for i, s in enumerate(steps):
                if i + 1 < len(steps) and steps[i + 1]["anchor"] != s["anchor"] + len(s["gen"]):
                    problems.append("anchor_chain")
                    break
            k = rec["k"]
            for s in steps:
                d, a, r = s["draft_ids"], s["anchor"], s["accepted"]
                if not d:
                    continue
                tot_steps += 1
                tot_draft += len(d)
                tot_acc += r
                acc_hist[r] += 1
                if out_ids[a : a + r] != d[:r]:
                    problems.append("invariant")
                if r < len(d):
                    first_rej[r + 1] += 1  # 1-based position of the first rejected token
                else:
                    first_rej["full"] += 1
            if problems:
                n_bad += 1
                for p in set(problems):
                    bad_kinds[p] += 1
            rec["ok"] = not problems
            rec["problems"] = sorted(set(problems))
            rec["draft_ckpt"] = args.draft_ckpt
            rec["round"] = args.round
            rid = rec["request_id"]
            resp = responses.get(rid) or next(
                (v for k, v in responses.items() if rid.endswith(k)), None
            )
            if resp is not None:
                rec["pool_index"] = resp["index"]
                rec["finish_reason_api"] = resp["finish_reason"]
            out.write(json.dumps(rec) + "\n")
    out.close()

    print(f"requests: {n}  bad: {n_bad} {dict(bad_kinds) if bad_kinds else ''}")
    if tot_steps:
        print(f"draft steps: {tot_steps}  mean accepted/step: {tot_acc / tot_steps:.3f}  "
              f"accept_rate: {tot_acc / tot_draft:.4f}  (k={k})")
        print("accepted-count histogram:", dict(sorted(acc_hist.items())))
        tot = sum(first_rej.values())
        print("first-rejection position (1-based; 'full' = whole block accepted):")
        for key in sorted(first_rej, key=lambda x: (isinstance(x, str), x)):
            print(f"  {key!s:>5}: {100 * first_rej[key] / tot:5.1f}%")
    print(f"wrote {args.out}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
