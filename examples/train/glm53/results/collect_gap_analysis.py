#!/usr/bin/env python
"""Copy the raw numbers behind ../SUMMARY.md (incoai gap analysis, 2026-11-10..12) into this dir.

  * glm53-ablations-fp8-v033/<exp>/  mtp_eval_summary.json + mtp_eval_details.jsonl
                                      + per_position.txt (from the vLLM server.log, which is too
                                      large to commit)
  * p0-sglang-card-fp8/<exp>/        mtp_eval_summary.json + mtp_eval_details.jsonl
  * val_metrics_gap_analysis.json    the trainer's end-of-epoch validation metrics per run/epoch

Re-run with the glm53_dflash2 env python (system python3 is 3.6):
    python examples/train/glm53/results/collect_gap_analysis.py
"""

import json
import re
import shutil
from pathlib import Path

W = Path("/import/ml-sc-scratch1/ravir/spec_decoding_work/glm53_dflash2")
HERE = Path(__file__).resolve().parent

VAL_RUNS = {  # run -> {epoch label: checkpoint dir}
    "A0_control": ("glm53_abl_A0_control", {"ep1": "0", "ep2": "1"}),
    "A1_incoai_taps": ("glm53_abl_A1_incoai_taps", {"ep1": "0", "ep2": "1"}),
    "A2_incoai_shape": ("glm53_abl_A2_incoai_shape", {"ep1": "0", "ep2": "1"}),
    "A3_incoai_shape_block8": ("glm53_abl_A3_incoai_shape_block8", {"ep1": "0", "ep2": "1"}),
    "ftA_onpolicy_ft_from_ep9": ("glm53_dflash2_onpolicyA_ft3ep_lr2e-5", {"ep1": "0", "ep2": "1", "ep3": "2"}),
    "incoai_as_released_valonly": ("incoai_valonly", {"val": "0"}),
    "incoai_ft_onpolicyA_half": ("incoai_ft_onpolicyA_half", {"half_ep": "0"}),
}

PAT = re.compile(r"Drafted: (\d+) tokens, Per-position acceptance rate: ([0-9., ]+)")


def per_position(server_log: Path) -> str:
    """Draft-step-weighted per-position acceptance over a whole vLLM server.log."""
    tot, steps = None, 0.0
    for line in server_log.open(errors="replace"):
        m = PAT.search(line)
        if not m:
            continue
        drafted = int(m.group(1))
        rates = [float(x) for x in m.group(2).replace(" ", "").strip(",").split(",") if x]
        if not rates or not drafted:
            continue
        n = drafted / len(rates)
        tot = [r * n for r in rates] if tot is None else [t + r * n for t, r in zip(tot, rates)]
        steps += n
    if not tot:
        return "no SpecDecoding metrics found\n"
    pos = [t / steps for t in tot]
    return (f"draft_steps {steps:.0f}\nper_position {' '.join(f'{p:.3f}' for p in pos)}\n"
            f"accept_len {1 + sum(pos):.3f}\n")


def main() -> None:
    for suite in ("glm53-ablations-fp8-v033", "p0-sglang-card-fp8"):
        for exp in sorted((W / "eval" / suite).iterdir()):
            out = HERE / suite / exp.name
            out.mkdir(parents=True, exist_ok=True)
            for f in ("mtp_eval_summary.json", "mtp_eval_details.jsonl"):
                if (exp / f).exists():
                    shutil.copy(exp / f, out / f)
            if (exp / "server.log").exists():
                (out / "per_position.txt").write_text(per_position(exp / "server.log"))
    val = {}
    for name, (run, epochs) in VAL_RUNS.items():
        val[name] = {}
        for label, ck in epochs.items():
            p = W / "output" / run / "checkpoints" / ck / "val_metrics.json"
            if p.exists():
                val[name][label] = json.loads(p.read_text())
    (HERE / "val_metrics_gap_analysis.json").write_text(json.dumps(val, indent=1) + "\n")
    print("collected into", HERE)


if __name__ == "__main__":
    main()
