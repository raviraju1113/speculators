#!/usr/bin/env python3
"""Read one AgentX (aiperf) run's metrics and print a `matrix.tsv` cell.

`run_agentx.sh` drives `aiperf profile --scenario inferencex-agentx-mvp` once per
concurrency level. aiperf writes `profile_export_aiperf.json` into its artifact
dir; this script reduces that to the handful of numbers the AgentX matrix wants.

Output: one tab-separated line, `decode_tok_s <TAB> out_tok_s <TAB> valid`, using
``NA`` for anything unavailable so a partial/failed cell still lines up in the
matrix instead of shifting columns.

Usage:
    python agentx_metrics.py <artifact-dir-or-profile_export_aiperf.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPORT_NAME = "profile_export_aiperf.json"
NA = "NA"

# Seconds per unit, for the time units aiperf stamps onto each metric result.
_SECONDS_PER = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "sec": 1.0, "s": 1.0}


def find_export(path: Path) -> Path | None:
    """Locate profile_export_aiperf.json at, under, or beside ``path``."""
    if path.is_file():
        return path
    if not path.is_dir():
        return None
    direct = path / EXPORT_NAME
    if direct.is_file():
        return direct
    # aiperf nests the per-run export one level down when it names the run dir.
    matches = sorted(path.glob(f"*/{EXPORT_NAME}")) or sorted(
        path.glob(f"**/{EXPORT_NAME}")
    )
    return matches[0] if matches else None


def _result(data: dict, tag: str) -> dict | None:
    """Metric results sit top-level in the per-run file, under `metrics` in the
    multi-run aggregate. Accept either."""
    for holder in (data, data.get("metrics") or {}):
        value = holder.get(tag)
        if isinstance(value, dict):
            return value
    return None


def _stat(data: dict, tag: str, field: str = "avg") -> float | None:
    r = _result(data, tag)
    if r is None:
        return None
    value = r.get(field)
    return float(value) if isinstance(value, (int, float)) else None


def _seconds(data: dict, tag: str, field: str) -> float | None:
    """Read a duration stat and convert to seconds using its declared unit."""
    r = _result(data, tag)
    if r is None:
        return None
    value = r.get(field)
    if not isinstance(value, (int, float)):
        return None
    scale = _SECONDS_PER.get(str(r.get("unit", "")).strip())
    if scale is None:
        return None
    return float(value) * scale


def decode_tok_s(data: dict) -> float | None:
    """Pooled decode throughput: total output tokens / total decode time.

    Matches the definition used elsewhere in this tree (`run_vllm_eval.py`'s
    ``decode_tok_s``, i.e. excluding TTFT) so AgentX numbers are comparable with
    the static-prompt benchmarks. aiperf's ``decode_duration`` is per request
    ``request_latency - ttft``, so summing it gives the same denominator.
    """
    total_osl = _stat(data, "total_osl")
    decode_s = _seconds(data, "decode_duration", "sum")
    if total_osl and decode_s and decode_s > 0:
        return total_osl / decode_s
    # Older/leaner exports may omit `sum`; avg x count is the same quantity.
    avg_s = _seconds(data, "decode_duration", "avg")
    count = _stat(data, "decode_duration", "count")
    if total_osl and avg_s and count and avg_s * count > 0:
        return total_osl / (avg_s * count)
    # Last resort: mean per-user decode speed (aiperf derives it as 1/ITL).
    return _stat(data, "output_token_throughput_per_user")


def collect(export: Path) -> dict:
    data = json.loads(export.read_text())
    meta = data.get("metadata") or {}
    valid = meta.get("submission_valid")
    return {
        "decode_tok_s": decode_tok_s(data),
        # Wall-clock system throughput. AgentX preserves inter-turn think time,
        # so idle gaps depress this -- keep it as a reference column only.
        "out_tok_s": _stat(data, "output_token_throughput"),
        "submission_valid": NA if valid is None else str(bool(valid)).lower(),
        "invalid_reasons": ",".join(meta.get("submission_invalid_reasons") or []),
        "context_overflow": _stat(data, "context_overflow_count"),
        "errors": _stat(data, "error_request_count"),
        "requests": _stat(data, "completed_request_count"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "path", type=Path, help=f"aiperf artifact dir, or a {EXPORT_NAME} file"
    )
    ap.add_argument(
        "--json", action="store_true", help="emit the full metric dict instead of a row"
    )
    args = ap.parse_args()

    export = find_export(args.path)
    if export is None:
        # Not fatal: the caller still writes a row so the matrix stays aligned.
        print(f"!! no {EXPORT_NAME} under {args.path}", file=sys.stderr)
        print(f"{NA}\t{NA}\t{NA}")
        return

    m = collect(export)
    if args.json:
        print(json.dumps(m, indent=2))
        return

    def fmt(value: float | None) -> str:
        return NA if value is None else f"{value:.1f}"

    print(f"{fmt(m['decode_tok_s'])}\t{fmt(m['out_tok_s'])}\t{m['submission_valid']}")
    if m["invalid_reasons"]:
        print(f"   submission_invalid_reasons: {m['invalid_reasons']}", file=sys.stderr)
    for key in ("context_overflow", "errors"):
        if m[key]:
            print(f"   {key}: {m[key]:.0f}", file=sys.stderr)


if __name__ == "__main__":
    main()
