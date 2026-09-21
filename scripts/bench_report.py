#!/usr/bin/env python3
"""Turn bench/results.jsonl into a before/after table, and check the answers.

Speed alone proves nothing: a kernel that drops a term is very fast. So every
row carries the greedy fingerprint the run produced, and a tag only counts as
an improvement when its fingerprint matches the baseline's for that model.

    scripts/bench_report.py --baseline baseline --tag kernels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_by_model(rows: list[dict], tag: str) -> dict[str, dict]:
    """Last run wins, so re-running one model does not need a file edit."""
    out: dict[str, dict] = {}
    for row in rows:
        if row.get("tag") == tag:
            out[row["model"]] = row
    return out


def _ids(row: dict) -> list[int] | None:
    fp = row.get("fingerprint")
    return fp.get("greedy_ids") if fp else None


def _verdict(base: dict, new: dict) -> str:
    a, b = _ids(base), _ids(new)
    if a is None or b is None:
        return "no fingerprint"
    if a == b:
        return "same"
    common = sum(1 for x, y in zip(a, b) if x == y)
    return f"DIFFERS (first {common}/{len(a)} match)"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=ROOT / "bench" / "results.jsonl")
    p.add_argument("--baseline", default="baseline")
    p.add_argument("--tag", action="append", default=None, help="repeatable")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any tag changed a model's greedy output",
    )
    args = p.parse_args()

    rows = load(args.results)
    tags = args.tag or sorted(
        {r["tag"] for r in rows if r.get("tag") != args.baseline}
    )
    base = latest_by_model(rows, args.baseline)

    lines: list[str] = []
    drifted: list[str] = []
    host = next((r.get("host") for r in rows if r.get("host")), "unknown")
    lines.append(f"# infer benchmarks — {host}")
    lines.append("")
    for tag in tags:
        new = latest_by_model(rows, tag)
        if not new:
            continue
        lines.append(f"## {tag} vs {args.baseline}")
        lines.append("")
        lines.append(
            "| model | params | decode tok/s base | decode tok/s new | speedup "
            "| prefill tok/s base | prefill tok/s new | speedup | output |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for model, row in sorted(new.items()):
            b = base.get(model)
            if b is None:
                lines.append(
                    f"| {model} | {row.get('params', 0) / 1e9:.2f}B | — | "
                    f"{row['decode_tok_s']} | — | — | {row['prefill_tok_s']} | — | "
                    "no baseline |"
                )
                continue
            d_gain = row["decode_tok_s"] / b["decode_tok_s"]
            p_gain = row["prefill_tok_s"] / b["prefill_tok_s"]
            verdict = _verdict(b, row)
            # Quantization changes the weights, so its output is expected to
            # move; a kernel or a captured graph has no such excuse.
            if verdict.startswith("DIFFERS") and not row.get("quant"):
                drifted.append(f"{tag}/{model}: {verdict}")
            lines.append(
                f"| {model} | {row.get('params', 0) / 1e9:.2f}B "
                f"| {b['decode_tok_s']} | {row['decode_tok_s']} | {d_gain:.2f}x "
                f"| {b['prefill_tok_s']} | {row['prefill_tok_s']} | {p_gain:.2f}x "
                f"| {verdict} |"
            )
        lines.append("")

    if drifted:
        lines.append("## output drift")
        lines.append("")
        lines.extend(f"- {item}" for item in drifted)
        lines.append("")

    text = "\n".join(lines)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    if drifted and args.strict:
        print("\nFAIL: a faster run changed the answer", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
