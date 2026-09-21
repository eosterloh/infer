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


def _tolerance(logits: list[float]) -> float:
    """What counts as the same logit in BF16.

    Reassociating a reduction moves a logit a little, and the kernels do exactly
    that — an fp32 accumulator over a different order of the same terms. The
    allowance scales with magnitude because the error does.
    """
    return 0.05 + 0.01 * max((abs(v) for v in logits), default=0.0)


def _logit_delta(base: dict, new: dict) -> float | None:
    """Largest gap between the two runs' top-5 logits, or None if incomparable."""
    fa, fb = base.get("fingerprint"), new.get("fingerprint")
    if not fa or not fb:
        return None
    ids_a, ids_b = fa.get("top5_ids"), fb.get("top5_ids")
    la, lb = fa.get("top5_logits"), fb.get("top5_logits")
    if not ids_a or not ids_b or not la or not lb:
        return None
    if set(ids_a) != set(ids_b):
        return float("inf")
    by_id = dict(zip(ids_b, lb))
    return max(abs(value - by_id[token]) for token, value in zip(ids_a, la))


def _verdict(base: dict, new: dict) -> tuple[str, bool]:
    """A label for the table, and whether it should fail the run.

    The gate is the first token and the logits behind it, not the whole greedy
    chain. A 16-token chain amplifies one near-tie into total divergence, so
    holding a kernel to an exact chain match would fail on arithmetic that is
    within BF16 noise. A wrong kernel does not sit inside that noise.
    """
    a, b = _ids(base), _ids(new)
    if a is None or b is None:
        return "no fingerprint", False
    delta = _logit_delta(base, new)
    tol = _tolerance(base.get("fingerprint", {}).get("top5_logits") or [])
    if delta is None:
        return "no logits", False
    if delta == float("inf"):
        return "DIFFERS (top-5 set changed)", True
    if a[0] != b[0]:
        return f"DIFFERS (first token, Δlogit {delta:.3f})", True
    if delta > tol:
        return f"DIFFERS (Δlogit {delta:.3f} > {tol:.3f})", True
    if a == b:
        return f"same (Δlogit {delta:.3f})", False
    common = sum(1 for x, y in zip(a, b) if x == y)
    return f"same head, chain splits at {common}/{len(a)} (Δlogit {delta:.3f})", False


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
            verdict, bad = _verdict(b, row)
            # Quantization changes the weights, so its output is expected to
            # move; a kernel or a captured graph has no such excuse.
            if bad and not row.get("quant"):
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
