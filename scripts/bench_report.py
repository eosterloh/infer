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


# What each recipe puts under load. A model can sit in more than one bucket —
# Nemotron-H is both a Mamba-2 hybrid and a sparse MoE — and the point of the
# grouping is to show which kernels the sweep actually exercised.
FAMILIES: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "hybrid recurrent (Mamba2 / GDN scan)",
        frozenset({"nemotron_h", "jamba", "qwen3_next", "olmo_hybrid", "qwen3_5", "qwen3_5_moe"}),
    ),
    ("MLA (compressed KV cache)", frozenset({"deepseek_v2", "deepseek_v3"})),
    (
        "sparse MoE dispatch",
        frozenset({
            "mixtral", "qwen2_moe", "qwen3_moe", "qwen3_5_moe", "gpt_oss", "llama4", "dbrx",
            "olmoe", "flex_olmo", "granitemoe", "granitemoe_swa", "granitemoeshared",
            "phimoe", "hunyuan_v1_moe", "ernie4_5_moe", "cohere2_moe", "glm4_moe",
            "exaone_moe", "nemotron_h", "nemotron",
        }),
    ),
    (
        "legacy attention (learned norms / parallel residual)",
        frozenset({
            "gpt2", "gptj", "gpt_neo", "gpt_neox", "gpt_bigcode", "opt", "bloom",
            "falcon", "mpt",
        }),
    ),
    ("sliding window / soft-capped attention", frozenset({"gemma2", "gemma3", "gpt_oss"})),
)
DENSE = "dense attention + MLP"


def families_for(recipe: str | None) -> list[str]:
    """Every bucket this recipe belongs to; dense is the fallback, not a bucket."""
    hits = [name for name, ids in FAMILIES if recipe in ids]
    return hits or [DENSE]


def coverage(rows: list[dict], tag: str | None = None) -> list[str]:
    """Which model families the sweep measured, and which it did not.

    Across every tag, not one of them: a family counts as covered if any run
    reached it, and a quantized sweep that skipped a model should not erase it.
    """
    seen: dict[str, set[str]] = {name: set() for name, _ in FAMILIES}
    seen[DENSE] = set()
    measured = latest_by_model(rows, tag).values() if tag else rows
    for row in measured:
        for family in families_for(row.get("recipe")):
            seen[family].add(row["model"])
    lines = ["## family coverage", ""]
    lines.append("| family | models measured |")
    lines.append("|---|---|")
    for family, models in seen.items():
        lines.append(f"| {family} | {', '.join(sorted(models)) if models else '**none**'} |")
    lines.append("")
    return lines


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


def _logit_delta(base: dict, new: dict, tol: float) -> float | None:
    """Largest gap between the two runs' top-5 logits, or None if incomparable.

    A top-5 list has a cut, and two tokens within a BF16 ulp of it trade places
    for no reason worth reporting — Qwen2.5 does exactly that here, with its
    first four logits bit-identical and rank five held by two tokens 0.125
    apart. So a token only one run listed is fine while it sits at that cut, and
    ``inf`` when it stands clear of it, which is what dropping a term looks like.
    """
    fa, fb = base.get("fingerprint"), new.get("fingerprint")
    if not fa or not fb:
        return None
    ids_a, ids_b = fa.get("top5_ids"), fb.get("top5_ids")
    la, lb = fa.get("top5_logits"), fb.get("top5_logits")
    if not ids_a or not ids_b or not la or not lb:
        return None
    a_by, b_by = dict(zip(ids_a, la)), dict(zip(ids_b, lb))
    shared = [token for token in ids_a if token in b_by]
    if not shared:
        return float("inf")
    for listed, other in ((a_by, b_by), (b_by, a_by)):
        cut = min(other.values())
        for token, value in listed.items():
            if token not in other and value - cut > tol:
                return float("inf")
    return max(abs(a_by[token] - b_by[token]) for token in shared)


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
    tol = _tolerance(base.get("fingerprint", {}).get("top5_logits") or [])
    delta = _logit_delta(base, new, tol)
    if delta is None:
        return "no logits", False
    if delta == float("inf"):
        return "DIFFERS (a top-5 token moved clear of the cut)", True
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
    measured = {tag: latest_by_model(rows, tag) for tag in tags}
    # A model that crashed under one configuration leaves no row at all, and a
    # table only shows what it has: the sweep would read as clean while a family
    # went unmeasured. So the models any configuration managed are what every
    # configuration is held to.
    expected = {model for got in measured.values() for model in got}

    lines: list[str] = []
    drifted: list[str] = []
    absent: list[str] = []
    host = next((r.get("host") for r in rows if r.get("host")), "unknown")
    lines.append(f"# infer benchmarks — {host}")
    lines.append("")
    lines.extend(coverage(rows))
    for tag in tags:
        new = measured[tag]
        for model in sorted(expected - set(new)):
            absent.append(f"{tag}/{model}: no row (the run failed or was skipped)")
        if not new:
            absent.append(f"{tag}: produced no rows at all")
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
                # The pre-work revision cannot load every checkpoint here — some
                # of these recipes are part of the work — so "no before" is a
                # real answer for those, not a hole in the measurement.
                lines.append(
                    f"| {model} | {(row.get('params') or 0) / 1e9:.2f}B | — | "
                    f"{row['decode_tok_s']} | — | — | {row['prefill_tok_s']} | — | "
                    f"no {args.baseline} row |"
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
                f"| {model} | {(row.get('params') or 0) / 1e9:.2f}B "
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

    if absent:
        lines.append("## missing rows")
        lines.append("")
        lines.extend(f"- {item}" for item in absent)
        lines.append("")

    text = "\n".join(lines)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    if args.strict and (drifted or absent):
        if drifted:
            print("\nFAIL: a faster run changed the answer", file=sys.stderr)
        if absent:
            print("FAIL: a configuration produced no row for a model", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
