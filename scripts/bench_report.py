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


def checkpoints_on_disk(root: Path) -> dict[str, str]:
    """``{folder name: model_type}`` for every checkpoint under ``root``.

    The model_type in config.json is the recipe id for all but a couple of
    recipes, and it is enough to say which families this host could have
    measured — which is the difference between a family this machine has no
    checkpoint for and one whose checkpoint quietly produced no row.
    """
    found: dict[str, str] = {}
    if not root.is_dir():
        return found
    for folder in sorted(root.iterdir()):
        config = folder / "config.json"
        if not config.is_file():
            continue
        try:
            with config.open() as handle:
                model_type = json.load(handle).get("model_type")
        except (OSError, ValueError):
            continue
        if model_type:
            found[folder.name] = str(model_type)
    return found


def coverage(
    rows: list[dict], tag: str | None = None, models_dir: Path | None = None
) -> list[str]:
    """Which model families the sweep measured, and which it did not.

    Across every tag, not one of them: a family counts as covered if any run
    reached it, and a quantized sweep that skipped a model should not erase it.

    With a models directory, an empty row says which of the two empties it is. A
    family this host has no checkpoint for is a limit of the machine; a family
    whose checkpoint is sitting right there and produced nothing is a result
    missing from the sweep, and the two should not print the same way.
    """
    seen: dict[str, set[str]] = {name: set() for name, _ in FAMILIES}
    seen[DENSE] = set()
    measured = latest_by_model(rows, tag).values() if tag else rows
    for row in measured:
        for family in families_for(row.get("recipe")):
            seen[family].add(row["model"])

    on_disk: dict[str, set[str]] = {name: set() for name in seen}
    for name, model_type in (
        checkpoints_on_disk(models_dir) if models_dir else {}
    ).items():
        for family in families_for(model_type):
            on_disk[family].add(name)

    lines = ["## family coverage", "", "| family | models measured |", "|---|---|"]
    for family, models in seen.items():
        if models:
            note = ", ".join(sorted(models))
        elif models_dir is None:
            note = "**none**"
        elif on_disk[family]:
            note = f"**none — {', '.join(sorted(on_disk[family]))} on disk, no row**"
        else:
            note = "*no checkpoint of this family on this host*"
        lines.append(f"| {family} | {note} |")
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


def _lossy(base: dict, new: dict) -> bool:
    """Whether the two rows hold different weights, not just different code.

    A 4-bit run is a different model, by design: NVFP4 rounds every weight to one
    of sixteen values, and the answer moves. Reporting that as a regression makes
    the gate meaningless — it would fail on the one configuration it is there to
    ship — so a precision change is measured and shown, not judged.
    """
    return (base.get("quant") or "none") != (new.get("quant") or "none")


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
    lossy = _lossy(base, new)
    if lossy:
        kept = sum(1 for x, y in zip(a, b) if x == y)
        drift = "top-5 reordered" if delta == float("inf") else f"Δlogit {delta:.3f}"
        quant = new.get("quant") or "none"
        return f"{quant}: {kept}/{len(a)} tokens kept, {drift}", False
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


_TAG_MEANINGS = {
    "origin": "the engine before any of this work, at the revision named in the sweep",
    "baseline": "this engine with the extension off, so the Python rework shows separately",
    "kernels": "the compiled kernels, BF16 weights",
    "graph": "the kernels plus a captured CUDA graph for the decode step",
    "nvfp4": "the kernels with weights packed to NVFP4 at load",
}


def _preamble(baseline: str, tags: list[str], rows: list[dict]) -> list[str]:
    """What the reader needs to trust the tables, in the file rather than a chat.

    A committed artifact outlives the session that produced it, and a column of
    ratios means nothing without knowing what changed between the two runs and
    what "same" was allowed to mean.
    """
    git = next((r.get("git") for r in reversed(rows) if r.get("git")), None)
    out = [
        "Every number here was measured on the host named above. Each row is one "
        "checkpoint from disk: a prefill of the stated length, then a fixed number "
        "of greedy decode steps, best of the repetitions.",
        "",
        f"Each table compares one configuration against `{baseline}`:",
        "",
    ]
    for tag in [baseline, *tags]:
        meaning = _TAG_MEANINGS.get(tag)
        if meaning:
            out.append(f"- `{tag}` — {meaning}")
    out += [
        "",
        "The **output** column is the correctness check, and it is deliberately not "
        "an exact match on the whole greedy chain. Reassociating a reduction moves a "
        "logit by about a BF16 ulp, and sixteen steps of greedy decoding turn one "
        "near-tie into a completely different sentence. So a configuration passes on "
        "its first token plus its top-5 logits staying within a tolerance that scales "
        "with their magnitude, and the table says where the chain split when it did. "
        "A configuration that changes precision is reported and never failed: its job "
        "is to be faster and cheaper, and how much answer that costs is the finding, "
        "not a bug.",
        "",
    ]
    if git:
        out += [f"Engine at `{git}`.", ""]
    return out


_MOE = next(ids for name, ids in FAMILIES if name.startswith("sparse MoE"))


def _roofline(
    measured: dict[str, dict[str, dict]], bandwidth: float
) -> list[str]:
    """How much of the memory bus each BF16 run reached, and so what is left.

    Decode at batch one reads every weight once and does two flops per byte, so
    ``params * 2 / bandwidth`` is a hard ceiling no kernel can beat. Without it a
    table of speedups cannot distinguish a kernel that gained nothing because it
    is badly written from one that gained nothing because the model was already
    against the wall — which is the honest reading of Qwen3.8 below.

    Only the BF16 rows: a packed row's bytes per weight are not two, and an MoE
    row reads its routed experts rather than all of them, so total parameters is
    the wrong divisor for both and the active count is not recorded here.
    """
    best: dict[str, tuple[float, float]] = {}
    for got in measured.values():
        for model, row in got.items():
            if row.get("quant") or row.get("recipe") in _MOE:
                continue
            params = row.get("params") or 0
            if not params:
                continue
            ceiling = bandwidth * 1e9 / (params * 2)
            keep = best.get(model)
            if keep is None or row["decode_tok_s"] > keep[0]:
                best[model] = (row["decode_tok_s"], ceiling)
    if not best:
        return []
    lines = [
        "## how much of the memory bus is left",
        "",
        f"Measured read bandwidth on this host is **{bandwidth} GB/s** "
        "(`scripts/bench_bandwidth.py`), against 273 GB/s on the specification. "
        "A BF16 decode step reads every weight once, so the ceiling below is "
        "`params x 2 bytes / bandwidth` and no kernel goes past it. Packed and "
        "MoE runs are left out: neither moves two bytes per parameter.",
        "",
        "| model | GB read per token | ceiling tok/s | best measured | of ceiling |",
        "|---|---|---|---|---|",
    ]
    # Least headroom first: that is the row a reader is about to ask why the
    # kernels did nothing for.
    for model, (got, ceiling) in sorted(best.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
        gb = bandwidth / ceiling
        lines.append(
            f"| {model} | {gb:.2f} | {ceiling:.1f} | {got} | {got / ceiling * 100:.0f}% |"
        )
    lines.append("")
    return lines


def _compare(
    tag: str, new: dict[str, dict], base: dict[str, dict], base_name: str
) -> tuple[list[str], list[str]]:
    """One tag's table against one baseline, plus the drift it should fail for."""
    lines = [
        f"## {tag} vs {base_name}",
        "",
        "| model | params | decode tok/s base | decode tok/s new | speedup "
        "| prefill tok/s base | prefill tok/s new | speedup | output |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    drifted: list[str] = []
    for model, row in sorted(new.items()):
        b = base.get(model)
        params = f"{(row.get('params') or 0) / 1e9:.2f}B"
        if b is None:
            # The pre-work revision cannot load every checkpoint here — some of
            # these recipes are part of the work — so "no before" is a real
            # answer for those, not a hole in the measurement.
            lines.append(
                f"| {model} | {params} | — | {row['decode_tok_s']} | — | — "
                f"| {row['prefill_tok_s']} | — | no {base_name} row |"
            )
            continue
        verdict, bad = _verdict(b, row)
        # Quantization changes the weights, so its output is expected to move; a
        # kernel or a captured graph has no such excuse.
        if bad and not row.get("quant"):
            drifted.append(f"{tag} vs {base_name}/{model}: {verdict}")
        lines.append(
            f"| {model} | {params} "
            f"| {b['decode_tok_s']} | {row['decode_tok_s']} "
            f"| {row['decode_tok_s'] / b['decode_tok_s']:.2f}x "
            f"| {b['prefill_tok_s']} | {row['prefill_tok_s']} "
            f"| {row['prefill_tok_s'] / b['prefill_tok_s']:.2f}x | {verdict} |"
        )
    lines.append("")
    return lines, drifted


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=ROOT / "bench" / "results.jsonl")
    p.add_argument("--baseline", default="baseline")
    p.add_argument("--tag", action="append", default=None, help="repeatable")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--bandwidth-gbs",
        type=float,
        default=247.3,
        help="measured read bandwidth; scripts/bench_bandwidth.py prints it "
        "(default is the GB10 figure, 91%% of the 273 GB/s specification)",
    )
    p.add_argument(
        "--against",
        default="baseline",
        help="second baseline, compared after the first; the anchor every model has",
    )
    p.add_argument(
        "--models",
        type=Path,
        default=Path.home() / "models",
        help="checkpoint folder, to tell a family this host lacks from one it skipped",
    )
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
    lines.extend(_preamble(args.baseline, tags, rows))
    lines.extend(coverage(rows, models_dir=args.models))
    lines.extend(_roofline(measured, args.bandwidth_gbs))
    for tag in tags:
        new = measured[tag]
        for model in sorted(expected - set(new)):
            absent.append(f"{tag}/{model}: no row (the run failed or was skipped)")
        if not new:
            absent.append(f"{tag}: produced no rows at all")
            continue
        body, moved = _compare(tag, new, base, args.baseline)
        lines.extend(body)
        drifted.extend(moved)

    # Then against the same code with the extension off, which is the only anchor
    # every checkpoint has. The pre-work revision cannot load the hybrid, the
    # sliding-window or the legacy families at all — support for several of them
    # is part of this work — so a table anchored only there leaves three of the
    # five families measured here with no before/after at all.
    second = args.against
    if second and second != args.baseline and second in {r.get("tag") for r in rows}:
        against = latest_by_model(rows, second)
        if against:
            lines += [
                f"# versus `{second}`",
                "",
                f"Same engine, same checkpoints, `{second}` as the before. This is "
                "what the kernels are worth on their own, and unlike the tables "
                "above it covers every model measured.",
                "",
            ]
            for tag in tags:
                if tag == second or not measured[tag]:
                    continue
                body, moved = _compare(tag, measured[tag], against, second)
                lines.extend(body)
                drifted.extend(moved)

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
