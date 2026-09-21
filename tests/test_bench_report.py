"""The gate that decides whether a speedup is allowed to count.

bench_report.py --strict is the last check in the Spark verification, so it has
to fail on a kernel that changed the answer and pass on arithmetic that merely
reassociated. Both halves matter: a gate that fires on BF16 noise gets disabled
the first time it cries wolf, and one that never fires proves nothing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bench_report", ROOT / "scripts" / "bench_report.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def _row(ids: list[int], top5: list[int], logits: list[float], **extra) -> dict:
    row = {
        "tag": extra.pop("tag", "kernels"),
        "model": "m",
        "decode_tok_s": 10.0,
        "prefill_tok_s": 100.0,
        "fingerprint": {"greedy_ids": ids, "top5_ids": top5, "top5_logits": logits},
    }
    row.update(extra)
    return row


BASE = _row([5, 6, 7, 8], [5, 6, 7, 8, 9], [20.0, 19.5, 18.0, 17.0, 16.0], tag="origin")


def test_identical_output_passes() -> None:
    label, bad = report._verdict(BASE, BASE)
    assert not bad and label.startswith("same")


def test_reassociated_arithmetic_passes() -> None:
    """Small logit moves, and a greedy chain that splits later, are not failures."""
    wobble = _row([5, 6, 99, 100], [5, 6, 7, 8, 9], [20.02, 19.48, 18.01, 17.0, 15.99])
    label, bad = report._verdict(BASE, wobble)
    assert not bad
    assert "chain splits at 2/4" in label


def test_a_changed_first_token_fails() -> None:
    other = _row([6, 6, 7, 8], [5, 6, 7, 8, 9], [20.0, 19.5, 18.0, 17.0, 16.0])
    label, bad = report._verdict(BASE, other)
    assert bad and "first token" in label


def test_a_moved_logit_fails_even_when_the_token_survives() -> None:
    """The token can be right by luck; the logits say whether the math was."""
    skewed = _row([5, 6, 7, 8], [5, 6, 7, 8, 9], [20.0, 19.5, 18.0, 17.0, 14.0])
    label, bad = report._verdict(BASE, skewed)
    assert bad and "Δlogit" in label


def test_a_swap_at_the_cut_passes_but_a_reordering_above_it_fails() -> None:
    """Which of two equal logits makes rank five is arbitrary; rank two is not."""
    at_the_cut = _row([5, 6, 7, 8], [5, 6, 7, 8, 11], [20.0, 19.5, 18.0, 17.0, 16.0])
    label, bad = report._verdict(BASE, at_the_cut)
    assert not bad, label

    above_it = _row([5, 6, 7, 8], [5, 11, 7, 8, 9], [20.0, 19.5, 18.0, 17.0, 16.0])
    label, bad = report._verdict(BASE, above_it)
    assert bad and "clear of the cut" in label


def test_tolerance_tracks_magnitude() -> None:
    assert report._tolerance([1.0]) < report._tolerance([100.0])
    assert report._tolerance([]) > 0


def test_strict_run_fails_on_drift(tmp_path: Path, capsys, monkeypatch) -> None:
    """End to end: a drifting tag makes the report exit non-zero."""
    import json

    results = tmp_path / "results.jsonl"
    bad = _row([6, 6, 7, 8], [5, 6, 7, 8, 9], [20.0, 19.5, 18.0, 17.0, 16.0])
    with results.open("w") as fh:
        for row in (BASE, bad):
            fh.write(json.dumps(row) + "\n")

    monkeypatch.setattr(
        "sys.argv",
        ["bench_report", "--results", str(results), "--baseline", "origin", "--strict"],
    )
    assert report.main() == 1
    assert "output drift" in capsys.readouterr().out

    quantized = dict(bad)
    quantized["quant"] = "nvfp4"
    with results.open("w") as fh:
        for row in (BASE, quantized):
            fh.write(json.dumps(row) + "\n")
    assert report.main() == 0, "quantization is allowed to change the answer"


def test_family_coverage_names_every_kernel_group() -> None:
    """A sweep that missed a family should say so, not stay quiet about it."""
    assert report.families_for("nemotron_h") == [
        "hybrid recurrent (Mamba2 / GDN scan)",
        "sparse MoE dispatch",
    ]
    assert report.families_for("llama") == ["dense attention + MLP"]
    assert report.families_for("deepseek_v3") == ["MLA (compressed KV cache)"]
    assert "sliding window / soft-capped attention" in report.families_for("gemma3")

    rows = [
        dict(BASE, model="llama1b", recipe="llama", tag="kernels"),
        dict(BASE, model="nano30b", recipe="nemotron_h", tag="kernels"),
    ]
    text = "\n".join(report.coverage(rows, "kernels"))
    assert "| dense attention + MLP | llama1b |" in text
    assert "nano30b" in text
    assert "| MLA (compressed KV cache) | **none** |" in text


def test_strict_fails_when_a_tag_has_no_row(tmp_path, capsys, monkeypatch) -> None:
    """A configuration that crashed on a model must not read as a clean sweep.

    No row means nothing appears in the table, so without this the loudest
    possible failure — a model that does not run at all — is the quietest.
    """
    import json

    ids, top5, logits = [5, 6], [5, 6, 7, 8, 9], [20.0, 19.5, 18.0, 17.0, 16.0]
    rows = []
    for tag, models in (
        ("origin", ("alpha", "beta")),
        ("kernels", ("alpha", "beta")),
        ("graph", ("alpha",)),
    ):
        for model in models:
            rows.append(_row(ids, top5, logits, tag=tag, model=model))
    results = tmp_path / "results.jsonl"
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_report",
            "--results", str(results),
            "--baseline", "origin",
            "--tag", "kernels",
            "--tag", "graph",
            "--strict",
        ],
    )
    code = report.main()
    printed = capsys.readouterr().out
    assert code == 1
    assert "graph/beta" in printed, "the missing model has to be named"
    assert "kernels/beta" not in printed


def test_a_tie_at_the_bottom_of_the_top5_is_not_drift() -> None:
    """Rank five trading places between two near-equal tokens says nothing.

    These are the real numbers from Qwen2.5-1.5B on the Spark: the first four
    logits identical to the bit, and the fifth held by two different tokens
    0.125 apart — one BF16 ulp at that magnitude. Failing the run for that would
    mean failing it for arithmetic.
    """
    base = _row(
        [21718, 13, 15277], [21718, 32671, 1083, 1304, 30743],
        [22.125, 19.0, 19.0, 18.75, 17.875], tag="origin",
    )
    kernels = _row(
        [21718, 13, 15277], [21718, 32671, 1083, 1304, 47506],
        [22.125, 19.0, 19.0, 18.75, 18.0],
    )
    label, bad = report._verdict(base, kernels)
    assert not bad, label


def test_a_token_that_should_have_made_the_list_is_drift() -> None:
    """The cut only excuses a tie; a logit well clear of it is a real change."""
    base = _row(
        [5, 6], [5, 6, 7, 8, 9], [20.0, 19.5, 18.0, 17.0, 16.0], tag="origin"
    )
    intruder = _row([5, 6], [5, 6, 7, 8, 99], [20.0, 19.5, 18.0, 17.0, 19.2])
    label, bad = report._verdict(base, intruder)
    assert bad, label
