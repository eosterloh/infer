"""The gate that decides whether a speedup is allowed to count.

bench_report.py --strict is the last check in the Spark verification, so it has
to fail on a kernel that changed the answer and pass on arithmetic that merely
reassociated. Both halves matter: a gate that fires on BF16 noise gets disabled
the first time it cries wolf, and one that never fires proves nothing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

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


def test_an_empty_family_says_whether_this_host_could_have_measured_it(
    tmp_path: Path,
) -> None:
    """"none" is two different results and only one of them is a gap.

    This host has no DeepSeek checkpoint, so MLA going unmeasured is a fact about
    the machine. A Gemma-2 sitting in the same folder with no row is a sweep that
    skipped a family it had, which the per-tag row check cannot see: it holds
    every tag to the models *some* tag managed, and a model no tag ever loaded is
    absent from that set too.
    """
    models = tmp_path / "models"
    for name, model_type in (("gemma-2-2b-it", "gemma2"), ("llama1b", "llama")):
        (models / name).mkdir(parents=True)
        (models / name / "config.json").write_text(f'{{"model_type": "{model_type}"}}')
    (models / "not-a-checkpoint").mkdir()

    rows = [dict(BASE, model="llama1b", recipe="llama", tag="kernels")]
    text = "\n".join(report.coverage(rows, "kernels", models_dir=models))
    assert "gemma-2-2b-it on disk, no row" in text
    assert "| MLA (compressed KV cache) | *no checkpoint of this family on this host* |" in text
    assert report.checkpoints_on_disk(models) == {
        "gemma-2-2b-it": "gemma2",
        "llama1b": "llama",
    }
    assert report.checkpoints_on_disk(tmp_path / "nowhere") == {}


def test_the_second_anchor_covers_what_the_first_cannot_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A model with no origin row still needs a measured before/after.

    The pre-work revision cannot load the hybrid, sliding-window or legacy
    checkpoints — support for several of them is this work — so anchoring only
    there leaves most of the families with a dash where their speedup goes.
    """
    rows = [
        # Only the dense model exists at origin.
        dict(BASE, model="llama1b", recipe="llama", tag="origin", decode_tok_s=10.0),
        dict(BASE, model="llama1b", recipe="llama", tag="baseline", decode_tok_s=11.0),
        dict(BASE, model="nano30b", recipe="nemotron_h", tag="baseline", decode_tok_s=9.0),
        dict(BASE, model="llama1b", recipe="llama", tag="kernels", decode_tok_s=22.0),
        dict(BASE, model="nano30b", recipe="nemotron_h", tag="kernels", decode_tok_s=27.0),
    ]
    results = tmp_path / "results.jsonl"
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "REPORT.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_report.py", "--results", str(results), "--baseline", "origin",
            "--tag", "baseline", "--tag", "kernels", "--out", str(out), "--strict",
        ],
    )
    assert report.main() == 0
    text = out.read_text()

    # Against origin the hybrid has no before, which the table states plainly.
    assert "| nano30b | 0.00B | — | 27.0 | — |" in text
    # Against the anchor every model has, it gets its speedup: 9 -> 27.
    assert "## kernels vs baseline" in text
    assert "| nano30b | 0.00B | 9.0 | 27.0 | 3.00x" in text
    assert "| llama1b | 0.00B | 11.0 | 22.0 | 2.00x" in text


def test_the_ceiling_says_a_flat_speedup_was_already_at_the_wall() -> None:
    """A model reading 55 GB per token cannot beat 4.45 tok/s, kernels or not."""
    measured = {
        "kernels": {
            # 27.78B bf16 = 55.6 GB/token, so ~4.5 tok/s is the whole budget.
            "big": dict(BASE, model="big", params=27.78e9, decode_tok_s=4.46),
            "small": dict(BASE, model="small", params=1.24e9, decode_tok_s=78.5),
            # Neither of these moves two bytes per parameter.
            "packed": dict(BASE, model="packed", params=1.5e9, decode_tok_s=173.0, quant="nvfp4"),
            "moe": dict(BASE, model="moe", params=31.6e9, recipe="nemotron_h", decode_tok_s=32.2),
        }
    }
    lines = report._roofline(measured, 247.3)
    text = "\n".join(lines)

    assert "| big | 55.56 | 4.5 | 4.46 | 100% |" in text
    assert "| small | 2.48 | 99.7 | 78.5 | 79% |" in text
    assert "packed" not in text and "moe" not in text
    # The one with the least headroom is listed first, since that is the one a
    # reader is about to ask why the kernels did nothing for.
    assert text.index("| big |") < text.index("| small |")


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


def test_a_quantized_row_reports_its_drift_without_failing() -> None:
    """NVFP4 changes the answer by construction; that is not a regression.

    Holding a 4-bit run to the bf16 logits fails the one configuration the
    quantized path exists to ship, so the verdict measures the drift and the
    gate stays on the runs that claim to be arithmetically equivalent.
    """
    base = _row([5, 6, 7, 8], [5, 6, 7, 8, 9], [9.0, 8.0, 7.0, 6.0, 5.0])
    new = _row([5, 6, 1, 2], [5, 6, 7, 8, 9], [9.5, 8.1, 7.2, 6.4, 5.1], quant="nvfp4")
    label, bad = report._verdict(base, new)
    assert not bad
    assert "nvfp4" in label and "2/4 tokens kept" in label


def test_the_same_drift_at_the_same_precision_still_fails() -> None:
    base = _row([5, 6, 7, 8], [5, 6, 7, 8, 9], [9.0, 8.0, 7.0, 6.0, 5.0])
    new = _row([5, 6, 1, 2], [5, 6, 7, 8, 9], [9.5, 8.1, 7.2, 6.4, 5.1])
    _, bad = report._verdict(base, new)
    assert bad
