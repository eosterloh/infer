"""Auto-detect which recipe a model folder uses.

Agents never register models. They pass a directory; we read config.json
and pick a recipe the engine already implements (llama or nemotron_h).
"""

from __future__ import annotations

from typing import Any

KNOWN_RECIPES = ("llama", "nemotron_h")


class UnsupportedRecipeError(ValueError):
    """config.json is a family this engine cannot run yet."""


def detect_recipe_id(raw: dict[str, Any]) -> str:
    """Return 'llama' or 'nemotron_h' from a parsed config.json dict."""
    mt = str(raw.get("model_type") or "").lower().replace("-", "_")
    arches = tuple(str(a).lower() for a in (raw.get("architectures") or []))

    if (
        mt in {"nemotron_h", "nemotronh"}
        or any("nemotronh" in a or "nemotron_h" in a for a in arches)
        or raw.get("hybrid_override_pattern")
    ):
        return "nemotron_h"

    if mt == "llama" or any(a.startswith("llama") for a in arches):
        return "llama"

    raise UnsupportedRecipeError(
        f"this folder's config is model_type={raw.get('model_type')!r} "
        f"architectures={list(raw.get('architectures') or [])}; "
        f"engine can run: {', '.join(KNOWN_RECIPES)}"
    )


def detect_quant_flags(
    raw: dict[str, Any], folder_name: str = ""
) -> tuple[bool, bool, bool]:
    """Return (nvfp4, fp8, mtp) advertised by config or directory name."""
    name = folder_name.lower()
    quant = raw.get("quantization_config")
    quant_s = str(quant).lower() if quant is not None else ""
    quant_method = str(raw.get("quant_method", "")).lower()
    blob = f"{quant_s} {quant_method} {name}"

    nvfp4 = "nvfp4" in blob or "fp4" in blob
    fp8 = "fp8" in blob and not nvfp4
    mtp = bool(raw.get("num_nextn_predict_layers") or raw.get("mtp_num_layers")) or (
        "mtp" in name
    )
    return nvfp4, fp8, mtp


def detect_missing(
    raw: dict[str, Any], folder_name: str = ""
) -> tuple[str, ...]:
    """Runtime pieces advertised but not implemented yet."""
    nvfp4, fp8, mtp = detect_quant_flags(raw, folder_name)
    missing: list[str] = []
    if nvfp4:
        missing.append("nvfp4_runtime")
    if fp8:
        missing.append("fp8_runtime")
    if mtp:
        missing.append("mtp_decode")
    return tuple(missing)


def can_run(recipe_id: str, missing: tuple[str, ...]) -> bool:
    """True if we can load BF16/FP16/FP32 weights and greedy-generate.

    MTP missing does not block greedy. NVFP4/FP8 load paths are not wired yet.
    """
    if recipe_id not in KNOWN_RECIPES:
        return False
    blocking = {"nvfp4_runtime", "fp8_runtime"}
    return not any(m in blocking for m in missing)
