"""Auto-detect which recipe a model folder uses.

Agents never register models. They pass a directory; we read config.json
(or GGUF metadata) and pick a recipe the engine already implements.
"""

from __future__ import annotations

from typing import Any

KNOWN_RECIPES = (
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
    "yi",
    "gemma",
    "phi3",
    "mixtral",
    "llama4",
    "gpt2",
    "gpt_neox",
    "gpt_oss",
    "deepseek_v3",
    "nemotron_h",
)


class UnsupportedRecipeError(ValueError):
    """config.json is a family this engine cannot run yet."""


def _arches(raw: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(a).lower().replace("-", "_") for a in (raw.get("architectures") or []))


def _mt(raw: dict[str, Any]) -> str:
    return str(raw.get("model_type") or "").lower().replace("-", "_")


def detect_recipe_id(raw: dict[str, Any]) -> str:
    """Return a KNOWN_RECIPES id from a parsed config.json (or flattened GGUF meta)."""
    mt = _mt(raw)
    arches = _arches(raw)
    blob = " ".join((mt, *arches, str(raw.get("general.architecture") or "")))

    if (
        mt in {"nemotron_h", "nemotronh"}
        or any("nemotronh" in a or "nemotron_h" in a for a in arches)
        or raw.get("hybrid_override_pattern")
    ):
        return "nemotron_h"

    if mt in {"mixtral"} or any("mixtral" in a for a in arches):
        return "mixtral"
    if mt in {"llama4", "llama4_text"} or any("llama4" in a for a in arches):
        return "llama4"
    if mt in {"gpt_oss", "gptoss"} or any("gptoss" in a or "gpt_oss" in a for a in arches):
        return "gpt_oss"
    if (
        mt.startswith("deepseek")
        or any("deepseek" in a for a in arches)
        or raw.get("q_lora_rank") is not None
        or raw.get("kv_lora_rank") is not None
    ):
        return "deepseek_v3"
    if mt in {"phi3", "phi"} or any(a.startswith("phi") for a in arches):
        return "phi3"
    if mt.startswith("gemma") or any("gemma" in a for a in arches):
        return "gemma"
    if mt.startswith("qwen") and ("moe" in mt or any("moe" in a for a in arches)):
        return "mixtral"
    if mt in {"qwen3"} or any("qwen3" in a for a in arches):
        return "qwen3"
    if mt.startswith("qwen") or any("qwen" in a for a in arches):
        return "qwen2"
    if mt in {"yi"} or any(a.startswith("yi") for a in arches):
        return "yi"
    if mt in {"mistral"} or any("mistral" in a and "mixtral" not in a for a in arches):
        return "mistral"
    if mt in {"gpt2", "gpt_2"} or any(a.startswith("gpt2") for a in arches):
        return "gpt2"
    if mt in {"gpt_neox", "gptneox"} or any("gptneox" in a or "gpt_neox" in a for a in arches):
        return "gpt_neox"
    if mt == "llama" or any(a.startswith("llama") for a in arches):
        return "llama"
    if "llama" in blob:
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
    fp8 = ("fp8" in blob and not nvfp4) or str(raw.get("torch_dtype", "")).lower() in {
        "float8_e4m3fn",
        "float8_e5m2",
        "fp8",
    }
    mtp = bool(raw.get("num_nextn_predict_layers") or raw.get("mtp_num_layers")) or (
        "mtp" in name
    )
    return nvfp4, fp8, mtp


def detect_missing(
    raw: dict[str, Any], folder_name: str = ""
) -> tuple[str, ...]:
    """Runtime pieces advertised but not on the greedy path.

    NVFP4/FP8 dequant-on-load is implemented, so those no longer block.
    MTP speculative decode is still not wired; greedy uses the main LM head.
    """
    _nvfp4, _fp8, mtp = detect_quant_flags(raw, folder_name)
    missing: list[str] = []
    if mtp:
        missing.append("mtp_decode")
    return tuple(missing)


def can_run(recipe_id: str, missing: tuple[str, ...]) -> bool:
    """True if we can load weights (including NVFP4/FP8 dequant) and greedy-generate.

    MTP missing does not block greedy.
    """
    del missing
    return recipe_id in KNOWN_RECIPES
