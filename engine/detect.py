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
    "phi",
    "mixtral",
    "llama4",
    "gpt2",
    "gpt_neox",
    "gpt_oss",
    "deepseek_v3",
    "nemotron_h",
    "qwen3_5",
    "granite",
    "granite_swa",
    "granitemoe",
    "granitemoeshared",
    "olmo",
    "olmo2",
    "olmo3",
    "olmoe",
    "smollm3",
    "starcoder2",
    "nemotron",
    "gemma2",
    "gemma3",
    "qwen3_moe",
    "qwen2_moe",
    "cohere",
    "glm",
    "stablelm",
    "exaone4",
    "arcee",
)


class UnsupportedRecipeError(ValueError):
    """config.json is a family this engine cannot run yet."""


def _arches(raw: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(a).lower().replace("-", "_") for a in (raw.get("architectures") or []))


def _mt(raw: dict[str, Any]) -> str:
    return str(raw.get("model_type") or "").lower().replace("-", "_")


def _has_moe_hint(mt: str, arches: tuple[str, ...]) -> bool:
    blob = " ".join((mt, *arches))
    return "moe" in blob


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
    if mt in {"phimoe"} or any("phimoe" in a for a in arches):
        raise UnsupportedRecipeError("phimoe is not implemented")
    if mt in {"phi3"} or any("phi3" in a for a in arches):
        return "phi3"
    if mt in {"phi"} or any(
        a in {"phiforcausallm", "phi"} or a.startswith("phifor") for a in arches
    ):
        return "phi"

    if mt in {"qwen3_next", "qwen3_next_text"} or any(
        "qwen3next" in a or "qwen3_next" in a for a in arches
    ):
        raise UnsupportedRecipeError(
            "qwen3-next uses fused GDN projections and is not the qwen3_5 recipe"
        )
    if mt in {"qwen3_5_moe", "qwen3_5_moe_text"} or any(
        "qwen3_5" in a and "moe" in a for a in arches
    ):
        raise UnsupportedRecipeError(
            "qwen3_5 MoE scheduling and expert weights are not implemented"
        )

    # Llama-identical families — before generic llama / qwen / gemma fallbacks.
    if (
        mt in {"helium"}
        or any("heliumforcausallm" in a or a == "helium" for a in arches)
        or any(a.startswith("helium") and "moe" not in a for a in arches)
    ):
        return "llama"
    if mt in {"hunyuan_v1_dense", "hunyuandensev1"} or any(
        "hunyuandense" in a or "hunyuan_v1_dense" in a for a in arches
    ):
        return "llama"
    if mt in {"seed_oss", "seedoss"} or any("seedoss" in a or "seed_oss" in a for a in arches):
        return "llama"
    if mt in {"cwm"} or any(a == "cwmforcausallm" or a.startswith("cwm") for a in arches):
        return "llama"
    if mt in {"ernie4_5", "ernie_4_5"} or any(
        "ernie4_5forcausallm" in a or a.startswith("ernie4_5") for a in arches
    ):
        if not _has_moe_hint(mt, arches):
            return "llama"
    if mt in {"ministral", "ministral3"} or any(
        "ministral" in a and "mixtral" not in a for a in arches
    ):
        return "mistral"
    if mt in {"mistral3"} or any("mistral3" in a for a in arches):
        return "mistral"
    if mt in {"arcee"} or any("arceeforcausallm" in a or a == "arcee" for a in arches):
        return "arcee"

    if mt in {"granitemoehybrid", "granite_moehybrid"} or any(
        "granitemoehybrid" in a or "granite_moehybrid" in a for a in arches
    ):
        raise UnsupportedRecipeError(
            "granitemoehybrid is a Mamba hybrid and is not the granitemoe recipe"
        )
    if mt in {"granitemoe_swa", "granitemoeswa"} or any(
        "granitemoe_swa" in a or "granitemoeswa" in a for a in arches
    ):
        raise UnsupportedRecipeError("granitemoe_swa is not implemented")
    if mt in {"olmo_hybrid", "olmohybrid", "flex_olmo", "flexolmo"} or any(
        "olmohybrid" in a or "flexolmo" in a or "flex_olmo" in a for a in arches
    ):
        raise UnsupportedRecipeError("olmo hybrid / FlexOlmo is not implemented")
    if mt in {"granitemoeshared"} or any("granitemoeshared" in a for a in arches):
        return "granitemoeshared"
    if mt in {"granitemoe"} or any("granitemoe" in a and "shared" not in a and "hybrid" not in a for a in arches):
        return "granitemoe"
    if mt in {"granite_swa", "graniteswa"} or any(
        "graniteswa" in a or "granite_swa" in a for a in arches
    ):
        return "granite_swa"
    if mt in {"granite"} or any("graniteforcausallm" in a or a == "granite" for a in arches):
        return "granite"
    if mt in {"olmoe"} or any("olmoe" in a for a in arches):
        return "olmoe"
    if mt in {"olmo3"} or any("olmo3" in a for a in arches):
        return "olmo3"
    if mt in {"olmo2"} or any("olmo2" in a for a in arches):
        return "olmo2"
    if mt in {"olmo"} or any(a.startswith("olmo") and "olmo2" not in a and "moe" not in a for a in arches):
        return "olmo"
    if mt in {"smollm3", "smol_lm3"} or any("smollm3" in a for a in arches):
        return "smollm3"
    if mt in {"starcoder2", "starcoder_2"} or any("starcoder2" in a for a in arches):
        return "starcoder2"
    if mt in {"nemotron"} or any(
        "nemotronforcausallm" in a and "nemotronh" not in a for a in arches
    ):
        return "nemotron"

    if mt in {"gemma3", "gemma3_text"} or any(
        "gemma3forcausallm" in a
        or "gemma3forconditionalgeneration" in a
        or a.startswith("gemma3")
        for a in arches
    ):
        return "gemma3"
    if mt in {"gemma2", "gemma2_text"} or any("gemma2" in a for a in arches):
        return "gemma2"
    if mt in {"gemma"} or any(a in {"gemmaforcausallm", "gemma"} or a.startswith("gemmafor") for a in arches):
        return "gemma"

    if mt in {"qwen3_moe"} or any("qwen3moe" in a or "qwen3_moe" in a for a in arches):
        return "qwen3_moe"
    if mt in {"qwen2_moe"} or any("qwen2moe" in a or "qwen2_moe" in a for a in arches):
        return "qwen2_moe"

    if mt in {"cohere2"} or any("cohere2" in a for a in arches):
        return "cohere"
    if mt in {"cohere"} or any("cohereforcausallm" in a or a == "cohere" for a in arches):
        return "cohere"
    if mt in {"glm", "glm4"} or any(
        a in {"glmforcausallm", "glm4forcausallm"} or (a.startswith("glm") and "moe" not in a)
        for a in arches
    ):
        return "glm"
    if mt in {"stablelm"} or any("stablelm" in a for a in arches):
        return "stablelm"
    if mt in {"exaone4"} or any("exaone4" in a for a in arches):
        return "exaone4"

    if (
        mt in {"qwen3_5", "qwen3_5_text"}
        or any("qwen3_5" in a for a in arches)
        or (
            not mt
            and isinstance(raw.get("layer_types"), list)
            and "linear_attention" in raw["layer_types"]
        )
    ):
        return "qwen3_5"
    if mt.startswith("qwen") and ("moe" in mt or any("moe" in a for a in arches)):
        return "mixtral"
    if mt in {"diffllama"} or any("diffllama" in a for a in arches):
        raise UnsupportedRecipeError(
            "diffllama uses differential attention and is not the llama recipe"
        )
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
    mtp = bool(
        raw.get("num_nextn_predict_layers")
        or raw.get("mtp_num_layers")
        or raw.get("mtp_num_hidden_layers")
    ) or ("mtp" in name)
    return nvfp4, fp8, mtp


def detect_missing(
    raw: dict[str, Any], folder_name: str = ""
) -> tuple[str, ...]:
    """Runtime pieces advertised but not implemented for the detected recipe."""
    _nvfp4, _fp8, mtp = detect_quant_flags(raw, folder_name)
    recipe_id = detect_recipe_id(raw)
    missing: list[str] = []
    if mtp and recipe_id != "qwen3_5":
        missing.append("mtp_decode")
    if (
        raw.get("vision_config")
        and not raw.get("language_model_only")
        and recipe_id != "qwen3_5"
    ):
        missing.append("vision")
    return tuple(missing)


def can_run(recipe_id: str, missing: tuple[str, ...]) -> bool:
    """True if we can load weights (including NVFP4/FP8 dequant) and greedy-generate.

    MTP missing does not block greedy.
    """
    del missing
    return recipe_id in KNOWN_RECIPES
