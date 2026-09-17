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
    "phi4",
    "phi3",
    "phi",
    "mixtral",
    "llama4",
    "gpt2",
    "gptj",
    "gpt_neo",
    "gpt_neox",
    "gpt_oss",
    "gpt_bigcode",
    "opt",
    "bloom",
    "falcon",
    "mpt",
    "deepseek_v2",
    "deepseek_v3",
    "nemotron_h",
    "qwen3_5",
    "granite",
    "granite_swa",
    "granitemoe",
    "granitemoe_swa",
    "granitemoeshared",
    "olmo",
    "olmo2",
    "olmo3",
    "olmoe",
    "flex_olmo",
    "smollm3",
    "starcoder2",
    "nemotron",
    "gemma2",
    "gemma3",
    "qwen3_moe",
    "qwen2_moe",
    "cohere",
    "glm",
    "glm4_moe",
    "stablelm",
    "exaone4",
    "exaone_moe",
    "arcee",
    "phimoe",
    "bitnet",
    "hunyuan_v1_moe",
    "ernie4_5_moe",
    "dbrx",
    "cohere2_moe",
    "diffllama",
    "olmo_hybrid",
    "qwen3_next",
    "qwen3_5_moe",
    "jamba",
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
    if mt in {"deepseek_v2", "deepseekv2"} or any(
        "deepseekv2" in a or "deepseek_v2" in a for a in arches
    ):
        return "deepseek_v2"
    if (
        mt.startswith("deepseek")
        or any("deepseek" in a for a in arches)
        or raw.get("q_lora_rank") is not None
        or raw.get("kv_lora_rank") is not None
    ):
        return "deepseek_v3"
    if mt in {"phimoe"} or any("phimoe" in a for a in arches):
        return "phimoe"
    if mt in {"phi4", "phi4_multimodal"} or any(
        "phi4" in a for a in arches
    ):
        return "phi4"
    if mt in {"phi3"} or any("phi3" in a for a in arches):
        return "phi3"
    if mt in {"phi"} or any(
        a in {"phiforcausallm", "phi"} or a.startswith("phifor") for a in arches
    ):
        return "phi"

    if mt in {"qwen3_next", "qwen3_next_text"} or any(
        "qwen3next" in a or "qwen3_next" in a for a in arches
    ):
        return "qwen3_next"
    if mt in {"qwen3_5_moe", "qwen3_5_moe_text"} or any(
        "qwen3_5" in a and "moe" in a for a in arches
    ):
        return "qwen3_5_moe"

    # Llama-identical families — before generic llama / qwen / gemma fallbacks.
    if (
        mt in {"helium"}
        or any("heliumforcausallm" in a or a == "helium" for a in arches)
        or any(a.startswith("helium") and "moe" not in a for a in arches)
    ):
        return "llama"
    if mt in {"hunyuan_v1_moe", "hunyuanmoev1"} or any(
        "hunyuanmoe" in a or "hunyuan_v1_moe" in a for a in arches
    ):
        return "hunyuan_v1_moe"
    if mt in {"hunyuan_v1_dense", "hunyuandensev1"} or any(
        "hunyuandense" in a or "hunyuan_v1_dense" in a for a in arches
    ):
        return "llama"
    if mt in {"seed_oss", "seedoss"} or any("seedoss" in a or "seed_oss" in a for a in arches):
        return "llama"
    if mt in {"cwm"} or any(a == "cwmforcausallm" or a.startswith("cwm") for a in arches):
        return "llama"
    if mt in {"ernie4_5_moe"} or any(
        "ernie4_5moe" in a or "ernie4_5_moe" in a for a in arches
    ):
        return "ernie4_5_moe"
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
        return "granitemoe_swa"
    if mt in {"olmo_hybrid", "olmohybrid"} or any(
        "olmohybrid" in a or "olmo_hybrid" in a for a in arches
    ):
        return "olmo_hybrid"
    if mt in {"flex_olmo", "flexolmo"} or any(
        "flexolmo" in a or "flex_olmo" in a for a in arches
    ):
        return "flex_olmo"
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
    if mt in {"gpt_bigcode", "gptbigcode", "starcoder"} or any(
        "gptbigcode" in a or "gpt_bigcode" in a for a in arches
    ):
        return "gpt_bigcode"
    if mt in {"nemotron"} or any(
        "nemotronforcausallm" in a and "nemotronh" not in a for a in arches
    ):
        return "nemotron"

    if mt in {"gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text"} or any(
        "gemma4" in a for a in arches
    ):
        raise UnsupportedRecipeError(
            "gemma4 uses per-layer embeddings (PLE) and optional KV sharing; not a gemma3 alias"
        )
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

    if mt in {"cohere2_moe"} or any("cohere2moe" in a or "cohere2_moe" in a for a in arches):
        return "cohere2_moe"
    if mt in {"cohere2"} or any("cohere2" in a for a in arches):
        return "cohere"
    if mt in {"cohere"} or any("cohereforcausallm" in a or a == "cohere" for a in arches):
        return "cohere"
    if mt in {"glm_moe_dsa"} or any("glmmoedsa" in a or "glm_moe_dsa" in a for a in arches):
        raise UnsupportedRecipeError(
            "glm_moe_dsa uses DSA attention and is not the glm4_moe recipe"
        )
    if mt in {"glm4_moe"} or any("glm4moe" in a or "glm4_moe" in a for a in arches):
        return "glm4_moe"
    if mt in {"glm", "glm4"} or any(
        a in {"glmforcausallm", "glm4forcausallm"} or (a.startswith("glm") and "moe" not in a)
        for a in arches
    ):
        return "glm"
    if mt in {"stablelm"} or any("stablelm" in a for a in arches):
        return "stablelm"
    if mt in {"exaone_moe"} or any("exaonemoe" in a or "exaone_moe" in a for a in arches):
        return "exaone_moe"
    if mt in {"exaone4_5", "exaone4_5_text"} or any("exaone4_5" in a for a in arches):
        return "exaone4"
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
        return "diffllama"
    if mt in {"qwen2_vl", "qwen2_vl_text", "qwen2_5_vl", "qwen2_5_vl_text"} or any(
        "qwen2vl" in a or "qwen2_vl" in a or "qwen2_5_vl" in a for a in arches
    ):
        return "qwen2"
    if mt in {"internvl"} or any("internvl" in a for a in arches):
        text_mt = ""
        text = raw.get("text_config")
        if isinstance(text, dict):
            text_mt = str(text.get("model_type") or "").lower().replace("-", "_")
        if text_mt.startswith("qwen"):
            return "qwen2"
        if text_mt in {"llama", "internlm", "internlm2", "internlm3"}:
            return "llama"
        return "qwen2"
    if mt in {"kimi_k25", "kimi_k2", "moonshot"} or any(
        "kimik25" in a or "kimi_k25" in a for a in arches
    ):
        return "deepseek_v3"
    if mt in {"qwen3"} or any("qwen3" in a for a in arches):
        return "qwen3"
    if mt.startswith("qwen") or any("qwen" in a for a in arches):
        return "qwen2"
    if mt in {"yi"} or any(a.startswith("yi") for a in arches):
        return "yi"
    if mt in {"mistral"} or any("mistral" in a and "mixtral" not in a for a in arches):
        return "mistral"
    if mt in {"gptj", "gpt_j"} or any("gptj" in a or "gpt_j" in a for a in arches):
        return "gptj"
    if mt in {"gpt_neo", "gptneo"} or any(
        ("gptneo" in a or "gpt_neo" in a) and "neox" not in a for a in arches
    ):
        return "gpt_neo"
    if mt in {"gpt2", "gpt_2"} or any(a.startswith("gpt2") for a in arches):
        return "gpt2"
    if mt in {"gpt_neox", "gptneox"} or any("gptneox" in a or "gpt_neox" in a for a in arches):
        return "gpt_neox"
    if mt in {"opt"} or any(a.startswith("opt") for a in arches):
        return "opt"
    if mt in {"bloom"} or any("bloom" in a for a in arches):
        return "bloom"
    if mt in {"falcon"} or any("falcon" in a and "mamba" not in a and "h1" not in a for a in arches):
        return "falcon"
    if mt in {"mpt"} or any(a.startswith("mpt") for a in arches):
        return "mpt"
    if mt in {"bitnet"} or any("bitnet" in a for a in arches):
        return "bitnet"
    if mt in {"jamba"} or any("jamba" in a for a in arches):
        return "jamba"
    if mt in {"minimax", "minimax_m2"} or any("minimax" in a for a in arches):
        raise UnsupportedRecipeError("minimax lightning attention is not implemented")
    if mt in {"dbrx"} or any("dbrx" in a for a in arches):
        return "dbrx"
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
        or raw.get("num_mtp_layers")
    ) or ("mtp" in name)
    return nvfp4, fp8, mtp


MTP_DECODE_RECIPES = frozenset({"qwen3_5", "deepseek_v3", "deepseek_v2", "nemotron_h"})
VISION_FAMILIES = frozenset(
    {"qwen3_5", "qwen2_vl", "qwen2_5_vl", "gemma3", "llama4", "mistral3"}
)


def detect_vision_family(raw: dict[str, Any], recipe_id: str | None = None) -> str | None:
    """Return the implemented vision tower family, or None if we cannot run it."""
    if recipe_id is None:
        recipe_id = detect_recipe_id(raw)
    if not raw.get("vision_config") or raw.get("language_model_only"):
        return None
    mt = _mt(raw)
    arches = _arches(raw)
    if recipe_id == "qwen3_5":
        return "qwen3_5"
    if mt in {"qwen2_vl", "qwen2_vl_text"} or any(
        "qwen2vl" in a and "qwen2_5" not in a for a in arches
    ):
        return "qwen2_vl"
    if mt in {"qwen2_5_vl", "qwen2_5_vl_text"} or any("qwen2_5_vl" in a for a in arches):
        return "qwen2_5_vl"
    if recipe_id == "gemma3":
        return "gemma3"
    if recipe_id == "llama4":
        return "llama4"
    if mt in {"mistral3"} or any("mistral3" in a for a in arches):
        return "mistral3"
    return None


def detect_missing(
    raw: dict[str, Any], folder_name: str = ""
) -> tuple[str, ...]:
    """Runtime pieces advertised but not implemented for the detected recipe."""
    _nvfp4, _fp8, mtp = detect_quant_flags(raw, folder_name)
    recipe_id = detect_recipe_id(raw)
    missing: list[str] = []
    if mtp and recipe_id not in MTP_DECODE_RECIPES:
        missing.append("mtp_decode")
    if (
        raw.get("vision_config")
        and not raw.get("language_model_only")
        and detect_vision_family(raw, recipe_id) is None
    ):
        missing.append("vision")
    return tuple(missing)


def can_run(recipe_id: str, missing: tuple[str, ...]) -> bool:
    """True if we can load weights (including NVFP4/FP8 dequant) and greedy-generate.

    MTP missing does not block greedy.
    """
    del missing
    return recipe_id in KNOWN_RECIPES
