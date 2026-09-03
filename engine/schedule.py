"""Layer schedule — which mixer / FFN each layer uses.

Llama: every layer is attention + dense MLP.
Nemotron-H: hybrid_override_pattern — one block per layer:
  M = Mamba2, * = Attention, E = MoE, - = dense MLP.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.config import ModelConfig


class MixerKind(str, Enum):
    ATTENTION = "attention"
    MAMBA2 = "mamba2"
    GATED_DELTANET = "gated_deltanet"
    NONE = "none"


class FfnKind(str, Enum):
    DENSE_MLP = "dense_mlp"
    MOE = "moe"
    NONE = "none"


@dataclass(frozen=True)
class LayerSpec:
    """One decoder layer's recipe."""

    index: int
    mixer: MixerKind
    ffn: FfnKind

    def summary(self) -> str:
        return f"{self.index}:{self.mixer.value}+{self.ffn.value}"


def llama_dense_schedule(num_layers: int) -> tuple[LayerSpec, ...]:
    """Classic Llama / dense Transformer: Attn + SwiGLU every layer."""
    return tuple(
        LayerSpec(i, MixerKind.ATTENTION, FfnKind.DENSE_MLP) for i in range(num_layers)
    )


def hybrid_pattern_schedule(pattern: str) -> tuple[LayerSpec, ...]:
    """Parse Nemotron-H hybrid_override_pattern into LayerSpecs.

    Pattern chars (len == num_hidden_layers):
      M → mamba2 only
      * → attention only
      E → moe only
      - → dense mlp only
    """
    specs: list[LayerSpec] = []
    for i, ch in enumerate(pattern):
        if ch == "M":
            specs.append(LayerSpec(i, MixerKind.MAMBA2, FfnKind.NONE))
        elif ch == "*":
            specs.append(LayerSpec(i, MixerKind.ATTENTION, FfnKind.NONE))
        elif ch == "E":
            specs.append(LayerSpec(i, MixerKind.NONE, FfnKind.MOE))
        elif ch == "-":
            specs.append(LayerSpec(i, MixerKind.NONE, FfnKind.DENSE_MLP))
        else:
            raise ValueError(
                f"unknown hybrid_override_pattern char {ch!r} at index {i}; "
                "expected one of M, *, E, -"
            )
    return tuple(specs)


def layer_types_schedule(layer_types: list[str] | tuple[str, ...]) -> tuple[LayerSpec, ...]:
    """Qwen3.5 / Qwen3.8 hybrid: layer_types[i] is linear_attention or full_attention."""
    specs: list[LayerSpec] = []
    for i, kind in enumerate(layer_types):
        k = str(kind).lower().replace("-", "_")
        if k in {"linear_attention", "gated_deltanet", "linear"}:
            specs.append(LayerSpec(i, MixerKind.GATED_DELTANET, FfnKind.DENSE_MLP))
        elif k in {"full_attention", "attention", "full"}:
            specs.append(LayerSpec(i, MixerKind.ATTENTION, FfnKind.DENSE_MLP))
        else:
            raise ValueError(
                f"unknown layer_types[{i}]={kind!r}; "
                "expected linear_attention or full_attention"
            )
    return tuple(specs)


def dense_or_moe_schedule(
    num_layers: int, moe_layers: set[int]
) -> tuple[LayerSpec, ...]:
    return tuple(
        LayerSpec(
            i,
            MixerKind.ATTENTION,
            FfnKind.MOE if i in moe_layers else FfnKind.DENSE_MLP,
        )
        for i in range(num_layers)
    )


def build_schedule(config: ModelConfig) -> tuple[LayerSpec, ...]:
    """Build layer schedule from already-detected recipe + config fields."""
    if config.layers:
        return config.layers

    recipe = getattr(config, "recipe_id", None)
    n = config.num_hidden_layers
    raw = config.raw or {}

    if recipe == "nemotron_h" or (config.model_type or "").lower() in {
        "nemotron_h",
        "nemotronh",
    }:
        pattern = getattr(config, "hybrid_override_pattern", None) or (
            raw.get("hybrid_override_pattern")
        )
        if not pattern:
            raise ValueError(
                "nemotron_h config missing hybrid_override_pattern — cannot build schedule"
            )
        if len(pattern) != n:
            raise ValueError(
                f"hybrid_override_pattern length {len(pattern)} != "
                f"num_hidden_layers {n}"
            )
        return hybrid_pattern_schedule(pattern)

    if recipe in {"mixtral", "gpt_oss"}:
        return tuple(LayerSpec(i, MixerKind.ATTENTION, FfnKind.MOE) for i in range(n))

    if recipe == "llama4":
        listed = raw.get("moe_layers")
        if listed is not None:
            moe_set = {int(x) for x in listed}
        else:
            step = int(raw.get("interleave_moe_layer_step", 1) or 1)
            moe_set = {i for i in range(n) if step > 0 and (i + 1) % step == 0}
        return dense_or_moe_schedule(n, moe_set)

    if recipe == "deepseek_v3":
        first = int(raw.get("first_k_dense_replace") or 0)
        moe_set = {i for i in range(n) if i >= first}
        return dense_or_moe_schedule(n, moe_set)

    if recipe == "qwen3_5" or (config.model_type or "").lower() in {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_next",
    }:
        types = getattr(config, "layer_types", None) or raw.get("layer_types")
        if not types:
            interval = int(raw.get("full_attention_interval") or 4)
            types = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(n)
            ]
        if len(types) != n:
            raise ValueError(
                f"layer_types length {len(types)} != num_hidden_layers {n}"
            )
        return layer_types_schedule(types)

    if recipe in {
        "llama",
        "mistral",
        "qwen2",
        "qwen3",
        "yi",
        "gemma",
        "phi3",
        "gpt2",
        "gpt_neox",
        None,
        "",
    } or (config.model_type or "").lower() in {"llama", ""}:
        return llama_dense_schedule(n)

    raise ValueError(
        f"no schedule for recipe={recipe!r} model_type={config.model_type!r}"
    )
