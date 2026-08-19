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


def build_schedule(config: ModelConfig) -> tuple[LayerSpec, ...]:
    """Build layer schedule from already-detected recipe + config fields."""
    if config.layers:
        return config.layers

    recipe = getattr(config, "recipe_id", None)
    if recipe == "nemotron_h" or (config.model_type or "").lower() in {
        "nemotron_h",
        "nemotronh",
    }:
        pattern = getattr(config, "hybrid_override_pattern", None) or (
            config.raw.get("hybrid_override_pattern") if config.raw else None
        )
        if not pattern:
            raise ValueError(
                "nemotron_h config missing hybrid_override_pattern — cannot build schedule"
            )
        if len(pattern) != config.num_hidden_layers:
            raise ValueError(
                f"hybrid_override_pattern length {len(pattern)} != "
                f"num_hidden_layers {config.num_hidden_layers}"
            )
        return hybrid_pattern_schedule(pattern)

    if recipe == "llama" or (config.model_type or "").lower() in {"llama", ""}:
        return llama_dense_schedule(config.num_hidden_layers)

    raise ValueError(
        f"no schedule for recipe={recipe!r} model_type={config.model_type!r}"
    )
