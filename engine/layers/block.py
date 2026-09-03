"""Scheduled decoder block — dispatches mixer + FFN by LayerSpec."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from engine.config import ModelConfig
from engine.layers.attention import attention_from_weights
from engine.layers.gdn import gated_delta_net
from engine.layers.mamba2 import mamba2
from engine.layers.mlp import mlp_from_weights
from engine.layers.moe import moe
from engine.layers.norm import apply_norm
from engine.schedule import FfnKind, LayerSpec, MixerKind

if TYPE_CHECKING:
    from engine.cache import KVCache, RuntimeState


def decoder_block(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    spec: LayerSpec,
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: ModelConfig,
    cache: KVCache | RuntimeState | None = None,
    *,
    use_rope: bool = True,
) -> torch.Tensor:
    """One scheduled layer: optional mixer residual + optional FFN residual."""
    p = f"layers.{spec.index}"
    layer = spec.index
    kind = config.norm_kind

    if spec.mixer == MixerKind.ATTENTION:
        h = apply_norm(x, weights, f"{p}.input_norm", config.rms_norm_eps, kind)
        h = attention_from_weights(
            h, weights, layer, cos, sin, config, cache=cache, use_rope=use_rope
        )
        x = x + h
    elif spec.mixer == MixerKind.MAMBA2:
        h = apply_norm(x, weights, f"{p}.input_norm", config.rms_norm_eps, kind)
        h = mamba2(h, weights, layer, config, cache=cache)
        x = x + h
    elif spec.mixer == MixerKind.GATED_DELTANET:
        h = apply_norm(x, weights, f"{p}.input_norm", config.rms_norm_eps, kind)
        h = gated_delta_net(h, weights, layer, config, cache=cache)
        x = x + h
    elif spec.mixer == MixerKind.NONE:
        pass
    else:
        raise ValueError(f"unknown mixer: {spec.mixer}")

    if spec.ffn == FfnKind.DENSE_MLP:
        nkey = f"{p}.input_norm" if spec.mixer == MixerKind.NONE else f"{p}.post_attn_norm"
        h = apply_norm(x, weights, nkey, config.rms_norm_eps, kind)
        act = config.mlp_hidden_act or config.hidden_act
        h = mlp_from_weights(h, weights, layer, act)
        x = x + h
    elif spec.ffn == FfnKind.MOE:
        nkey = f"{p}.input_norm" if spec.mixer == MixerKind.NONE else f"{p}.post_attn_norm"
        h = apply_norm(x, weights, nkey, config.rms_norm_eps, kind)
        h = moe(h, weights, layer, config)
        x = x + h
    elif spec.ffn == FfnKind.NONE:
        pass
    else:
        raise ValueError(f"unknown ffn: {spec.ffn}")

    return x


def transformer_block(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: ModelConfig,
    cache: KVCache | RuntimeState | None = None,
    *,
    use_rope: bool = True,
) -> torch.Tensor:
    """Backward-compatible dense Llama block (attention + dense MLP)."""
    from engine.schedule import LayerSpec

    spec = LayerSpec(layer, MixerKind.ATTENTION, FfnKind.DENSE_MLP)
    return decoder_block(
        x, weights, spec, cos, sin, config, cache=cache, use_rope=use_rope
    )
