"""Scheduled decoder block — dispatches mixer + FFN by LayerSpec."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from engine.config import ModelConfig
from engine.layers.attention import attention
from engine.layers.mamba2 import mamba2
from engine.layers.mlp import mlp
from engine.layers.moe import moe
from engine.layers.norm import rms_norm
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

    if spec.mixer == MixerKind.ATTENTION:
        h = rms_norm(x, weights[f"{p}.input_norm.weight"], config.rms_norm_eps)
        h = attention(
            h,
            weights[f"{p}.attn.q.weight"],
            weights[f"{p}.attn.k.weight"],
            weights[f"{p}.attn.v.weight"],
            weights[f"{p}.attn.o.weight"],
            cos,
            sin,
            nq=config.num_attention_heads,
            nkv=config.num_key_value_heads,
            hd=config.head_dim,
            cache=cache,
            layer=layer,
            use_rope=use_rope,
        )
        x = x + h
    elif spec.mixer == MixerKind.MAMBA2:
        h = rms_norm(x, weights[f"{p}.input_norm.weight"], config.rms_norm_eps)
        h = mamba2(h, weights, layer, config, cache=cache)
        x = x + h
    elif spec.mixer == MixerKind.NONE:
        pass
    else:
        raise ValueError(f"unknown mixer: {spec.mixer}")

    if spec.ffn == FfnKind.DENSE_MLP:
        # Llama: post-attn norm. Nemotron-H MLP-only layer: prenorm is input_norm
        # (and mixer is NONE — already applied above would be wrong).
        if spec.mixer == MixerKind.NONE:
            h = rms_norm(x, weights[f"{p}.input_norm.weight"], config.rms_norm_eps)
        else:
            h = rms_norm(x, weights[f"{p}.post_attn_norm.weight"], config.rms_norm_eps)
        h = mlp(
            h,
            weights[f"{p}.mlp.gate.weight"],
            weights[f"{p}.mlp.up.weight"],
            weights[f"{p}.mlp.down.weight"],
        )
        x = x + h
    elif spec.ffn == FfnKind.MOE:
        if spec.mixer == MixerKind.NONE:
            h = rms_norm(x, weights[f"{p}.input_norm.weight"], config.rms_norm_eps)
        else:
            h = rms_norm(x, weights[f"{p}.post_attn_norm.weight"], config.rms_norm_eps)
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
