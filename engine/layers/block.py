"""Scheduled decoder block — dispatches mixer + FFN by LayerSpec."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from engine.config import ModelConfig
from engine.layers.attention import attention_from_weights
from engine.layers.gdn import gated_delta_net
from engine.layers.mamba1 import mamba1
from engine.layers.mamba2 import mamba2
from engine.layers.mlp import mlp_from_weights
from engine.layers.moe import moe
from engine.layers.norm import add_and_norm, apply_norm
from engine.schedule import FfnKind, LayerSpec, MixerKind

# Every one of these runs the same norm → mix → add; the kind only picks which
# mixer `run_mixer` dispatches to, which it already decides for itself.
_SEQUENTIAL_MIXERS = (
    MixerKind.ATTENTION,
    MixerKind.MAMBA2,
    MixerKind.MAMBA1,
    MixerKind.GATED_DELTANET,
    MixerKind.NONE,
)

if TYPE_CHECKING:
    from engine.cache import KVCache, RuntimeState


def _layer_use_rope(config: ModelConfig, layer: int, default: bool) -> bool:
    no_rope = getattr(config, "no_rope_layers", ()) or ()
    if layer < len(no_rope):
        return bool(no_rope[layer])
    return default


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
    attention_mask: torch.Tensor | None = None,
    kv_mask: torch.Tensor | None = None,
    q_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """One scheduled layer: optional mixer residual + optional FFN residual."""
    p = f"layers.{spec.index}"
    layer = spec.index
    kind = config.norm_kind
    residual_kind = getattr(config, "residual_kind", "sequential") or "sequential"
    scale = float(getattr(config, "residual_multiplier", 1.0) or 1.0)
    layer_rope = _layer_use_rope(config, layer, use_rope)

    def run_mixer(h: torch.Tensor) -> torch.Tensor:
        if spec.mixer == MixerKind.ATTENTION:
            return attention_from_weights(
                h,
                weights,
                layer,
                cos,
                sin,
                config,
                cache=cache,
                use_rope=layer_rope,
                attention_mask=attention_mask,
                kv_mask=kv_mask,
                q_positions=q_positions,
            )
        if spec.mixer == MixerKind.MAMBA2:
            return mamba2(h, weights, layer, config, cache=cache)
        if spec.mixer == MixerKind.MAMBA1:
            return mamba1(h, weights, layer, config, cache=cache)
        if spec.mixer == MixerKind.GATED_DELTANET:
            return gated_delta_net(
                h, weights, layer, config, cache=cache, attention_mask=attention_mask
            )
        if spec.mixer == MixerKind.NONE:
            raise ValueError("run_mixer called with mixer=NONE")
        raise ValueError(f"unknown mixer: {spec.mixer}")

    def run_ffn(h: torch.Tensor) -> torch.Tensor:
        act = config.mlp_hidden_act or config.hidden_act
        if spec.ffn == FfnKind.DENSE_MLP:
            return mlp_from_weights(h, weights, layer, act)
        if spec.ffn == FfnKind.MOE:
            return moe(h, weights, layer, config)
        if spec.ffn == FfnKind.NONE:
            raise ValueError("run_ffn called with ffn=NONE")
        raise ValueError(f"unknown ffn: {spec.ffn}")

    if residual_kind == "parallel":
        h = apply_norm(x, weights, f"{p}.input_norm", config.rms_norm_eps, kind)
        delta = torch.zeros_like(x)
        if spec.mixer != MixerKind.NONE:
            delta = delta + run_mixer(h)
        if spec.ffn != FfnKind.NONE:
            delta = delta + run_ffn(h)
        return x + delta * scale

    if residual_kind == "parallel_dual":
        delta = torch.zeros_like(x)
        if spec.mixer != MixerKind.NONE:
            h = apply_norm(x, weights, f"{p}.input_norm", config.rms_norm_eps, kind)
            delta = delta + run_mixer(h)
        if spec.ffn != FfnKind.NONE:
            ff_key = f"{p}.post_attn_norm" if f"{p}.post_attn_norm.weight" in weights else f"{p}.input_norm"
            h = apply_norm(x, weights, ff_key, config.rms_norm_eps, kind)
            delta = delta + run_ffn(h)
        return x + delta * scale

    if residual_kind == "olmo_hybrid":
        if spec.mixer == MixerKind.GATED_DELTANET:
            residual_kind = "sequential"
        else:
            residual_kind = "post_norm"

    if residual_kind == "post_norm":
        if spec.mixer != MixerKind.NONE:
            h = run_mixer(x)
            h = apply_norm(h, weights, f"{p}.post_attn_norm", config.rms_norm_eps, kind)
            x = x + h * scale
        if spec.ffn != FfnKind.NONE:
            h = run_ffn(x)
            ff_key = f"{p}.post_ff_norm" if f"{p}.post_ff_norm.weight" in weights else f"{p}.post_attn_norm"
            h = apply_norm(h, weights, ff_key, config.rms_norm_eps, kind)
            x = x + h * scale
        return x

    if residual_kind == "gemma2":
        if spec.mixer != MixerKind.NONE:
            residual = x
            h = apply_norm(x, weights, f"{p}.input_norm", config.rms_norm_eps, kind)
            h = run_mixer(h)
            h = apply_norm(h, weights, f"{p}.post_attn_norm", config.rms_norm_eps, kind)
            x = residual + h * scale
        if spec.ffn != FfnKind.NONE:
            residual = x
            pre_key = f"{p}.pre_ff_norm" if f"{p}.pre_ff_norm.weight" in weights else f"{p}.post_attn_norm"
            h = apply_norm(x, weights, pre_key, config.rms_norm_eps, kind)
            h = run_ffn(h)
            post_key = f"{p}.post_ff_norm" if f"{p}.post_ff_norm.weight" in weights else f"{p}.post_attn_norm"
            h = apply_norm(h, weights, post_key, config.rms_norm_eps, kind)
            x = residual + h * scale
        return x

    if spec.mixer not in _SEQUENTIAL_MIXERS:
        raise ValueError(f"unknown mixer: {spec.mixer}")
    if spec.ffn not in (FfnKind.DENSE_MLP, FfnKind.MOE, FfnKind.NONE):
        raise ValueError(f"unknown ffn: {spec.ffn}")

    mixed = None
    if spec.mixer != MixerKind.NONE:
        h = apply_norm(x, weights, f"{p}.input_norm", config.rms_norm_eps, kind)
        mixed = run_mixer(h)

    if spec.ffn == FfnKind.NONE:
        return x if mixed is None else x + mixed * scale

    nkey = f"{p}.input_norm" if spec.mixer == MixerKind.NONE else f"{p}.post_attn_norm"
    if mixed is None:
        h = apply_norm(x, weights, nkey, config.rms_norm_eps, kind)
    else:
        # The mixer's residual add and the FFN's norm read and write the same
        # hidden state back to back, which is two passes over it per layer per
        # token on a path where nothing else touches it in between.
        h, x = add_and_norm(
            mixed, x, weights, nkey, config.rms_norm_eps, kind, scale=scale
        )
    return x + run_ffn(h) * scale


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
