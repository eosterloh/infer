"""Norms: RMS (Llama), Gemma RMS (weight+1), LayerNorm (GPT-2 / NeoX)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


from engine.kernels import rms_norm as kernel_rms_norm


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in fp32 reduce, cast back to x.dtype."""
    return kernel_rms_norm(x, weight, eps)


def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma: normalize then multiply (1 + weight)."""
    orig_dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    x_f = x_f * torch.rsqrt(var + eps)
    return (x_f * (1.0 + weight.float())).to(orig_dtype)


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def olmo_layer_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Affine-free LayerNorm (OLMo)."""
    orig_dtype = x.dtype
    return F.layer_norm(x.float(), (x.shape[-1],), None, None, eps).to(orig_dtype)


def cohere_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Mean-center then RMS-style scale (Command-R)."""
    orig_dtype = x.dtype
    x_f = x.float()
    mean = x_f.mean(dim=-1, keepdim=True)
    var = (x_f - mean).pow(2).mean(dim=-1, keepdim=True)
    x_f = (x_f - mean) * torch.rsqrt(var + eps)
    return (weight.float() * x_f).to(orig_dtype)


def apply_norm(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    key: str,
    eps: float,
    kind: str,
) -> torch.Tensor:
    if kind == "olmo":
        return olmo_layer_norm(x, eps)
    w = weights[f"{key}.weight"]
    if kind == "gemma_rms":
        return gemma_rms_norm(x, w, eps)
    if kind == "layer":
        return layer_norm(x, w, weights.get(f"{key}.bias"), eps)
    if kind == "layer_1p":
        bias = weights.get(f"{key}.bias")
        return F.layer_norm(x, (x.shape[-1],), w + 1.0, bias, eps)
    if kind == "cohere":
        return cohere_norm(x, w, eps)
    return rms_norm(x, w, eps)
