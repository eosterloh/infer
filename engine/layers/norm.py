"""Norms: RMS (Llama), Gemma RMS (weight+1), LayerNorm (GPT-2 / NeoX)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in fp32 reduce, cast back to x.dtype."""
    orig_dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    x_f = x_f * torch.rsqrt(var + eps)
    return (x_f * weight.float()).to(orig_dtype)


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


def apply_norm(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    key: str,
    eps: float,
    kind: str,
) -> torch.Tensor:
    w = weights[f"{key}.weight"]
    if kind == "gemma_rms":
        return gemma_rms_norm(x, w, eps)
    if kind == "layer":
        return layer_norm(x, w, weights.get(f"{key}.bias"), eps)
    return rms_norm(x, w, eps)
