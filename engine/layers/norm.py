"""Norms: RMS (Llama), Gemma RMS (weight+1), LayerNorm (GPT-2 / NeoX)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


from engine.kernels import fused_add_rms_norm as kernel_add_rms_norm
from engine.kernels import rms_norm as kernel_rms_norm

# The offset each RMS-like kind folds into its weight. Only these fuse with the
# residual add; LayerNorm subtracts a mean and Cohere centers, neither of which
# the fused kernel does, and OLMo has no weight to pass it.
_RMS_OFFSETS = {"rms": 0.0, "gemma_rms": 1.0}


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in fp32 reduce, cast back to x.dtype."""
    return kernel_rms_norm(x, weight, eps)


def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma: normalize then multiply (1 + weight)."""
    return kernel_rms_norm(x, weight, eps, weight_offset=1.0)


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


def add_and_norm(
    delta: torch.Tensor,
    residual: torch.Tensor,
    weights: dict[str, torch.Tensor],
    key: str,
    eps: float,
    kind: str,
    *,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual + delta``, then normalized. Returns (normed, residual).

    Both halves read and write the whole hidden state, and between them nothing
    else touches it, so the RMS kinds do it in one pass. The residual is written
    in place, which is why the caller has to own it — see DecoderModel.forward,
    which copies an embedding handed in from outside before the first layer.

    A recipe that scales its residual, or normalizes with anything the fused
    kernel does not implement, takes the two-step form and is still correct.
    """
    offset = _RMS_OFFSETS.get(kind)
    if offset is not None and scale == 1.0 and delta.shape == residual.shape:
        normed, residual = kernel_add_rms_norm(
            delta, residual, weights[f"{key}.weight"], eps, weight_offset=offset
        )
        return normed, residual
    residual = residual + delta * scale
    return apply_norm(residual, weights, key, eps, kind), residual


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
