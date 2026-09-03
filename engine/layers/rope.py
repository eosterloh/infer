"""Llama / NeoX-style RoPE."""

from __future__ import annotations

import math

import torch

from engine.config import ModelConfig


def _inv_freq_default(dim: int, base: float, device: torch.device) -> torch.Tensor:
    return 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64, device=device).to(torch.float32)
            / dim
        )
    )


def _inv_freq_llama3(
    dim: int,
    base: float,
    device: torch.device,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> torch.Tensor:
    """Match transformers.modeling_rope_utils._compute_llama3_parameters."""
    inv_freq = _inv_freq_default(dim, base, device)
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)


def build_inv_freq(config: ModelConfig, device: torch.device) -> torch.Tensor:
    dim = int(config.head_dim * float(getattr(config, "partial_rotary_factor", 1.0) or 1.0))
    if dim < 2:
        dim = config.head_dim
    base = float(config.rope_theta)
    scaling = config.rope_scaling
    if scaling and scaling.get("rope_type") == "llama3":
        return _inv_freq_llama3(
            dim=dim,
            base=base,
            device=device,
            factor=float(scaling["factor"]),
            low_freq_factor=float(scaling["low_freq_factor"]),
            high_freq_factor=float(scaling["high_freq_factor"]),
            old_context_len=float(scaling["original_max_position_embeddings"]),
        )
    return _inv_freq_default(dim, base, device)


def build_rope_cos_sin(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """position_ids: [B, S] int/long → cos/sin [B, S, head_dim] in dtype."""
    inv_freq_expanded = inv_freq[None, :, None].float().expand(
        position_ids.shape[0], -1, 1
    )
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [B, S, dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # Llama/neox style
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)
    return cos, sin


def build_mrope_cos_sin(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    dtype: torch.dtype,
    mrope_section: list[int] | tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen multimodal RoPE from temporal/height/width positions.

    ``position_ids`` is [3, B, S]. Text positions have three identical rows.
    Frequency slots are interleaved T/H/W according to ``mrope_section``.
    """
    if position_ids.dim() != 3 or position_ids.shape[0] != 3:
        raise ValueError(
            f"expected MRoPE position_ids [3,B,S], got {tuple(position_ids.shape)}"
        )
    b = position_ids.shape[1]
    inv = inv_freq[None, None, :, None].float().expand(3, b, -1, 1)
    pos = position_ids[:, :, None, :].float()
    freqs = (inv @ pos).transpose(2, 3)
    mixed = freqs[0].clone()
    for source, offset in ((1, 1), (2, 2)):
        length = int(mrope_section[source]) * 3
        mixed[..., offset:length:3] = freqs[source, ..., offset:length:3]
    emb = torch.cat((mixed, mixed), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q/k: [B, heads, S, hd]; cos/sin: [B, S, rotary_dim] (may be < hd)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    if rotary_dim < q.shape[-1]:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_embed = torch.cat(
            ((q_rot * cos) + (rotate_half(q_rot) * sin), q_pass), dim=-1
        )
        k_embed = torch.cat(
            ((k_rot * cos) + (rotate_half(k_rot) * sin), k_pass), dim=-1
        )
        return q_embed, k_embed
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
