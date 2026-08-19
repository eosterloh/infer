"""Causal GQA attention."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from engine.layers.rope import apply_rope

if TYPE_CHECKING:
    from engine.cache import KVCache, RuntimeState


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, n_kv, S, hd] → [B, n_kv * n_rep, S, hd]."""
    if n_rep == 1:
        return x
    b, n_kv, s, hd = x.shape
    x = x[:, :, None, :, :].expand(b, n_kv, n_rep, s, hd)
    return x.reshape(b, n_kv * n_rep, s, hd)


def attention(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nq: int,
    nkv: int,
    hd: int,
    cache: KVCache | RuntimeState | None = None,
    layer: int | None = None,
    use_rope: bool = True,
) -> torch.Tensor:
    """Causal GQA attention. x: [B, S_new, H].

    Nemotron-H attention layers use use_rope=False (no positional embeddings).
    """
    b, s_new, _ = x.shape
    q = F.linear(x, w_q)
    k = F.linear(x, w_k)
    v = F.linear(x, w_v)

    q = q.view(b, s_new, nq, hd).transpose(1, 2)
    k = k.view(b, s_new, nkv, hd).transpose(1, 2)
    v = v.view(b, s_new, nkv, hd).transpose(1, 2)

    if use_rope:
        if cos.numel() == 0 or sin.numel() == 0:
            raise ValueError("use_rope=True but cos/sin are empty")
        q, k = apply_rope(q, k, cos, sin)

    if cache is not None:
        if layer is None:
            raise ValueError("layer index required when cache is provided")
        k, v = cache.update(layer, k, v)
    s_total = k.shape[2]

    k = repeat_kv(k, nq // nkv)
    v = repeat_kv(v, nq // nkv)

    scale = 1.0 / math.sqrt(hd)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

    if s_new == s_total:
        causal = torch.triu(
            torch.full(
                (s_new, s_total), float("-inf"), device=x.device, dtype=scores.dtype
            ),
            diagonal=1,
        )
    else:
        q_pos = torch.arange(
            s_total - s_new, s_total, device=x.device, dtype=torch.long
        )[:, None]
        k_pos = torch.arange(s_total, device=x.device, dtype=torch.long)[None, :]
        causal = torch.where(
            k_pos > q_pos,
            torch.tensor(float("-inf"), device=x.device, dtype=scores.dtype),
            torch.tensor(0.0, device=x.device, dtype=scores.dtype),
        )
    scores = scores + causal
    weights = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
    out = torch.matmul(weights, v)
    out = out.transpose(1, 2).contiguous().view(b, s_new, nq * hd)
    return F.linear(out, w_o)
