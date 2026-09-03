"""Causal GQA attention, plus fused-QKV / GPT-2 / attention-sink variants."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from engine.layers.norm import gemma_rms_norm, rms_norm
from engine.layers.rope import apply_rope

if TYPE_CHECKING:
    from engine.cache import KVCache, RuntimeState
    from engine.config import ModelConfig


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, n_kv, S, hd] → [B, n_kv * n_rep, S, hd]."""
    if n_rep == 1:
        return x
    b, n_kv, s, hd = x.shape
    x = x[:, :, None, :, :].expand(b, n_kv, n_rep, s, hd)
    return x.reshape(b, n_kv * n_rep, s, hd)


def _split_qkv(
    qkv: torch.Tensor,
    nq: int,
    nkv: int,
    hd: int,
    *,
    gpt_neox: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, s, _ = qkv.shape
    if gpt_neox:
        # [B, S, nq, 3, hd] interleaved
        qkv = qkv.view(b, s, nq, 3, hd)
        q, k, v = qkv.unbind(dim=3)
        return (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
    q_size = nq * hd
    kv_size = nkv * hd
    q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
    q = q.view(b, s, nq, hd).transpose(1, 2)
    k = k.view(b, s, nkv, hd).transpose(1, 2)
    v = v.view(b, s, nkv, hd).transpose(1, 2)
    return q, k, v


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
    *,
    b_q: torch.Tensor | None = None,
    b_k: torch.Tensor | None = None,
    b_v: torch.Tensor | None = None,
    b_o: torch.Tensor | None = None,
    q_norm: torch.Tensor | None = None,
    k_norm: torch.Tensor | None = None,
    sliding_window: int | None = None,
    sinks: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    output_gate: bool = False,
    qk_gemma: bool = False,
) -> torch.Tensor:
    """Causal GQA attention. x: [B, S_new, H]."""
    b, s_new, _ = x.shape
    q = F.linear(x, w_q, b_q)
    k = F.linear(x, w_k, b_k)
    v = F.linear(x, w_v, b_v)

    gate = None
    if output_gate:
        q = q.view(b, s_new, nq, hd * 2)
        q, gate_h = q.split(hd, dim=-1)
        gate = gate_h.reshape(b, s_new, nq * hd)
        q = q.transpose(1, 2)
    else:
        q = q.view(b, s_new, nq, hd).transpose(1, 2)
    k = k.view(b, s_new, nkv, hd).transpose(1, 2)
    v = v.view(b, s_new, nkv, hd).transpose(1, 2)

    if q_norm is not None:
        qn = gemma_rms_norm if qk_gemma else rms_norm
        q = qn(q, q_norm, rms_eps)
    if k_norm is not None:
        kn = gemma_rms_norm if qk_gemma else rms_norm
        k = kn(k, k_norm, rms_eps)

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
        if sliding_window:
            causal = torch.where(
                k_pos < (q_pos - sliding_window + 1),
                torch.tensor(float("-inf"), device=x.device, dtype=scores.dtype),
                causal,
            )
    if sliding_window and s_new == s_total:
        band = torch.tril(
            torch.ones((s_new, s_total), device=x.device, dtype=torch.bool),
            diagonal=0,
        ) & torch.triu(
            torch.ones((s_new, s_total), device=x.device, dtype=torch.bool),
            diagonal=1 - sliding_window,
        )
        causal = torch.where(
            band,
            torch.zeros((), device=x.device, dtype=scores.dtype),
            torch.tensor(float("-inf"), device=x.device, dtype=scores.dtype),
        )

    scores = scores + causal
    if sinks is not None:
        sink = sinks.reshape(1, -1, 1, 1).to(dtype=scores.dtype)
        scores = torch.cat([scores, sink.expand(b, nq, s_new, 1)], dim=-1)
        weights = torch.softmax(scores, dim=-1)[..., :-1].to(dtype=v.dtype)
    else:
        weights = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
    out = torch.matmul(weights, v)
    out = out.transpose(1, 2).contiguous().view(b, s_new, nq * hd)
    if gate is not None:
        out = out * torch.sigmoid(gate)
    return F.linear(out, w_o, b_o)


def attention_from_weights(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    spec_index: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: ModelConfig,
    cache: KVCache | RuntimeState | None = None,
    *,
    use_rope: bool = True,
) -> torch.Tensor:
    p = f"layers.{spec_index}"
    nq, nkv, hd = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    kind = config.attention_kind

    if kind == "gpt2":
        qkv = F.linear(x, weights[f"{p}.attn.c_attn.weight"], weights.get(f"{p}.attn.c_attn.bias"))
        q, k, v = _split_qkv(qkv, nq, nkv, hd)
        b, s_new, _ = x.shape
        if cache is not None:
            k, v = cache.update(spec_index, k, v)
        s_total = k.shape[2]
        scale = 1.0 / math.sqrt(hd)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        if s_new == s_total:
            causal = torch.triu(
                torch.full((s_new, s_total), float("-inf"), device=x.device, dtype=scores.dtype),
                diagonal=1,
            )
        else:
            q_pos = torch.arange(s_total - s_new, s_total, device=x.device)[:, None]
            k_pos = torch.arange(s_total, device=x.device)[None, :]
            causal = torch.where(
                k_pos > q_pos,
                torch.tensor(float("-inf"), device=x.device, dtype=scores.dtype),
                torch.zeros((), device=x.device, dtype=scores.dtype),
            )
        weights_s = torch.softmax(scores + causal, dim=-1).to(dtype=v.dtype)
        out = torch.matmul(weights_s, v).transpose(1, 2).contiguous().view(b, s_new, nq * hd)
        return F.linear(out, weights[f"{p}.attn.c_proj.weight"], weights.get(f"{p}.attn.c_proj.bias"))

    if kind == "fused_qkv":
        qkv = F.linear(x, weights[f"{p}.attn.qkv.weight"], weights.get(f"{p}.attn.qkv.bias"))
        q, k, v = _split_qkv(qkv, nq, nkv, hd, gpt_neox=config.recipe_id == "gpt_neox")
        if use_rope and cos.numel():
            q, k = apply_rope(q, k, cos, sin)
        if cache is not None:
            k, v = cache.update(spec_index, k, v)
        k = repeat_kv(k, nq // max(nkv, 1)) if nkv else k
        v = repeat_kv(v, nq // max(nkv, 1)) if nkv else v
        # gpt_neox fused uses nq==nkv typically
        if config.recipe_id == "gpt_neox":
            nkv = nq
        scale = 1.0 / math.sqrt(hd)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        s_new, s_total = q.shape[2], k.shape[2]
        causal = torch.triu(
            torch.full((s_new, s_total), float("-inf"), device=x.device, dtype=scores.dtype),
            diagonal=1 if s_new == s_total else s_total - s_new + 1,
        )
        if s_new != s_total:
            q_pos = torch.arange(s_total - s_new, s_total, device=x.device)[:, None]
            k_pos = torch.arange(s_total, device=x.device)[None, :]
            causal = torch.where(
                k_pos > q_pos,
                torch.tensor(float("-inf"), device=x.device, dtype=scores.dtype),
                torch.zeros((), device=x.device, dtype=scores.dtype),
            )
        attn_w = torch.softmax(scores + causal, dim=-1).to(dtype=v.dtype)
        out = torch.matmul(attn_w, v).transpose(1, 2).contiguous().view(x.shape[0], s_new, nq * hd)
        return F.linear(out, weights[f"{p}.attn.o.weight"], weights.get(f"{p}.attn.o.bias"))

    if kind == "mla":
        from engine.layers.mla import mla_attention

        return mla_attention(x, weights, spec_index, cos, sin, config, cache=cache)

    return attention(
        x,
        weights[f"{p}.attn.q.weight"],
        weights[f"{p}.attn.k.weight"],
        weights[f"{p}.attn.v.weight"],
        weights[f"{p}.attn.o.weight"],
        cos,
        sin,
        nq=nq,
        nkv=nkv,
        hd=hd,
        cache=cache,
        layer=spec_index,
        use_rope=use_rope,
        b_q=weights.get(f"{p}.attn.q.bias"),
        b_k=weights.get(f"{p}.attn.k.bias"),
        b_v=weights.get(f"{p}.attn.v.bias"),
        b_o=weights.get(f"{p}.attn.o.bias"),
        q_norm=weights.get(f"{p}.attn.q_norm.weight"),
        k_norm=weights.get(f"{p}.attn.k_norm.weight"),
        sliding_window=config.sliding_window,
        sinks=weights.get(f"{p}.attn.sinks"),
        rms_eps=config.rms_norm_eps,
        output_gate=config.attn_output_gate,
        qk_gemma=config.norm_kind == "gemma_rms",
    )
