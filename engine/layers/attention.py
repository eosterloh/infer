"""Causal GQA attention, plus fused-QKV / GPT-2 / attention-sink variants."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from engine import kernels
from engine.layers.linear import dense
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
    falcon: bool = False,
    new_decoder: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, s, _ = qkv.shape
    if falcon and new_decoder:
        group = nq // max(nkv, 1) + 2
        qkv = qkv.view(b, s, nkv, group, hd)
        q = qkv[:, :, :, :-2].reshape(b, s, nq, hd).transpose(1, 2)
        k = qkv[:, :, :, -2].transpose(1, 2)
        v = qkv[:, :, :, -1].transpose(1, 2)
        return q, k, v
    if falcon:
        qkv = qkv.view(b, s, nq + 2, hd)
        q = qkv[:, :, :-2].transpose(1, 2)
        k = qkv[:, :, -2:-1].transpose(1, 2)
        v = qkv[:, :, -1:].transpose(1, 2)
        return q, k, v
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


def _bloom_alibi_slopes(num_heads: int, device: torch.device) -> torch.Tensor:
    closest = 2 ** int(math.floor(math.log2(num_heads)))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest) - 3))), device=device, dtype=torch.float32
    )
    slopes = torch.pow(base, torch.arange(1, 1 + closest, device=device, dtype=torch.int32))
    if closest != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest) - 3))),
            device=device,
            dtype=torch.float32,
        )
        remaining = min(closest, num_heads - closest)
        extra = torch.arange(1, 1 + 2 * remaining, 2, device=device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra)], dim=0)
    return slopes[:num_heads]


def _mpt_alibi_slopes(
    num_heads: int, device: torch.device, alibi_bias_max: float = 8.0
) -> torch.Tensor:
    n2 = 2 ** math.ceil(math.log2(num_heads))
    base = torch.arange(1, n2 + 1, device=device, dtype=torch.float32)
    base = base * (alibi_bias_max / n2)
    slopes = (1.0 / torch.pow(2, base)).view(n2)
    if n2 != num_heads:
        slopes = torch.cat([slopes[1::2], slopes[::2]], dim=0)[:num_heads]
    return slopes


def build_alibi(
    nq: int,
    s_new: int,
    s_total: int,
    device: torch.device,
    kind: str,
    alibi_bias_max: float = 8.0,
) -> torch.Tensor:
    """Return additive bias [1, nq, s_new, s_total] before softmax."""
    q_pos = torch.arange(s_total - s_new, s_total, device=device, dtype=torch.float32)
    k_pos = torch.arange(s_total, device=device, dtype=torch.float32)
    if kind == "mpt":
        slopes = _mpt_alibi_slopes(nq, device, alibi_bias_max)
        rel = torch.arange(1 - s_total, 1, device=device, dtype=torch.float32)
        # MPT builds a max-length table then slices the trailing query/key window.
        table = slopes.view(nq, 1, 1) * rel.view(1, 1, s_total)
        return table[:, -s_new:, :].unsqueeze(0)
    slopes = _bloom_alibi_slopes(nq, device)
    # Bloom/Falcon: slope * key_index (softmax-invariant vs slope*(k-q)).
    bias = slopes.view(1, nq, 1, 1) * k_pos.view(1, 1, 1, s_total)
    return bias.expand(1, nq, s_new, s_total)


def _attention_mode() -> str:
    """``auto`` uses SDPA where the math allows it; ``eager`` forces Python."""
    return os.environ.get("INFER_ATTENTION", "auto").strip().lower()


def causal_keep_mask(
    s_new: int,
    s_total: int,
    device: torch.device,
    sliding_window: int | None = None,
    q_start: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """[s_new, s_total] bool — True where a query may read a key.

    ``q_start`` is the position of the first query. It defaults to the end of the
    key buffer, which holds whenever the buffer is exactly the live length, but
    CUDA-graph decode reads a fixed buffer that runs past it: there the window has
    to be measured from the query's own position or it excludes every real key.
    It may be a device scalar so the mask stays capturable.
    """
    if q_start is None:
        q_start = s_total - s_new
    offset = torch.arange(s_new, device=device).unsqueeze(1)
    if isinstance(q_start, torch.Tensor):
        q_pos = q_start.reshape(-1)[:1].to(device=device, dtype=torch.long) + offset
    else:
        q_pos = int(q_start) + offset
    k_pos = torch.arange(s_total, device=device).unsqueeze(0)
    keep = k_pos <= q_pos
    if sliding_window:
        keep = keep & (k_pos > q_pos - sliding_window)
    return keep


def decode_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    sliding_window: int | None = None,
    key_mask: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    softcap: float | None = None,
) -> torch.Tensor | None:
    """One-query attention through the fused kernel, or None to use SDPA.

    Worth its own path because a decode step is all cache read: the kernel takes
    K/V exactly as the cache stores them, so nothing of length ``s_total`` — no
    mask, no GQA copy, no score row — is materialized per layer per token.
    """
    if q.shape[2] != 1 or _attention_mode() == "eager":
        return None
    s_total = k.shape[2]
    window = int(sliding_window or 0)
    if window and key_mask is not None and s_total > window:
        # The kernel measures its window back from the end of the buffer, and a
        # mask means the live length is below that, so the band would sit past
        # the real keys. SDPA gets the query's position and can place it.
        return None
    out = kernels.attn_decode(
        q[:, :, 0],
        k,
        v,
        scale=scale,
        kv_mask=None if key_mask is None else key_mask[:, -s_total:],
        sinks=sinks,
        window=window if window < s_total else 0,
        softcap=float(softcap or 0.0),
    )
    return None if out is None else out.unsqueeze(2)


def sdpa_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    sliding_window: int | None = None,
    padding_mask: torch.Tensor | None = None,
    q_start: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Fused attention over [B, heads, S, D] tensors. GQA stays unexpanded.

    Keeping K/V at their stored head count means flash attention reads the
    cache once instead of materializing an nq-head copy of it, and the scores
    never exist in memory.
    """
    nq, nkv = q.shape[1], k.shape[1]
    s_new, s_total = q.shape[2], k.shape[2]
    attn_mask: torch.Tensor | None = None
    is_causal = False
    if padding_mask is None and not sliding_window and q_start is None:
        if s_new == s_total and s_new > 1:
            is_causal = True
        elif s_new > 1:
            attn_mask = causal_keep_mask(s_new, s_total, q.device)[None, None]
    else:
        keep = causal_keep_mask(
            s_new, s_total, q.device, sliding_window, q_start=q_start
        )[None, None]
        if padding_mask is not None:
            keep = keep & padding_mask[:, -s_total:].bool()[:, None, None, :]
        attn_mask = keep
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=nq != nkv and nkv > 0,
    )
    if padding_mask is not None:
        # Rows whose every key is masked come back NaN; the eager path zeroes them.
        out = torch.nan_to_num(out)
    return out


def _causal_mask(
    s_new: int,
    s_total: int,
    device: torch.device,
    dtype: torch.dtype,
    sliding_window: int | None = None,
) -> torch.Tensor:
    if s_new == s_total:
        causal = torch.triu(
            torch.full((s_new, s_total), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )
        if sliding_window:
            band = torch.tril(
                torch.ones((s_new, s_total), device=device, dtype=torch.bool),
                diagonal=0,
            ) & torch.triu(
                torch.ones((s_new, s_total), device=device, dtype=torch.bool),
                diagonal=1 - sliding_window,
            )
            causal = torch.zeros((s_new, s_total), device=device, dtype=dtype)
            causal = causal.masked_fill(~band, float("-inf"))
        return causal
    # masked_fill takes the scalar as an argument; torch.tensor(-inf, device=cuda)
    # is an unpinned host copy, which is illegal inside a graph capture.
    keep = causal_keep_mask(s_new, s_total, device, sliding_window)
    causal = torch.zeros((s_new, s_total), device=device, dtype=dtype)
    return causal.masked_fill(~keep, float("-inf"))


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
    q_norm_bias: torch.Tensor | None = None,
    k_norm_bias: torch.Tensor | None = None,
    sliding_window: int | None = None,
    sinks: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    output_gate: bool = False,
    qk_gemma: bool = False,
    attention_mask: torch.Tensor | None = None,
    attention_multiplier: float | None = None,
    query_pre_attn_scalar: float | None = None,
    attn_logit_softcapping: float | None = None,
    clip_qkv: float | None = None,
    rope_interleaved: bool = False,
    alibi: bool = False,
    alibi_kind: str = "bloom",
    alibi_bias_max: float = 8.0,
    qk_norm_after_rope: bool = False,
    sub_norm: torch.Tensor | None = None,
    kv_mask: torch.Tensor | None = None,
    q_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal GQA attention. x: [B, S_new, H].

    ``attention_mask`` is the usual [B, S] key-padding mask, whose trailing
    entries also mark which *queries* are real. ``kv_mask`` marks key validity
    only: CUDA-graph decode reads a fixed KV window that runs past the live
    length, and the query it is decoding is always real.
    """
    b, s_new, _ = x.shape
    q = dense(x, w_q, b_q)
    k = dense(x, w_k, b_k)
    v = dense(x, w_v, b_v)
    if clip_qkv is not None:
        q = q.clamp(-clip_qkv, clip_qkv)
        k = k.clamp(-clip_qkv, clip_qkv)
        v = v.clamp(-clip_qkv, clip_qkv)

    gate = None
    qn = gemma_rms_norm if qk_gemma else rms_norm
    kn = gemma_rms_norm if qk_gemma else rms_norm
    # Olmo2: RMS over the concatenated Q/K vector before the head split.
    if q_norm is not None and q_norm.numel() == q.shape[-1]:
        q = qn(q, q_norm, rms_eps)
        q_norm = None
    if k_norm is not None and k_norm.numel() == k.shape[-1]:
        k = kn(k, k_norm, rms_eps)
        k_norm = None

    if output_gate:
        q = q.view(b, s_new, nq, hd * 2)
        q, gate_h = q.split(hd, dim=-1)
        gate = gate_h.reshape(b, s_new, nq * hd)
        q = q.transpose(1, 2)
    else:
        q = q.view(b, s_new, nq, hd).transpose(1, 2)
    k = k.view(b, s_new, nkv, hd).transpose(1, 2)
    v = v.view(b, s_new, nkv, hd).transpose(1, 2)

    def _apply_qk_norm() -> tuple[torch.Tensor, torch.Tensor]:
        qq, kk = q, k
        if q_norm is not None:
            if q_norm_bias is not None:
                qq = F.layer_norm(qq, (hd,), q_norm, q_norm_bias, rms_eps)
            else:
                qq = qn(qq, q_norm, rms_eps)
        if k_norm is not None:
            if k_norm_bias is not None:
                kk = F.layer_norm(kk, (hd,), k_norm, k_norm_bias, rms_eps)
            else:
                kk = kn(kk, k_norm, rms_eps)
        return qq, kk

    if not qk_norm_after_rope:
        q, k = _apply_qk_norm()

    if use_rope:
        if cos.numel() == 0 or sin.numel() == 0:
            raise ValueError("use_rope=True but cos/sin are empty")
        q, k = apply_rope(q, k, cos, sin, interleaved=rope_interleaved)

    if qk_norm_after_rope:
        q, k = _apply_qk_norm()

    if cache is not None:
        if layer is None:
            raise ValueError("layer index required when cache is provided")
        k, v = cache.update(layer, k, v)
    s_total = k.shape[2]

    if attention_multiplier is not None:
        scale = float(attention_multiplier)
    elif query_pre_attn_scalar is not None:
        scale = float(query_pre_attn_scalar) ** -0.5
    else:
        scale = 1.0 / math.sqrt(hd)

    if attention_mask is not None and (
        attention_mask.dim() != 2 or attention_mask.shape[0] != b
    ):
        raise ValueError(
            f"expected attention_mask [B,S], got {tuple(attention_mask.shape)}"
        )
    key_mask = attention_mask
    if kv_mask is not None:
        key_mask = kv_mask if key_mask is None else (key_mask.bool() & kv_mask.bool())

    # A kv_mask is the signal that the key buffer runs past the live length, which
    # is the one case where the query does not sit at its end. Everywhere else
    # q_start stays None and the mask is built exactly as before.
    q_start = q_positions if (kv_mask is not None and q_positions is not None) else None

    fused_ok = _attention_mode() != "eager" and not alibi and q.is_floating_point()
    out = None
    if fused_ok:
        # The decode kernel covers sinks and soft-capping; SDPA does not.
        out = decode_attend(
            q,
            k,
            v,
            scale=scale,
            sliding_window=sliding_window,
            key_mask=key_mask,
            sinks=sinks,
            softcap=attn_logit_softcapping,
        )
        if out is None and sinks is None and not attn_logit_softcapping:
            out = sdpa_attend(
                q,
                k,
                v,
                scale=scale,
                sliding_window=sliding_window,
                padding_mask=key_mask,
                q_start=q_start,
            )
    if out is not None:
        out = out.transpose(1, 2).contiguous().view(b, s_new, nq * hd)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        if sub_norm is not None:
            out = rms_norm(out, sub_norm, rms_eps)
        out = dense(out, w_o, b_o)
        if attention_mask is not None:
            out = out * attention_mask[:, -s_new:, None].to(dtype=out.dtype)
        return out

    k = repeat_kv(k, nq // nkv)
    v = repeat_kv(v, nq // nkv)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
    if alibi:
        bias = build_alibi(nq, s_new, s_total, x.device, alibi_kind, alibi_bias_max)
        if alibi_kind == "falcon":
            scores = (scores + bias) * scale
        else:
            scores = scores * scale + bias
    else:
        scores = scores * scale
    if attn_logit_softcapping:
        cap = float(attn_logit_softcapping)
        scores = torch.tanh(scores / cap) * cap

    if s_new == s_total:
        causal = torch.triu(
            torch.full(
                (s_new, s_total), float("-inf"), device=x.device, dtype=scores.dtype
            ),
            diagonal=1,
        )
    else:
        keep = causal_keep_mask(
            s_new, s_total, x.device, sliding_window, q_start=q_start
        )
        causal = torch.zeros((s_new, s_total), device=x.device, dtype=scores.dtype)
        causal = causal.masked_fill(~keep, float("-inf"))
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
    if key_mask is not None:
        keep = key_mask[:, -s_total:].to(device=x.device, dtype=torch.bool)
        scores = scores.masked_fill(~keep[:, None, None, :], float("-inf"))
    if sinks is not None:
        sink = sinks.reshape(1, -1, 1, 1).to(dtype=scores.dtype)
        scores = torch.cat([scores, sink.expand(b, nq, s_new, 1)], dim=-1)
        weights = torch.softmax(scores, dim=-1)[..., :-1].to(dtype=v.dtype)
    else:
        weights = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
    weights = torch.nan_to_num(weights)
    out = torch.matmul(weights, v)
    out = out.transpose(1, 2).contiguous().view(b, s_new, nq * hd)
    if gate is not None:
        out = out * torch.sigmoid(gate)
    if sub_norm is not None:
        out = rms_norm(out, sub_norm, rms_eps)
    out = dense(out, w_o, b_o)
    if attention_mask is not None:
        out = out * attention_mask[:, -s_new:, None].to(dtype=out.dtype)
    return out


def _layer_sliding_window(config: ModelConfig, layer: int) -> int | None:
    types = getattr(config, "layer_types", ()) or ()
    if layer < len(types):
        kind = str(types[layer]).lower().replace("-", "_")
        if kind in {"sliding_attention", "sliding"}:
            return config.sliding_window
        if kind in {"full_attention", "full"}:
            return None
    return config.sliding_window


def differential_attention(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: ModelConfig,
    cache: KVCache | RuntimeState | None = None,
    *,
    use_rope: bool = True,
    attention_mask: torch.Tensor | None = None,
    key_mask: torch.Tensor | None = None,
    q_start: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Differential Transformer attention (DiffLlama)."""
    p = f"layers.{layer}"
    nq, nkv, hd = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    b, s_new, _ = x.shape
    q = F.linear(x, weights[f"{p}.attn.q.weight"], weights.get(f"{p}.attn.q.bias"))
    k = F.linear(x, weights[f"{p}.attn.k.weight"], weights.get(f"{p}.attn.k.bias"))
    v = F.linear(x, weights[f"{p}.attn.v.weight"], weights.get(f"{p}.attn.v.bias"))
    q = q.view(b, s_new, nq, hd).transpose(1, 2)
    k = k.view(b, s_new, nkv, hd).transpose(1, 2)
    v = v.view(b, s_new, nkv, hd).transpose(1, 2)
    if use_rope and cos.numel() > 0:
        q, k = apply_rope(q, k, cos, sin)
    if cache is not None:
        k, v = cache.update(layer, k, v)
    v1, v2 = (half.repeat(1, 2, 1, 1) for half in torch.chunk(v, 2, dim=1))
    k = repeat_kv(k, nq // nkv)
    v1 = repeat_kv(v1, nq // nkv)
    v2 = repeat_kv(v2, nq // nkv)
    s_total = k.shape[2]
    scale = 1.0 / math.sqrt(hd)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if s_new == s_total and q_start is None:
        causal = _causal_mask(s_new, s_total, x.device, scores.dtype)
    else:
        keep = causal_keep_mask(s_new, s_total, x.device, q_start=q_start)
        # masked_fill, not where(..., tensor(-inf)): building a device scalar out
        # of a Python float is an unpinned host copy, which a capture rejects.
        causal = torch.zeros((s_new, s_total), device=x.device, dtype=scores.dtype)
        causal = causal.masked_fill(~keep, float("-inf"))
    scores = scores + causal
    # Neither mask reached this path before, so a left-padded batch read its
    # padding and a captured graph read the zeroed tail of a fixed buffer.
    valid = key_mask if key_mask is not None else attention_mask
    if valid is not None:
        keep_k = valid[:, -s_total:].to(device=x.device, dtype=torch.bool)
        scores = scores.masked_fill(~keep_k[:, None, None, :], float("-inf"))
    attn = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
    out1 = torch.matmul(attn, v1).transpose(1, 2).contiguous()
    out2 = torch.matmul(attn, v2).transpose(1, 2).contiguous()
    # [B, S, nq, 2hd] then pair first/last head halves.
    packed = torch.cat([out1, out2], dim=-1)
    half1, half2 = torch.chunk(packed, 2, dim=2)
    lq1 = weights[f"{p}.attn.lambda_q1"]
    lk1 = weights[f"{p}.attn.lambda_k1"]
    lq2 = weights[f"{p}.attn.lambda_q2"]
    lk2 = weights[f"{p}.attn.lambda_k2"]
    lam1 = torch.exp((lq1 * lk1).sum(dim=-1, dtype=torch.float32)).to(dtype=q.dtype)
    lam2 = torch.exp((lq2 * lk2).sum(dim=-1, dtype=torch.float32)).to(dtype=q.dtype)
    lam_init = 0.8 - 0.6 * math.exp(-0.3 * layer)
    lam = lam1 - lam2 + lam_init
    out = half1 - lam * half2
    out_f = out.float()
    var = out_f.pow(2).mean(dim=-1, keepdim=True)
    out = (out_f * torch.rsqrt(var + config.rms_norm_eps) * (1.0 - lam_init)).to(dtype=x.dtype)
    out = out.reshape(b, s_new, nq * hd)
    return F.linear(out, weights[f"{p}.attn.o.weight"], weights.get(f"{p}.attn.o.bias"))


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
    attention_mask: torch.Tensor | None = None,
    kv_mask: torch.Tensor | None = None,
    q_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    p = f"layers.{spec_index}"
    nq, nkv, hd = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    kind = config.attention_kind
    sliding_window = _layer_sliding_window(config, spec_index)
    # The packed-QKV branches below used to mask on kv_mask alone, which left a
    # left-padded batch attending to its padding.
    key_mask = attention_mask
    if kv_mask is not None:
        key_mask = kv_mask if key_mask is None else (key_mask.bool() & kv_mask.bool())
    q_start = q_positions if (kv_mask is not None and q_positions is not None) else None

    def _mask_out(out: torch.Tensor, s_new: int) -> torch.Tensor:
        if attention_mask is None:
            return out
        return out * attention_mask[:, -s_new:, None].to(dtype=out.dtype)

    if kind in {"gpt2", "gpt_bigcode"}:
        qkv = dense(x, weights[f"{p}.attn.c_attn.weight"], weights.get(f"{p}.attn.c_attn.bias"))
        q, k, v = _split_qkv(qkv, nq, nkv, hd)
        b, s_new, _ = x.shape
        if cache is not None:
            k, v = cache.update(spec_index, k, v)
        scale = 1.0 / math.sqrt(hd)
        if _attention_mode() != "eager" and q.is_floating_point():
            out = decode_attend(q, k, v, scale=scale, key_mask=key_mask)
            if out is None:
                out = sdpa_attend(
                    q, k, v, scale=scale, padding_mask=key_mask, q_start=q_start
                )
        else:
            k = repeat_kv(k, nq // max(nkv, 1)) if nkv else k
            v = repeat_kv(v, nq // max(nkv, 1)) if nkv else v
            s_total = k.shape[2]
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            causal = _causal_mask(s_new, s_total, x.device, scores.dtype)
            scores = scores + causal
            if key_mask is not None:
                keep = key_mask[:, -s_total:].to(device=x.device, dtype=torch.bool)
                scores = scores.masked_fill(~keep[:, None, None, :], float("-inf"))
            weights_s = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
            out = torch.matmul(weights_s, v)
        out = out.transpose(1, 2).contiguous().view(b, s_new, nq * hd)
        out = dense(
            out, weights[f"{p}.attn.c_proj.weight"], weights.get(f"{p}.attn.c_proj.bias")
        )
        return _mask_out(out, s_new)

    if kind == "fused_qkv":
        qkv = dense(x, weights[f"{p}.attn.qkv.weight"], weights.get(f"{p}.attn.qkv.bias"))
        recipe = config.recipe_id
        raw = config.raw or {}
        if getattr(config, "clip_qkv", None):
            qkv = qkv.clamp(-float(config.clip_qkv), float(config.clip_qkv))
        q, k, v = _split_qkv(
            qkv,
            nq,
            nkv,
            hd,
            gpt_neox=recipe in {"gpt_neox", "bloom"},
            falcon=recipe == "falcon",
            new_decoder=bool(raw.get("new_decoder_architecture", False)),
        )
        layer_rope = use_rope and (not getattr(config, "alibi", False)) and cos.numel() > 0
        if layer_rope:
            q, k = apply_rope(
                q, k, cos, sin, interleaved=bool(getattr(config, "rope_interleaved", False))
            )
        if cache is not None:
            k, v = cache.update(spec_index, k, v)
        scale = 1.0 / math.sqrt(hd)
        s_new = q.shape[2]
        if (
            _attention_mode() != "eager"
            and not getattr(config, "alibi", False)
            and q.is_floating_point()
        ):
            out = decode_attend(
                q,
                k,
                v,
                scale=scale,
                sliding_window=sliding_window,
                key_mask=key_mask,
            )
            if out is None:
                out = sdpa_attend(
                    q,
                    k,
                    v,
                    scale=scale,
                    sliding_window=sliding_window,
                    padding_mask=key_mask,
                    q_start=q_start,
                )
            out = out.transpose(1, 2).contiguous().view(x.shape[0], s_new, nq * hd)
            out = dense(
                out, weights[f"{p}.attn.o.weight"], weights.get(f"{p}.attn.o.bias")
            )
            return _mask_out(out, s_new)
        if recipe not in {"gpt_neox", "bloom"}:
            k = repeat_kv(k, nq // max(nkv, 1)) if nkv else k
            v = repeat_kv(v, nq // max(nkv, 1)) if nkv else v
        s_total = k.shape[2]
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
        if getattr(config, "alibi", False):
            bias = build_alibi(
                nq,
                s_new,
                s_total,
                x.device,
                getattr(config, "alibi_kind", "bloom"),
                float((raw.get("attn_config") or {}).get("alibi_bias_max") or raw.get("alibi_bias_max") or 8),
            )
            if getattr(config, "alibi_kind", "bloom") == "falcon":
                scores = (scores + bias) * scale
            else:
                scores = scores * scale + bias
        else:
            scores = scores * scale
        causal = _causal_mask(s_new, s_total, x.device, scores.dtype, sliding_window)
        scores = scores + causal
        if key_mask is not None:
            keep = key_mask[:, -s_total:].to(device=x.device, dtype=torch.bool)
            scores = scores.masked_fill(~keep[:, None, None, :], float("-inf"))
        attn_w = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
        out = torch.matmul(attn_w, v).transpose(1, 2).contiguous().view(x.shape[0], s_new, nq * hd)
        out = dense(out, weights[f"{p}.attn.o.weight"], weights.get(f"{p}.attn.o.bias"))
        return _mask_out(out, s_new)

    if kind == "mla":
        from engine.layers.mla import mla_attention

        return mla_attention(
            x,
            weights,
            spec_index,
            cos,
            sin,
            config,
            cache=cache,
            key_mask=key_mask,
            q_start=q_start,
        )

    if kind == "diff":
        return differential_attention(
            x,
            weights,
            spec_index,
            cos,
            sin,
            config,
            cache=cache,
            use_rope=use_rope,
            attention_mask=attention_mask,
            key_mask=key_mask,
            q_start=q_start,
        )

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
        q_norm_bias=weights.get(f"{p}.attn.q_norm.bias"),
        k_norm_bias=weights.get(f"{p}.attn.k_norm.bias"),
        sliding_window=sliding_window,
        sinks=weights.get(f"{p}.attn.sinks"),
        rms_eps=config.rms_norm_eps,
        output_gate=config.attn_output_gate,
        qk_gemma=config.norm_kind == "gemma_rms",
        attention_mask=attention_mask,
        attention_multiplier=getattr(config, "attention_multiplier", None),
        query_pre_attn_scalar=getattr(config, "query_pre_attn_scalar", None),
        attn_logit_softcapping=getattr(config, "attn_logit_softcapping", None),
        clip_qkv=getattr(config, "clip_qkv", None),
        rope_interleaved=bool(getattr(config, "rope_interleaved", False)),
        alibi=bool(getattr(config, "alibi", False)),
        alibi_kind=getattr(config, "alibi_kind", "bloom"),
        alibi_bias_max=float(
            ((config.raw or {}).get("attn_config") or {}).get("alibi_bias_max")
            or (config.raw or {}).get("alibi_bias_max")
            or 8
        ),
        qk_norm_after_rope=config.recipe_id == "hunyuan_v1_moe",
        sub_norm=weights.get(f"{p}.attn.sub_norm.weight"),
        kv_mask=kv_mask,
        q_positions=q_positions,
    )
