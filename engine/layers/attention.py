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
            causal = torch.where(
                band,
                torch.zeros((), device=device, dtype=dtype),
                torch.tensor(float("-inf"), device=device, dtype=dtype),
            )
        return causal
    q_pos = torch.arange(s_total - s_new, s_total, device=device, dtype=torch.long)[:, None]
    k_pos = torch.arange(s_total, device=device, dtype=torch.long)[None, :]
    causal = torch.where(
        k_pos > q_pos,
        torch.tensor(float("-inf"), device=device, dtype=dtype),
        torch.zeros((), device=device, dtype=dtype),
    )
    if sliding_window:
        causal = torch.where(
            k_pos < (q_pos - sliding_window + 1),
            torch.tensor(float("-inf"), device=device, dtype=dtype),
            causal,
        )
    return causal


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
) -> torch.Tensor:
    """Causal GQA attention. x: [B, S_new, H]."""
    b, s_new, _ = x.shape
    q = F.linear(x, w_q, b_q)
    k = F.linear(x, w_k, b_k)
    v = F.linear(x, w_v, b_v)
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

    k = repeat_kv(k, nq // nkv)
    v = repeat_kv(v, nq // nkv)

    if attention_multiplier is not None:
        scale = float(attention_multiplier)
    elif query_pre_attn_scalar is not None:
        scale = float(query_pre_attn_scalar) ** -0.5
    else:
        scale = 1.0 / math.sqrt(hd)
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
    if attention_mask is not None:
        if attention_mask.dim() != 2 or attention_mask.shape[0] != b:
            raise ValueError(
                f"expected attention_mask [B,S], got {tuple(attention_mask.shape)}"
            )
        key_mask = attention_mask[:, -s_total:].to(device=x.device, dtype=torch.bool)
        scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
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
    out = F.linear(out, w_o, b_o)
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
    causal = _causal_mask(s_new, s_total, x.device, scores.dtype)
    scores = scores + causal
    if attention_mask is not None:
        key_mask = attention_mask[:, -s_total:].to(device=x.device, dtype=torch.bool)
        scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
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
) -> torch.Tensor:
    p = f"layers.{spec_index}"
    nq, nkv, hd = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    kind = config.attention_kind
    sliding_window = _layer_sliding_window(config, spec_index)

    if kind in {"gpt2", "gpt_bigcode"}:
        qkv = F.linear(x, weights[f"{p}.attn.c_attn.weight"], weights.get(f"{p}.attn.c_attn.bias"))
        q, k, v = _split_qkv(qkv, nq, nkv, hd)
        b, s_new, _ = x.shape
        if cache is not None:
            k, v = cache.update(spec_index, k, v)
        k = repeat_kv(k, nq // max(nkv, 1)) if nkv else k
        v = repeat_kv(v, nq // max(nkv, 1)) if nkv else v
        s_total = k.shape[2]
        scale = 1.0 / math.sqrt(hd)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        causal = _causal_mask(s_new, s_total, x.device, scores.dtype)
        weights_s = torch.softmax(scores + causal, dim=-1).to(dtype=v.dtype)
        out = torch.matmul(weights_s, v).transpose(1, 2).contiguous().view(b, s_new, nq * hd)
        return F.linear(out, weights[f"{p}.attn.c_proj.weight"], weights.get(f"{p}.attn.c_proj.bias"))

    if kind == "fused_qkv":
        qkv = F.linear(x, weights[f"{p}.attn.qkv.weight"], weights.get(f"{p}.attn.qkv.bias"))
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
        if recipe not in {"gpt_neox", "bloom"}:
            k = repeat_kv(k, nq // max(nkv, 1)) if nkv else k
            v = repeat_kv(v, nq // max(nkv, 1)) if nkv else v
        scale = 1.0 / math.sqrt(hd)
        s_new, s_total = q.shape[2], k.shape[2]
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
        causal = _causal_mask(s_new, s_total, x.device, scores.dtype)
        attn_w = torch.softmax(scores + causal, dim=-1).to(dtype=v.dtype)
        out = torch.matmul(attn_w, v).transpose(1, 2).contiguous().view(x.shape[0], s_new, nq * hd)
        return F.linear(out, weights[f"{p}.attn.o.weight"], weights.get(f"{p}.attn.o.bias"))

    if kind == "mla":
        from engine.layers.mla import mla_attention

        return mla_attention(x, weights, spec_index, cos, sin, config, cache=cache)

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
    )
