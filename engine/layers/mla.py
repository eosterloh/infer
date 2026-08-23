"""DeepSeek Multi-head Latent Attention (MLA)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from engine.layers.norm import rms_norm
from engine.layers.rope import apply_rope

if TYPE_CHECKING:
    from engine.cache import KVCache, RuntimeState
    from engine.config import ModelConfig


def mla_attention(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: ModelConfig,
    cache: KVCache | RuntimeState | None = None,
) -> torch.Tensor:
    """x: [B, S, H] → [B, S, H]. Caches expanded K/V for greedy decode."""
    p = f"layers.{layer}"
    b, s, _ = x.shape
    nq = config.num_attention_heads
    nope = int(config.qk_nope_head_dim or config.head_dim)
    rope_d = int(config.qk_rope_head_dim or config.head_dim)
    vdh = int(config.v_head_dim or config.head_dim)
    kv_lora = int(config.kv_lora_rank or x.shape[-1])
    qk = nope + rope_d

    if f"{p}.attn.q_a.weight" in weights:
        q = F.linear(x, weights[f"{p}.attn.q_a.weight"])
        q = rms_norm(q, weights[f"{p}.attn.q_a_norm.weight"], config.rms_norm_eps)
        q = F.linear(q, weights[f"{p}.attn.q_b.weight"])
    else:
        q = F.linear(x, weights[f"{p}.attn.q.weight"])
    q = q.view(b, s, nq, qk).transpose(1, 2)
    q_pass, q_rot = torch.split(q, [nope, rope_d], dim=-1)

    compressed = F.linear(x, weights[f"{p}.attn.kv_a.weight"])
    kv_nope, k_rot = torch.split(compressed, [kv_lora, rope_d], dim=-1)
    kv_nope = rms_norm(kv_nope, weights[f"{p}.attn.kv_a_norm.weight"], config.rms_norm_eps)

    if cos.numel():
        # RoPE only on the rotary slice; broadcast cos/sin to rope_d.
        if cos.shape[-1] != rope_d:
            cos_r = cos[..., :rope_d]
            sin_r = sin[..., :rope_d]
        else:
            cos_r, sin_r = cos, sin
        q_rot, k_rot_t = apply_rope(
            q_rot,
            k_rot.view(b, s, 1, rope_d).transpose(1, 2),
            cos_r,
            sin_r,
        )
        k_rot = k_rot_t.transpose(1, 2).reshape(b, s, rope_d)
    k_rot = k_rot.view(b, s, 1, rope_d).expand(b, s, nq, rope_d).transpose(1, 2)

    kv = F.linear(kv_nope, weights[f"{p}.attn.kv_b.weight"])
    kv = kv.view(b, s, nq, nope + vdh).transpose(1, 2)
    k_nope, v = torch.split(kv, [nope, vdh], dim=-1)
    k = torch.cat((k_nope, q_rot.new_empty(b, nq, s, rope_d)), dim=-1)
    k[..., :nope] = k_nope
    k[..., nope:] = k_rot
    q = torch.cat((q_pass, q_rot), dim=-1)

    if cache is not None:
        k, v = cache.update(layer, k, v)

    scale = 1.0 / math.sqrt(qk)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    s_new, s_total = q.shape[2], k.shape[2]
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
    attn = torch.softmax(scores + causal, dim=-1).to(dtype=v.dtype)
    out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s_new, nq * vdh)
    return F.linear(out, weights[f"{p}.attn.o.weight"])
