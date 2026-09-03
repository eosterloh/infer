"""Gated DeltaNet mixer (Qwen3.5 / Qwen3.8) — sequential PyTorch path.

Matches transformers `Qwen3_5GatedDeltaNet` + `torch_recurrent_gated_delta_rule`.
Prefill walks the sequence; decode is one recurrent step. Not the chunk kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from engine.config import ModelConfig
from engine.layers.mamba2 import _depthwise_conv1d
from engine.layers.norm import rms_norm

if TYPE_CHECKING:
    from engine.cache import RuntimeState


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _gated_delta_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_log: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """query/key/value: [B, S, H, D]; g_log/beta: [B, S, H]; state: [B, H, Dk, Dv]."""
    query = _l2norm(query)
    key = _l2norm(key)
    q, k, v, beta_f, g_f = (
        t.transpose(1, 2).contiguous().float()
        for t in (query, key, value, beta, g_log)
    )
    b, n_heads, seq_len, k_dim = k.shape
    v_dim = v.shape[-1]
    scale = k_dim ** -0.5
    q = q * scale

    if state is None:
        rec = torch.zeros(b, n_heads, k_dim, v_dim, device=k.device, dtype=torch.float32)
    else:
        rec = state.float()

    outs = []
    for i in range(seq_len):
        q_t = q[:, :, i]
        k_t = k[:, :, i]
        v_t = v[:, :, i]
        g_t = g_f[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta_f[:, :, i].unsqueeze(-1)
        rec = rec * g_t
        kv_mem = (rec * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        rec = rec + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((rec * q_t.unsqueeze(-1)).sum(dim=-2))
    out = torch.stack(outs, dim=2).transpose(1, 2).contiguous().to(dtype=query.dtype)
    return out, rec


def gated_delta_net(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer: int,
    config: ModelConfig,
    cache: RuntimeState | None = None,
) -> torch.Tensor:
    """Qwen3.5 linear-attention mixer. x: [B, S, H] (already prenormed)."""
    if (
        config.linear_num_key_heads is None
        or config.linear_num_value_heads is None
        or config.linear_key_head_dim is None
        or config.linear_value_head_dim is None
        or config.linear_conv_kernel_dim is None
    ):
        raise ValueError("Gated DeltaNet requires linear_* dims on config")

    p = f"layers.{layer}"
    b, s, _ = x.shape
    n_k = config.linear_num_key_heads
    n_v = config.linear_num_value_heads
    dk = config.linear_key_head_dim
    dv = config.linear_value_head_dim
    key_dim = n_k * dk
    value_dim = n_v * dv
    kernel = config.linear_conv_kernel_dim
    dtype = x.dtype

    mixed = F.linear(x, weights[f"{p}.gdn.in_proj_qkv.weight"])
    z = F.linear(x, weights[f"{p}.gdn.in_proj_z.weight"]).view(b, s, n_v, dv)
    beta_raw = F.linear(x, weights[f"{p}.gdn.in_proj_b.weight"])
    a_raw = F.linear(x, weights[f"{p}.gdn.in_proj_a.weight"])
    conv_w = weights[f"{p}.gdn.conv1d.weight"]
    a_log = weights[f"{p}.gdn.A_log"]
    dt_bias = weights[f"{p}.gdn.dt_bias"]
    norm_w = weights[f"{p}.gdn.norm.weight"]
    out_proj = weights[f"{p}.gdn.out_proj.weight"]

    decode = (
        cache is not None
        and cache.conv_states[layer] is not None
        and cache.mamba_ready(layer)
        and s == 1
    )

    if decode:
        assert cache is not None
        conv_state = cache.update_conv_step(layer, mixed)
        mixed_out = torch.sum(conv_state * conv_w.squeeze(1), dim=-1)
        mixed_out = F.silu(mixed_out)[:, None, :]
    else:
        if cache is not None and cache.conv_states[layer] is not None:
            mixed_t = mixed.transpose(1, 2)
            if s >= kernel:
                conv_state = mixed_t[:, :, -kernel:].contiguous()
            else:
                conv_state = F.pad(mixed_t, (kernel - s, 0))
            cache.update_conv_prefill(layer, conv_state)
        mixed_out = F.silu(_depthwise_conv1d(mixed, conv_w, None, kernel))

    query, key, value = mixed_out.split((key_dim, key_dim, value_dim), dim=-1)
    query = query.view(b, s, n_k, dk)
    key = key.view(b, s, n_k, dk)
    value = value.view(b, s, n_v, dv)
    if n_v // n_k > 1:
        query = query.repeat_interleave(n_v // n_k, dim=2)
        key = key.repeat_interleave(n_v // n_k, dim=2)

    beta = torch.sigmoid(beta_raw)
    g_log = -a_log.float().exp() * F.softplus(a_raw.float() + dt_bias.float())

    initial = None
    if decode:
        assert cache is not None
        initial = cache.ssm_states[layer]
    elif cache is not None and cache.mamba_ready(layer) and cache.ssm_states[layer] is not None:
        initial = cache.ssm_states[layer]

    core, rec = _gated_delta_recurrent(query, key, value, g_log, beta, initial)
    if cache is not None:
        cache.update_ssm(layer, rec)

    core = core.reshape(b, s, n_v, dv)
    core_n = rms_norm(core, norm_w, config.rms_norm_eps)
    core_n = core_n * F.silu(z.float()).to(dtype=dtype)
    return F.linear(core_n.reshape(b, s, value_dim), out_proj)
