"""Gated DeltaNet mixer (Qwen3.5 / Qwen3.8).

Matches transformers `Qwen3_5GatedDeltaNet` + `torch_recurrent_gated_delta_rule`.
The recurrent path is the reference; CUDA prefill can use the equivalent
64-token chunk rule to reduce Python launch overhead.
"""

from __future__ import annotations

import os
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
    trajectory: list[torch.Tensor] | None = None,
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
        if trajectory is not None:
            trajectory.append(rec)
        outs.append((rec * q_t.unsqueeze(-1)).sum(dim=-2))
    out = torch.stack(outs, dim=2).transpose(1, 2).contiguous().to(dtype=query.dtype)
    return out, rec


def _gated_delta_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_log: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallel-within-chunk form of the same gated delta recurrence."""
    original_dtype = query.dtype
    query, key = _l2norm(query), _l2norm(key)
    q, k, v, beta_f, g = (
        tensor.transpose(1, 2).contiguous().float()
        for tensor in (query, key, value, beta, g_log)
    )
    b, heads, seq_len, k_dim = k.shape
    v_dim = v.shape[-1]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    q, k, v = [F.pad(tensor, (0, 0, 0, pad)) for tensor in (q, k, v)]
    beta_f, g = [F.pad(tensor, (0, pad)) for tensor in (beta_f, g)]
    total = seq_len + pad
    q = q * (k_dim**-0.5)

    v_beta = v * beta_f.unsqueeze(-1)
    k_beta = k * beta_f.unsqueeze(-1)
    q, k, v, k_beta, v_beta = [
        tensor.reshape(b, heads, -1, chunk_size, tensor.shape[-1])
        for tensor in (q, k, v, k_beta, v_beta)
    ]
    g = g.reshape(b, heads, -1, chunk_size).cumsum(dim=-1)
    causal = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device)
    )
    decay = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()
    transform = -((k_beta @ k.transpose(-1, -2)) * decay).masked_fill(
        causal, 0
    )
    for i in range(1, chunk_size):
        row = transform[..., i, :i].clone()
        prior = transform[..., :i, :i].clone()
        transform[..., i, :i] = row + (row.unsqueeze(-1) * prior).sum(-2)
    transform = transform + torch.eye(
        chunk_size, dtype=transform.dtype, device=transform.device
    )
    v = transform @ v_beta
    k_decay = transform @ (k_beta * g.exp().unsqueeze(-1))
    rec = (
        torch.zeros(b, heads, k_dim, v_dim, device=q.device, dtype=torch.float32)
        if state is None
        else state.float()
    )
    out = torch.zeros_like(v)
    for i in range(total // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        local = q_i @ k_i.transpose(-1, -2) * decay[:, :, i]
        v_new = v_i - k_decay[:, :, i] @ rec
        inter = (q_i * g[:, :, i, :, None].exp()) @ rec
        out[:, :, i] = inter + local @ v_new
        rec = (
            rec * g[:, :, i, -1, None, None].exp()
            + (
                k_i
                * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
            ).transpose(-1, -2)
            @ v_new
        )
    out = out.reshape(b, heads, total, v_dim)[:, :, :seq_len]
    return out.transpose(1, 2).contiguous().to(original_dtype), rec


def gated_delta_net(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer: int,
    config: ModelConfig,
    cache: RuntimeState | None = None,
    attention_mask: torch.Tensor | None = None,
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
    if attention_mask is not None:
        x = x * attention_mask[:, -s:, None].to(device=x.device, dtype=x.dtype)

    mixed = F.linear(x, weights[f"{p}.gdn.in_proj_qkv.weight"])
    z = F.linear(x, weights[f"{p}.gdn.in_proj_z.weight"]).view(b, s, n_v, dv)
    beta_raw = F.linear(x, weights[f"{p}.gdn.in_proj_b.weight"])
    a_raw = F.linear(x, weights[f"{p}.gdn.in_proj_a.weight"])
    conv_w = weights[f"{p}.gdn.conv1d.weight"]
    a_log = weights[f"{p}.gdn.A_log"]
    dt_bias = weights[f"{p}.gdn.dt_bias"]
    norm_w = weights[f"{p}.gdn.norm.weight"]
    out_proj = weights[f"{p}.gdn.out_proj.weight"]

    continuation = (
        cache is not None
        and cache.conv_states[layer] is not None
        and cache.mamba_ready(layer)
    )

    if continuation:
        assert cache is not None
        previous = cache.conv_states[layer]
        assert previous is not None
        joined = torch.cat((previous, mixed.transpose(1, 2)), dim=-1)
        mixed_out = F.conv1d(
            joined,
            conv_w,
            bias=None,
            padding=0,
            groups=joined.shape[1],
        )[..., -s:]
        cache.update_conv_prefill(layer, joined[..., -kernel:].contiguous())
        mixed_out = F.silu(mixed_out.transpose(1, 2))
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
    if continuation:
        assert cache is not None
        initial = cache.ssm_states[layer]
    elif cache is not None and cache.mamba_ready(layer) and cache.ssm_states[layer] is not None:
        initial = cache.ssm_states[layer]

    trajectory: list[torch.Tensor] | None = (
        [] if cache is not None and cache.is_speculating else None
    )
    use_chunk = (
        x.is_cuda
        and s >= 64
        and trajectory is None
        and os.environ.get("INFER_GDN_CHUNK", "1") != "0"
    )
    if use_chunk:
        core, rec = _gated_delta_chunk(query, key, value, g_log, beta, initial)
    else:
        core, rec = _gated_delta_recurrent(
            query, key, value, g_log, beta, initial, trajectory
        )
    if cache is not None:
        cache.update_ssm(layer, rec)
        if trajectory is not None:
            cache.record_gdn_speculation(layer, trajectory, mixed)

    core = core.reshape(b, s, n_v, dv)
    core_n = rms_norm(core, norm_w, config.rms_norm_eps)
    core_n = core_n * F.silu(z.float()).to(dtype=dtype)
    return F.linear(core_n.reshape(b, s, value_dim), out_proj)
