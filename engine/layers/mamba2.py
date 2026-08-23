"""Mamba-2 mixer (Nemotron-H) — pure PyTorch sequential SSM + depthwise conv.

Readable DIY path for agents/learning. Prefer fused kernels later for Spark throughput.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from engine.config import ModelConfig

if TYPE_CHECKING:
    from engine.cache import RuntimeState


def _rms_norm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
    group_size: int,
) -> torch.Tensor:
    """Match mamba_ssm rmsnorm_fn with norm_before_gate=False (group RMS)."""
    # y = rmsnorm(x * silu(gate)) * weight, RMS over each group
    y = x.float() * F.silu(gate.float())
    orig = y.shape
    d = orig[-1]
    if d % group_size != 0:
        raise ValueError(f"hidden {d} not divisible by group_size {group_size}")
    y = y.reshape(*orig[:-1], d // group_size, group_size)
    var = y.pow(2).mean(dim=-1, keepdim=True)
    y = y * torch.rsqrt(var + eps)
    y = y.reshape(orig) * weight.float()
    return y.to(dtype=x.dtype)


def _depthwise_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    kernel: int,
) -> torch.Tensor:
    """x: [B, S, C] → [B, S, C] causal depthwise conv (pad left)."""
    # weight: [C, 1, K]
    b, s, c = x.shape
    x_t = x.transpose(1, 2)  # [B, C, S]
    y = F.conv1d(x_t, weight, bias=bias, padding=kernel - 1, groups=c)
    y = y[..., :s]
    return y.transpose(1, 2)


def mamba2(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer: int,
    config: ModelConfig,
    cache: RuntimeState | None = None,
) -> torch.Tensor:
    """Nemotron-H Mamba-2 mixer. x: [B, S, H] (already prenormed)."""
    if (
        config.mamba_num_heads is None
        or config.mamba_head_dim is None
        or config.ssm_state_size is None
        or config.n_groups is None
        or config.conv_kernel is None
    ):
        raise ValueError("Mamba-2 requires mamba_* / ssm / n_groups / conv_kernel on config")

    p = f"layers.{layer}"
    b, s, _ = x.shape
    n_heads = config.mamba_num_heads
    head_dim = config.mamba_head_dim
    n_groups = config.n_groups
    state_size = config.ssm_state_size
    kernel = config.conv_kernel
    inter = config.mamba_intermediate
    conv_dim = config.mamba_conv_dim
    dtype = x.dtype
    device = x.device

    in_proj = weights[f"{p}.mamba.in_proj.weight"]
    out_proj = weights[f"{p}.mamba.out_proj.weight"]
    conv_w = weights[f"{p}.mamba.conv1d.weight"]
    conv_b = weights.get(f"{p}.mamba.conv1d.bias")
    A_log = weights[f"{p}.mamba.A_log"]
    D = weights[f"{p}.mamba.D"]
    dt_bias = weights[f"{p}.mamba.dt_bias"]
    norm_w = weights[f"{p}.mamba.norm.weight"]

    projected = F.linear(x, in_proj)
    # d_mlp is 0 for Nemotron-H: split = gate | B_C | dt
    gate, bc, dt = projected.split([inter, conv_dim, n_heads], dim=-1)

    # Decode = single new token after we've already primed mamba state.
    decode = (
        cache is not None
        and cache.conv_states[layer] is not None
        and cache.mamba_ready(layer)
        and s == 1
    )

    if decode:
        assert cache is not None
        conv_state = cache.update_conv_step(layer, bc)
        # Depthwise FIR on the window: sum_k state[...,k] * w[...,k]
        bc_out = torch.sum(conv_state * conv_w.squeeze(1), dim=-1)
        if conv_b is not None:
            bc_out = bc_out + conv_b
        bc_out = F.silu(bc_out)[:, None, :]  # [B, 1, conv_dim]
    else:
        if cache is not None and cache.conv_states[layer] is not None:
            # Prefill: store last `kernel` tokens of bc (left-pad if short)
            bc_t = bc.transpose(1, 2)  # [B, C, S]
            if s >= kernel:
                conv_state = bc_t[:, :, -kernel:].contiguous()
            else:
                conv_state = F.pad(bc_t, (kernel - s, 0))
            cache.update_conv_prefill(layer, conv_state)
        bc_out = F.silu(_depthwise_conv1d(bc, conv_w, conv_b, kernel))

    x_ssm, B, C = bc_out.split(
        [inter, n_groups * state_size, n_groups * state_size], dim=-1
    )

    A = -torch.exp(A_log.float())  # [n_heads]
    dt = F.softplus(dt.float() + dt_bias.float())
    # Clamp using config limits if present
    raw = config.raw or {}
    dt_limit = raw.get("time_step_limit") or raw.get("mamba_dt_limit")
    if dt_limit is not None and len(dt_limit) == 2:
        lo, hi = float(dt_limit[0]), float(dt_limit[1])
        if hi != float("inf"):
            dt = torch.clamp(dt, lo, hi)
        elif lo > 0:
            dt = torch.clamp(dt, min=lo)

    # Reshape for multi-head SSM
    # x: [B,S,H,D], B/C: [B,S,n_groups,N] → expand to heads
    x_h = x_ssm.reshape(b, s, n_heads, head_dim).float()
    B_g = B.reshape(b, s, n_groups, state_size).float()
    C_g = C.reshape(b, s, n_groups, state_size).float()
    reps = n_heads // n_groups
    B_h = B_g.repeat_interleave(reps, dim=2)  # [B,S,H,N]
    C_h = C_g.repeat_interleave(reps, dim=2)

    if decode:
        assert cache is not None and cache.ssm_states[layer] is not None
        # Single-step recurrence (matches HF decode branch math)
        dt_t = dt[:, 0, :]  # [B, H]
        dt_exp = dt_t[:, :, None].expand(b, n_heads, head_dim)  # [B,H,D]
        A_exp = A[None, :, None, None].expand(b, n_heads, head_dim, state_size)
        dA = torch.exp(dt_exp[..., None] * A_exp)  # [B,H,D,N]
        B_t = B_h[:, 0]  # [B,H,N]
        dB = dt_exp[..., None] * B_t[:, :, None, :]  # [B,H,D,N]
        x_t = x_h[:, 0]  # [B,H,D]
        dBx = dB * x_t[..., None]
        prev = cache.ssm_states[layer].float()
        new_state = prev * dA + dBx
        cache.update_ssm(layer, new_state)

        C_t = C_h[:, 0]  # [B,H,N]
        y = torch.einsum("bhdn,bhn->bhd", new_state, C_t)
        y = y + x_t * D.float()[None, :, None]
        y = y.reshape(b, 1, inter)
    else:
        # Sequential scan (correct, DIY-readable; not the chunk SSD fast path)
        if cache is not None and cache.mamba_ready(layer) and cache.ssm_states[layer] is not None:
            state = cache.ssm_states[layer].float().clone()
        else:
            state = torch.zeros(
                b, n_heads, head_dim, state_size, device=device, dtype=torch.float32
            )
        ys = []
        A_exp = A[None, :, None, None]  # [1,H,1,1] broadcasts with D,N
        for t in range(s):
            dt_t = dt[:, t, :]  # [B,H]
            dt_exp = dt_t[:, :, None].expand(b, n_heads, head_dim)
            dA = torch.exp(dt_exp[..., None] * A_exp.expand(b, n_heads, head_dim, state_size))
            B_t = B_h[:, t]
            dB = dt_exp[..., None] * B_t[:, :, None, :]
            x_t = x_h[:, t]
            state = state * dA + dB * x_t[..., None]
            C_t = C_h[:, t]
            y_t = torch.einsum("bhdn,bhn->bhd", state, C_t)
            y_t = y_t + x_t * D.float()[None, :, None]
            ys.append(y_t)
        y = torch.stack(ys, dim=1).reshape(b, s, inter)
        if cache is not None:
            cache.update_ssm(layer, state)

    group_size = inter // n_groups
    scan_out = _rms_norm_gated(
        y.to(dtype),
        norm_w,
        gate,
        eps=config.rms_norm_eps,
        group_size=group_size,
    )
    return F.linear(scan_out, out_proj)
