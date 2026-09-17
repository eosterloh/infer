"""Mamba-1 mixer (Jamba) — selective SSM with x_proj / dt_proj."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from engine.config import ModelConfig
from engine.layers.mamba2 import _depthwise_conv1d
from engine.layers.norm import rms_norm

if TYPE_CHECKING:
    from engine.cache import RuntimeState


def mamba1(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer: int,
    config: ModelConfig,
    cache: RuntimeState | None = None,
) -> torch.Tensor:
    """Jamba Mamba-1 mixer. x: [B, S, H] (already prenormed)."""
    raw = config.raw or {}
    p = f"layers.{layer}"
    b, s, h = x.shape
    expand = int(raw.get("mamba_expand") or 2)
    inter = expand * h
    state = int(raw.get("mamba_d_state") or config.ssm_state_size or 16)
    kernel = int(raw.get("mamba_d_conv") or config.conv_kernel or 4)
    dtype = x.dtype
    device = x.device
    act = str(config.hidden_act or "silu")

    projected = F.linear(
        x,
        weights[f"{p}.mamba1.in_proj.weight"],
        weights.get(f"{p}.mamba1.in_proj.bias"),
    )
    hidden_bc, gate = projected.split(inter, dim=-1)
    conv_w = weights[f"{p}.mamba1.conv1d.weight"]
    conv_b = weights.get(f"{p}.mamba1.conv1d.bias")

    decode = (
        cache is not None
        and cache.conv_states[layer] is not None
        and cache.mamba_ready(layer)
        and s == 1
    )
    if decode:
        assert cache is not None
        conv_state = cache.update_conv_step(layer, hidden_bc)
        hidden_bc = torch.sum(conv_state * conv_w.squeeze(1), dim=-1)
        if conv_b is not None:
            hidden_bc = hidden_bc + conv_b
        hidden_bc = hidden_bc[:, None, :]
        if act == "silu":
            hidden_bc = F.silu(hidden_bc)
        else:
            hidden_bc = F.gelu(hidden_bc)
    else:
        if cache is not None and cache.conv_states[layer] is not None:
            bc_t = hidden_bc.transpose(1, 2)
            if s >= kernel:
                conv_state = bc_t[:, :, -kernel:].contiguous()
            else:
                conv_state = F.pad(bc_t, (kernel - s, 0))
            cache.update_conv_prefill(layer, conv_state)
        hidden_bc = _depthwise_conv1d(hidden_bc, conv_w, conv_b, kernel)
        hidden_bc = F.silu(hidden_bc) if act == "silu" else F.gelu(hidden_bc)

    dt_rank = weights[f"{p}.mamba1.dt_proj.weight"].shape[1]
    projected_bc = F.linear(hidden_bc, weights[f"{p}.mamba1.x_proj.weight"])
    time_step, B, C = projected_bc.split((dt_rank, state, state), dim=-1)
    time_step = rms_norm(time_step, weights[f"{p}.mamba1.dt_norm.weight"], config.rms_norm_eps)
    B = rms_norm(B, weights[f"{p}.mamba1.b_norm.weight"], config.rms_norm_eps)
    C = rms_norm(C, weights[f"{p}.mamba1.c_norm.weight"], config.rms_norm_eps)
    # dt_proj without bias, then add bias inside softplus (HF decode/prefill).
    time_step = F.linear(time_step, weights[f"{p}.mamba1.dt_proj.weight"])
    dt_bias = weights[f"{p}.mamba1.dt_proj.bias"].float()
    A = -torch.exp(weights[f"{p}.mamba1.A_log"].float())
    D = weights[f"{p}.mamba1.D"].float()

    x_ssm = hidden_bc.float()
    B_f = B.float()
    C_f = C.float()
    dt = F.softplus(time_step.float() + dt_bias)
    gate_f = gate.float()

    if decode:
        assert cache is not None and cache.ssm_states[layer] is not None
        prev = cache.ssm_states[layer].float()
        dt_t = dt[:, 0]
        x_t = x_ssm[:, 0]
        B_t = B_f[:, 0]
        C_t = C_f[:, 0]
        dA = torch.exp(dt_t.unsqueeze(-1) * A)
        dB = dt_t.unsqueeze(-1) * B_t
        new_state = prev * dA + dB * x_t.unsqueeze(-1)
        cache.update_ssm(layer, new_state)
        y = (new_state * C_t.unsqueeze(1)).sum(dim=-1) + D * x_t
        y = y[:, None, :]
    else:
        if cache is not None and cache.mamba_ready(layer) and cache.ssm_states[layer] is not None:
            state = cache.ssm_states[layer].float().clone()
        else:
            state = torch.zeros(b, inter, state, device=device, dtype=torch.float32)
        ys = []
        for t in range(s):
            dt_t = dt[:, t]
            x_t = x_ssm[:, t]
            B_t = B_f[:, t]
            C_t = C_f[:, t]
            dA = torch.exp(dt_t.unsqueeze(-1) * A)
            dB = dt_t.unsqueeze(-1) * B_t
            state = state * dA + dB * x_t.unsqueeze(-1)
            y_t = (state * C_t.unsqueeze(1)).sum(dim=-1) + D * x_t
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        if cache is not None:
            cache.update_ssm(layer, state)

    y = y * F.silu(gate_f)
    return F.linear(
        y.to(dtype),
        weights[f"{p}.mamba1.out_proj.weight"],
        weights.get(f"{p}.mamba1.out_proj.bias"),
    )
