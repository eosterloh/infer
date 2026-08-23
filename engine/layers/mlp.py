"""Dense FFNs: SwiGLU, GELU (GPT-2), fused gate_up (Phi-3), relu2."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mlp(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    act: str = "silu",
    b_gate: torch.Tensor | None = None,
    b_up: torch.Tensor | None = None,
    b_down: torch.Tensor | None = None,
) -> torch.Tensor:
    if act in {"silu", "swiglu"}:
        return F.linear(
            F.silu(F.linear(x, w_gate, b_gate)) * F.linear(x, w_up, b_up),
            w_down,
            b_down,
        )
    if act in {"gelu", "gelu_new", "gelu_pytorch_tanh"}:
        return F.linear(F.gelu(F.linear(x, w_up, b_up)), w_down, b_down)
    if act in {"relu2", "relu_squared", "squared_relu"}:
        h = F.linear(x, w_up, b_up)
        return F.linear(torch.square(F.relu(h)), w_down, b_down)
    raise ValueError(f"unsupported mlp act {act!r}")


def mlp_from_weights(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer: int,
    act: str,
) -> torch.Tensor:
    p = f"layers.{layer}"
    if f"{p}.mlp.gate_up.weight" in weights:
        gate, up = weights[f"{p}.mlp.gate_up.weight"].chunk(2, dim=0)
        return mlp(x, gate, up, weights[f"{p}.mlp.down.weight"], act="silu")
    if f"{p}.mlp.c_fc.weight" in weights:
        return mlp(
            x,
            weights[f"{p}.mlp.c_fc.weight"],
            weights[f"{p}.mlp.c_fc.weight"],
            weights[f"{p}.mlp.c_proj.weight"],
            act="gelu",
            b_up=weights.get(f"{p}.mlp.c_fc.bias"),
            b_down=weights.get(f"{p}.mlp.c_proj.bias"),
        )
    gate = weights.get(f"{p}.mlp.gate.weight")
    up = weights.get(f"{p}.mlp.up.weight")
    down = weights[f"{p}.mlp.down.weight"]
    if gate is None:
        return mlp(x, up, up, down, act=act or "relu2")
    return mlp(
        x,
        gate,
        up,
        down,
        act=act or "silu",
        b_gate=weights.get(f"{p}.mlp.gate.bias"),
        b_up=weights.get(f"{p}.mlp.up.bias"),
        b_down=weights.get(f"{p}.mlp.down.bias"),
    )
