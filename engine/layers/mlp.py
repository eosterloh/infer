"""Dense FFNs: SwiGLU, GELU (GPT-2), fused gate_up (Phi-3), relu2."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.kernels import act_and_mul, act_mul
from engine.layers.linear import dense


def _gelu(x: torch.Tensor, act: str) -> torch.Tensor:
    if act in {"gelu_pytorch_tanh", "gelu_new", "gelu_fast"}:
        return F.gelu(x, approximate="tanh")
    return F.gelu(x)


def _fused_gelu_kind(act: str) -> str:
    return "gelu_tanh" if act in {"gelu_pytorch_tanh", "gelu_new", "gelu_fast"} else "gelu"


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
    gated = w_gate is not w_up
    if act in {"silu", "swiglu"}:
        return dense(
            act_mul(dense(x, w_gate, b_gate), dense(x, w_up, b_up), "silu"),
            w_down,
            b_down,
        )
    if act in {"gelu", "gelu_new", "gelu_pytorch_tanh"}:
        up = dense(x, w_up, b_up)
        if gated:
            up = act_mul(dense(x, w_gate, b_gate), up, _fused_gelu_kind(act))
        else:
            up = _gelu(up, act)
        return dense(up, w_down, b_down)
    if act in {"relu2", "relu_squared", "squared_relu"}:
        h = dense(x, w_up, b_up)
        return dense(torch.square(F.relu(h)), w_down, b_down)
    if act == "relu":
        h = dense(x, w_up, b_up)
        return dense(F.relu(h), w_down, b_down)
    raise ValueError(f"unsupported mlp act {act!r}")


def mlp_from_weights(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer: int,
    act: str,
) -> torch.Tensor:
    p = f"layers.{layer}"
    sub = weights.get(f"{p}.mlp.sub_norm.weight")
    if f"{p}.mlp.gate_up.weight" in weights:
        # One GEMM over the packed projection, then split inside the kernel.
        h = act_and_mul(dense(x, weights[f"{p}.mlp.gate_up.weight"]), "silu")
        if sub is not None:
            from engine.layers.norm import rms_norm

            h = rms_norm(h, sub, 1e-5)
        return dense(h, weights[f"{p}.mlp.down.weight"])
    if f"{p}.mlp.c_fc.weight" in weights:
        return mlp(
            x,
            weights[f"{p}.mlp.c_fc.weight"],
            weights[f"{p}.mlp.c_fc.weight"],
            weights[f"{p}.mlp.c_proj.weight"],
            act=act or "gelu",
            b_up=weights.get(f"{p}.mlp.c_fc.bias"),
            b_down=weights.get(f"{p}.mlp.c_proj.bias"),
        )
    gate = weights.get(f"{p}.mlp.gate.weight")
    up = weights.get(f"{p}.mlp.up.weight")
    down = weights[f"{p}.mlp.down.weight"]
    if gate is None:
        return mlp(
            x,
            up,
            up,
            down,
            act=act or "relu2",
            b_up=weights.get(f"{p}.mlp.up.bias"),
            b_down=weights.get(f"{p}.mlp.down.bias"),
        )
    if sub is not None:
        from engine.layers.norm import rms_norm

        h = act_mul(
            dense(x, gate, weights.get(f"{p}.mlp.gate.bias")),
            dense(x, up, weights.get(f"{p}.mlp.up.bias")),
            "silu",
        )
        h = rms_norm(h, sub, 1e-5)
        return dense(h, down, weights.get(f"{p}.mlp.down.bias"))
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
