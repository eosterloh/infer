"""Dense SwiGLU MLP (Llama FFN)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mlp(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)
