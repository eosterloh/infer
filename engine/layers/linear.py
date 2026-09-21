"""Projection helper: cuBLAS for batches, the engine's GEMV for one token.

cuBLAS reaches roughly two thirds of the Spark's memory bandwidth on a
matrix-vector product. Decode is nothing but matrix-vector products, so single
row inputs take the engine's own kernel and everything else stays on cuBLAS,
which wins as soon as there is a tile of work per weight tile.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.kernels import gemv


def dense(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    if type(weight) is not torch.Tensor and hasattr(weight, "kind"):
        from engine.qweight import qlinear

        return qlinear(x, weight, bias)
    if x.numel() == x.shape[-1]:
        out = gemv(x, weight, bias)
        if out is not None:
            return out
    return F.linear(x, weight, bias)
