"""Dequant NVFP4 / FP8 checkpoints to compute dtypes on load.

Fused kernels are not required for a working greedy path: unpack onto the
device as BF16/FP16/FP32, then run the existing decoder.
"""

from __future__ import annotations

import torch

# NVIDIA FP4 E2M1 codes (nibble 0..15) → float.
_FP4_E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def pack_nvfp4(values: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack float weights to NVFP4 nibbles + per-group FP32 scales (test helper)."""
    orig_shape = tuple(values.shape)
    flat = values.detach().float().reshape(-1)
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    grouped = flat.view(-1, group_size)
    scales = grouped.abs().amax(dim=-1).clamp(min=1e-8) / 6.0
    q = grouped / scales.unsqueeze(-1)
    table = _FP4_E2M1.to(device=q.device)
    idx = (q.unsqueeze(-1) - table.view(1, 1, -1)).abs().argmin(dim=-1)
    low = idx[:, 0::2]
    high = idx[:, 1::2]
    packed = (low + (high << 4)).to(torch.uint8).reshape(-1)
    if orig_shape and orig_shape[-1] % 2 == 0:
        packed = packed.view(*orig_shape[:-1], orig_shape[-1] // 2)
    return packed, scales.float()


def dequant_nvfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    shape: tuple[int, ...],
    group_size: int = 16,
) -> torch.Tensor:
    """Unpack uint8 nibbles + per-group scales → float32 tensor of `shape`."""
    table = _FP4_E2M1.to(device=packed.device)
    packed_u = packed.reshape(-1).to(torch.uint8)
    low = packed_u & 0x0F
    high = packed_u >> 4
    codes = torch.stack((low, high), dim=1).reshape(-1)
    n = 1
    for d in shape:
        n *= d
    codes = codes[:n]
    vals = table[codes.long()]
    n_pad = ((n + group_size - 1) // group_size) * group_size
    if vals.numel() < n_pad:
        vals = torch.nn.functional.pad(vals, (0, n_pad - vals.numel()))
    grouped = vals[:n_pad].view(-1, group_size)
    scales = scales.reshape(-1).float()[: grouped.shape[0]]
    out = (grouped * scales.unsqueeze(-1)).reshape(-1)[:n]
    return out.view(*shape)


def dequant_fp8(tensor: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Interpret uint8/float8 storage as E4M3FN values, optional per-tensor scale."""
    if tensor.dtype in {torch.float16, torch.bfloat16, torch.float32}:
        out = tensor.float()
    else:
        # E4M3FN: 1 sign, 4 exp, 3 mantissa, bias 7. NaN when exp=15 and mant!=0.
        x = tensor.view(torch.uint8).to(torch.int32)
        sign = torch.where((x & 0x80) != 0, -1.0, 1.0)
        exp = (x >> 3) & 0x0F
        mant = x & 0x07
        is_zero = (exp == 0) & (mant == 0)
        is_sub = (exp == 0) & (mant != 0)
        is_nan = (exp == 15) & (mant != 0)
        mag = torch.where(
            is_sub,
            (mant.float() / 8.0) * (2.0 ** -6),
            torch.where(
                is_zero,
                torch.zeros_like(sign),
                (1.0 + mant.float() / 8.0) * torch.pow(torch.tensor(2.0, device=x.device), exp.float() - 7),
            ),
        )
        out = torch.where(is_nan, torch.full_like(sign, float("nan")), sign * mag)
    if scale is not None:
        out = out * scale.float().reshape(-1)[0]
    return out


def maybe_dequant_state(
    hf_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Replace packed NVFP4/FP8 weights with float tensors; drop aux scale keys."""
    from engine.maps import is_quant_aux

    scales: dict[str, torch.Tensor] = {}
    for name, tensor in hf_state.items():
        if name.endswith(".weight_scale") or name.endswith(".weight_scale_inv"):
            base = name.rsplit(".", 1)[0]
            if name.endswith("_inv"):
                base = name[: -len(".weight_scale_inv")] + ".weight"
            else:
                base = name[: -len(".weight_scale")] + ".weight"
            scales[base] = tensor

    out: dict[str, torch.Tensor] = {}
    for name, tensor in hf_state.items():
        if is_quant_aux(name):
            continue
        scale = scales.get(name)
        if tensor.dtype == torch.uint8 and scale is not None:
            if tensor.dim() >= 1:
                unpacked_shape = (*tuple(tensor.shape[:-1]), int(tensor.shape[-1]) * 2)
            else:
                unpacked_shape = (tensor.numel() * 2,)
            out[name] = dequant_nvfp4(tensor, scale, unpacked_shape)
        elif str(tensor.dtype).startswith("torch.float8") or (
            tensor.dtype == torch.uint8 and scale is None and "fp8" in name.lower()
        ):
            out[name] = dequant_fp8(tensor, scale)
        elif scale is not None and tensor.dtype in {torch.float16, torch.bfloat16, torch.float32}:
            out[name] = tensor.float() * scale.float().reshape(-1)[0]
        else:
            out[name] = tensor
    return out
