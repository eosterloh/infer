"""Packed weights that stay packed.

Decode at batch one is bandwidth bound: a step reads every weight exactly once,
so 4-bit weights decode about four times faster than BF16 on the same GPU. The
engine keeps the packed bytes resident and dequantizes inside the GEMV, instead
of unpacking the checkpoint into BF16 at load time.

Formats
    int4    group-wise affine, ``w = q * scale + zero``, uint8 nibble pairs
    nvfp4   NVIDIA FP4 (E2M1) with an FP8 E4M3 scale per 16 values and one
            FP32 global scale — the format NVFP4 checkpoints ship in
    fp8      E4M3 bytes with a per-row or per-tensor FP32 scale
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from engine.kernels import _ops

KIND_INT4 = 0
KIND_NVFP4 = 1
KIND_FP8 = 2

_KIND_CODE = {"int4": KIND_INT4, "nvfp4": KIND_NVFP4, "fp8": KIND_FP8}
_BITS = {"int4": 4.0, "nvfp4": 4.0, "fp8": 8.0}


@dataclass
class QuantWeight:
    """A packed [out, in] projection weight.

    Carries enough of the tensor API (``shape``, ``dtype``, ``device``,
    ``numel``) that the engine's weight dict and parameter counting keep
    working without special cases.
    """

    kind: str
    qweight: torch.Tensor
    scales: torch.Tensor | None
    zeros: torch.Tensor | None
    channel_scale: torch.Tensor | None
    global_scale: float
    out_features: int
    in_features: int
    group_size: int
    compute_dtype: torch.dtype
    # Set when this weight is a stack of MoE experts flattened to [E * N, K],
    # so the grouped GEMV can find row ``expert * expert_cols + n``.
    experts: int = 1
    expert_cols: int = 0

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.out_features, self.in_features))

    @property
    def dtype(self) -> torch.dtype:
        return self.compute_dtype

    @property
    def device(self) -> torch.device:
        return self.qweight.device

    @property
    def is_cuda(self) -> bool:
        return self.qweight.is_cuda

    def dim(self) -> int:
        return 2

    def size(self, index: int | None = None):
        if index is None:
            return self.shape
        return self.shape[index]

    def numel(self) -> int:
        return self.out_features * self.in_features

    def stored_bytes(self) -> int:
        total = self.qweight.numel() * self.qweight.element_size()
        for extra in (self.scales, self.zeros, self.channel_scale):
            if extra is not None:
                total += extra.numel() * extra.element_size()
        return total

    def _code(self) -> int:
        return _KIND_CODE[self.kind]

    def dequantize(self, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Unpack to a dense [out, in] tensor (prefill and reference path)."""
        dtype = out_dtype or self.compute_dtype
        ops = _ops()
        # The kernel writes bf16 or fp16 and nothing else, and its dtype argument
        # is a single flag, so asking it for fp32 used to hand back bf16 without
        # complaint. Other dtypes go through the reference unpack, which works in
        # fp32 throughout and is the only way to get an exact answer out of this.
        if ops is not None and dtype in (torch.bfloat16, torch.float16):
            try:
                return ops.dequant(
                    self.qweight,
                    self.scales,
                    self.zeros,
                    self.channel_scale,
                    self._code(),
                    self.group_size,
                    self.out_features,
                    self.in_features,
                    float(self.global_scale),
                    1 if dtype == torch.float16 else 0,
                )
            except Exception:
                pass
        return python_dequantize(self).to(dtype)

    def linear(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        return qlinear(x, self, bias)

    def expert_view(self, index: int) -> "QuantWeight":
        """One expert of a stacked block, as views (packing is per row)."""
        if self.expert_cols <= 0:
            raise ValueError("not a stacked expert weight")
        n = self.expert_cols
        lo, hi = index * n, (index + 1) * n
        return QuantWeight(
            kind=self.kind,
            qweight=self.qweight[lo:hi],
            scales=None if self.scales is None else self.scales[lo:hi],
            zeros=None if self.zeros is None else self.zeros[lo:hi],
            channel_scale=(
                None if self.channel_scale is None else self.channel_scale[lo:hi]
            ),
            global_scale=self.global_scale,
            out_features=n,
            in_features=self.in_features,
            group_size=self.group_size,
            compute_dtype=self.compute_dtype,
        )


def concat_quant_weights(parts: list[QuantWeight], *, free: bool = False) -> QuantWeight:
    """Join same-shaped packed weights row-wise into one ``[sum N, K]`` weight.

    Packing is per row and per group, so the bytes concatenate exactly — no
    re-encoding, no second dense copy. This is how independently packed MoE
    experts become the single block the grouped GEMV indexes into. ``free``
    releases each part as it is copied, for a streaming load.
    """
    if not parts:
        raise ValueError("nothing to concatenate")
    first = parts[0]
    for part in parts:
        if (
            part.kind != first.kind
            or part.in_features != first.in_features
            or part.out_features != first.out_features
            or part.group_size != first.group_size
        ):
            raise ValueError("expert blocks must be packed identically to concatenate")

    n = first.out_features
    if first.channel_scale is not None:
        channel = torch.cat([p.channel_scale.float() for p in parts])
    elif first.kind == "nvfp4":
        # NVFP4 folds each part's per-tensor global scale into a per-row scale,
        # since the joined block can only carry one global.
        channel = torch.cat(
            [
                torch.full(
                    (n,), float(p.global_scale), dtype=torch.float32, device=p.device
                )
                for p in parts
            ]
        )
    else:
        channel = None

    def join(field: str) -> torch.Tensor | None:
        if any(getattr(p, field) is None for p in parts):
            return None
        sample = getattr(first, field)
        out = torch.empty(
            (sample.shape[0] * len(parts), *sample.shape[1:]),
            dtype=sample.dtype,
            device=sample.device,
        )
        del sample
        at = 0
        for part in parts:
            value = getattr(part, field)
            out[at : at + value.shape[0]].copy_(value)
            at += value.shape[0]
            del value
            if free:
                # Release each part as it lands so a streaming load never holds
                # both the parts and the block.
                setattr(part, field, None)
        return out
    joined = QuantWeight(
        kind=first.kind,
        qweight=join("qweight"),
        scales=join("scales"),
        zeros=join("zeros"),
        channel_scale=channel,
        global_scale=first.global_scale,
        out_features=n * len(parts),
        in_features=first.in_features,
        group_size=first.group_size,
        compute_dtype=first.compute_dtype,
        experts=len(parts),
        expert_cols=n,
    )
    return joined


def python_dequantize(qw: QuantWeight) -> torch.Tensor:
    """Reference unpack in pure PyTorch; the parity target for the kernel."""
    n, k = qw.out_features, qw.in_features
    if qw.kind == "fp8":
        values = _fp8_e4m3_decode(qw.qweight.reshape(n, k))
        if qw.channel_scale is not None:
            return values * qw.channel_scale.float().reshape(n, 1)
        return values * float(qw.global_scale)

    packed = qw.qweight.reshape(n, k // 2).to(torch.int32)
    codes = torch.stack((packed & 0x0F, (packed >> 4) & 0x0F), dim=-1).reshape(n, k)
    groups = k // qw.group_size
    if qw.kind == "int4":
        assert qw.scales is not None
        value = codes.float().reshape(n, groups, qw.group_size) * qw.scales.float().reshape(
            n, groups, 1
        )
        if qw.zeros is not None:
            value = value + qw.zeros.float().reshape(n, groups, 1)
        return value.reshape(n, k)
    assert qw.scales is not None
    table = _FP4_E2M1.to(device=codes.device)
    value = table[codes.reshape(-1).long()].reshape(n, groups, qw.group_size)
    scale = _fp8_e4m3_decode(qw.scales.reshape(n, groups)).unsqueeze(-1)
    if qw.channel_scale is not None:
        scale = scale * qw.channel_scale.float().reshape(n, 1, 1)
    else:
        scale = scale * float(qw.global_scale)
    return (value * scale).reshape(n, k)


_FP4_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _fp8_e4m3_decode(bytes_: torch.Tensor) -> torch.Tensor:
    x = bytes_.to(torch.int32)
    sign = torch.where((x & 0x80) != 0, -1.0, 1.0)
    exp = ((x >> 3) & 0x0F).float()
    man = (x & 0x07).float()
    sub = man * 0.001953125
    norm = (1.0 + man / 8.0) * torch.pow(torch.tensor(2.0, device=x.device), exp - 7.0)
    return sign * torch.where(exp == 0, sub, norm)


# Midpoints between the eight E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}.
_FP4_BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def _fp4_e2m1_encode(values: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 nibble per element, without a 16-wide search tensor."""
    magnitude = torch.bucketize(
        values.abs(), _FP4_BOUNDARIES.to(device=values.device, dtype=values.dtype)
    )
    return magnitude + 8 * (values < 0).to(magnitude.dtype)


def _fp8_e4m3_encode(values: torch.Tensor) -> torch.Tensor:
    """Round FP32 values to E4M3 bytes."""
    return _fp8_e4m3_encode_signed(values.detach().float())


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """[N, K] codes in 0..15 → uint8 [N, K/2], even index in the low nibble."""
    codes = codes.to(torch.uint8)
    low = codes[:, 0::2]
    high = codes[:, 1::2]
    return (low | (high << 4)).contiguous()


def quantize_int4(
    weight: torch.Tensor, *, group_size: int = 128, compute_dtype: torch.dtype | None = None
) -> QuantWeight:
    """Group-wise affine 4-bit: min/max per group, ``w = q * scale + zero``."""
    n, k = weight.shape
    if k % group_size or group_size % 32:
        raise ValueError(f"int4 needs k % group_size == 0 and group % 32 == 0 ({k}, {group_size})")
    dtype = compute_dtype or (
        weight.dtype if weight.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    )
    groups = weight.float().reshape(n, k // group_size, group_size)
    lo = groups.amin(dim=-1, keepdim=True)
    hi = groups.amax(dim=-1, keepdim=True)
    scale = ((hi - lo) / 15.0).clamp_min(1e-8)
    # Round the zero offset through the storage dtype so dequant matches exactly.
    scale_q = scale.to(dtype).float()
    zero_q = lo.to(dtype).float()
    codes = ((groups - zero_q) / scale_q).round_().clamp_(0, 15)
    return QuantWeight(
        kind="int4",
        qweight=_pack_nibbles(codes.reshape(n, k)),
        scales=scale_q.reshape(n, k // group_size).to(dtype).contiguous(),
        zeros=zero_q.reshape(n, k // group_size).to(dtype).contiguous(),
        channel_scale=None,
        global_scale=1.0,
        out_features=n,
        in_features=k,
        group_size=group_size,
        compute_dtype=dtype,
    )


def quantize_nvfp4(
    weight: torch.Tensor, *, group_size: int = 16, compute_dtype: torch.dtype | None = None
) -> QuantWeight:
    """NVFP4: E2M1 values, one E4M3 scale per 16, one FP32 global scale."""
    n, k = weight.shape
    if k % group_size:
        raise ValueError(f"nvfp4 needs k % {group_size} == 0, got {k}")
    dtype = compute_dtype or (
        weight.dtype if weight.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    )
    groups = weight.float().reshape(n, k // group_size, group_size)
    amax = groups.abs().amax(dim=-1)
    # The global scale maps the largest block scale onto the top of E4M3 (448),
    # which is how NVIDIA's packers keep the per-block scales in range.
    global_amax = amax.max().clamp_min(1e-12)
    global_scale = float(global_amax / (6.0 * 448.0))
    block = (amax / (6.0 * global_scale)).clamp(1e-12, 448.0)
    block_codes = _fp8_e4m3_encode(block)
    block_real = _fp8_e4m3_decode(block_codes.to(torch.int32)) * global_scale
    normalized = groups / block_real.clamp_min(1e-12).unsqueeze(-1)
    codes = _fp4_e2m1_encode(normalized)
    return QuantWeight(
        kind="nvfp4",
        qweight=_pack_nibbles(codes.reshape(n, k)),
        scales=block_codes.reshape(n, k // group_size).contiguous(),
        zeros=None,
        channel_scale=None,
        global_scale=global_scale,
        out_features=n,
        in_features=k,
        group_size=group_size,
        compute_dtype=dtype,
    )


def quantize_fp8(
    weight: torch.Tensor,
    *,
    compute_dtype: torch.dtype | None = None,
    group_size: int | None = None,
) -> QuantWeight:
    """Per-row E4M3: near lossless, half the bytes of BF16.

    ``group_size`` is accepted and ignored — an FP8 row is its own group — so
    callers can pass the same arguments for every kind.
    """
    del group_size
    n, k = weight.shape
    dtype = compute_dtype or (
        weight.dtype if weight.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    )
    amax = weight.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / 448.0
    scaled = (weight.float() / scale).clamp(-448.0, 448.0)
    codes = _fp8_e4m3_encode_signed(scaled)
    return QuantWeight(
        kind="fp8",
        qweight=codes.reshape(n, k).contiguous(),
        scales=None,
        zeros=None,
        channel_scale=scale.reshape(n).float().contiguous(),
        global_scale=1.0,
        out_features=n,
        in_features=k,
        group_size=k,
        compute_dtype=dtype,
    )


def _fp8_e4m3_encode_signed(values: torch.Tensor) -> torch.Tensor:
    """Nearest E4M3 byte for signed values, via torch's own float8 cast."""
    try:
        return values.to(torch.float8_e4m3fn).view(torch.uint8)
    except Exception:
        # Chunked nearest-code search: the full [numel, 256] difference would be
        # 64 bytes per weight, which does not fit for a large checkpoint.
        table = _fp8_e4m3_decode(
            torch.arange(256, device=values.device, dtype=torch.int32)
        )
        table = torch.where(torch.isfinite(table), table, torch.full_like(table, 1e30))
        flat = values.reshape(-1)
        out = torch.empty(flat.numel(), dtype=torch.uint8, device=values.device)
        step = max(1, 1 << 20)
        for start in range(0, flat.numel(), step):
            chunk = flat[start : start + step]
            idx = (chunk.unsqueeze(1) - table.unsqueeze(0)).abs().argmin(dim=1)
            out[start : start + step] = idx.to(torch.uint8)
        return out.reshape(values.shape)


QUANTIZERS = {
    "int4": quantize_int4,
    "nvfp4": quantize_nvfp4,
    "fp8": quantize_fp8,
}


def quantize(weight: torch.Tensor, kind: str, **kwargs) -> QuantWeight:
    if kind not in QUANTIZERS:
        raise ValueError(f"unknown quantization {kind!r}; want one of {sorted(QUANTIZERS)}")
    return QUANTIZERS[kind](weight, **kwargs)


def bits_per_weight(kind: str) -> float:
    return _BITS[kind]


# Rows the kernel itself can hold: each lane keeps one accumulator per row, so
# past this the register file, not the algorithm, is the limit.
KERNEL_ROW_LIMIT = 32


def _max_fused_rows() -> int:
    """Rows the fused path keeps; INFER_QGEMV_MAX_ROWS to sweep the crossover."""
    try:
        rows = int(os.environ.get("INFER_QGEMV_MAX_ROWS", "32"))
    except ValueError:
        return KERNEL_ROW_LIMIT
    return max(0, min(rows, 4096))


def fused_qlinear(x: torch.Tensor, qw: "QuantWeight") -> torch.Tensor | None:
    """``x @ w.T`` straight from the packed bytes, or None for shapes it cannot do.

    Above the kernel's register budget this walks the rows in chunks, which reads
    the packed weight once per chunk. That is the same total arithmetic a wider
    kernel would do and a fraction of the traffic the unpack path pays, so where
    the two meet stays a question for measurement rather than a register count.
    """
    ops = _ops()
    if (
        ops is None
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.shape[-1] != qw.in_features
        or qw.in_features % 64 != 0
    ):
        return None
    flat = x.reshape(-1, x.shape[-1])
    rows = flat.shape[0]
    args = (
        qw.qweight, qw.scales, qw.zeros, qw.channel_scale, _KIND_CODE[qw.kind],
        qw.group_size, qw.out_features, qw.in_features, float(qw.global_scale),
    )
    try:
        if rows <= KERNEL_ROW_LIMIT:
            out = ops.qgemv(flat, *args)
        else:
            out = torch.empty(rows, qw.out_features, device=x.device, dtype=x.dtype)
            for start in range(0, rows, KERNEL_ROW_LIMIT):
                stop = min(start + KERNEL_ROW_LIMIT, rows)
                out[start:stop] = ops.qgemv(flat[start:stop], *args)
    except Exception:
        return None
    return out.reshape(*x.shape[:-1], qw.out_features)


def qlinear(
    x: torch.Tensor, qw: QuantWeight, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ w.T + bias`` against packed weights.

    Small row counts go through the fused GEMV, which never materializes the
    BF16 weight. Wide prefills unpack it once and hand the work to cuBLAS, where
    tensor cores dominate; the unpacked copy is transient and the allocator hands
    the same block back every layer.

    Where "small" ends is a real crossover, not a guess: the fused path pays
    ``2·M`` scalar FLOPs per weight element, the dequantize path pays about four
    extra bytes of traffic per weight element and then runs on tensor cores. On
    GB10 those meet in the low hundreds of rows. The cap sits well inside that
    at 32, which is also where a routed expert lands during a sparse prefill —
    512 tokens over 128 experts at top-8 is 32 rows each, so the MoE prefill
    stops dequantizing entirely.
    """
    rows = x.numel() // x.shape[-1]
    if qw.qweight.is_cuda and x.is_cuda and rows <= _max_fused_rows():
        out = fused_qlinear(x, qw)
        if out is not None:
            return out if bias is None else out + bias
    weight = qw.dequantize(out_dtype=x.dtype)
    return torch.nn.functional.linear(x, weight, bias)
