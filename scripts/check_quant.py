#!/usr/bin/env python3
"""Parity and bandwidth checks for the quantized GEMV and MoE kernels."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.kernels import moe_combine, moe_gemv  # noqa: E402
from engine.qweight import python_dequantize, qlinear, quantize  # noqa: E402


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9))


def check_dequant(kind: str, n: int, k: int, device: str) -> None:
    torch.manual_seed(0)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.02
    qw = quantize(w, kind)
    kernel = qw.dequantize()
    reference = python_dequantize(qw).to(torch.bfloat16)
    err = rel_err(kernel, reference)
    round_trip = rel_err(reference, w)
    print(
        f"  dequant {kind:6s} [{n},{k}] kernel-vs-python {err:.2e}  "
        f"quantization error {round_trip:.4f}"
    )
    assert err < 3e-3, f"{kind} dequant mismatch: {err}"


def check_qgemv(kind: str, n: int, k: int, device: str, rows: int = 1) -> None:
    torch.manual_seed(0)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.02
    x = torch.randn(rows, k, device=device, dtype=torch.bfloat16)
    qw = quantize(w, kind)
    got = qlinear(x, qw)
    want = torch.nn.functional.linear(x.float(), python_dequantize(qw)).to(torch.bfloat16)
    err = rel_err(got, want)
    dense = torch.nn.functional.linear(x, w)
    print(
        f"  qgemv   {kind:6s} [{rows},{k}]x[{n},{k}] vs-dequant {err:.2e}  "
        f"vs-bf16 {rel_err(got, dense):.4f}"
    )
    assert err < 5e-3, f"{kind} qgemv mismatch: {err}"


def bench_qgemv(kind: str | None, n: int, k: int, device: str, iters: int = 200) -> float:
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.02
    x = torch.randn(1, k, device=device, dtype=torch.bfloat16)
    if kind is None:
        run = lambda: torch.nn.functional.linear(x, w)  # noqa: E731
        moved = n * k * 2
    else:
        qw = quantize(w, kind)
        run = lambda: qlinear(x, qw)  # noqa: E731
        moved = qw.stored_bytes()
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / iters
    gbs = moved / per / 1e9
    label = kind or "bf16"
    print(f"  {label:6s} [{n},{k}] {per * 1e6:8.1f} us  {gbs:7.1f} GB/s  {moved / 1e6:7.1f} MB")
    return per


def check_moe(device: str) -> None:
    torch.manual_seed(0)
    experts, inter, hidden, topk, tokens = 16, 256, 320, 4, 3
    up = torch.randn(experts, inter, hidden, device=device, dtype=torch.bfloat16) * 0.05
    down = torch.randn(experts, hidden, inter, device=device, dtype=torch.bfloat16) * 0.05
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    idx = torch.randint(0, experts, (tokens, topk), device=device)
    wts = torch.rand(tokens, topk, device=device, dtype=torch.bfloat16)

    row_expert = idx.reshape(-1).to(torch.int32)
    row_input = torch.arange(tokens, device=device, dtype=torch.int32).repeat_interleave(topk)
    h = moe_gemv(x, up, row_expert, row_input)
    assert h is not None, "moe_gemv unavailable"
    h = torch.square(torch.relu(h))
    out = moe_gemv(h.contiguous(), down, row_expert, None)
    got = moe_combine(out, wts, topk)

    want = torch.zeros(tokens, hidden, device=device, dtype=torch.float32)
    for t in range(tokens):
        for j in range(topk):
            e = int(idx[t, j])
            hv = torch.square(torch.relu(x[t].float() @ up[e].float().T))
            want[t] += float(wts[t, j]) * (hv @ down[e].float().T)
    err = rel_err(got, want)
    print(f"  moe     [{tokens} tok, top{topk}, {experts} experts] vs-reference {err:.2e}")
    assert err < 2e-2, f"moe mismatch: {err}"


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print("parity:")
    for kind in ("int4", "nvfp4", "fp8"):
        check_dequant(kind, 512, 256, device)
        check_qgemv(kind, 512, 256, device)
        check_qgemv(kind, 4096, 2048, device, rows=4)
    if device == "cuda":
        check_moe(device)
        print("bandwidth (batch 1):")
        for n, k in ((4096, 4096), (11008, 4096), (4096, 11008), (128256, 2048)):
            for kind in (None, "fp8", "int4", "nvfp4"):
                bench_qgemv(kind, n, k, device)
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
