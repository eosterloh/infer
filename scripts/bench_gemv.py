#!/usr/bin/env python3
"""How close to memory peak does a decode GEMV get?

Decode with batch 1 is a stream of matrix-vector products, so the only number
that matters is achieved bytes/second against the machine's peak. This compares
a large copy (the practical ceiling), cuBLAS through F.linear, and the engine's
own kernel on the shapes real checkpoints use.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.kernels import gemv, load_extension  # noqa: E402


def timed(fn, iters: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def peak_copy_bandwidth() -> float:
    n = 512 * 1024 * 1024 // 2  # 512 MB of bf16
    src = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    seconds = timed(lambda: dst.copy_(src), iters=20)
    return 2 * src.numel() * 2 / seconds


SHAPES = [
    # (name, in_features, out_features)
    ("llama1b q_proj", 2048, 2048),
    ("llama1b gate/up", 2048, 8192),
    ("llama1b down", 8192, 2048),
    ("llama1b lm_head", 2048, 128256),
    ("llama8b q_proj", 4096, 4096),
    ("llama8b gate/up", 4096, 14336),
    ("llama8b down", 14336, 4096),
    ("nemotron shared", 4480, 8960),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    ext = load_extension()
    print(f"device       {torch.cuda.get_device_name()}")
    print(f"extension    {'loaded' if ext else 'missing'}")
    peak = peak_copy_bandwidth()
    print(f"copy peak    {peak / 1e9:.1f} GB/s (read+write)\n")

    header = f"{'shape':<20}{'K':>7}{'N':>8}{'cuBLAS':>11}{'kernel':>11}{'best %peak':>12}"
    print(header)
    for name, k, n in SHAPES:
        x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        bytes_moved = w.numel() * 2
        t_blas = timed(lambda: F.linear(x, w), iters=args.iters)
        blas_bw = bytes_moved / t_blas
        try:
            out = gemv(x, w)
            assert out is not None
            t_mine = timed(lambda: gemv(x, w), iters=args.iters)
            mine_bw = bytes_moved / t_mine
            err = (out.float() - F.linear(x, w).float()).abs().max().item()
        except Exception as exc:
            mine_bw = 0.0
            err = float("nan")
            print(f"  kernel unavailable: {exc}")
        best = max(blas_bw, mine_bw)
        print(
            f"{name:<20}{k:>7}{n:>8}{blas_bw / 1e9:>10.1f}G"
            f"{mine_bw / 1e9:>10.1f}G{100 * best / peak:>11.0f}%"
            + (f"   maxerr {err:.4f}" if mine_bw else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
