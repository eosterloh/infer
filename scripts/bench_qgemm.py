#!/usr/bin/env python3
"""Where does a packed weight stop wanting the fused kernel?

Two ways to multiply by a quantized weight:

  fused      the packed GEMV reads the bytes once and dequantizes in registers.
             Bandwidth-optimal, but the arithmetic runs on plain FP32 FMAs, so
             the cost grows with every row of activations.
  dequantize unpack the whole weight into a BF16 scratch and hand it to cuBLAS.
             Costs about four extra bytes of traffic per weight element — write
             the scratch, read it back — and then runs on tensor cores.

One row favors fused by a wide margin; a thousand rows favor cuBLAS, because the
dequantize tax is paid once while the tensor-core advantage scales with rows.
The crossover is a property of this machine, not something to guess at, and it is
what INFER_QGEMV_MAX_ROWS should be set to. This finds it, and prints the rows
per expert a sparse prefill actually produces so the answer can be read against
the shapes that matter.

    scripts/bench_qgemm.py --kind nvfp4
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

from engine.kernels import load_extension  # noqa: E402
from engine.quantize import group_for_quant  # noqa: E402
from engine.qweight import fused_qlinear, quantize  # noqa: E402

SHAPES = [
    ("llama8b q", 4096, 4096),
    ("llama8b gate/up", 4096, 14336),
    ("llama8b down", 14336, 4096),
    ("nemotron expert up", 4480, 2 * 768),
    ("nemotron expert down", 768, 4480),
    ("qwen3 lm_head", 4096, 151936),
]
ROWS = [1, 2, 4, 8, 16, 32, 48, 64, 128, 256, 512]
# Enough to exercise every branch on a laptop; the timings mean nothing there.
SELF_TEST_SHAPES = [("tiny q", 256, 128), ("tiny down", 128, 256)]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def timed(fn, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters


def _fused(x: torch.Tensor, qw) -> torch.Tensor | None:
    """The packed path exactly as production runs it, row cap ignored.

    Past the kernel's 32-row register budget this chunks, so the sweep can ask
    what 64 or 512 rows would cost instead of stopping where the kernel does.
    """
    return fused_qlinear(x, qw)


def _dequant_then_blas(x: torch.Tensor, qw) -> torch.Tensor:
    return F.linear(x, qw.dequantize(out_dtype=x.dtype))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="nvfp4", choices=["int4", "nvfp4", "fp8"])
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--rows", type=int, nargs="*", default=None)
    p.add_argument(
        "--self-test",
        action="store_true",
        help="tiny shapes on whatever device exists, to check this script runs",
    )
    args = p.parse_args()

    if DEVICE == "cpu" and not args.self_test:
        print("no CUDA device; this measures GB10 (use --self-test to check the script)")
        return 1
    name = torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu"
    print(f"device     {name}")
    print(f"extension  {'loaded' if load_extension() else 'MISSING'}")
    print(f"kind       {args.kind} (group {args.group})")
    print()

    rows_list = args.rows or ([1, 4, 32] if args.self_test else ROWS)
    shapes = SELF_TEST_SHAPES if args.self_test else SHAPES
    if args.self_test:
        args.iters = 2
    crossovers = []
    for name, k, n in shapes:
        # The same group the loader would pick, so this measures production.
        group = group_for_quant(args.kind, k, args.group)
        if group is None:
            print(f"{name}: cannot pack {n}x{k} as {args.kind}")
            continue
        w = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16) * 0.02
        qw = quantize(w, kind=args.kind, group_size=group)
        bf16_bytes = w.numel() * 2
        packed_bytes = qw.stored_bytes()
        print(f"{name}  [{n} x {k}]  group {group}  bf16 {bf16_bytes / 1e6:.0f} MB -> "
              f"packed {packed_bytes / 1e6:.0f} MB")
        print(f"  {'rows':>5}{'fused ms':>11}{'dequant ms':>12}{'bf16 ms':>10}"
              f"{'fused GB/s':>12}{'winner':>10}")
        wins: list[tuple[int, bool]] = []
        for rows in rows_list:
            x = torch.randn(rows, k, device=DEVICE, dtype=torch.bfloat16)
            t_blas = timed(lambda: F.linear(x, w), args.iters)
            t_deq = timed(lambda: _dequant_then_blas(x, qw), args.iters)
            fused = _fused(x, qw)
            if fused is None:
                t_fused = float("nan")
                gbs = float("nan")
                winner = "dequant"
            else:
                ref = _dequant_then_blas(x, qw)
                err = (fused.float() - ref.float()).abs().max().item()
                scale = ref.float().abs().max().item() or 1.0
                assert err / scale < 0.05, f"{name} rows={rows} rel err {err / scale:.3f}"
                t_fused = timed(lambda: _fused(x, qw), args.iters)
                gbs = packed_bytes / t_fused
                # A 3% edge is not a reason to take a different code path.
                won = t_deq > t_fused * 1.03
                wins.append((rows, won))
                winner = "fused" if won else "dequant"
            print(f"  {rows:>5}{1000 * t_fused:>11.3f}{1000 * t_deq:>12.3f}"
                  f"{1000 * t_blas:>10.3f}{gbs / 1e9:>12.1f}{winner:>10}")
        # Read the prefix, not the first loss: a single noisy row in the middle
        # should not be reported as the crossover.
        holds = 0
        for rows, won in wins:
            if not won:
                break
            holds = rows
        crossovers.append((name, holds))
        del qw, w
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print()
    if crossovers:
        for name, rows in crossovers:
            print(f"crossover  {name}: fused holds through {rows} rows")
        held = sorted(rows for _, rows in crossovers)
        worst = held[0]
        # The median, not the minimum: these shapes are not equally hot. The
        # narrow expert projections are the ones that give up early, and a sparse
        # prefill already leaves the grouped GEMV before it reaches those row
        # counts (INFER_MOE_FUSED_ROWS_PER_EXPERT), so cutting every wide
        # projection down to the narrowest shape's crossover costs more than it
        # saves. The minimum is printed so the trade is visible.
        cap = max(1, held[len(held) // 2])
        print(f"\nset INFER_QGEMV_MAX_ROWS={cap}   (narrowest shape gave up at {worst})")
        if worst >= max(rows_list):
            print("fused won everywhere measured; raise --rows to find the edge")
    print(
        "\nrows per expert in a sparse prefill = tokens * top_k / n_experts; "
        "512 tokens at top-8 over 128 experts is 32"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
