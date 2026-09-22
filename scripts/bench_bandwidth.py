#!/usr/bin/env python3
"""What the memory bus actually gives, which is what decode is limited by.

At batch one every projection reads a weight and multiplies it by a vector, so a
decode step moves the whole model through the bus and does almost no arithmetic
per byte. The ceiling on tokens per second is therefore bytes-per-token over
achievable bandwidth, and "achievable" is not the number on the box: the GB10 is
advertised at 273 GB/s and a dependent read stream does not reach that. Measuring
it is the only way the benchmark report can say whether a kernel that gained
nothing was badly written or already done.

Run it on the machine you are reporting about; it prints the figure to pass to
bench_report.py --bandwidth-gbs.
"""

from __future__ import annotations

import argparse
import json

import torch


def measure(gb: float, repeat: int) -> dict[str, float]:
    """Read-dominated and copy bandwidth over a buffer too big to cache."""
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    n = int(gb * 1e9) // 2  # bf16
    src = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    bytes_read = src.numel() * src.element_size()

    def timed(fn, moved: int) -> float:
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        seconds = start.elapsed_time(end) / 1e3 / repeat
        return moved / seconds / 1e9

    # A sum is one pass of reads and a register accumulation, which is the closest
    # simple stand-in for a GEMV's traffic. The copy moves twice the bytes and is
    # reported because it is the usual STREAM-style figure.
    read = timed(lambda: torch.sum(src), bytes_read)
    copy = timed(lambda: dst.copy_(src), bytes_read * 2)
    return {"read_gbs": round(read, 1), "copy_gbs": round(copy, 1), "buffer_gb": gb}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gb", type=float, default=8.0, help="buffer size, GB")
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    got = measure(args.gb, args.repeat)
    if args.json:
        print(json.dumps(got))
    else:
        print(f"device      {torch.cuda.get_device_name(0)}")
        print(f"buffer      {got['buffer_gb']:.0f} GB bf16")
        print(f"read        {got['read_gbs']} GB/s   (one pass, accumulate)")
        print(f"copy        {got['copy_gbs']} GB/s   (read + write)")
        print(f"\npass --bandwidth-gbs {got['read_gbs']} to bench_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
