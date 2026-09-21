#!/usr/bin/env python3
"""Where does one decode step go? Launch count, CPU time, and top kernels."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.agent_api import load_engine  # noqa: E402


@torch.inference_mode()
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prefill", type=int, default=128)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--top", type=int, default=18)
    args = p.parse_args()

    engine = load_engine(args.model, device="cuda")
    model = engine.model
    vocab = int(model.config.vocab_size)
    ids = [(i * 7919) % max(vocab - 16, 8) + 8 for i in range(args.prefill)]
    tokens = torch.tensor([ids], dtype=torch.long, device=model.device)
    cache = model.make_cache(batch_size=1, device=model.device, dtype=model.dtype)
    model.forward(tokens, cache=cache, logits_to_keep=1)
    step = torch.tensor([[1]], dtype=torch.long, device=model.device)
    for _ in range(4):
        model.forward(step, cache=cache, logits_to_keep=1)
    torch.cuda.synchronize()

    # CPU-side submit time vs total wall time: if they match, we are launch bound.
    t0 = time.perf_counter()
    for _ in range(args.steps):
        model.forward(step, cache=cache, logits_to_keep=1)
    t_submit = time.perf_counter() - t0
    torch.cuda.synchronize()
    t_total = time.perf_counter() - t0

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            model.forward(step, cache=cache, logits_to_keep=1)
        torch.cuda.synchronize()

    events = [e for e in prof.key_averages() if e.device_type.name == "CUDA" or e.self_device_time_total]
    events.sort(key=lambda e: e.self_device_time_total, reverse=True)
    launches = sum(e.count for e in events if e.self_device_time_total > 0)
    gpu_us = sum(e.self_device_time_total for e in events)

    print(f"model            {args.model.name}")
    print(f"layers           {model.config.num_hidden_layers}")
    print(f"steps            {args.steps}")
    print(f"cpu submit       {1000 * t_submit / args.steps:.3f} ms/step")
    print(f"wall total       {1000 * t_total / args.steps:.3f} ms/step")
    print(f"gpu busy         {gpu_us / 1000 / args.steps:.3f} ms/step")
    print(f"kernel launches  {launches / args.steps:.1f} per step")
    print()
    print(f"{'kernel':<62}{'ms/step':>9}{'count/step':>12}")
    for e in events[: args.top]:
        if e.self_device_time_total <= 0:
            continue
        name = e.key[:60]
        print(
            f"{name:<62}{e.self_device_time_total / 1000 / args.steps:>9.3f}"
            f"{e.count / args.steps:>12.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
