#!/usr/bin/env python3
"""Where does one decode step go? Launch count, CPU time, top kernels, roofline.

Batch-1 decode reads the weights it needs and does almost no arithmetic with
them, so the honest ceiling is bandwidth: bytes the step must touch divided by
what the bus delivers. The roofline line turns "12 ms/token" into "62% of what
this machine can do", which is the number that says whether a kernel is worth
writing or already finished.
"""

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
from engine.layers.moe import EXPERT_STACK_KEY  # noqa: E402


def _bytes_of(value) -> int:
    stored = getattr(value, "stored_bytes", None)
    if callable(stored):
        return int(stored())
    return int(value.numel() * value.element_size())


def _resident_weight_bytes(model, engine) -> int:
    """Weight bytes one decode step has to read.

    Not the parameter count: the embedding contributes one row, and a sparse
    model reads only the experts it routed to, which on a 30B A3B checkpoint is
    a tenth of the weights.
    """
    config = model.config
    experts_per_tok = int(getattr(config, "num_experts_per_tok", 0) or 0)
    n_experts = int(getattr(config, "num_experts", 0) or 0)
    sparse = experts_per_tok / n_experts if experts_per_tok and n_experts else 1.0

    total = 0.0
    for name, value in model.weights.items():
        if name.startswith("_") or not hasattr(value, "numel"):
            continue
        if name == "embed.weight":
            continue  # one row per token
        share = sparse if ".experts." in name else 1.0
        total += _bytes_of(value) * share

    for block in (model.weights.get(EXPERT_STACK_KEY) or {}).values():
        for value in block.values():
            held = int(getattr(value, "experts", 0) or 0)
            if not held and getattr(value, "dim", None) and value.dim() == 3:
                held = int(value.shape[0])
            share = (experts_per_tok / held) if (held and experts_per_tok) else 1.0
            total += _bytes_of(value) * min(share, 1.0)

    if "lm_head.weight" not in model.weights and "embed.weight" in model.weights:
        total += _bytes_of(model.weights["embed.weight"])  # tied, still read in full
    return int(total)


def _kv_bytes(model, cache) -> int:
    """Cache bytes the step reads: the live KV window plus any mixer state."""
    kv = getattr(cache, "kv", cache)
    length = kv.seq_len() if hasattr(kv, "seq_len") else 0
    total = 0
    for buf in list(getattr(kv, "_k_buf", [])) + list(getattr(kv, "_v_buf", [])):
        if buf is None:
            continue
        per_position = buf.shape[0] * buf.shape[1] * buf.shape[3] * buf.element_size()
        total += length * per_position
    states = list(getattr(cache, "conv_states", []) or [])
    states += list(getattr(cache, "ssm_states", []) or [])
    for state in states:
        if state is not None:
            total += _bytes_of(state)
    return total


@torch.inference_mode()
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prefill", type=int, default=128)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--top", type=int, default=18)
    p.add_argument("--quant", default=None, choices=["int4", "nvfp4", "fp8"])
    p.add_argument(
        "--bandwidth-gbs",
        type=float,
        default=273.0,
        help="peak memory bandwidth; GB10 is 273 GB/s",
    )
    args = p.parse_args()

    engine = load_engine(args.model, device="cuda", quant=args.quant)
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

    # What the step is obliged to read: the weights it touches plus the cache.
    # For a sparse model only the routed experts are read, so the dense param
    # count would overstate the floor by an order of magnitude.
    weight_bytes = _resident_weight_bytes(model, engine)
    kv_bytes = _kv_bytes(model, cache)
    per_step = weight_bytes + kv_bytes
    ms = 1000 * t_total / args.steps
    floor_ms = 1000 * per_step / (args.bandwidth_gbs * 1e9)

    print(f"model            {args.model.name}")
    print(f"layers           {model.config.num_hidden_layers}")
    print(f"quant            {args.quant or 'none'}")
    print(f"steps            {args.steps}")
    print(f"cpu submit       {1000 * t_submit / args.steps:.3f} ms/step")
    print(f"wall total       {ms:.3f} ms/step")
    print(f"gpu busy         {gpu_us / 1000 / args.steps:.3f} ms/step")
    print(f"kernel launches  {launches / args.steps:.1f} per step")
    print(f"bytes to read    {per_step / 1e9:.3f} GB/step "
          f"({weight_bytes / 1e9:.3f} weights + {kv_bytes / 1e9:.3f} cache)")
    print(f"roofline         {floor_ms:.3f} ms/step at {args.bandwidth_gbs:.0f} GB/s")
    print(f"achieved         {per_step / (ms / 1000) / 1e9:.1f} GB/s "
          f"= {100 * floor_ms / ms:.0f}% of peak")
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
