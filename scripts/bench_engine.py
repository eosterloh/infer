#!/usr/bin/env python3
"""Prefill / decode throughput plus a greedy fingerprint for one model folder.

Two jobs in one pass so a kernel change can be judged on speed and on output:

  speed        prefill tok/s and decode tok/s, median over repeats
  fingerprint  the greedy token ids a fixed prompt produces, and the last
               position's top logits, so "faster" can be checked against
               "same answer"

Results append to a JSONL file; ``--tag`` labels the run (baseline, kernels, …).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.agent_api import load_engine  # noqa: E402

try:  # The pre-kernel revision has no extension; this script benchmarks it too.
    from engine import kernels
except ImportError:
    kernels = None

FINGERPRINT_PROMPT = "The capital of France is Paris, and the capital of Italy is"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def _fresh_prefill(model, tokens: torch.Tensor) -> float:
    cache = model.make_cache(batch_size=1, device=model.device, dtype=model.dtype)
    _sync(model.device)
    t0 = time.perf_counter()
    model.forward(tokens, cache=cache)
    _sync(model.device)
    return time.perf_counter() - t0


@torch.inference_mode()
def _decode_loop(model, tokens: torch.Tensor, steps: int, graph: bool = False) -> float:
    cache = model.make_cache(batch_size=1, device=model.device, dtype=model.dtype)
    logits = model.forward(tokens, cache=cache)
    next_id = int(torch.argmax(logits[0, -1, :]).item())
    decoder = None
    if graph:
        from engine.graph import GraphDecoder

        decoder = GraphDecoder.create(
            model, cache, length=cache.seq_len(), budget=steps + 1
        )
        if decoder is not None and not decoder.capture(next_id):
            decoder = None
        if decoder is None:
            raise SystemExit("graph capture failed; rerun without --graph")
    _sync(model.device)
    t0 = time.perf_counter()
    if decoder is not None:
        for _ in range(steps):
            next_id = decoder.step()
    else:
        for _ in range(steps):
            step = torch.tensor([[next_id]], dtype=torch.long, device=model.device)
            logits = model.forward(step, cache=cache)
            next_id = int(torch.argmax(logits[0, -1, :]).item())
    _sync(model.device)
    elapsed = time.perf_counter() - t0
    if decoder is not None:
        decoder.release()
    return elapsed


@torch.inference_mode()
def fingerprint(engine, new_tokens: int) -> dict:
    """Greedy ids + last-step top logits from a fixed prompt."""
    model, tokenizer = engine.model, engine.tokenizer
    ids = tokenizer.encode(FINGERPRINT_PROMPT, add_special_tokens=True)
    tokens = torch.tensor([ids], dtype=torch.long, device=model.device)
    cache = model.make_cache(batch_size=1, device=model.device, dtype=model.dtype)
    logits = model.forward(tokens, cache=cache)
    top = torch.topk(logits[0, -1, :].float(), k=5)
    out = [int(torch.argmax(logits[0, -1, :]).item())]
    for _ in range(new_tokens - 1):
        step = torch.tensor([[out[-1]]], dtype=torch.long, device=model.device)
        logits = model.forward(step, cache=cache)
        out.append(int(torch.argmax(logits[0, -1, :]).item()))
    return {
        "prompt": FINGERPRINT_PROMPT,
        "prompt_ids": ids,
        "greedy_ids": out,
        "text": tokenizer.decode(out, skip_special_tokens=False),
        "top5_ids": [int(i) for i in top.indices.tolist()],
        "top5_logits": [round(float(v), 4) for v in top.values.tolist()],
    }


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prefill", type=int, default=512)
    p.add_argument("--decode", type=int, default=32)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default=None)
    p.add_argument("--tag", default="run")
    p.add_argument("--quant", default=None, choices=["int4", "nvfp4", "fp8"])
    p.add_argument("--graph", action="store_true", help="time the captured decode step")
    p.add_argument("--out", type=Path, default=ROOT / "bench" / "results.jsonl")
    p.add_argument("--fingerprint-tokens", type=int, default=16)
    p.add_argument("--skip-fingerprint", action="store_true")
    args = p.parse_args()

    if args.graph:
        import os

        os.environ["INFER_CUDA_GRAPH"] = "1"

    t_load = time.perf_counter()
    extra = {"quant": args.quant} if args.quant else {}
    engine = load_engine(args.model, device=args.device, dtype=args.dtype, **extra)
    load_s = time.perf_counter() - t_load
    model = engine.model

    # Deterministic token ids that exist in every vocab.
    vocab = int(model.config.vocab_size)
    ids = [(i * 7919) % max(vocab - 16, 8) + 8 for i in range(args.prefill)]
    tokens = torch.tensor([ids], dtype=torch.long, device=model.device)

    for _ in range(args.warmup):
        _fresh_prefill(model, tokens)
        _decode_loop(model, tokens[:, :8], 2)

    prefill_times = [_fresh_prefill(model, tokens) for _ in range(args.reps)]
    decode_times = [
        _decode_loop(model, tokens, args.decode, graph=args.graph)
        for _ in range(args.reps)
    ]

    prefill_s = statistics.median(prefill_times)
    decode_s = statistics.median(decode_times)
    record = {
        "tag": args.tag,
        "git": _git_rev(),
        "host": platform.node(),
        "model": args.model.name,
        "model_dir": str(args.model),
        "recipe": model.config.recipe_id,
        "params": getattr(engine, "n_params", None),
        "quant": args.quant,
        "graph": bool(args.graph),
        "kernels": bool(kernels is not None and kernels.available()),
        "dtype": str(model.dtype),
        "device": str(model.device),
        "load_seconds": round(load_s, 2),
        "prefill_tokens": args.prefill,
        "decode_tokens": args.decode,
        "prefill_seconds": round(prefill_s, 5),
        "decode_seconds": round(decode_s, 5),
        "prefill_tok_s": round(args.prefill / prefill_s, 2),
        "decode_tok_s": round(args.decode / decode_s, 2),
        "ms_per_decode_token": round(1000 * decode_s / args.decode, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if torch.cuda.is_available():
        record["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    if getattr(engine, "quantization", None):
        record["quantization"] = engine.quantization
    if not args.skip_fingerprint:
        record["fingerprint"] = fingerprint(engine, args.fingerprint_tokens)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
