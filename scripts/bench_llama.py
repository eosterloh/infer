#!/usr/bin/env python3
"""WikiText-2 perplexity + generate throughput for the DIY engine.

Matches llama.cpp `llama-perplexity -c 512 --ppl-stride 512` as closely as
possible: non-overlapping 512-token windows, teacher-forced NLL, no chat
template. Writes JSON so we can compare with a llama.cpp run of the same file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.agent_api import load_engine  # noqa: E402


def _tokenize_corpus(tokenizer, text: str) -> list[int]:
    # PPL is a completion metric — do not wrap in the instruct chat template.
    return tokenizer.encode(text, add_special_tokens=True)


@torch.inference_mode()
def perplexity(
    model,
    ids: list[int],
    *,
    ctx: int,
    stride: int,
    max_tokens: int | None,
    score_last: int | None = None,
) -> dict:
    """Teacher-forced NLL over windows.

    If score_last is set (llama.cpp --ppl-stride mode), each window is `ctx`
    tokens and we only score the last `score_last` next-token predictions.
    llama.cpp with `-c 512 --ppl-stride 512` bumps ctx to 768 and scores 512.
    """
    if max_tokens is not None:
        ids = ids[: max_tokens]
    if len(ids) < 2:
        raise ValueError("corpus too short")

    device = model.device
    nll_sum = 0.0
    token_count = 0
    windows = 0
    t0 = time.perf_counter()

    for start in range(0, len(ids) - 1, stride):
        window = ids[start : start + ctx]
        if len(window) < 2:
            break
        tokens = torch.tensor([window], dtype=torch.long, device=device)
        logits = model.forward(tokens, cache=None)
        pred = logits[0, :-1, :].float()
        target = tokens[0, 1:]
        if score_last is not None:
            keep = min(score_last, pred.shape[0])
            pred = pred[-keep:]
            target = target[-keep:]
        nll = F.cross_entropy(pred, target, reduction="sum")
        nll_sum += float(nll.item())
        token_count += int(target.numel())
        windows += 1

    elapsed = time.perf_counter() - t0
    ppl = math.exp(nll_sum / token_count) if token_count else float("inf")
    return {
        "perplexity": ppl,
        "nll_nats": nll_sum,
        "token_count": token_count,
        "windows": windows,
        "ctx": ctx,
        "stride": stride,
        "score_last": score_last,
        "seconds": elapsed,
        "tokens_per_sec_prefill": token_count / elapsed if elapsed else 0.0,
    }
    if max_tokens is not None:
        ids = ids[: max_tokens]
    if len(ids) < 2:
        raise ValueError("corpus too short")

    device = model.device
    nll_sum = 0.0
    token_count = 0
    windows = 0
    t0 = time.perf_counter()

    for start in range(0, len(ids) - 1, stride):
        window = ids[start : start + ctx]
        if len(window) < 2:
            break
        tokens = torch.tensor([window], dtype=torch.long, device=device)
        logits = model.forward(tokens, cache=None)
        # Predict window[1:] from logits[:-1]
        pred = logits[0, :-1, :].float()
        target = tokens[0, 1:]
        nll = F.cross_entropy(pred, target, reduction="sum")
        nll_sum += float(nll.item())
        token_count += int(target.numel())
        windows += 1

    elapsed = time.perf_counter() - t0
    ppl = math.exp(nll_sum / token_count) if token_count else float("inf")
    return {
        "perplexity": ppl,
        "nll_nats": nll_sum,
        "token_count": token_count,
        "windows": windows,
        "ctx": ctx,
        "stride": stride,
        "seconds": elapsed,
        "tokens_per_sec_prefill": token_count / elapsed if elapsed else 0.0,
    }


@torch.inference_mode()
def generate_throughput(
    model,
    tokenizer,
    prompt: str | None,
    *,
    max_new_tokens: int,
    prompt_ids: list[int] | None = None,
) -> dict:
    if prompt_ids is None:
        if prompt is None:
            raise ValueError("prompt or prompt_ids required")
        ids = tokenizer.encode(prompt, add_special_tokens=True)
    else:
        ids = list(prompt_ids)
    device = model.device
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    cache = model.make_cache(batch_size=1, device=device, dtype=model.dtype)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model.forward(tokens, cache=cache)
    next_id = int(torch.argmax(logits[0, -1, :]).item())
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_prefill = time.perf_counter()
    n_gen = 1
    for _ in range(max_new_tokens - 1):
        step = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits = model.forward(step, cache=cache)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        n_gen += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    prefill_s = t_prefill - t0
    decode_s = t_end - t_prefill
    return {
        "prompt_tokens": len(ids),
        "new_tokens": n_gen,
        "prefill_seconds": prefill_s,
        "decode_seconds": decode_s,
        "prefill_tok_s": len(ids) / prefill_s if prefill_s else 0.0,
        "decode_tok_s": (n_gen - 1) / decode_s if decode_s and n_gen > 1 else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--wiki", type=Path, required=True, help="wiki.test.raw")
    p.add_argument("--ctx", type=int, default=768, help="window size (llama.cpp strided PPL uses 768)")
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--score-last", type=int, default=512, help="only score last N tokens of each window")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--gen-tokens", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    text = args.wiki.read_text(encoding="utf-8")
    print(f"loading {args.model} …", flush=True)
    eng = load_engine(args.model, device=args.device)
    ids = _tokenize_corpus(eng.tokenizer, text)
    print(f"corpus_tokens={len(ids)} using_first={min(len(ids), args.max_tokens)}", flush=True)

    ppl = perplexity(
        eng.model,
        ids,
        ctx=args.ctx,
        stride=args.stride,
        max_tokens=args.max_tokens,
        score_last=args.score_last if args.score_last > 0 else None,
    )
    gen_prompt_len = 512
    # Same 512-token prefill llama-bench uses by default (first corpus window).
    gen = generate_throughput(
        eng.model,
        eng.tokenizer,
        prompt=None,
        max_new_tokens=args.gen_tokens,
        prompt_ids=ids[:gen_prompt_len],
    )

    result = {
        "engine": "infer-diy",
        "model": str(args.model),
        "wiki": str(args.wiki),
        "perplexity": ppl,
        "generate": gen,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
